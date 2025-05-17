from typing import Union
from tensordict.tensordict import TensorDict
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CONCAT.embedding import LotStageEmbedding
from model.CONCAT.context import loadport_context, pm_context, single_armed_robot_context, dual_armed_robot_context
from model.CONCAT.ncts_model_sub import MixedScore_MultiHeadAttention, AddAndInstanceNormalization, FeedForward
from model.CONCAT.utils import gather_by_index


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


# ================================= Model =====================================
class CONCATNet(nn.Module):
    def __init__(self, env, **params):
        super().__init__()
        self.params = params
        self.encoder = Encoder(**params)
        self.decoder = Decoder(env, **params)

    def encoding(self, env, state):
        # get init embedding vector
        self.row_embed, self.col_embed, self.cost_mat = self.encoder.init_embedding(env, state)

        # get encoded embedding vector of row(=wafer), col(=loc)
        self.row_embed, self.col_embed =\
              self.encoder(self.row_embed, self.col_embed, self.cost_mat)

    def decoding(self, env, state):
        # get probability of action through decoder
        # --------------------------------------------------------
        prob = self.decoder(self.row_embed, self.col_embed, env, state)
        # shape (batch, action_cnt)

        # training mode action selection
        # --------------------------------------------------------
        if self.training or self.params['eval_type'] == 'softmax':
            while True:
                selected = torch.multinomial(prob, num_samples=1).squeeze(1)
                # shape: (batch,)
                selected_prob = prob[state.batch_idx, selected]
                # shape: (batch, )
                selected_prob[state.done] = 1
                # to fix pytorch.multinomial bug on selecting 0 probability elements
                if (selected_prob != 0).all(): break


        # evaluate mode action selection (greedy selection)
        # --------------------------------------------------------
        else:
            selected = prob.argmax(dim=1)
            # shape: (batch, pomo)
            selected_prob = prob[state.batch_idx, selected]
            # shape: (batch,)
            selected_prob[state.done] = 1

        return selected, selected_prob

    def forward(self, env, state):
        selected, prob = self.decoding(env, state)

        return selected, prob

# ================================= Encoder =====================================
class Encoder(nn.Module):
    def __init__(self, **params) -> None:
        super().__init__()
        encoder_layer_num = params['encoder_layer_num']

        self.init_embedding = LotStageEmbedding(**params)

        self.layers = nn.ModuleList(
            [EncoderLayer(**params) for _ in range(encoder_layer_num)]
        )

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**params) # F_A
        self.col_encoding_block = EncodingBlock(**params) # F_B

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out

class EncodingBlock(nn.Module):
    def __init__(self, **params) -> None:
        super().__init__()
        self.params = params
        embedding_dim = self.params['embedding_dim']
        head_num = self.params['head_num']
        qkv_dim = self.params['qkv_dim']
        device = self.params['device']

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**params)
        self.feed_forward = FeedForward(**params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**params)

        self.to(device)

    def forward(self, row_emb, col_emb, cost_mat):
        # row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)

# ================================= Decoder =====================================
class Decoder(nn.Module):
    def __init__(self, env, **params):
        super().__init__()
        self.params = params
        self.context_type = params.get('context_type', 'default')
        self.input_action = params.get('input_action', 'wafer')

        embedding_dim = self.params['embedding_dim']
        head_num = self.params['head_num']
        qkv_dim = self.params['qkv_dim']
        device = self.params['device']

        # PM embedding
        self.pm_dynamic_linear = nn.Linear(2, embedding_dim, bias=False)
        self.pm_concat_liear = nn.Linear(embedding_dim*3, embedding_dim, bias=False)

        # Action embedding
        self.action_time_linear = nn.Linear(1, embedding_dim, bias=False)
        #self.action_concat_linear = nn.Linear(embedding_dim*3,embedding_dim, bias=False)

        # context setting
        self.loadport_context = loadport_context(embedding_dim)
        self.pm_context = pm_context(env, embedding_dim) # self attention, transformer encoder

        if self.params['arm_type'] == 'single':
            self.robot_context = single_armed_robot_context(embedding_dim)

        elif self.params['arm_type'] == 'dual':
            self.robot_context = dual_armed_robot_context(embedding_dim)
            self.linear_action = nn.Linear(embedding_dim*3, embedding_dim, bias=False)

        self.concat_context = nn.Sequential(
                nn.Linear(embedding_dim*3, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        # MHA, SHA
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        if self.params['arm_type'] == 'single':
            self.Wk = nn.Linear(3*embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv = nn.Linear(3*embedding_dim, head_num * qkv_dim, bias=False)
            self.Wshk =nn.Linear(3*embedding_dim, embedding_dim, bias=False)
        else:
            self.Wk = nn.Linear(4*embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv = nn.Linear(4*embedding_dim, head_num * qkv_dim, bias=False)
            self.Wshk =nn.Linear(4*embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

        self.to(device)

    def get_PM_embedding(self, encoded_row, encoded_col, env, state):
        num_loc = state.loc_id.size(-1)
        clock = state.clock[:, None].repeat(1, num_loc) # (batch, num_loc)

        # PM dynamic embedding
        remain_process_time = torch.where(state.loc_process_end_time > clock,
                        state.loc_process_end_time - clock, 0)
        if env.purge_constraint:
            remain_purge_time = torch.where(state.loc_purge_end_time > clock,
                                            state.loc_purge_end_time - clock, 0)
        else:
            remain_purge_time = torch.zeros_like(remain_process_time)
        pm_dyanamic_features = torch.cat([remain_process_time[:, :, None],
                                          remain_purge_time[:, :, None]], dim=-1)[:, 1:-1]
        # (batch, num_pm, 2)
        pm_dyna_embed = self.pm_dynamic_linear(pm_dyanamic_features)
        # (batch, num_pm, embedding)

        # PM stage, wafer embedding
        encoded_row_add_dummy = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
        pm_lot_idx = env.wafer.get_recipe(state.loc_hold_wafer[:, 1:-1])
        pm_lot_idx = torch.where(pm_lot_idx <= env.num_lot_type, pm_lot_idx, encoded_row_add_dummy.size(1)-1) # (batch, num_stage)
        pm_wafer_embed = gather_by_index(encoded_row_add_dummy, pm_lot_idx) # (batch, num_stage, embedding)
        pm_stage_embed = gather_by_index(encoded_col, state.loc_stage[:, 1:-1]-1)

        if pm_wafer_embed.ndim == 2:
            pm_wafer_embed = pm_wafer_embed[None, :, :]
        if pm_stage_embed.ndim == 2:
            pm_stage_embed = pm_stage_embed[None, :, :]

        # concat PM embedding
        pm_embeddings = self.pm_concat_liear(
            torch.cat([pm_stage_embed, pm_wafer_embed, pm_dyna_embed], dim=-1)
        )
        # (batch, num_pm, embedding)

        return pm_embeddings, pm_wafer_embed

    def get_single_arm_action_duration_embedding(self, state, env):
        num_loc = state.loc_id.size(-1)
        clock = state.clock[:, None].repeat(1, num_loc)
        pkup_time =  clock + env.robot.move_time *\
            (state.loc_id != state.robot_loc[:, None].repeat(1,num_loc))

        eust = torch.where(state.loc_process_end_time > pkup_time,
                        state.loc_process_end_time, pkup_time) # earliest unload start time

        eat = eust + env.robot.unload_time + env.robot.move_time # earliest arrival time
        elst = eat # earliest load start time

        if env.purge_constraint:
            # next stage의 min purge time이 고려 되어야 함.
            # next stage의 PM 중에 바로 load 할 수 있는 idle한 PM이 있을 경우
            # earliest load start time == earlist arrival time
            # 없을 경우, next stage의 min purge time까지 기다려야 함.
            for loc in range(env.loc.num_loc):
                loc_stage = env.loc.stage[0, loc]
                next_stage = loc_stage + 1
                next_stage_loc_load_delay_time = torch.ones_like(state.loc_purge_end_time)
                next_stage_idle_loc = torch.logical_and(env.loc.stage == next_stage,
                                                        env.loc.status == env.loc.status_dict['unload'])
                next_stage_purge_loc = torch.logical_and(env.loc.stage == next_stage,
                                                         env.loc.status == env.loc.status_dict['purge'])

                next_stage_loc_load_delay_time[next_stage_idle_loc] = 0 # idle한 PM
                next_stage_loc_load_delay_time[next_stage_purge_loc] =\
                      torch.where(state.loc_purge_end_time[next_stage_purge_loc] > eat[next_stage_purge_loc],
                                  state.loc_purge_end_time[next_stage_purge_loc] - eat[next_stage_purge_loc], 0)

                min_next_stage_delay_time = next_stage_loc_load_delay_time.min(dim=-1).values

                elst[:, loc] += min_next_stage_delay_time

        elet = elst + env.robot.load_time  # earliest load end time
        elet_duration_time = (elet - clock)[:, :-1, None]

        action_time_embedding = self.action_time_linear(elet_duration_time)

        return action_time_embedding

    def get_dual_arm_action_duration_embedding(self, state, env):
        # unload, load action duration embedding
        num_loc = state.loc_id.size(-1)
        clock = state.clock[:, None].repeat(1, num_loc)

        # earliest unload end time
        empty_robot_loc = torch.where(state.robot_hold_wafer == -1, state.robot_loc, -1)
        pkup_time = clock + env.robot.move_time * ~torch.isin(state.loc_id, empty_robot_loc)
        eust = torch.where(state.loc_process_end_time > pkup_time,
                           state.loc_process_end_time, pkup_time)
        euet = eust + env.robot.unload_time

        euet_duration_time = (euet-clock)[:,:-1, None]
        unload_time_embedding = self.action_time_linear(euet_duration_time)
        ll_unload_time_embedding = unload_time_embedding[:, 0]
        pm_unload_time_embedding = unload_time_embedding[:, 1:, :]


        # earliest load end time
        arm1_move_time = clock + env.robot.move_time * (state.loc_id != state.robot_loc[:, 0][:, None].repeat(1, num_loc))
        arm2_move_time = clock + env.robot.move_time * (state.loc_id != state.robot_loc[:, 1][:, None].repeat(1, num_loc))

        if env.purge_constraint:
            arm1_elet = torch.where(state.loc_purge_end_time > arm1_move_time,
                                    state.loc_purge_end_time, arm1_move_time) + env.robot.load_time
            arm2_elet = torch.where(state.loc_purge_end_time > arm2_move_time,
                                    state.loc_purge_end_time, arm2_move_time) + env.robot.load_time

        else:
            arm1_elet = arm1_move_time + env.robot.load_time
            arm2_elet = arm2_move_time + env.robot.load_time

        arm1_elet_duration_time = (arm1_elet-clock)[:, :, None]
        arm2_elet_duration_time = (arm2_elet-clock)[:, :, None]
        arm1_load_time_embedding = self.action_time_linear(arm1_elet_duration_time[:, 1:, :]) # except in loadlock
        arm2_load_time_embedding = self.action_time_linear(arm2_elet_duration_time[:, 1:, :])

        return ll_unload_time_embedding, pm_unload_time_embedding, arm1_load_time_embedding, arm2_load_time_embedding

    def get_action_embedding(self, encoded_row, encoded_col, env, state, pm_embeddings, pm_wafer_embeddings):
        if env.arm_type == 'single':
            # get action duration
            unload_dur_embedding = self.get_single_arm_action_duration_embedding(state, env)
            ll_unload_dur_embeddings = unload_dur_embedding[:, 0]  # (batch, embedding)
            pm_unload_dur_embeddings = unload_dur_embedding[:, 1:, :]  # (batch, num_pm, embedding)

            # LL unload action
            if self.input_action == 'wafer':
                #release_foup_idx = 1
                #wafer_types = env.wafer.recipe[:, release_foup_idx, :].to(state.device())
                ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1) #FIXME 
                wafer_types = env.wafer.recipe[env.batch_idx, ll_foup].to(state.device())
                wafer_embedding = gather_by_index(encoded_row, wafer_types)
                ll_embedding = torch.zeros_like(wafer_embedding) # (batch, num_lot_type, embedding)
                ll_unload_dur_embeddings =\
                    ll_unload_dur_embeddings[:, None, :].repeat(1, wafer_embedding.size(1), 1)

                ll_unload_embedding = torch.cat([ll_embedding, wafer_embedding, ll_unload_dur_embeddings], dim=-1)


            elif self.input_action == 'type':
                wafer_type_embedding = encoded_row # (batch, num_lot_type, embedding)
                ll_embedding = torch.zeros_like(wafer_type_embedding) # (batch, num_lot_type, embedding)
                ll_unload_dur_embeddings =\
                    ll_unload_dur_embeddings[:, None, :].repeat(1,wafer_type_embedding.size(1), 1)

                ll_unload_embedding =torch.cat([ll_embedding, wafer_type_embedding, ll_unload_dur_embeddings], dim=-1)

            else:
                print(">>> [Error] Invalid action embedding!")

            # PM unload action
            pm_unload_embedding =  torch.cat([pm_embeddings, pm_wafer_embeddings, pm_unload_dur_embeddings], dim=-1)
            # (batch, num_pm, 3*embedding)

            action_embedding = torch.cat([ll_unload_embedding, pm_unload_embedding], dim=1)

        elif env.arm_type == 'dual':
            ll_unload_dur_embeddings, pm_unload_dur_embeddings, arm1_load_dur_embedding, arm2_load_dur_embedding =\
                self.get_dual_arm_action_duration_embedding(state, env)

            # unload actions
            if self.input_action == 'wafer':
                #release_foup_idx = 1
                #wafer_types = env.wafer.recipe[:, release_foup_idx, :].to(state.device())
                ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1) # FIXME 
                #wafer_types = env.wafer.recipe[env.batch_idx, env.wafer.exit_foup].to(state.device())
                wafer_types = env.wafer.recipe[env.batch_idx, ll_foup].to(state.device())
                wafer_embedding = gather_by_index(encoded_row, wafer_types)
                ll_embedding = torch.zeros_like(wafer_embedding)
                ll_unload_dur_embeddings =\
                    ll_unload_dur_embeddings[:, None, :].repeat(1, wafer_types.size(-1), 1)
                robot_embedding = torch.ones_like(ll_embedding)

                if ll_embedding.ndim == 2:
                    ll_embedding = ll_embedding[None, :, :]
                    robot_embedding = robot_embedding[None, :, :]
                    wafer_embedding = wafer_embedding[None, :, :]

                ll_unload_embedding = torch.cat([ll_embedding,
                                                 robot_embedding,
                                                 wafer_embedding,
                                                 ll_unload_dur_embeddings], dim=-1)

            elif self.input_action == 'type':
                print(">>> [Error] Invalid action embedding!")
            else:
                print(">>> [Error] Invalid action embedding!")

            robot_embedding = torch.ones_like(pm_embeddings)
            pm_unload_embedding = torch.cat([pm_embeddings,
                                             robot_embedding,
                                             pm_wafer_embeddings,
                                             pm_unload_dur_embeddings], dim=-1)

            # load actions
            encoded_row_add_dummy = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
            arm1_hold_wafer_idx = torch.where(state.robot_hold_wafer[:, 0]!=-1,
                                              env.wafer.get_recipe(state.robot_hold_wafer[:, 0]),
                                              encoded_row_add_dummy.size(1)-1)
            arm2_hold_wafer_idx = torch.where(state.robot_hold_wafer[:, 1]!=-1,
                                              env.wafer.get_recipe(state.robot_hold_wafer[:, 1]),
                                              encoded_row_add_dummy.size(1)-1)

            arm1_hold_wafer_embed = gather_by_index(encoded_row_add_dummy, arm1_hold_wafer_idx)
            arm2_hold_wafer_embed = gather_by_index(encoded_row_add_dummy, arm2_hold_wafer_idx)
            if arm1_hold_wafer_embed.ndim == 1:
                arm1_hold_wafer_embed = arm1_hold_wafer_embed[None, :]
                arm2_hold_wafer_embed = arm2_hold_wafer_embed[None, :]

            robot_embedding = torch.ones_like(pm_embeddings[:, 0, :])
            ll_embedding = torch.zeros_like(robot_embedding)


            arm1_pm_load_embedding = torch.cat([
                robot_embedding[:, None, :].repeat(1, pm_embeddings.size(1),1),
                pm_embeddings,
                arm1_hold_wafer_embed[:, None, :].repeat(1, pm_embeddings.size(1),1),
                arm1_load_dur_embedding[:, :-1]
            ], dim=-1)

            arm2_pm_load_embedding = torch.cat([
                robot_embedding[:, None, :].repeat(1, pm_embeddings.size(1),1),
                pm_embeddings,
                arm2_hold_wafer_embed[:, None, :].repeat(1, pm_embeddings.size(1),1),
                arm2_load_dur_embedding[:, :-1]
            ], dim=-1)

            arm1_ll_load_embedding = torch.cat([
                robot_embedding,
                ll_embedding,
                arm1_hold_wafer_embed,
                arm1_load_dur_embedding[:, -1]
            ], dim=-1)[:, None, :]

            arm2_ll_load_embedding = torch.cat([
                robot_embedding,
                ll_embedding,
                arm2_hold_wafer_embed,
                arm2_load_dur_embedding[:, -1]
            ], dim=-1)[:, None, :]

            action_embedding = torch.cat([ll_unload_embedding,
                                          pm_unload_embedding,
                                          arm1_pm_load_embedding,
                                          arm1_ll_load_embedding,
                                          arm2_pm_load_embedding,
                                          arm2_ll_load_embedding], dim=1)

        else:
            print(">>> [Error] Invalid robot arm type for action embedding")


        return action_embedding

    def _multi_head_attention_for_decoder(self, q, k, v,
                                          rank2_ninf_mask=None,
                                          rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, qkv_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch, head_num, job_cnt+1, qkv_dim)
        # rank2_ninf_mask.shape: (batch, job_cnt+1)
        # rank3_ninf_mask.shape: (batch, n, job_cnt+1)

        # my implementation
        # q shape: (batch*n, head_num, 1, qkv_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch*n, head_num, action_cnt, qkv_dim)
        # rank2_ninf_mask.shape: (batch, action_cnt)
        # rank3_ninf_mask.shape: (batch*n, action_cnt) -> action mask

        head_num = self.params['head_num']
        qkv_dim = self.params['qkv_dim']
        sqrt_qkv_dim = self.params['sqrt_qkv_dim']

        batch_pomo_size = q.size(0)
        action_cnt = k.size(2)

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch*n, head_num, 1, action_cnt)

        score_scaled = score / sqrt_qkv_dim

        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + \
                rank3_ninf_mask[:, None, :, :].expand(batch_pomo_size, head_num, 1, action_cnt)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch*n, head_num, 1, action_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch*n, head_num, 1, qkv_dim)

        out_concat = out.reshape(batch_pomo_size, head_num * qkv_dim)
        # shape: (batch*n, head_num*qkv_dim)

        return out_concat

    def forward(self, encoded_row, encoded_col, env, state, ninf_mask=None):
        """
        ------------------------------------------------------------------------------------------
        # 1. Generate PM embedding = [stage embed, wafer embed, remain process time] TODO: residency time
        # 2. Generate action embedding
        #      single arm action:
        #           unload from inLL with type(or wafer)
        #           => [inLL embedding | wafer type embedding | action dur] \in R^3d
        #           unload from PM & load to the next stage
        #           => [PM embedding  | action dur] \in R^3d
        #      dual arm action:
        #           unload from inLL with type(or wafer) + PM
        #           load to the PM & outLL from arm1, arm2

        # 3. Generate context embedding
        # Context = [Loadlock context | PM context | Robot context]

        # 4. MHA: Query <- context embedding, Key, Value (action embedding)

        # 5. SHA: Get action probability
        ------------------------------------------------------------------------------------------
        """
        batch_size = encoded_row.size(0)
        head_num = self.params['head_num']
        sqrt_embedding_dim = self.params['sqrt_embedding_dim']
        logit_clipping = self.params['logit_clipping']

        # 1. Generate PM embeddings = [stage embed, wafer embed, remain process time, remain purge time]
        # TODO: residency time
        # ------------------------------------------------------------------------
        pm_embeddings, pm_wafer_embeddings = self.get_PM_embedding(encoded_row, encoded_col, env, state)
        # (batch, num_pm, embedding)

        # 2. Generate action embedding
        # ------------------------------------------------------------------------
        action_embeddings = self.get_action_embedding(encoded_row, encoded_col,
                                                      env, state, pm_embeddings, pm_wafer_embeddings)
        # wafer based: (batch, num_wafer + num_pm, embedding)
        # type based: (batch, num_wafer_type + num_pm, embedding)

        # 3. Generate context embedding
        # Context = [Loadlock context | PM context | Robot context]
        # ------------------------------------------------------------------------
        ll_context = self.loadport_context(env, encoded_row, state)
        pm_context = self.pm_context(pm_embeddings)
        robot_context = self.robot_context(pm_embeddings, state) if env.arm_type == 'single' else \
            self.robot_context(env, encoded_row, encoded_col, state)

        context = self.concat_context(
            torch.cat([ll_context, pm_context, robot_context], dim=1)
        )[:, None, :]
        # (batch, 1, embedding)

        # 4. MHA: Query <- context embedding, Key, Value (action embedding)
        # ------------------------------------------------------------------------
        num_wafer = env.foup_size
        exit_foup = torch.clamp(env.wafer.exit_foup, max=env.num_foup-1)
        ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1)
        ll_wafer_avail = (env.wafer.loc[env.batch_idx, ll_foup] == 0).to(state.device())
        ll_wafer_avail = ll_wafer_avail * (state.action_mask[:, :env.num_lot_type].any(dim=-1)[:, None].repeat(1, num_wafer))
        #ll_mask = (~(state.action_mask[:, :env.num_lot_type].any(dim=-1)[:, None].repeat(1, num_wafer)) &\
        #            (env.wafer.loc[env.batch_idx, exit_foup] != 0).to(state.device()))
        #ll_wafer_mask = (-1e10 * ll_mask)
        ll_wafer_mask = (-1e10 * (~ll_wafer_avail))
        pm_wafer_mask = (-1e10 * (~state.action_mask[:, env.num_lot_type:]))
        ninf_mask = torch.cat([ll_wafer_mask, pm_wafer_mask], dim=-1)[:, None, :]

        #ninf_mask = -1e10 * (~state.action_mask[:, None, :])
        # action mask가 True가 아닌 action들에 대해서 -inf 값으로 선택 확률 0 처리

        self.k = reshape_by_heads(self.Wk(action_embeddings), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(action_embeddings), head_num=head_num)
        # shape: (batch * pomo, head_num, action_cnt, qkv_dim)
        self.single_head_key = self.Wshk(action_embeddings).transpose(1, 2)
        # shape: (batch * pomo, embedding, action_cnt)
        q = reshape_by_heads(self.Wq_3(context), head_num=head_num)

        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v,
                                                            rank3_ninf_mask=ninf_mask)
        # shape: (batch*pomo, head_num*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat)[:, None, :]
        # shape: (batch*pomo, 1, embedding)


        # 5. SHA: Get action probability withg attention score
        # ------------------------------------------------------------------------
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch*pomo, 1, row_cnt)
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch*pomo, 1, row_cnt)
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask
        score_masked = score_masked.reshape(score_masked.size(0),-1)
        # shape: (batch*pomo, row_cnt)

        probs = F.softmax(score_masked, dim=-1)
        # shape (batch*pomo, action_space)

        return probs
