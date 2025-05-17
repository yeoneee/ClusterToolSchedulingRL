from typing import Union
from tensordict.tensordict import TensorDict
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import LotStageEmbedding
from model.context import loadport_context, pm_context, dual_armed_robot_context
from model.ncts_model_sub import MixedScore_MultiHeadAttention, AddAndInstanceNormalization, FeedForward
from model.utils import gather_by_index


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
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.encoder = Encoder(**params)
        self.decoder = Decoder(**params)

    def encoding(self, state):
        # get init embedding vector
        self.row_embed, self.col_embed, self.cost_mat = self.encoder.init_embedding(state)

        # get encoded embedding vector of row(=wafer), col(=loc)
        self.row_embed, self.col_embed =\
              self.encoder(self.row_embed, self.col_embed, self.cost_mat)

    def decoding(self, state):
        # get probability of action through decoder
        # --------------------------------------------------------
        prob = self.decoder(self.row_embed, self.col_embed, state)
        # shape (batch, action_cnt)

        # training mode action selection
        # --------------------------------------------------------
        if self.training or self.params['eval_type'] == 'softmax':
            while True:
                selected = torch.multinomial(prob, num_samples=1).squeeze(1)
                # shape: (batch,)
                selected_prob = prob[state.batch_idx, selected]
                # shape: (batch, )
                selected_prob[state.done.squeeze()] = 1
                # to fix pytorch.multinomial bug on selecting 0 probability elements
                if (selected_prob != 0).all(): break


        # evaluate mode action selection (greedy selection)
        # --------------------------------------------------------
        else:
            selected = prob.argmax(dim=1)
            # shape: (batch, pomo)
            selected_prob = prob[state.batch_idx, selected]
            # shape: (batch,)
            selected_prob[state.done.squeeze()] = 1

        return selected, selected_prob

    def forward(self, state):
        selected, prob = self.decoding(state)

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
    def __init__(self, **params):
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

        # Robot embedding
        self.robot_embed_linear = nn.Linear(embedding_dim*3, embedding_dim, bias=False)

        # Action embedding
        self.action_time_linear = nn.Linear(1, embedding_dim, bias=False)

        # context setting
        self.loadport_context = loadport_context(embedding_dim)
        self.pm_context = pm_context(embedding_dim) # self attention, transformer encoder
        self.robot_context = dual_armed_robot_context(embedding_dim)
        self.concat_context = nn.Sequential(
                nn.Linear(embedding_dim*3, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

        # MHA, SHA
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wk = nn.Linear(4*embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(4*embedding_dim, head_num * qkv_dim, bias=False)
        self.Wshk =nn.Linear(4*embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

        self.to(device)

    def get_PM_embedding(self, encoded_row, encoded_col, state):
        num_pm = state.loc_id.size(-1)
        clock = state.clock.repeat(1, num_pm) # (batch, num_pm)

        # PM dynamic embedding
        norm_scale = 300
        remain_prs_time = torch.where(state.loc_process_end_time > clock, state.loc_process_end_time - clock, 0) / norm_scale
        remain_purge_time = torch.zeros_like(remain_prs_time) / norm_scale
        pm_dynamic_features = torch.cat([remain_prs_time[:, :, None], remain_purge_time[:, :, None]], dim=-1)
        pm_dyna_embed = self.pm_dynamic_linear(pm_dynamic_features)
        # (batch, num_pm, embedding)

        # PM stage, wafer embedding
        encoded_row_add_dummy = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
        pm_lot_idx = state.loc_hold_wafer
        pm_lot_idx = torch.where(pm_lot_idx >= 0, pm_lot_idx, encoded_row_add_dummy.size(1)-1).to(torch.int64)
        stage_idx = (state.loc_stage -1).to(torch.int64)

        pm_wafer_embed = gather_by_index(encoded_row_add_dummy, pm_lot_idx) # (batch, num_stage, embedding)
        pm_stage_embed = gather_by_index(encoded_col, stage_idx)


        # concat PM embedding
        pm_embeddings = self.pm_concat_liear(
            torch.cat([pm_stage_embed, pm_wafer_embed, pm_dyna_embed], dim=-1)
        )
        # (batch, num_pm, embedding)

        return pm_embeddings, pm_wafer_embed

    def get_robot_embedding(self, encoded_row, encoded_col, state, pm_embedding):
        def get_loc_embedding(pm_embedding):
            # in LL + PMs + out LL embedding
            loc_embedding = torch.cat(
                [torch.zeros_like(pm_embedding[:, 0:1]), pm_embedding], dim=1)
            loc_embedding = torch.cat(
                [loc_embedding, torch.ones_like(pm_embedding[:, 0:1])], dim=1)
            return loc_embedding

        def get_hold_wafer_embedding(encoded_row, state, arm_hold_wafer_recipe):
            encoded_row = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
            arm_wafer = torch.where(arm_hold_wafer_recipe >= 0, arm_hold_wafer_recipe, encoded_row.size(1)-1).to(torch.int64)
            return gather_by_index(encoded_row, arm_wafer)

        def get_next_stage_embedding(encoded_col, state, arm_hold_wafer_next_stage):
            encoded_col = torch.cat([torch.zeros_like(encoded_col[:, 0:1]), encoded_col], dim=1)
            encoded_col = torch.cat([encoded_col, torch.zeros_like(encoded_col[:, 0:1])], dim=1)
            arm_next_stage = torch.where(arm_hold_wafer_next_stage >= 0, arm_hold_wafer_next_stage, 0).to(torch.int64)
            return gather_by_index(encoded_col, arm_next_stage)

        loc_embedding = get_loc_embedding(pm_embedding)
        arm1_loc_embedding = gather_by_index(loc_embedding, state.robot_arm1_loc)
        arm2_loc_embedding = gather_by_index(loc_embedding, state.robot_arm2_loc)

        arm1_hold_wafer_embedding = get_hold_wafer_embedding(encoded_row, state, state.robot_arm1_hold_wafer_recipe)
        arm2_hold_wafer_embedding = get_hold_wafer_embedding(encoded_row, state, state.robot_arm2_hold_wafer_recipe)

        arm1_next_stage_embedding = get_next_stage_embedding(encoded_col, state, state.robot_arm1_hold_wafer_next_stage)
        arm2_next_stage_embedding = get_next_stage_embedding(encoded_col, state, state.robot_arm2_hold_wafer_next_stage)

        arm1_embedding = self.robot_embed_linear(torch.cat([arm1_loc_embedding, arm1_hold_wafer_embedding, arm1_next_stage_embedding], dim=-1))
        arm2_embedding = self.robot_embed_linear(torch.cat([arm2_loc_embedding, arm2_hold_wafer_embedding, arm2_next_stage_embedding], dim=-1))

        return arm1_embedding, arm2_embedding

    def get_dual_arm_action_duration_embedding(self, state):
        def cal_unload_action_duration(state, loc_id, loc_process_end_time, move_time, unload_time):
            num_loc = loc_id.size(-1)
            clock = state.clock.repeat(1, num_loc)
            faced_arm = torch.logical_or((state.robot_arm1_loc.repeat(1, num_loc) == loc_id),
                                           (state.robot_arm2_loc.repeat(1, num_loc) == loc_id))
            pkup_time = clock + move_time * ~faced_arm
            eust = torch.where(loc_process_end_time > pkup_time, loc_process_end_time, pkup_time)  # earliest unload start time
            euet = eust + unload_time
            return euet - clock

        def cal_load_action_duration(state, loc_id, move_time, load_time):
            num_loc = loc_id.size(-1)
            clock = state.clock.repeat(1, num_loc)

            arm1_move_time = clock + move_time * (loc_id != state.robot_arm1_loc.repeat(1, num_loc))
            arm2_move_time = clock + move_time * (loc_id != state.robot_arm2_loc.repeat(1, num_loc))

            arm1_elet = arm1_move_time + load_time
            arm2_elet = arm2_move_time + load_time

            return arm1_elet - clock, arm2_elet - clock

        # earliest unload end time
        # inLL unload action duration
        inll_id = torch.zeros((state.batch_size(), 1), device=state.device())
        inll_process_end_time = torch.zeros((state.batch_size(), 1), device=state.device())
        inll_duration_time = cal_unload_action_duration(state, inll_id, inll_process_end_time, 3, 3)

        # PM unload action duration
        euet_duration_time = cal_unload_action_duration(state, state.loc_id, state.loc_process_end_time, 3, 3)
        euet_duration_time = torch.cat([inll_duration_time, euet_duration_time], dim=-1) # (batch, 1+num_pm)

        # earliest load end time
        outloadlock_id = state.loc_id.size(-1)+1
        loc_ids = torch.cat([state.loc_id, torch.full_like(state.loc_id[:, 0:1], fill_value=outloadlock_id)], dim=1) # add outloadlock
        arm1_elet_duration_time, arm2_elet_duration_time = cal_load_action_duration(state, loc_ids, 3, 3)

        # norm & embedding
        norm_scale = 300
        euet_duration_time = (euet_duration_time / norm_scale)[:, :, None]
        arm1_elet_duration_time = (arm1_elet_duration_time / norm_scale)[:, :, None]
        arm2_elet_duration_time = (arm2_elet_duration_time / norm_scale)[:, :, None]

        euet_duration_time = self.action_time_linear(euet_duration_time)
        arm1_elet_duration_time = self.action_time_linear(arm1_elet_duration_time)
        arm2_elet_duration_time = self.action_time_linear(arm2_elet_duration_time)

        return euet_duration_time, arm1_elet_duration_time, arm2_elet_duration_time

    def get_action_embedding(self, encoded_row, encoded_col, state,
                             pm_embeddings, pm_wafer_embeddings,
                             arm1_embedding, arm2_embedding, robot_context):
        # Get dual arm action duration embeddings
        unload_dur_embedding, arm1_load_dur_embedding, arm2_load_dur_embedding = self.get_dual_arm_action_duration_embedding(state)

        def get_ll_unload_embedding():
            group1_embedding = gather_by_index(encoded_row, state.loadlock1_wafer_recipe)[:, None, :]
            group2_embedding = gather_by_index(encoded_row, state.loadlock2_wafer_recipe)[:, None, :]
            group_embedding = torch.cat([group1_embedding, group2_embedding], dim=1)  # (batch, 2, embedding)
            robot_embedding = torch.ones_like(group_embedding)  # (batch, 2, embedding)
            ll_embedding = torch.zeros_like(group_embedding)  # (batch, 2, embedding)
            ll_unload_dur_embedding = unload_dur_embedding[:, 0:1].repeat(1, 2, 1)  # (batch, 2, embedding)
            return torch.cat([ll_embedding, robot_embedding, group_embedding, ll_unload_dur_embedding], dim=-1)  # (batch, 2, 4*embedding)

        def get_pm_unload_embedding():
            robot_embedding = torch.ones_like(pm_embeddings)  # (batch, num_pm, embedding)
            pm_unload_dur_embedding = unload_dur_embedding[:, 1:]  # (batch, num_pm, embedding)
            return torch.cat([pm_embeddings, robot_embedding, pm_wafer_embeddings, pm_unload_dur_embedding], dim=-1)  # (batch, num_pm, 4*embedding)

        def get_arm_load_embedding(arm_embedding, arm_hold_wafer_recipe, load_dur_embedding):
            encoded_row_add_dummy = torch.cat([encoded_row, torch.zeros_like(encoded_row[:, 0:1])], dim=1)
            arm_hold_wafer = torch.where(arm_hold_wafer_recipe >= 0, arm_hold_wafer_recipe, encoded_row_add_dummy.size(1)-1).to(torch.int64)
            arm_hold_wafer_embedding = gather_by_index(encoded_row_add_dummy, arm_hold_wafer)
            return torch.cat([
                arm_embedding[:, None, :].repeat(1, pm_embeddings.size(1)+1, 1),
                torch.cat([pm_embeddings, torch.zeros_like(pm_embeddings[:, 0:1])], dim=1),
                arm_hold_wafer_embedding[:, None, :].repeat(1, pm_embeddings.size(1)+1, 1),
                load_dur_embedding
            ], dim=-1)  # (batch, num_pm+1, 4*embedding)

        # Get action embeddings
        ll_unload_embedding = get_ll_unload_embedding()
        pm_unload_embedding = get_pm_unload_embedding()
        arm1_load_embedding = get_arm_load_embedding(arm1_embedding, state.robot_arm1_hold_wafer_recipe, arm1_load_dur_embedding)
        arm2_load_embedding = get_arm_load_embedding(arm2_embedding, state.robot_arm2_hold_wafer_recipe, arm2_load_dur_embedding)

        # Concatenate action embeddings
        action_embeddings = torch.cat([ll_unload_embedding, pm_unload_embedding, arm1_load_embedding, arm2_load_embedding], dim=1)

        return action_embeddings


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

    def forward(self, encoded_row, encoded_col, state, ninf_mask=None):
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
        # ------------------------------------------------------------------------
        pm_embeddings, pm_wafer_embeddings = self.get_PM_embedding(encoded_row, encoded_col, state)
        arm1_embeddings, arm2_embeddings = self.get_robot_embedding(encoded_row, encoded_col, state, pm_embeddings)
        # (batch, num_pm, embedding)


        # Generate context embedding
        # Context = [Loadlock context | PM context | Robot context]
        # ------------------------------------------------------------------------
        #ll_context = self.loadport_context(env, encoded_row, state)
        ll_context = self.loadport_context(encoded_row, state)
        pm_context = self.pm_context(pm_embeddings)
        robot_context = self.robot_context(arm1_embeddings, arm2_embeddings)
        context = self.concat_context(torch.cat([ll_context, pm_context, robot_context], dim=1))[:, None, :]
        # (batch, 1, embedding)

        # Generate action embedding
        # ------------------------------------------------------------------------
        action_embeddings = self.get_action_embedding(encoded_row, encoded_col, state,
                                                      pm_embeddings, pm_wafer_embeddings,
                                                      arm1_embeddings, arm2_embeddings, robot_context)

        # wafer based: (batch, num_wafer + num_pm, embedding)
        # type based: (batch, num_wafer_type + num_pm, embedding)



        # 4. MHA: Query <- context embedding, Key, Value (action embedding)
        # ------------------------------------------------------------------------
        """ wafer type based
        ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1)
        ll_wafer_avail = (env.wafer.loc[env.batch_idx, ll_foup] == 0).to(state.device())
        # ANCHOR
        """
        num_lot_type = 2
        lm_mask = (-1e10 * (~state.action_mask[:, :num_lot_type]))
        pm_mask = (-1e10 * (~state.action_mask[:, num_lot_type:]))
        ninf_mask = torch.cat([lm_mask, pm_mask], dim=-1)[:, None, :]

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
