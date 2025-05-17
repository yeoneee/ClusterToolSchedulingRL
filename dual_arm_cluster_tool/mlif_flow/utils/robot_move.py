import copy
import torch

def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)


"================================================loadport input sequence policy =========================================="
""""
"""
class lot_input_seq_policy:
    def __init__(self, rule:str='random', env: object=None) -> None:
        self.rule = rule
        self.prev_lot_type = None
        self.env = env
        self.recipe = env.recipe_table['process_time']
        self.prev = True

    def __call__(self, env, state):
        input_lots = torch.zeros(state.batch_size(), self.env.num_lot_type).to(state.device())
        input_lots_cnt = torch.zeros(state.batch_size(), self.env.num_lot_type, state.wafer_loc.size(-1)).to(state.device())
        curr_foups = torch.zeros(state.batch_size(), self.env.num_foup+1).to(state.device())

        bidx, widx = (state.wafer_loc == 0).nonzero(as_tuple=True)

        # check the input loadloak has wafer
        if (state.wafer_loc == 0).any():
            foup_idx = self.env.wafer.get_foup(state.wafer_name[bidx, widx])
            curr_foups[bidx, foup_idx] = 1
            no_input_wafer_batch_idx = curr_foups.sum(dim=-1) == 0
            exit_foup_idx = self.env.wafer.exit_foup.to(state.device())[no_input_wafer_batch_idx]
            #ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1)
            curr_foups[no_input_wafer_batch_idx, exit_foup_idx+1] = 1 # no input loadlock batch idx #FIXME

            bidx, fidx = (curr_foups.cumsum(dim=-1).cumsum(dim=-1) == 1).nonzero(as_tuple=True)
            curr_foup_wafers = self.env.wafer.get_foup(state.wafer_name) ==\
                fidx[:, None].repeat(1, self.env.loadport_capacity*self.env.foup_size)

            bidx, widx = torch.logical_and(state.wafer_loc == 0, curr_foup_wafers).nonzero(as_tuple=True)
            ridx = state.wafer_recipe[bidx, widx].to(torch.long)
            input_lots[bidx, ridx] = 1
            input_lots_cnt[bidx, ridx, widx] = 1


            action_mask = torch.ones_like(state.action_mask)
            action_mask[state.batch_idx, :self.env.num_lot_type] = False

            if self.rule == 'random':
                selected_lot = torch.multinomial(input_lots, num_samples=1).squeeze() # (batch,)
                action_mask[state.batch_idx, selected_lot] = True

            elif self.rule == 'spt':
                fpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :, 1], 1e10) # total process time
                min_tpt, _ = fpt.min(dim=-1)
                input_mask = fpt == min_tpt[:, None].repeat(1, self.env.num_lot_type)
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'lpt':
                fpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :, 1], 0) # total process time
                max_tpt, _ = fpt.max(dim=-1)
                input_mask = fpt == max_tpt[:, None].repeat(1, self.env.num_lot_type)
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'stpt':
                tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :].sum(dim=-1), 1e10) # total process time
                min_tpt, _ = tpt.min(dim=-1)
                input_mask = tpt == min_tpt[:, None].repeat(1, self.env.num_lot_type)
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'ltpt':
                tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :].sum(dim=-1), 0) # total process time
                max_tpt, _ = tpt.max(dim=-1)
                input_mask = tpt == max_tpt[:, None].repeat(1, self.env.num_lot_type)
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'switch_spt_lpt':
                if self.prev == True:
                    tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :, 1], 0) # total process time
                    max_tpt, _ = tpt.max(dim=-1)
                    input_mask = tpt == max_tpt[:, None].repeat(1, self.env.num_lot_type)
                    self.prev = False
                else:
                    tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :, 1], 1e10) # total process time
                    min_tpt, _ = tpt.min(dim=-1)
                    input_mask = tpt == min_tpt[:, None].repeat(1, self.env.num_lot_type)
                    self.prev = True

                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'switch_stpt_ltpt':
                if self.prev == True:
                    tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :].sum(dim=-1), 0) # total process time
                    max_tpt, _ = tpt.max(dim=-1)
                    input_mask = tpt == max_tpt[:, None].repeat(1, self.env.num_lot_type)
                    self.prev = False
                else:
                    tpt = torch.where(input_lots == 1, self.recipe[state.batch_idx, :].sum(dim=-1), 1e10) # total process time
                    min_tpt, _ = tpt.min(dim=-1)
                    input_mask = tpt == min_tpt[:, None].repeat(1, self.env.num_lot_type)
                    self.prev = True

                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

            elif self.rule == 'workload':
                # workload balancing을 위한 input sequencing
                wafer_type_cnt = input_lots_cnt.sum(dim=-1) #(batch, num_type)
                pass

            elif self.rule == 'neh':
                pass

                """
                # 현재 neh_sequence에서 pop을 해서 다음 job type 선택
                # 해당 job type만 select 가능
                # neh sequence 전체 실행 시 스케줄 종료 필요._
                if self.env.kwargs['next_release_wafer_idx'][0] < self.env.kwargs['release_sequence'].size(-1):
                    lot_types = torch.arange(input_lots.size(1))[None, :].repeat(input_lots.size(0), 1)
                    # release wafer 의 input sequence 상 순서 확인
                    release_wafer_id = self.env.kwargs['release_sequence'][
                        self.env.batch_idx, self.env.kwargs['next_release_wafer_idx']]
                    # 해당 순서의 사이즈를 보고 FOUP number 확인 -> Foup index, wafer id index를 통해 recipe 확인
                    release_lot_type = self.env.wafer.recipe[
                        self.env.batch_idx,
                        self.env.kwargs['next_release_wafer_idx'] // self.env.foup_size + 1,
                        release_wafer_id][:, None].repeat(1, input_lots.size(1))

                    #release_lot_type = gather_by_index(self.env.wafer.recipe[self.env.batch_idx, self.env.wafer.exit_foup],
                    #                                   release_wafer_id)[:, None].repeat(1, input_lots.size(1))

                    input_mask = (input_lots == 1) & (lot_types == release_lot_type)
                else:
                    input_mask = torch.zeros_like(input_lots, dtype=torch.bool)
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask
                """

            elif self.rule == 'fix':
                input_idx = torch.where(env.input_idx >= env.input_seq.size(1), 0, env.input_idx)
                ll_wafer_idx = gather_by_index(env.input_seq, input_idx)
                ll_foup = torch.argmax((env.wafer.loc == 0).any(dim=-1).float(), dim=1)
                ll_wafer_recipe = env.wafer.recipe[env.batch_idx, ll_foup].to(state.device())[state.batch_idx, ll_wafer_idx]

                input_mask = torch.zeros_like(input_lots, dtype=torch.bool)
                input_mask[state.batch_idx, ll_wafer_recipe] = True
                action_mask[state.batch_idx, :self.env.num_lot_type] = input_mask

        else:
            action_mask = torch.ones_like(state.action_mask)

        return action_mask

"================================================Robot arm move sequence policy =========================================="

class Random_policy:
    def __init__(self, env):
        pass

    def __call__(self, state, additional_mask=None):
        action_mask = (torch.logical_and(state.action_mask, additional_mask)
                       if additional_mask is not None else state.action_mask)
        action = torch.multinomial((action_mask).float(), 1).squeeze(-1)

        return action

class Backward_policy:
    def __init__(self, env) -> None:
        # env info
        self.batch_size = env.batch_size
        self.num_wafer = env.loadport_capacity * env.wafer.foup_size
        self.num_loc = env.loc.num_loc
        self.num_stage = env.loc.num_stage
        self.num_lot_type = env.num_lot_type
        # target stage index
        self.n = torch.full(size=self.batch_size, fill_value=self.num_stage, dtype=torch.long)
        self.lot_release = lot_input_seq_policy(rule=env.lot_release_rule, env=env)

    def to(self, device):
        self.n = self.n.to(device)
        self.lot_release.recipe = self.lot_release.recipe.to(device)


    def __call__(self, env, state):
        # get backward rule based action mask
        backward_mask = self.get_backward_mask(env, state)

        # input sequence masking
        input_rule = True if self.lot_release.rule != 'random' else False
        if input_rule:
            input_mask = self.lot_release(env, state)
        else:
            input_mask = torch.ones_like(state.action_mask, dtype=torch.bool)

        # skip stage if empty stage
        while (~(state.action_mask & backward_mask & input_mask).any(dim=-1)).any():
            jump_stage_batch_idx = (~(state.action_mask & backward_mask & input_mask).any(dim=-1))
            # update backward stage index
            self.n[jump_stage_batch_idx] -= 1
            self.n[self.n < 0] = self.num_stage

            backward_mask = self.get_backward_mask(env, state)

        # update backward stage index
        self.n -= 1

        # update action mask
        action_mask = state.action_mask & backward_mask & input_mask
        assert (action_mask).any(), "no available action"

        # action selection
        # to fix pytorch.multinomial bug on selecting 0 probability elements
        while True:
            prob = action_mask.float()
            action = torch.multinomial(prob, 1).squeeze(-1)
            selected_prob = prob[state.batch_idx, action]
            selected_prob[state.done] = 1
            if (selected_prob != 0).all(): break

        # fixed input sequence
        #action[action < env.num_lot_type] = 0
        #action[action >= env.num_lot_type] = action[action >= env.num_lot_type] - env.num_lot_type + 1

        return action

    def get_backward_mask(self, env, state):
        # step1: empty stage skip
        n_expand_wafer = self.n[:, None].repeat(1, self.num_wafer)
        exist_unloadable_wafer = (state.wafer_stage == n_expand_wafer).any(dim=-1)
        while not exist_unloadable_wafer.all():
            self.n[~exist_unloadable_wafer] -= 1
            self.n[self.n < 0] = self.num_stage
            n_expand_wafer = self.n[:, None].repeat(1, self.num_wafer)
            exist_unloadable_wafer = (state.wafer_stage == n_expand_wafer).any(dim=-1)

        # step2: set backward policy mask
        backward_mask = torch.zeros_like(state.action_mask, dtype=torch.bool)

        # n == 0. loadlock unload case
        ll_batch_idx = self.n == 0
        backward_mask[ll_batch_idx, :self.num_lot_type] = True

        # n > 0. PM unload case
        # find the "loaded" & "stage n" location
        n_expand_loc = self.n[:, None].repeat(1, self.num_loc)
        stage_n_loc = state.loc_stage == n_expand_loc
        loaded_pm_loc = state.loc_hold_wafer != -1
        lm_loc = state.loc_stage == 0   # exception case: loadlock
        loaded_loc = torch.logical_or(loaded_pm_loc, lm_loc)
        loaded_stage_n_loc = torch.logical_and(stage_n_loc, loaded_loc)

        # FIFO FOUP PM
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1):
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid,
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))

        loaded_stage_n_loc = loaded_stage_n_loc & FIFO_FOUP_loaded_pm

        # find the "earliest available" PM
        loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
        loc_batch_idx = loaded_stage_n_loc.nonzero()[:, 0]
        loc_idx = loaded_stage_n_loc.nonzero()[:, 1]
        loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
        est_ready_time = loc_est_ready_time.min(dim=-1)[0]

        est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
        est_loc = state.loc_process_end_time == est_ready_time_expand_loc # (b, loc)
        est_stage_n_loc = torch.logical_and(loaded_stage_n_loc, est_loc)
        est_stage_n_pm_loc = est_stage_n_loc[:, 1:-1] # exclude loadlock & unload

        backward_mask[~ll_batch_idx, self.num_lot_type:] = est_stage_n_pm_loc[~ll_batch_idx]

        return backward_mask

class Backward_z_policy:
    """
    Yu, T. S., Kim, H. J., & Lee, T. E. (2017). Scheduling single-armed cluster tools with
    chamber cleaning operations. IEEE Transactions on Automation Science and Engineering, 15(2),
    705-716.
    """
    def __init__(self, env) -> None:
        # env info
        self.batch_size = env.batch_size
        self.num_wafer = env.loadport_capacity * env.wafer.foup_size
        self.num_loc = env.loc.num_loc
        self.num_stage = env.loc.num_stage
        self.num_lot_type = env.num_lot_type
        # target stage index
        self.n = torch.full(size=self.batch_size, fill_value=self.num_stage, dtype=torch.long)
        # lot release
        self.lot_release = lot_input_seq_policy(rule=env.lot_release_rule, env=env)

    def to(self, device):
        self.n = self.n.to(device)
        self.lot_release.recipe = self.lot_release.recipe.to(device)

    def __call__(self, env, state):
        # get backward rule based action mask
        backward_mask = self.get_backward_z_mask(env, state)

        input_rule = True if self.lot_release.rule != 'random' else False
        if input_rule:
            input_mask = self.lot_release(env, state)
        else:
            input_mask = torch.ones_like(state.action_mask, dtype=torch.bool)

        # skip stage if empty stage
        while (~(state.action_mask & backward_mask & input_mask).any(dim=-1)).any():
            jump_stage_batch_idx = (~(state.action_mask & backward_mask & input_mask).any(dim=-1))
            # update backward stage index
            self.n[jump_stage_batch_idx] -= 1
            self.n[self.n < 0] = self.num_stage

            backward_mask = self.get_backward_z_mask(env, state)

        # update backward stage index
        self.n -= 1

        # update action mask
        action_mask = state.action_mask & backward_mask & input_mask
        assert (action_mask).any(), "no available action"

        # action selection
        # to fix pytorch.multinomial bug on selecting 0 probability elements
        while True:
            prob = action_mask.float()
            action = torch.multinomial(prob, 1).squeeze(-1)
            selected_prob = prob[state.batch_idx, action]
            selected_prob[state.done] = 1
            if (selected_prob != 0).all(): break

        # fixed input sequence
        #action[action < env.num_lot_type] = 0
        #action[action >= env.num_lot_type] = action[action >= env.num_lot_type] - env.num_lot_type + 1

        return action

    def get_backward_z_mask(self, env, state):
        # step1: empty stage skip
        n_expand_wafer = self.n[:, None].repeat(1, self.num_wafer)
        exist_unloadable_wafer = (state.wafer_stage == n_expand_wafer).any(dim=-1)
        while not exist_unloadable_wafer.all():
            self.n[~exist_unloadable_wafer] -= 1
            self.n[self.n < 0] = self.num_stage
            n_expand_wafer = self.n[:, None].repeat(1, self.num_wafer)
            exist_unloadable_wafer = (state.wafer_stage == n_expand_wafer).any(dim=-1)

        # step2: set backward policy mask
        backward_mask = torch.zeros_like(state.action_mask, dtype=torch.bool)

        # n == 0. loadlock unload case
        ll_batch_idx = self.n == 0
        backward_mask[ll_batch_idx, :self.num_lot_type] = True

        # n > 0. PM unload case
        # find the "loaded" & "stage n" location
        n_expand_loc = self.n[:, None].repeat(1, self.num_loc)
        stage_n_loc = state.loc_stage == n_expand_loc
        loaded_pm_loc = state.loc_hold_wafer != -1
        lm_loc = state.loc_stage == 0   # exception case: loadlock
        loaded_loc = torch.logical_or(loaded_pm_loc, lm_loc)
        loaded_stage_n_loc = torch.logical_and(stage_n_loc, loaded_loc)

        # FIFO FOUP PM
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1):
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid,
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))

        loaded_stage_n_loc = loaded_stage_n_loc & FIFO_FOUP_loaded_pm

        # find the "earliest available" PM
        loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
        loc_batch_idx = loaded_stage_n_loc.nonzero()[:, 0]
        loc_idx = loaded_stage_n_loc.nonzero()[:, 1]
        loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
        est_ready_time = loc_est_ready_time.min(dim=-1)[0]

        est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
        est_loc = state.loc_process_end_time == est_ready_time_expand_loc # (b, loc)
        est_stage_n_loc = torch.logical_and(loaded_stage_n_loc, est_loc)
        est_stage_n_pm_loc = est_stage_n_loc[:, 1:-1] # exclude loadlock & unload

        backward_mask[~ll_batch_idx, self.num_lot_type:] = est_stage_n_pm_loc[~ll_batch_idx]

        return backward_mask

class Swap_policy:
    def __init__(self, env) -> None:
        # env info
        self.batch_size = env.batch_size
        self.num_wafer = env.loadport_capacity * env.wafer.foup_size
        self.num_loc = env.loc.num_loc
        self.num_pm = env.loc.num_pm
        self.num_stage = env.loc.num_stage
        self.num_lot_type = env.num_lot_type
        self.purge_constraint = env.purge_constraint
        self.stage = env.loc.stage
        self.ts = torch.zeros(size=(*self.batch_size,), dtype=torch.int) # target stage index
        self.did_load = torch.ones(size=(*self.batch_size,), dtype=torch.bool) # previous action (0: unload, 1: load)
        self.need_swap = torch.zeros(size=(*self.batch_size,), dtype=torch.bool) # swap action flag


        self.ul_ll_end_idx = self.num_lot_type
        self.ul_start_idx = self.num_lot_type
        self.ul_end_idx = self.num_lot_type + self.num_pm
        self.l_arm1_start_idx = self.num_lot_type + self.num_pm
        self.l_arm1_end_idx = self.num_lot_type + self.num_pm * 2
        self.l_arm2_start_idx = self.num_lot_type + self.num_pm * 2 + 1
        self.l_arm2_end_idx = self.num_lot_type + self.num_pm * 3 + 1

        self.lot_release = lot_input_seq_policy(rule=env.lot_release_rule, env=env)

    def to(self, device):
        self.ts = self.ts.to(device)
        self.did_load = self.did_load.to(device)
        self.need_swap = self.need_swap.to(device)
        self.lot_release.recipe = self.lot_release.recipe.to(device)


    def __call__(self, env, state):
        # get swap rule based action mask
        swap_action_mask = self.get_swap_mask(env, state)

        # input sequence masking
        input_rule = True if self.lot_release.rule != 'random' else False
        if input_rule:
            input_mask = self.lot_release(env, state)
            #lot_release_batch_idx = action_mask[torch.arange(*self.batch_size), :self.num_lot_type].any(dim=-1)
            #action_mask[lot_release_batch_idx] = action_mask[lot_release_batch_idx] & input_mask[lot_release_batch_idx]
        else:
            input_mask = torch.ones_like(state.action_mask, dtype=torch.bool)

        # skip stage if empty stage
        while (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1)).any():
            jump_stage_batch_idx = (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1))
            self.ts[jump_stage_batch_idx] += 1
            self.ts[self.ts > self.num_stage + 1] = 0
            self.need_swap[jump_stage_batch_idx] = False
            swap_action_mask = self.get_swap_mask(env, state)

        # action mask update
        action_mask = state.action_mask & swap_action_mask & input_mask
        assert action_mask.any(), "no available action"


        # action selection
        while True:
            prob = action_mask.float()
            action = torch.multinomial(prob, 1).squeeze(-1)
            selected_prob = prob[state.batch_idx, action]
            selected_prob[state.done] = 1
            # to fix pytorch.multinomial bug on selecting 0 probability elements
            if (selected_prob != 0).all(): break

        # action mapping
        action_is_load = action >= self.l_arm1_start_idx
        action_is_unload_from_pm = (action >= self.ul_start_idx) & (action < self.ul_end_idx)
        action_is_unload_from_ll = action < self.ul_ll_end_idx

        # target stage update
        self.ts[action_is_load | action_is_unload_from_ll] += 1
        self.ts[self.ts > self.num_stage + 1] = 0 # comeback

        # prev action update
        self.need_swap[action_is_unload_from_pm] = True
        self.need_swap[~action_is_unload_from_pm] = False

        return action

        # input sequnece fix
        #action_is_load = action >= self.num_lot_type + self.num_pm
        #action_is_unload_from_pm = (action >= self.num_lot_type) & (action < self.num_lot_type+self.num_pm)
        #action_is_unload_from_ll = action < self.num_lot_type

        # target stage update
        #self.ts[action_is_load | action_is_unload_from_ll] += 1
        #self.ts[self.ts > self.num_stage + 1] = 0 # comeback

        # prev action update
        #self.need_swap[action_is_unload_from_pm] = True
        #self.need_swap[~action_is_unload_from_pm] = False

        # fixed input sequence
        #action[action_is_unload_from_ll] = 0
        #action[action_is_unload_from_pm] = action[action_is_unload_from_pm] - self.num_lot_type + 1
        #action[action_is_load] = action[action_is_load] - self.num_lot_type + 1

        #return action

    def get_swap_mask(self, env, state):
        """
        action case1: input loadlock
            - 1. pickup(move) unload from input loadlock

        action case2: PM
            - 1. pickup(move. PM of stage 1 with earliest available PM) + unload from PM
            - 2. rotate(move) + load to PM

        action case3: output loadlock
            - 1. rotate(move) + load to output loadlock
        """
        il_batch_idx = self.ts == 0                                          # input loadlock stage
        ol_batch_idx = self.ts == self.num_stage +1                          # output loadlock stage

        ll_batch_idx = torch.logical_or(il_batch_idx, ol_batch_idx)         # loadlock stage
        pm_batch_idx = (self.ts > 0) & (self.ts <= self.num_stage)           # process module stage


        swap_action_mask = torch.zeros_like(state.action_mask, dtype=torch.bool)

        # unload input loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        ull_batch_idx = ll_batch_idx & self.did_load
        swap_action_mask[ull_batch_idx, :self.ul_ll_end_idx] = True


        # load output loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        lll_batch_idx = ll_batch_idx & ~self.did_load
        swap_action_mask[lll_batch_idx, self.l_arm1_end_idx] = True
        swap_action_mask[lll_batch_idx, self.l_arm2_end_idx] = True

        self.did_load[ull_batch_idx] = False
        self.did_load[lll_batch_idx] = True

        # unload process module stage action batch
        # ------------------------------------------------------------------------------------------------------------
        tar_stage_loc_idx = self.stage == self.ts[:, None].repeat(1, self.num_loc)

        # 현재 PM에서 unload를 했을 때 -> load 실행
        load_to_pm_batch_idx = pm_batch_idx & self.need_swap
        swap_action_mask[load_to_pm_batch_idx, self.l_arm1_start_idx: self.l_arm1_end_idx] = tar_stage_loc_idx[load_to_pm_batch_idx, 1:-1]
        swap_action_mask[load_to_pm_batch_idx, self.l_arm2_start_idx: self.l_arm2_end_idx] = tar_stage_loc_idx[load_to_pm_batch_idx, 1:-1]
        self.did_load[load_to_pm_batch_idx] = True

        # 현재 PM에서 unload를 하지 않았을 때 -> unload 실행
        unload_from_pm_batch_idx = pm_batch_idx & ~self.need_swap

        # FIFO
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1):
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid,
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))
        # unload earliest process end PM mask
        unload_earliest_finish_pm = True
        if unload_earliest_finish_pm:
            tar_stage_hold_wafer_loc_idx = tar_stage_loc_idx & (state.loc_hold_wafer != -1) & FIFO_FOUP_loaded_pm
            loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
            loc_batch_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 0]
            loc_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 1]
            loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
            est_ready_time = loc_est_ready_time.min(dim=-1)[0]
            est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
            est_loc = state.loc_process_end_time == est_ready_time_expand_loc

            tar_stage_est_loc_idx = torch.logical_and(tar_stage_hold_wafer_loc_idx, est_loc)

            swap_action_mask[unload_from_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] =\
                tar_stage_est_loc_idx[unload_from_pm_batch_idx, 1:-1]

        else:
            swap_action_mask[unload_from_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] = tar_stage_loc_idx[unload_from_pm_batch_idx, 1:-1]
        self.did_load[unload_from_pm_batch_idx] = False

        # unload 할 필요 없이 load를 할 수 있을 때 -> load 실행
        # ------------------------------------------------------------------------------------------------------------
        return swap_action_mask

class Swap_z_policy:
    """
    Yu, T. S., & Lee, T. E. (2017). Scheduling dual-armed cluster tools with
    chamber cleaning operations. IEEE transactions on automation science and engineering,
    16(1), 218-228.

    z: partial loading stage
    """
    def __init__(self, env) -> None:
        # env info
        self.batch_size = env.batch_size
        self.num_wafer = env.loadport_capacity * env.wafer.foup_size
        self.num_loc = env.loc.num_loc
        self.num_pm = env.loc.num_pm
        self.num_stage = env.loc.num_stage
        self.num_lot_type = env.num_lot_type
        self.purge_constraint = env.purge_constraint
        self.stage = env.loc.stage
        self.ts = torch.zeros(size=(*self.batch_size,), dtype=torch.int) # target stage index
        self.did_load = torch.ones(size=(*self.batch_size,), dtype=torch.bool) # previous action (0: unload, 1: load)
        self.need_swap = torch.zeros(size=(*self.batch_size,), dtype=torch.bool) # swap action flag

        self.ul_ll_start_idx = 0
        self.ul_ll_end_idx = self.num_lot_type
        self.ul_start_idx = self.num_lot_type
        self.ul_end_idx = self.num_lot_type + self.num_pm
        self.l_arm1_start_idx = self.num_lot_type + self.num_pm
        self.l_arm1_end_idx = self.num_lot_type + self.num_pm * 2
        self.l_arm2_start_idx = self.num_lot_type + self.num_pm * 2 + 1
        self.l_arm2_end_idx = self.num_lot_type + self.num_pm * 3 + 1

        self.lot_release = lot_input_seq_policy(rule=env.lot_release_rule, env=env)

    def to(self, device):
        self.ts = self.ts.to(device)
        self.did_load = self.did_load.to(device)
        self.need_swap = self.need_swap.to(device)
        self.lot_release.recipe = self.lot_release.recipe.to(device)


    def __call__(self, env, state):
        # get swap rule based action mask
        swap_action_mask = self.get_swap_z_mask(state)

        # input sequence masking
        input_rule = True if self.lot_release.rule != 'random' else False
        if input_rule:
            input_mask = self.lot_release(env, state)
            #lot_release_batch_idx = action_mask[torch.arange(*self.batch_size), :self.num_lot_type].any(dim=-1)
            #action_mask[lot_release_batch_idx] = action_mask[lot_release_batch_idx] & input_mask[lot_release_batch_idx]
        else:
            input_mask = torch.ones_like(state.action_mask, dtype=torch.bool)

        # skip stage if empty stage
        while (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1)).any():
            jump_stage_batch_idx = (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1))
            self.ts[jump_stage_batch_idx] += 1
            self.ts[self.ts > self.num_stage + 1] = 0
            self.need_swap[jump_stage_batch_idx] = False
            swap_action_mask = self.get_swap_z_mask(state)

        # action mask update
        action_mask = state.action_mask & swap_action_mask & input_mask
        assert action_mask.any(), "no available action"

        # action selection
        while True:
            prob = action_mask.float()
            action = torch.multinomial(prob, 1).squeeze(-1)
            selected_prob = prob[state.batch_idx, action]
            selected_prob[state.done] = 1
            # to fix pytorch.multinomial bug on selecting 0 probability elements
            if (selected_prob != 0).all(): break


        # action mapping
        #action_is_load = action >= self.l_arm1_start_idx
        #action_is_unload_from_pm = (action >= self.ul_start_idx) & (action < self.ul_end_idx)
        #action_is_unload_from_ll = action < self.ul_ll_end_idx

        # input sequnece fix
        action_is_load = action >= self.num_lot_type + self.num_pm
        action_is_unload_from_pm = (action >= self.num_lot_type) & (action < self.num_lot_type+self.num_pm)
        action_is_unload_from_ll = action < self.num_lot_type


        # target stage update
        self.ts[action_is_load | action_is_unload_from_ll] += 1
        self.ts[self.ts > self.num_stage + 1] = 0 # comeback

        # prev action update
        self.need_swap[action_is_unload_from_pm] = True
        self.need_swap[~action_is_unload_from_pm] = False

        return action

    def get_swap_z_mask(self, state):
        """
        action case1: input loadlock
            - 1. pickup(move) unload from input loadlock

        action case2: PM
            - 1. pickup(move. PM of stage 1 with earliest available PM) + unload from PM
            - 2. rotate(move) + load to PM

        action case3: output loadlock
            - 1. rotate(move) + load to output loadlock
        """
        il_batch_idx = self.ts == 0                                          # input loadlock stage
        ol_batch_idx = self.ts == self.num_stage +1                          # output loadlock stage

        ll_batch_idx = torch.logical_or(il_batch_idx, ol_batch_idx)          # loadlock stage
        pm_batch_idx = (self.ts > 0) & (self.ts <= self.num_stage)           # process module stage


        # unload input loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        swap_action_mask = torch.zeros_like(state.action_mask, dtype=torch.bool)

        ull_batch_idx = ll_batch_idx & self.did_load
        swap_action_mask[ull_batch_idx, :self.ul_ll_end_idx] = True

        # load output loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        lll_batch_idx = ll_batch_idx & ~self.did_load
        swap_action_mask[lll_batch_idx, self.l_arm1_end_idx] = True
        swap_action_mask[lll_batch_idx, self.l_arm2_end_idx] = True

        # update
        self.did_load[ull_batch_idx] = False
        self.did_load[lll_batch_idx] = True

        # unload process module stage action batch
        # ------------------------------------------------------------------------------------------------------------
        tar_stage_loc_idx = self.stage == self.ts[:, None].repeat(1, self.num_loc)
        tar_stage_loc_purge_end_time = torch.where(tar_stage_loc_idx, state.loc_purge_end_time, 1e5) # only target stage PM search
        tar_stage_loc_purge_end_time = torch.where(state.loc_hold_wafer != -1, 1e5, tar_stage_loc_purge_end_time) # among the target stage PM, except processing PM
        earliest_purge_end_loc_idx, _ = tar_stage_loc_purge_end_time.min(dim=-1)
        load_pm_idx = tar_stage_loc_idx & (tar_stage_loc_purge_end_time == earliest_purge_end_loc_idx[:, None].repeat(1, self.num_loc))

        # 현재 PM에서 unload를 했을 때 -> load 실행
        load_to_pm_batch_idx = pm_batch_idx & self.need_swap
        swap_action_mask[load_to_pm_batch_idx, self.l_arm1_start_idx:self.l_arm1_end_idx] = load_pm_idx[load_to_pm_batch_idx, 1:-1]
        swap_action_mask[load_to_pm_batch_idx, self.l_arm2_start_idx:self.l_arm2_end_idx] = load_pm_idx[load_to_pm_batch_idx, 1:-1]
        self.did_load[load_to_pm_batch_idx] = True

        # 현재 PM에서 unload를 하지 않았을 때 -> unload 실행
        unload_from_pm_batch_idx = pm_batch_idx & ~self.need_swap

        # unload earliest process end PM mask
        unload_earliest_finish_pm = True
        if unload_earliest_finish_pm:
            tar_stage_hold_wafer_loc_idx = tar_stage_loc_idx & (state.loc_hold_wafer != -1)
            loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
            loc_batch_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 0]
            loc_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 1]
            loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
            est_ready_time = loc_est_ready_time.min(dim=-1)[0]
            est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
            est_loc = state.loc_process_end_time == est_ready_time_expand_loc

            tar_stage_est_loc_idx = torch.logical_and(tar_stage_hold_wafer_loc_idx, est_loc)
            swap_action_mask[unload_from_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] = tar_stage_est_loc_idx[unload_from_pm_batch_idx, 1:-1]

        else:
            swap_action_mask[unload_from_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] = tar_stage_loc_idx[unload_from_pm_batch_idx, 1:-1]
        self.did_load[unload_from_pm_batch_idx] = False


        return swap_action_mask

class Swap_a_z_policy:
    """
    Yu, T. S., & Lee, T. E. (2017). Scheduling dual-armed cluster tools with
    chamber cleaning operations. IEEE transactions on automation science and engineering,
    16(1), 218-228.

    z: partial loading stage
    a: push-and-wait stage
    We let a = 1 if a swap operation is used for process step i and a = 0 if a push-and-wait operation is used.

    z가 1>= 인 stage만 a=0 설정 가능. empty PM이 최소 1개는 있어야 push 가능.
    """
    def __init__(self, env) -> None:
        # env info
        self.batch_size = env.batch_size
        self.num_wafer = env.loadport_capacity * env.wafer.foup_size
        self.num_loc = env.loc.num_loc
        self.num_pm = env.loc.num_pm
        self.num_stage = env.loc.num_stage
        self.num_lot_type = env.num_lot_type
        self.purge_constraint = env.purge_constraint
        self.stage = env.loc.stage
        self.ts = torch.zeros(size=(*self.batch_size,), dtype=torch.int) # target stage index

        self.did_load = torch.ones(size=(*self.batch_size, self.num_stage+1), dtype=torch.bool) # (0: unload, 1: load)
        self.stage_strategy = torch.cat([torch.tensor([1]),torch.tensor(env.kwargs['strategy'])]) # 0: push-and-wait stage, 1: swap stage. First one is for loadlock stage
        self.did_load[:, self.stage_strategy==0] = False

        #assert ~((torch.tensor([1] + list(env.init_partial_loading)) + self.stage_strategy) == 0).any(), \
        #    "no empty PM for push-and-wait stage"
        assert ~((torch.tensor([1] + list(env.wafer.z)) + self.stage_strategy) == 0).any(), \
            "no empty PM for push-and-wait stage"

        self.ul_ll_start_idx = 0
        self.ul_ll_end_idx = self.num_lot_type
        self.ul_start_idx = self.num_lot_type
        self.ul_end_idx = self.num_lot_type + self.num_pm
        self.l_arm1_start_idx = self.num_lot_type + self.num_pm
        self.l_arm1_end_idx = self.num_lot_type + self.num_pm * 2
        self.l_arm2_start_idx = self.num_lot_type + self.num_pm * 2 + 1
        self.l_arm2_end_idx = self.num_lot_type + self.num_pm * 3 + 1

        lot_release_rule = env.kwargs.get('lot_release', 'random')
        self.lot_release = lot_input_seq_policy(rule=lot_release_rule, env=env)

    def __call__(self, env, state):
        # get swap rule based action mask
        swap_action_mask = self.get_swap_a_z_mask(env, state)

        # input sequence masking
        input_rule = True if self.lot_release.rule != 'random' else False
        if input_rule:
            input_mask = self.lot_release(env, state)
            #lot_release_batch_idx = action_mask[torch.arange(*self.batch_size), :self.num_lot_type].any(dim=-1)
            #action_mask[lot_release_batch_idx] = action_mask[lot_release_batch_idx] & input_mask[lot_release_batch_idx]
        else:
            input_mask = torch.ones_like(state.action_mask, dtype=torch.bool)

        # skip stage if empty stage
        while (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1)).any():
            jump_stage_batch_idx = (~(state.action_mask & swap_action_mask & input_mask).any(dim=-1))
            self.ts[jump_stage_batch_idx] += 1
            self.ts[self.ts >= self.num_stage + 1] = 0
            swap_action_mask = self.get_swap_a_z_mask(env, state)

        # action mask update
        action_mask = state.action_mask & swap_action_mask & input_mask
        assert action_mask.any(), "no available action"

        # action selection
        while True:
            prob = action_mask.float()
            action = torch.multinomial(prob, 1).squeeze(-1)
            selected_prob = prob[state.batch_idx, action]
            selected_prob[state.done] = 1
            # to fix pytorch.multinomial bug on selecting 0 probability elements
            if (selected_prob != 0).all(): break


        # action mapping
        is_push_stage = self.stage_strategy[None, :].repeat(*self.batch_size, 1)[torch.arange(*self.batch_size), self.ts] == 0
        is_out_loadlock = torch.logical_or(action == self.l_arm1_end_idx,
                                           action == self.l_arm2_end_idx)
        action_is_push_load_to_pm = (action >= self.l_arm1_start_idx) & is_push_stage & ~is_out_loadlock
        action_is_swap_load_to_pm = (action >= self.l_arm1_start_idx) & ~is_push_stage & ~is_out_loadlock
        action_is_push_unload_from_pm  = (action >= self.ul_start_idx) & (action < self.ul_end_idx) & is_push_stage
        action_is_swap_unload_from_pm = (action >= self.ul_start_idx) & (action < self.ul_end_idx) & ~is_push_stage
        action_is_unload_from_ll = action < self.ul_ll_end_idx


        # action update (loaded or unloaded)
        self.did_load[action_is_push_load_to_pm, self.ts[action_is_push_load_to_pm]] = True  # push stage
        self.did_load[action_is_swap_load_to_pm, self.ts[action_is_swap_load_to_pm]] = True  # swap stage
        self.did_load[~action_is_unload_from_ll, self.ts[~action_is_unload_from_ll]] = True  # out loadlock
        self.did_load[action_is_push_unload_from_pm, self.ts[action_is_push_unload_from_pm]] = False # push stage
        self.did_load[action_is_swap_unload_from_pm, self.ts[action_is_swap_unload_from_pm]] = False # swap stage
        self.did_load[action_is_unload_from_ll, self.ts[action_is_unload_from_ll]] = False # in loadlock

        # target stage update
        self.ts[action_is_unload_from_ll] += 1
        self.ts[action_is_push_unload_from_pm] += 1
        self.ts[action_is_swap_load_to_pm] += 1
        self.ts[self.ts >= self.num_stage + 1] = 0 # comeback

        return action

    def get_swap_a_z_mask(self, env, state):
        """
        action case1: input loadlock
            - 1. pickup(move) unload from input loadlock

        action case2: PM
            - 1. pickup(move. PM of stage 1 with earliest available PM) + unload from PM
            - 2. rotate(move) + load to PM

        action case3: output loadlock
            - 1. rotate(move) + load to output loadlock
        """
        il_batch_idx = self.ts == 0                                          # input loadlock stage
        ol_batch_idx = self.ts == self.num_stage +1                          # output loadlock stage

        ll_batch_idx = torch.logical_or(il_batch_idx, ol_batch_idx)          # loadlock stage
        pm_batch_idx = (self.ts > 0) & (self.ts <= self.num_stage)           # process module stage


        ##########################
        # loadloack
        ##########################
        # [unload in loadlock] unload input loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        swap_action_mask = torch.zeros_like(state.action_mask, dtype=torch.bool)
        before_loaded = self.did_load[torch.arange(*self.batch_size), self.ts]


        ull_batch_idx = ll_batch_idx & before_loaded
        swap_action_mask[ull_batch_idx, :self.ul_ll_end_idx] = True

        # [load out loadlock]  load output loadlock stage action batch
        # ------------------------------------------------------------------------------------------------------------
        lll_batch_idx = ll_batch_idx & ~before_loaded
        swap_action_mask[lll_batch_idx, self.l_arm1_end_idx] = True
        swap_action_mask[lll_batch_idx, self.l_arm2_end_idx] = True

        # update
        self.did_load[ull_batch_idx, self.ts[ull_batch_idx]] = False
        self.did_load[lll_batch_idx, self.ts[lll_batch_idx]] = True

        ##########################
        # PM
        ##########################
        # [load PM]
        # ------------------------------------------------------------------------------------------------------------
        # get earliest load PM of stage
        tar_stage_loc_idx = self.stage == self.ts[:, None].repeat(1, self.num_loc)
        tar_stage_loc_purge_end_time = torch.where(tar_stage_loc_idx, state.loc_purge_end_time, 1e5) # only target stage PM search
        tar_stage_loc_purge_end_time = torch.where(state.loc_hold_wafer != -1, 1e5, tar_stage_loc_purge_end_time) # among the target stage PM, except processing PM
        earliest_purge_end_loc_idx, _ = tar_stage_loc_purge_end_time.min(dim=-1)
        earliest_loadable_loc_idx = (tar_stage_loc_purge_end_time == earliest_purge_end_loc_idx[:, None].repeat(1, self.num_loc))
        load_pm_idx = tar_stage_loc_idx & earliest_loadable_loc_idx

        #
        is_push_stage = self.stage_strategy[None, :].repeat(*self.batch_size, 1)[torch.arange(*self.batch_size), self.ts] == 0

        # [push-and-wait stage load PM]
        load_to_push_pm_batch_idx = (pm_batch_idx & is_push_stage & ~before_loaded)
        swap_action_mask[load_to_push_pm_batch_idx, self.l_arm1_start_idx:self.l_arm1_end_idx] = load_pm_idx[load_to_push_pm_batch_idx, 1:-1]
        swap_action_mask[load_to_push_pm_batch_idx, self.l_arm2_start_idx:self.l_arm2_end_idx] = load_pm_idx[load_to_push_pm_batch_idx, 1:-1]
        self.did_load[load_to_push_pm_batch_idx, self.ts[load_to_push_pm_batch_idx]] = True

        # [swap stage load PM]
        load_to_swap_pm_batch_idx = (pm_batch_idx & ~is_push_stage & ~before_loaded)
        swap_action_mask[load_to_swap_pm_batch_idx, self.l_arm1_start_idx:self.l_arm1_end_idx] = load_pm_idx[load_to_swap_pm_batch_idx, 1:-1]
        swap_action_mask[load_to_swap_pm_batch_idx, self.l_arm2_start_idx:self.l_arm2_end_idx] = load_pm_idx[load_to_swap_pm_batch_idx, 1:-1]
        self.did_load[load_to_swap_pm_batch_idx, self.ts[load_to_swap_pm_batch_idx]] = True

        # [unload PM]
        # ------------------------------------------------------------------------------------------------------------
        # [push-and-wait stage unload PM]
        unload_from_push_pm_batch_idx = (pm_batch_idx & is_push_stage & before_loaded)

        # FIFO
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1):
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid,
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))

        # unload earliest process end PM mask
        tar_stage_hold_wafer_loc_idx = tar_stage_loc_idx & (state.loc_hold_wafer != -1) & FIFO_FOUP_loaded_pm
        loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
        loc_batch_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 0]
        loc_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 1]
        loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
        est_ready_time = loc_est_ready_time.min(dim=-1)[0]
        est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
        est_loc = state.loc_process_end_time == est_ready_time_expand_loc
        tar_stage_est_loc_idx = torch.logical_and(tar_stage_hold_wafer_loc_idx, est_loc)

        #
        swap_action_mask[unload_from_push_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] = tar_stage_est_loc_idx[unload_from_push_pm_batch_idx, 1:-1]
        self.did_load[unload_from_push_pm_batch_idx, self.ts[unload_from_push_pm_batch_idx]] = False

        # [swap stage unload PM]
        unload_from_swap_pm_batch_idx = (pm_batch_idx & ~is_push_stage & before_loaded)

        # unload earliest process end PM mask
        tar_stage_hold_wafer_loc_idx = tar_stage_loc_idx & (state.loc_hold_wafer != -1) & FIFO_FOUP_loaded_pm
        loc_est_ready_time = torch.full_like(state.loc_id, fill_value=1e9, dtype=torch.float)
        loc_batch_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 0]
        loc_idx = tar_stage_hold_wafer_loc_idx.nonzero()[:, 1]
        loc_est_ready_time[loc_batch_idx, loc_idx] = state.loc_process_end_time[loc_batch_idx, loc_idx]
        est_ready_time = loc_est_ready_time.min(dim=-1)[0]
        est_ready_time_expand_loc = est_ready_time[:, None].repeat(1, self.num_loc)
        est_loc = state.loc_process_end_time == est_ready_time_expand_loc
        tar_stage_est_loc_idx = torch.logical_and(tar_stage_hold_wafer_loc_idx, est_loc)

        #
        swap_action_mask[unload_from_swap_pm_batch_idx, self.ul_start_idx: self.ul_end_idx] = tar_stage_est_loc_idx[unload_from_swap_pm_batch_idx, 1:-1]
        self.did_load[unload_from_swap_pm_batch_idx, self.ts[unload_from_swap_pm_batch_idx]] = False

        return swap_action_mask


"========================================================================================================================="
ROBOT_POLICY_REGISTRY = {
    "random": Random_policy,
    "backward": Backward_policy,
    "backward_z": Backward_z_policy,
    "swap": Swap_policy,
    "swap_z": Swap_z_policy,
    "swap_a_z": Swap_a_z_policy
}


def get_policy(policy_name, env):
    """Get policy by name.
    Args:
        policy_name: Policy name
    Returns:
        Policy class
    """
    policy_class = ROBOT_POLICY_REGISTRY.get(policy_name, None)
    if policy_class is None:
        raise ValueError(
            f"Unknown policy {policy_name}. Available policies: {ROBOT_POLICY_REGISTRY.keys()}"
        )
    policy = policy_class(env) # Initialize the policy with the environment

    return policy


def rollout(env: object, state:object, policy: str='random'):
    robot_policy = get_policy(policy, env)
    # rollout
    while not state.done.all():
        action = robot_policy(state)
        state = env.step(action)

    return state

