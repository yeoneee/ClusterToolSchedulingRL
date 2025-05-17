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

"========================================================================================================================="
ROBOT_POLICY_REGISTRY = {
    "random": Random_policy,
    "backward_z": Backward_z_policy,

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

