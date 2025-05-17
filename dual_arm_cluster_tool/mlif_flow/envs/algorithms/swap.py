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
                # input sequencing for workload balancing
                wafer_type_cnt = input_lots_cnt.sum(dim=-1) #(batch, num_type)
                pass

            elif self.rule == 'neh':
                pass

                """
                # Select the next job type by popping from the current neh_sequence
                # Only the selected job type can be selected
                # Need to end the schedule when the entire neh sequence is executed.
                if self.env.kwargs['next_release_wafer_idx'][0] < self.env.kwargs['release_sequence'].size(-1):
                    lot_types = torch.arange(input_lots.size(1))[None, :].repeat(input_lots.size(0), 1)
                    # Check the order in the input sequence of the release wafer
                    release_wafer_id = self.env.kwargs['release_sequence'][
                        self.env.batch_idx, self.env.kwargs['next_release_wafer_idx']]
                    # Check the FOUP number by looking at the size of the order -> Check the recipe through Foup index, wafer id index
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

"========================================================================================================================="
ROBOT_POLICY_REGISTRY = {
    "random": Random_policy,
    "swap": Swap_policy,
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

