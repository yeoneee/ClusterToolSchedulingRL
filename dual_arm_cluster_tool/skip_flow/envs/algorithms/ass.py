from typing import Any
import copy
import numpy as np

class AlternatingSwapSequence:
    """
    TODO: group family for each wafer type plan
    ASS: Alternative swap sequence

    Fundamental cycle plan: (A, B), or (B, A)

    Reference:  Scheduling cluster tools for concurrent processing of two wafer types
    with PM sharing, IJPR 15'
    """
    def __init__(self, env):
        self.cycle_plan = [1,1]
        self._initialize_current_logic(env)


    def _initialize_current_logic(self, env):
        self.shared_n = env.shared_stage
        self.n = 0  # start from the 0 stage of group1 (forward)
        self.curr_group = 1  if len(self.shared_n) == 1 else 0 # group1
        self.curr_cycle = copy.deepcopy(self.cycle_plan)
        self.swap = False

    def __call__(self, env: object) -> int:
        ass_mask = self._create_ass_mask(env)
        action_mask = np.logical_and(env.action_mask,ass_mask)
        if not action_mask.any():
            print("no avail actions")
            env.get_action_mask()
            self._create_ass_mask(env)

        assert action_mask.any(), "no available action"

        action = self._select_action(action_mask)

        return action

    def _create_ass_mask(self, env):
        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(env.pms)
        num_load_action = len(env.pms) + 1
        start_idx = num_ll_unload_action + num_pm_unload_action
        ass_mask = np.zeros_like(env.action_mask)

        def _find_arm_id():
            for i, j in enumerate(env.robot.hold_wafer):
                if j is not None and env._get_next_stage_of_wafer(j) == self.n:
                    return i
            return None

        if self.n == env.outloadlock:  # produced wafer (=load to out loadlock)
            arm_id = _find_arm_id()
            if arm_id is not None:
                ass_mask[start_idx + (1 + arm_id) * num_load_action - 1] = 1
                self.n = 0 # return to the in-loadlock
                self.curr_cycle[self.curr_group] -= 1
                self.swap = False
                if len(self.shared_n) > 1:
                    self.curr_group = 1 - self.curr_group

        else:
            if self.swap:  # load only
                arm_id = _find_arm_id()
                if arm_id is not None:
                    for idx, pm in enumerate(env.pms, start=start_idx):
                        if pm.group in [self.curr_group,-1]  and self.n == pm.stage:
                            ass_mask[idx + arm_id * num_load_action] = 1

                if self.n in self.shared_n:
                   self.curr_group = 1- self.curr_group
                self.n = self._get_next_stage(env, self.n, self.curr_group)
                self.swap = False

            else:  # unload only
                if self.n == 0: # unload in-loadlock
                    ass_mask[self.curr_group] = 1
                    self.n = self._get_next_stage(env, self.n, self.curr_group)
                    self.swap = False

                else:  # unload pm
                    for idx, pm in enumerate(env.pms, start=num_ll_unload_action):
                        if pm.group in [self.curr_group,-1] and self.n == pm.stage:
                            ass_mask[idx] = 1
                    #if self.n in self.shared_n:
                    #   self.curr_group = 1- self.curr_group
                    self.swap = True

        # reset cycle
        if sum(self.curr_cycle) == 0:
            self._reset_cycle()

        return ass_mask

    def _select_action(self, action_mask):
        while True:
            prob = action_mask.astype(float)
            action = np.random.choice(np.flatnonzero(prob))
            if prob[action] != 0:
                return action

    def _get_next_stage(self, env, n, group):
        group_stage = env.group1_stage if group == 0 else env.group2_stage
        for i in range(n, len(env.stage)+1):
            if i == len(group_stage):
                return env.outloadlock
            if group_stage[i] != 0:
                return i+1


    def _reset_cycle(self):
        self.curr_cycle = copy.deepcopy(self.cycle_plan)