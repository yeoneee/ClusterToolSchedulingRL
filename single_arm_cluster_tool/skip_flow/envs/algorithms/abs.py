import copy
import numpy as np


class AlternatingBackwardSequence:
    """

    Fundamental cycle plan: (A, B), or (B, A)


    Reference:  Scheduling cluster tools for concurrent processing of two wafer types
    with PM sharing, IJPR 15'
    """


    def __init__(self, env, strategy='fundamental'):
        if strategy == 'fundamental':
            self.cycle_plan = [1, 1]
        else:
            NotImplementedError
        self._initialize_current_logic(env)

    def _initialize_current_logic(self, env):
        self.curr_group = 0
        self.curr_cycle = copy.deepcopy(self.cycle_plan)
        self.shared_n = env.shared_stage

        self.n = self._get_last_stage(env, self.curr_group)

    def __call__(self, env: object):
        abs_mask = self._create_abs_mask(env)
        action_mask = np.logical_and(env.action_mask, abs_mask)
        if action_mask.sum() == 0:
            print('no available action')
            env.get_action_mask()
            self._create_abs_mask(env)
        assert action_mask.any(), "no available action"

        action = self._select_action(action_mask)

        return action


    def _create_abs_mask(self, env):
        abs_mask = np.zeros_like(env.action_mask)
        if self.n == 0: # in LL
            abs_mask[0 if self.curr_group == 0 else 1] = 1 # mask update

            if len(self.shared_n) > 1:
                self.curr_group = 1 - self.curr_group # group update
            self.n = self._get_last_stage(env, self.curr_group) # n update

        elif self.n in self.shared_n: # shared
            # mask update
            for i, pm in enumerate(env.pms, start=2):
                if pm.group == -1 and self.n == pm.stage:
                    abs_mask[i] = 1

            # n update
            self.curr_group = 1 - self.curr_group
            self.n = self._get_prev_stage(env, self.n, self.curr_group)

        else:  # nonshared
            prod_wafer = False
            if self._is_last_stage(env, self.n, self.curr_group):
                prod_wafer = True

            # mask update
            for i, pm in enumerate(env.pms, start=2):
                if pm.group == self.curr_group and self.n == pm.stage:
                    abs_mask[i] = 1

            # n update
            self.n = self._get_prev_stage(env, self.n, self.curr_group)

            # cycle update
            if prod_wafer:
                self.curr_cycle[self.curr_group] -= 1

        # reset cycle
        if sum(self.curr_cycle) == 0:
            self._reset_cycle(env)

        return abs_mask

    def _select_action(self, action_mask):
        while True:
            prob = action_mask.astype(float)
            action = np.random.choice(np.flatnonzero(prob))
            if prob[action] != 0:
                return action


    def _is_last_stage(self, env, n, group):
        group_stage = env.group1_stage if group == 0 else env.group2_stage
        if sum(group_stage[:n]) == sum(group_stage) and group_stage[n-1] != 0:
            return True
        else:
            return False

    def _get_prev_stage(self, env, n, group):
        group_stage = env.group1_stage if group == 0 else env.group2_stage
        for i in range(n-1 -1, -1, -1):
            if group_stage[i] != 0:
                return i+1
        return 0  # in LL

    def _get_last_stage(self, env, group):
        group_stage = env.group1_stage if group == 0 else env.group2_stage
        for i in range(len(group_stage)-1, -1, -1):
            if group_stage[i] != 0:
                return i+1
        return -1

    def _reset_cycle(self, env):
        self.curr_cycle = copy.deepcopy(self.cycle_plan)
