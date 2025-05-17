from typing import Any
import copy
import numpy as np

class ConcurrentBackwardSequence:
    """
    TODO: group family for each wafer type plan
    CBS: Concurrent backward sequence

    Need to determine the cycle plan (w1, w2).
    The bottleneck PM PM^1, and PM^2 for each wafer type.
    The workload of PM^1, WL1 = p1 + 2u + 2l + 3t, and WL2 = p2 + 2u + 2l + 3t.

    if WL1 > WL2:
        plan1:  (1, ceil(WL1/WL2))
        plan2:  (1, floor(WL1/WL2))
    else:
        plan1:  (ceil(WL2/WL1), 1)
        plan2:  (floor(WL2/WL1), 1)

    select the better plan from these two options.

    Reference: Kim, H. J., & Lee, J. H. (2024). Scheduling Cluster Tools for Concurrent Processing:
    Deep Reinforcement Learning With Adaptive Search. IEEE Transactions on Automation Science and Engineering.
    """
    def __init__(self, env, strategy):
        self._initialize_workloads(env)
        self._set_cycle_plan(strategy)
        self._initialize_current_logic(env)

    def _initialize_workloads(self, env):
        self.group1_bottleneck_stage = np.argmax(env.recipes[0].time)
        p1 = env.recipes[0].time[self.group1_bottleneck_stage]
        self.wl1 = p1 + 2 * env.robot.unload_time + 2 * env.robot.load_time + 3 * env.robot.move_time

        self.group2_bottleneck_stage = np.argmax(env.recipes[1].time)
        p2 = env.recipes[1].time[self.group2_bottleneck_stage]
        self.wl2 = p2 + 2 * env.robot.unload_time + 2 * env.robot.load_time + 3 * env.robot.move_time

    def _set_cycle_plan(self, strategy):
        workload_ratio = (self.wl1 / self.wl2) if self.wl1 > self.wl2 else (self.wl2 / self.wl1)

        if self.wl1 > self.wl2:
            self.cycle_plan = [1, np.ceil(workload_ratio)] if strategy == 'ceil' else [1, np.floor(workload_ratio)]
        else:
            self.cycle_plan = [np.ceil(workload_ratio), 1] if strategy == 'ceil' else [np.floor(workload_ratio), 1]

        self.cycle_plan = [int(x) for x in self.cycle_plan]

    def _initialize_current_logic(self, env):
        self.n = len(env.group1_stage)  # start from the last stage of group1 (backward)
        self.curr_group = 0  # group1
        self.curr_cycle = copy.deepcopy(self.cycle_plan)

    def __call__(self, env: object) -> int:
        cbs_mask = self._create_cbs_mask(env)
        action_mask = np.logical_and(env.action_mask,cbs_mask)
        assert action_mask.any(), "no available action"

        action = self._select_action(action_mask)
        self._update_state(env)

        return action

    def _create_cbs_mask(self, env):
        # TODO: parallel stage

        cbs_mask = np.zeros_like(env.action_mask)

        # inLL
        if self.n == 0:
            cbs_mask[0 if self.curr_group == 0 else 1] = 1

        # PMs
        for idx, pm in enumerate(env.pms, start=2):
            if pm.group == self.curr_group and self.n == pm.stage:
                cbs_mask[idx] = 1
        return cbs_mask

    def _select_action(self, action_mask):
        while True:
            prob = action_mask.astype(float)
            action = np.random.choice(np.flatnonzero(prob))
            if prob[action] != 0:
                return action

    def _update_state(self, env):
        self.n -= 1
        if self.n == len(env.group1_stage) and self.curr_group == 1:
            self.n = 0

        if self.n < 0:  # finish the current group
            self.curr_cycle[self.curr_group] -= 1

            if self.curr_group == 0:
                if self.curr_cycle[0] >= 1:
                    self.n = len(env.group1_stage)

                elif self.curr_cycle[1] >= 1:
                    self.curr_group = 1
                    self.n = len(env.stage)

                else:
                    self._reset_cycle(env)

            elif self.curr_group == 1:
                if self.curr_cycle[1] >= 1:
                    self.n = len(env.stage)

                elif self.curr_cycle[0] >= 1:
                    self.curr_group = 0
                    self.n = len(env.group1_stage)

                else:
                    self._reset_cycle(env)

    def _reset_cycle(self, env):
        self.curr_cycle = copy.deepcopy(self.cycle_plan)
        self.n = len(env.group1_stage)
        self.curr_group = 0
