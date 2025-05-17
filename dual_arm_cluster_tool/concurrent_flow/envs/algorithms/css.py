from typing import Any
import copy
import numpy as np

class ConcurrentSwapSequence:
    """
    Concurrent swap sequence (CSS): Concurrent swap sequence

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
        self.n = 0  # start from the 0 stage of group1 (forward)
        self.curr_group = 0  # group1
        self.curr_cycle = copy.deepcopy(self.cycle_plan)

        self.swap = False

    def __call__(self, env: object) -> int:
        css_mask = self._create_css_mask(env)
        action_mask = np.logical_and(env.action_mask,css_mask)
        assert action_mask.any(), "no available action"

        action = self._select_action(action_mask)
        self._update_state(env)

        return action

    def _create_css_mask(self, env):
        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(env.pms)
        num_load_action = len(env.pms) + 1
        start_idx = num_ll_unload_action + num_pm_unload_action

        css_mask = np.zeros_like(env.action_mask)

        def find_arm_id():
            for i, j in enumerate(env.robot.hold_wafer):
                if j is not None and env._get_next_stage_of_wafer(j) == self.n:
                    return i
            return None

        if self.n == env.outloadlock:  # produced wafer
            arm_id = find_arm_id()
            if arm_id is not None:
                css_mask[start_idx + (1 + arm_id) * num_load_action - 1] = 1
        else:
            if self.swap:  # load only
                arm_id = find_arm_id()
                if arm_id is not None:
                    for idx, pm in enumerate(env.pms, start=start_idx):
                        if pm.group == self.curr_group and self.n == pm.stage:
                            css_mask[idx + arm_id * num_load_action] = 1
            else:  # unload only
                if self.n == 0:
                    css_mask[self.curr_group] = 1
                else:
                    for idx, pm in enumerate(env.pms, start=num_ll_unload_action):
                        if pm.group == self.curr_group and self.n == pm.stage:
                            css_mask[idx] = 1

        return css_mask

    def _select_action(self, action_mask):
        while True:
            prob = action_mask.astype(float)
            action = np.random.choice(np.flatnonzero(prob))
            if prob[action] != 0:
                return action

    def _update_state(self, env):
        num_stage = len(env.group1_stage) + len(env.group2_stage)

        if self.n == 0: # unloaded  from LL
            self.n = 1 if self.curr_group == 0 else len(env.group1_stage) + 1

        elif self.n == num_stage+1: # produced wafer
            self.n = 0
            self.curr_cycle[self.curr_group] -= 1

            if self.curr_cycle[self.curr_group] == 0:
                if self.curr_cycle[1 - self.curr_group] > 0:
                    self.curr_group = 1 - self.curr_group
                else:
                    # reset cycle
                    self.curr_cycle = copy.deepcopy(self.cycle_plan)
                    self.n = 0
                    self.curr_group = 0

        else:
            self.swap = not self.swap
            if not self.swap: # next do unload
                if self.curr_group == 0 and self.n == len(env.group1_stage):
                    self.n = num_stage+1
                else:
                    self.n += 1
