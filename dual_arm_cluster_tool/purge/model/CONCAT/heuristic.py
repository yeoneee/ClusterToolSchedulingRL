import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils

import copy
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

from envs.clustertool import NoncyclicClusterToolEnv as Env

@dataclass
class EnvNode:
    env: object = None # current environment
    state: object = None # current state

    parent: Optional['EnvNode'] = None # parent node
    child: Optional['EnvNode'] = None  # child node

    action: object = None        # selected action

    cands_action: torch.Tensor = None        # candidate action masks
    cands_start: torch.Tensor = None       # candidate durations
    cands_explor: torch.Tensor = None      # explored actions 1: True 0: False



class odest_framework:
    """ Perform ODEST algorithm with given robot move polcy
    El Amraoui, A., & Elhafsi, M. (2016). An efficient new heuristic
    for the hoist scheduling problem. Computers & Operations Research, 67, 184-192.

    Procedure:
        Loop
            1. Find the action by robot move policy
            2. Env step with the action
            3. Check the violation & violation time
            4. violation time > 0 --> backtracking batch. otherwise, forward batch

    """
    def __init__(self, env, policy) -> None:
        #self.robot_policy = get_policy(policy, env)


        self.root_node = EnvNode(env=env,
                                 state=env.reset(batch_size=1, device='cpu'),
                                 parent=None,
                                 child=None,
                                 action=None,
                                 cands_action=env.action_mask,
                                 cands_start=env.get_action_start_time(),
                                 cands_explor=torch.zeros_like(env.action_mask)
                                )
        # action excess time
        self.epsilon = torch.zeros(
            size=(env.num_foup,
                  env.foup_size,
                  env.loc.num_stage),
            dtype=torch.float32
        )

    def run(self):
        node = copy.deepcopy(self.root_node)
        while True:
            action_idx = self.get_earliest_start_time_action(node)

            if self.is_action_violated(node, action_idx):
                self.save_excess_time(node, action_idx)
                node = self.backtracking(node)

                if self.is_prev_op_of_violated_action(node):
                    self.delay_load_time(node)

                else:
                    self.update_candidate_mask(node)

            else:
                 node = self.forward(node, action_idx)

            if node.env.done:
                break

    def forward(self, node, action_idx):
        # Create next node
        next_node = copy.deepcopy(node)
        next_node.state = next_node.env.step(action_idx)
        next_node.env.render(node.env.get_action_mapping(action_idx), 0)
        next_node.parent = node

        next_node.cands_action = next_node.env.action_mask
        next_node.cands_start = next_node.env.get_action_start_time()
        next_node.cands_explor = torch.zeros_like(next_node.env.action_mask)

        # Update the current node
        action = node.env.get_action_mapping(action_idx)
        node.child = next_node
        node.action = action

        is_pm_unload = (node.action.unload_loc != 0) * node.env.num_lot_type
        node.cands_explor[0, is_pm_unload + node.action.unload_loc -1] = 1

        return next_node

    def update_candidate_mask(self, node):
        is_pm_unload = (node.action.unload_loc != 0) * node.env.num_lot_type
        node.cands_explor[0, is_pm_unload + node.action.unload_loc -1] = 1
        node.cands_action = torch.logical_and(node.cands_action, node.cands_explor==0)

    def is_action_violated(self, node, action_idx):
        # Check if the action is violated
        temp_node = copy.deepcopy(node)

        # action index is converted to action object
        action = temp_node.env.get_action_mapping(action_idx)
        temp_node.state = temp_node.env.step(action_idx)

        # residency, process, delay constraint check
        residency_start_time = torch.clamp(temp_node.env.wafer.residency_time[0, action.foup_idx, action.wafer_idx, action.unload_loc, 0],0)
        residency_end_time = temp_node.env.wafer.residency_time[0, action.foup_idx, action.wafer_idx, action.unload_loc, 1]
        residency_time = residency_end_time - residency_start_time

        process_start_time = temp_node.env.loc.process_start_time[0, action.unload_loc]
        process_end_time = temp_node.env.loc.process_end_time[0, action.unload_loc]
        process_time = process_end_time - process_start_time

        delay_limit_time = temp_node.env.loc.delay_limit[0, action.unload_loc]

        # is violated action?
        is_violated = residency_time > process_time + delay_limit_time

        # save the excess time
        self.action_excess_time = residency_time - (process_time + delay_limit_time) if is_violated else 0

        if action.foup_idx == 0 or action.unload_loc == 0:
            is_violated = False

        return is_violated

    def save_excess_time(self, node, action_idx):
        # line 9, Compute the upper-time and store
        action = node.env.get_action_mapping(action_idx)
        self.epsilon[action.foup_idx,
                     action.wafer_idx,
                     node.env.loc.stage[0, action.unload_loc]] = self.action_excess_time

    def backtracking(self, node):
        return node.parent

    def is_prev_op_of_violated_action(self, node):
        # Check if the backtraced step's action is the previous operation of the violated action
        action = node.action
        unload_stage = node.env.loc.stage[0, action.unload_loc]
        is_prev_op = self.epsilon[action.foup_idx,
                                  action.wafer_idx,
                                  max(unload_stage+1, node.env.num_stage)] != 0

        return is_prev_op

    def delay_load_time(self, node):
        node.cands_start[node.action] = node.cands_start[node.action] + self.action_excess_time

    def get_earliest_start_time_action(self, node):
        # TODO break tie rule
        cands_action = torch.logical_and(node.cands_action, node.cands_explor==0)
        est = node.cands_start[cands_action].min()[None]
        est_cands_action = torch.logical_and(node.cands_action, node.cands_start==est)
        action = torch.multinomial(est_cands_action.float(), 1).squeeze(-1)

        return action



if '__main__' == __name__:
    "======================= Seed fix =================================="
    SEED = 1000
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    "======================= Rollout test ================================"
    env = Env(arm_type='single',
              stage=[1,2,1],
              init_partial_loading=[0,0,0],
              min_process_time=50,
              max_process_time=300,
              loadport_capacity=2,
              num_foup=2,
              foup_size=5,
              num_lot_type=5,
              min_delay_time=30,
              max_delay_time=50,
            )


    odest = odest_framework(env, policy='backward')
    odest.run()
    print("Done")


