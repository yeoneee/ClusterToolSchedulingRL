import random
import copy
import torch
from typing import Union, Optional
from dataclasses import dataclass, fields

import pandas as pd

from tensordict import TensorDict
from torch import Tensor

from envs.module import Module
from envs.robot.single_arm import SingleArmedRobot
from envs.robot.dual_arm import DualArmedRobot
from envs.robot.ind_arm import IndArmedRobot
from envs.wafer import Wafer
from envs.render import render_schedule

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


class ScheduleModule:
    transport_module: list = []
    process_module: list = []
    init_status: list = []

    def update(self, env, action, state):
        robot_schedule = torch.vstack([
            action.foup_idx,
            action.wafer_idx,
            action.unload_loc,
            action.load_loc,
            env.robot.pkup_start_time,
            env.robot.pkup_end_time,
            env.robot.unload_start_time,
            env.robot.unload_end_time,
            env.robot.move_start_time,
            env.robot.move_end_time,
            env.robot.load_start_time,
            env.robot.load_end_time,
        ]).T

        pm_schedule = torch.vstack([
            action.load_loc,
            action.foup_idx,
            action.wafer_idx,
            env.loc.process_start_time[env.batch_idx, action.load_loc],
            env.loc.process_end_time[env.batch_idx, action.load_loc],
            action.unload_loc,
            env.loc.process_end_time[env.batch_idx, action.unload_loc],
            env.robot.unload_start_time
        ]).T

        if env.purge_constraint:
            pm_schedule = torch.hstack([pm_schedule, env.loc.purge_start_time[env.batch_idx, action.unload_loc][:, None]])
            pm_schedule = torch.hstack([pm_schedule, env.loc.purge_end_time[env.batch_idx, action.unload_loc][:, None]])

        self.transport_module.append(robot_schedule)
        self.process_module.append(pm_schedule)

    def reset(self):
        self.transport_module = []
        self.process_module = []
        self.init_status = []

@dataclass
class Action:
    idx = None
    foup_idx = None
    wafer_idx = None

    unload_loc = None   # unload module id,
    load_loc = None
    is_load = None
    # (batch, 1)

    robot_idx = None

    def __init__(self, action_idx):
        self.idx = action_idx
        self.foup_idx = -torch.ones_like(action_idx, dtype=torch.int64)
        self.wafer_idx = -torch.ones_like(action_idx, dtype=torch.int64)
        self.unload_loc = -torch.ones_like(action_idx, dtype=torch.int64)
        self.load_loc = -torch.ones_like(action_idx, dtype=torch.int64)
        self.is_load = torch.zeros_like(action_idx, dtype=torch.bool)
        self.robot_idx = torch.zeros_like(action_idx, dtype=torch.int64)

@dataclass
class State:
    i:torch.Tensor=None
    clock:torch.Tensor=None
    done:torch.Tensor=None
    batch_idx:torch.Tensor=None
    action_mask:torch.Tensor=None

    loc_id:torch.Tensor=None
    loc_stage:torch.Tensor=None
    loc_status:torch.Tensor=None
    loc_hold_wafer:torch.Tensor=None
    loc_process_start_time:torch.Tensor=None
    loc_process_end_time:torch.Tensor=None
    loc_purge_start_time:torch.Tensor=None
    loc_purge_end_time:torch.Tensor=None

    robot_loc:torch.Tensor=None
    robot_hold_wafer:torch.Tensor=None
    robot_pkup_start_time:torch.Tensor=None
    robot_pkup_end_time:torch.Tensor=None
    robot_unload_start_time:torch.Tensor=None
    robot_unload_end_time:torch.Tensor=None
    robot_move_start_time:torch.Tensor=None
    robot_move_end_time:torch.Tensor=None
    robot_load_start_time:torch.Tensor=None
    robot_load_end_time:torch.Tensor=None

    wafer_name:torch.Tensor=None
    wafer_status:torch.Tensor=None
    wafer_recipe:torch.Tensor=None
    wafer_stage:torch.Tensor=None
    wafer_loc:torch.Tensor=None
    wafer_ready_time:torch.Tensor=None
    wafer_residency_time:torch.Tensor=None

    def set(self, env):
        for k, v in self.__dict__.items():
            if 'loc_' in k:
                ek = k.split('loc_')[-1]
                if ek in env.loc.__dict__:
                    setattr(self, k, env.loc.__dict__[ek])
            elif 'robot_' in k:
                ek = k.split('robot_')[-1]
                if ek in env.robot.__dict__:
                    setattr(self, k, env.robot.__dict__[ek])
            elif 'wafer_' in k:
                ek = k.split('wafer_')[-1]
                if ek in env.wafer.__dict__:
                    """only loadport loaded lot wafer state"""

                    intool_wafer_idx = env.wafer.get_wip_wafer()
                    # (batch, FOUP_num, FOUP_size)
                    num_FOUP = intool_wafer_idx.all(dim=-1).sum(dim=-1)
                    assert (num_FOUP == env.loadport_capacity).all(), \
                        'The number of FOUPs in the loadport should not \
                            exceed the loadport capacity.'

                    wafer_val_flatten =\
                        env.wafer.__dict__[ek][intool_wafer_idx].reshape(*env.batch_size, -1)
                    # (batch, FOUP_num * FOUP_size)
                    setattr(self, k, wafer_val_flatten)
            else:
                if k in env.__dict__:
                    setattr(self, k, env.__dict__[k])

    def to(self, device):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[attr] = copy.deepcopy(value).to(device)

    def batch_size(self):
        return self.i.size(0)

    def device(self):
        return self.i.device

class NoncyclicClusterToolEnv:
    def __init__(self,
                 arm_type: str = 'single',
                 stage: list = [3,2],
                 td_params: TensorDict = None,
                 **kwargs) -> None:
        self.kwargs = kwargs

        # cluster tool configuration
        self.arm_type = arm_type
        self.stage = stage
        self.init_partial_loading = kwargs.get('init_partial_loading', [0]*len(stage))
        assert len(stage) == len(self.init_partial_loading), \
            'The length of the stage and the initial partial loading should be the same.'

        self.loadport_capacity = kwargs.get('loadport_capacity', 2)

        # Lot configuration
        self.num_foup = kwargs.get('num_foup', 2)
        self.foup_size = kwargs.get('foup_size', 25)
        self.num_lot_type = kwargs.get('num_lot_type', 5)
        self.release_foup_idx = 1

        # additional constraints for the environment
        self.purge_constraint = kwargs['max_purge_time'] != 0
        self.delay_constraint = 'min_delay_time' in kwargs
        self.setup_constraint = 'min_setup_time' in kwargs

        # state
        self.state = State()

        # log
        self.schedule = ScheduleModule()
        self.action_log = []

    "========================================  core function  ============================================"""
    def step(self, action: torch.Tensor, load_delay_time: Optional[torch.Tensor] = None, rule=False, show=False) -> TensorDict:

        # convert action id -> robot move control information
        action = self.get_action_mapping(action, rule)
        # unload action
        self.robot.unload(self, action)

        # load action
        if self.arm_type == 'single':
            action.is_load = ~action.is_load
        self.robot.load(self, action, load_delay_time)

        # update clock
        self.clock = {
            'single': self.robot.load_end_time.clone(),
            'dual': self.robot.unload_end_time * (action.is_load == False) \
                    + self.robot.load_end_time * (action.is_load == True)
        }.get(self.arm_type)

        # update PM, LL
        self.loc.update(self)
        self.wafer.update(self)

        wafer_status_flatten = self.wafer.status.reshape(*self.batch_size, -1)
        self.done = torch.logical_or((wafer_status_flatten == self.wafer.status_dict['outloadport']),
                                     (wafer_status_flatten == self.wafer.status_dict['exit'])).all(dim=-1)
        self.action_mask = self.get_action_mask()

        # episode step count
        self.i[~self.done] += 1

        # state update
        self.state.set(self)
        self.state.to(self.device)

        # render
        if show:
            self.render(env=self, action=action, render_batch_idx=1)  # 15

        # log
        self.schedule.update(self, action, self.state)


        return self.state

    def reset(self, batch_size: list = None, device: str = 'cpu') -> TensorDict:
        self.device = device
        self.batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_idx = torch.arange(*self.batch_size)

        # recipe configuration
        if 'recipe_file_dir' in self.kwargs:
            self.recipe_table = self.read_recipe_table()
        else: self.recipe_table = self.generate_recipe_table()

        # module init
        self.loc = Module(self)
        self.wafer = Wafer(self)

        ROBOT_ARM_REGISTRY = {
            "single": SingleArmedRobot,
            "dual": DualArmedRobot,
        }
        self.robot = ROBOT_ARM_REGISTRY.get(self.arm_type)(self)

        # set initial loading
        self.wafer.set_init_status(self)
        self.loc.set_init_status(self)

        # action space configuration
        if self.arm_type == 'single':
            # unload actions L+M
            self.num_action = self.num_lot_type + self.loc.num_pm

        elif self.arm_type == 'dual':
            # unload actions: L+M
            # load actions: (M+1)*2 <- arm1,2
            self.num_action = self.num_lot_type + self.loc.num_pm +\
                              2 * (self.loc.num_pm + 1)


        # Initialize simulation clock time index, counter(=i)
        self.i = torch.zeros(size=(*self.batch_size,), dtype=torch.int64)
        self.clock = torch.zeros(size=(*self.batch_size,),dtype=torch.float)
        self.done = torch.zeros(size=(*self.batch_size,),dtype=torch.bool)
        self.action_mask = self.get_action_mask()

        # state define
        self.state.set(self)
        self.state.to(self.device)

        # save init state
        self.schedule.reset()
        self.action_log = []

        self.schedule.init_status.append([self.state.loc_hold_wafer, self.state.loc_process_end_time])
        if self.purge_constraint:
            self.schedule.init_status.append(self.state.loc_purge_end_time)

        return self.state

    def get_action_mask(self) -> torch.Tensor:

        if self.arm_type == 'single':
            action_mask = self.robot.valid_unload_action(self)

        elif self.arm_type == 'dual':
            unload_action_mask = self.robot.valid_unload_action(self)
            load_action_mask = self.robot.valid_load_action(self)
            action_mask = torch.cat([unload_action_mask, load_action_mask], dim=-1)

        # if done schedule, mask all action TODO
        done = (self.wafer.loc == self.loc.num_pm+1)\
                .reshape(*self.batch_size, -1).all(dim=-1)
        action_mask[done] = True

        return action_mask

    def get_reward(self) -> torch.Tensor:
        negative_cmax = -self.clock.to(torch.float).to(self.device)

        return negative_cmax

    "========================================  instance generate function  ============================================"""
    def generate_recipe_table(self) -> torch.Tensor:
        # lot recipe setting
        lot_process_time = self.gen_lot_process_time()
        lot_flow = self.gen_lot_flow()
        recipe_table = {'process_time': lot_process_time,
                        'flow': lot_flow}
        # (batch, num_lot_type, num_step)

        return recipe_table

    def gen_lot_process_time(self):
        lot_variance = self.kwargs.get('lot_variance', False)
        consider_lot_type = self.kwargs.get('consider_lot_type', 5)
        min_process_time = self.kwargs.get('min_process_time', 10)
        max_process_time = self.kwargs.get('max_process_time', 100)
        balanced_process_time = self.kwargs.get('balanced_process_time', False)
        min_time_unit = self.kwargs.get('min_time_unit', 2)
        max_time_unit = max_process_time

        self.num_stage = len(self.stage)
        num_extra_step = 3 # Dummy
        self.num_step = self.num_stage + num_extra_step

        # sampling
        if consider_lot_type > 1 and lot_variance:
            num_short_lot = round(self.num_lot_type / 2)
            num_long_lot = self.num_lot_type - num_short_lot

            mid_process_time = int((min_process_time+max_process_time)/2)

            short_process_time = torch.randint(min_process_time, mid_process_time,
                                        (*self.batch_size, num_short_lot, self.num_step))

            long_process_time = torch.randint(mid_process_time, max_process_time,
                                        (*self.batch_size, num_long_lot, self.num_step))

            process_time = torch.cat([short_process_time, long_process_time], dim=1)

            indices = torch.randperm(process_time.size(1))
            process_time = process_time[:, indices, :]

        else:
            if str(self.stage) == '[2, 3]':
                process_time = torch.concat([
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 200, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 300, (*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1))
                ], dim=-1)

            elif str(self.stage) == '[1, 2, 1]':
                process_time = torch.concat([
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 150, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 300, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 150, (*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1))
                ], dim=-1)

            elif str(self.stage) == '[1, 3, 2]':
                process_time = torch.concat([
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 100, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 300, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 200, (*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1))
                ], dim=-1)

            elif str(self.stage) == '[1, 2, 2, 1]':
                process_time = torch.concat([
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 150, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 300, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 300, (*self.batch_size, self.num_lot_type, 1)),
                    torch.randint(10, 150, (*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1)),
                    torch.zeros((*self.batch_size, self.num_lot_type, 1))
                ], dim=-1)
            else:
                process_time = torch.randint(min_process_time, max_process_time,
                                            (*self.batch_size, self.num_lot_type, self.num_step))


        # set lot type ratio per batch
        single_lot_batch = False
        if single_lot_batch:
            num_single_lot_batch = self.batch_size[0] // 2 if isinstance(self.batch_size, list)\
                else self.batch_size // 2
            single_lot_batch_idx = torch.randperm(*self.batch_size)[:num_single_lot_batch]
            process_time[single_lot_batch_idx, :] =\
                process_time[single_lot_batch_idx, 0, :][:, None, :].repeat(1, self.num_lot_type, 1)


        # Min-max normalization
        self._process_time = copy.deepcopy(process_time).int()
        if self.kwargs.get('norm_time', True):
            process_time = (process_time - min_time_unit) / (max_time_unit - min_time_unit)

        # loadlock time
        process_time[:, :, 0] = 0
        process_time[:, :, self.num_stage+1:] = 0

        return process_time

    def gen_lot_flow(self):
        """ Multi flow
        # each lot flow number
        num_serial = env.kwargs.get('num_serial_flow_recipe', 30)
        num_skip = env.kwargs.get('num_skip_flow_recipe', 0)
        num_reenter = env.kwargs.get('num_reenter_flow_recipe', 0)
        assert self.num_lot_type == num_serial + num_skip + num_reenter, \
            "num_recipe should be equal to the sum of each flow type."

        flow = torch.full((*env.batch_size, env.num_lot_type, env.num_step),
                          env.num_stage+1, dtype=torch.int64)

        # serial
        serial = torch.arange(env.num_stage+2, dtype=torch.int64)
        flow[:, :num_serial, :env.num_stage+2] = serial

        # skip
        skip_stage = torch.randint(1, env.num_stage+1, (num_skip,))
        for i in range(num_skip):
            skip = serial[serial!=skip_stage[i]]
            flow[:, num_serial+i, :env.num_stage+1] = skip

        # reenter

        return flow
        """
        # serial flow only
        num_serial = self.num_lot_type
        flow = torch.full((*self.batch_size, self.num_lot_type, self.num_step),
                            self.num_stage+1, dtype=torch.int64)

        # serial
        serial_flow = torch.arange(self.num_stage+2, dtype=torch.int64)
        flow[:, :num_serial, :self.num_stage+2] = serial_flow

        return flow

    "========================================  action function  ============================================"""
    def get_action_mapping(self, action_idx: torch.Tensor, rule=False) -> TensorDict:
        """ maps action index to the corresponding control wafer, PM, robot"""
        if self.arm_type == 'single':
            action_idx = action_idx.to('cpu') if action_idx.device != 'cpu' else action_idx
            if not rule:
                # action 1. insert wafer from LL -> PM action
                # action 2. move wafer from PM -> PM(or LL) action
                # ll wafer based model
                batch_idx = torch.arange(*self.batch_size)
                num_ll_wafer = self.foup_size
                num_wafer_type = self.num_lot_type
                pm_action_idx = action_idx >= num_ll_wafer
                ll_action_idx = action_idx < num_ll_wafer
                #exit_foup = torch.clamp(self.wafer.exit_foup, max=self.num_foup-1) # FIXME
                ll_foup = torch.argmax((self.wafer.loc == 0).any(dim=-1).float(), dim=1)
                wafer_recipes = self.wafer.recipe[batch_idx, ll_foup]
                action_idx[pm_action_idx] = action_idx[pm_action_idx] - num_ll_wafer + num_wafer_type
                action_idx[ll_action_idx] = gather_by_index(wafer_recipes[ll_action_idx], action_idx[ll_action_idx])
                #
            action = Action(action_idx)
            self.get_lm_unload_action_mapping(action)
            self.get_pm_unload_action_mapping(action)


        elif self.arm_type == 'dual':
            action_idx = action_idx.to('cpu') if action_idx.device != 'cpu' else action_idx
            if not rule:
                batch_idx = torch.arange(*self.batch_size)
                num_ll_wafer = self.foup_size
                num_wafer_type = self.num_lot_type
                ll_action_idx = action_idx < num_ll_wafer
                #exit_foup = torch.clamp(self.wafer.exit_foup, max=self.num_foup-1) # FIXME
                ll_foup = torch.argmax((self.wafer.loc == 0).any(dim=-1).float(), dim=1)
                wafer_recipes = self.wafer.recipe[batch_idx, ll_foup]
                action_idx[ll_action_idx] = gather_by_index(wafer_recipes[ll_action_idx], action_idx[ll_action_idx])
                action_idx[~ll_action_idx] = action_idx[~ll_action_idx] - num_ll_wafer + num_wafer_type

            action = Action(action_idx)
            # ================================== unload from the Loadport action =====================================
            # get the wafer id of the target type lot
            unload_ll_batch_idx = action_idx < self.num_lot_type
            action_wafer_type = action_idx[:, None, None].repeat(1, self.num_foup, self.foup_size)
            is_selected_type_wafer = self.wafer.get_recipe(self.wafer.name) == action_wafer_type
            is_in_lm_wafer = self.wafer.status == self.wafer.status_dict.get('inloadport')

            lm_action_wafers = torch.logical_and(is_selected_type_wafer, is_in_lm_wafer) #(bathc, foup, foup_size)
            #descending = torch.arange(self.foup_size, 0, -1)[None, None, :].repeat(*self.batch_size, self.num_foup, 1)
            descending = (torch.arange(self.foup_size * self.num_foup, 0, -1)[None, :]
                      .repeat(*self.batch_size, 1)
                      .reshape(*self.batch_size, self.num_foup, self.foup_size))

            # select the wafer with the lowest index
            lm_action_wafers = (lm_action_wafers * descending).reshape(*self.batch_size, -1)
            lm_action_foup_idx = torch.argmax(lm_action_wafers, dim=-1)[unload_ll_batch_idx] // self.foup_size
            lm_action_wafer_idx = torch.argmax(lm_action_wafers, dim=-1)[unload_ll_batch_idx] % self.foup_size

            action.foup_idx[unload_ll_batch_idx] = lm_action_foup_idx
            action.wafer_idx[unload_ll_batch_idx] = lm_action_wafer_idx
            action.unload_loc[unload_ll_batch_idx] = 0 # LL
            action.is_load[unload_ll_batch_idx] = False

            # ================================== unload from the PM action ===========================================
            # get the wafer id of the PM
            unload_pm_batch_idx = torch.logical_and(action_idx >= self.num_lot_type,
                                                    action_idx < self.num_lot_type + self.loc.num_pm)
            unload_pm = action_idx[unload_pm_batch_idx] - self.num_lot_type + 1
            unload_pm = unload_pm[:, None, None].repeat(1, self.num_foup, self.foup_size)
            unload_wafer_idx = (self.wafer.loc[unload_pm_batch_idx] == unload_pm).nonzero()

            wafer_batch_idx = self.batch_idx[unload_pm_batch_idx]
            wafer_foup_idx = unload_wafer_idx[:, 1]
            wafer_order_idx = unload_wafer_idx[:, 2]

            action.foup_idx[unload_pm_batch_idx] = wafer_foup_idx
            action.wafer_idx[unload_pm_batch_idx] = wafer_order_idx
            action.unload_loc[unload_pm_batch_idx] = self.wafer.loc[wafer_batch_idx, wafer_foup_idx, wafer_order_idx]
            action.is_load[unload_pm_batch_idx] = False


            # ================================== load to the PM action ===============================================
            load_batch_idx = action_idx >= self.num_lot_type + self.loc.num_pm
            load_arm_1 = torch.logical_and(action_idx >= self.num_lot_type + self.loc.num_pm,
                                           action_idx < self.num_lot_type + self.loc.num_pm + self.loc.num_pm + 1)

            load_loc = action_idx[load_batch_idx] - (self.num_lot_type + self.loc.num_pm) + 1\
                  - (self.loc.num_pm + 1) * (~load_arm_1)[load_batch_idx]
            load_wafer_name = self.robot.hold_wafer[load_batch_idx, (~load_arm_1).long()[load_batch_idx]]
            load_wafer_idx = (self.wafer.name[load_batch_idx, :, :] == load_wafer_name[:, None, None]).nonzero()

            wafer_batch_idx = self.batch_idx[load_batch_idx]
            wafer_foup_idx = load_wafer_idx[:, 1]
            wafer_order_idx = load_wafer_idx[:, 2]

            action.foup_idx[load_batch_idx] = wafer_foup_idx
            action.wafer_idx[load_batch_idx] = wafer_order_idx
            action.load_loc[load_batch_idx] = load_loc
            action.is_load[load_batch_idx] = True

        return action

    def get_lm_unload_action_mapping(self, action: object):
        # ================================== unload from Loadport action ==========================================
        # get the wafer id of the target type lot
        action_idx = action.idx
        lm_batch_idx = action_idx < self.num_lot_type

        action_wafer_type = action_idx[:, None, None].repeat(1, self.num_foup, self.foup_size)
        is_selected_type_wafer = self.wafer.get_recipe(self.wafer.name) == action_wafer_type
        is_in_lm_wafer = self.wafer.status == self.wafer.status_dict.get('inloadport')

        lm_action_wafers = torch.logical_and(is_selected_type_wafer, is_in_lm_wafer) #(bathc, foup, foup_size)

        # select the wafer with the lowest index
        #descending = torch.arange(self.foup_size, 0, -1)[None, None, :].repeat(*self.batch_size, self.num_foup, 1)
        descending = (torch.arange(self.foup_size * self.num_foup, 0, -1)[None, :]
                      .repeat(*self.batch_size, 1)
                      .reshape(*self.batch_size, self.num_foup, self.foup_size))
        lm_action_wafers = (lm_action_wafers * descending).reshape(*self.batch_size, -1)

        lm_action_foup_idx = torch.argmax(lm_action_wafers, dim=-1)[lm_batch_idx] // self.foup_size
        lm_action_wafer_idx = torch.argmax(lm_action_wafers, dim=-1)[lm_batch_idx] % self.foup_size

        # load pm
        lm_action_batch_idx = self.batch_idx[lm_batch_idx]
        action_wafer_recipe = self.wafer.get_recipe(
            self.wafer.name[lm_action_batch_idx, lm_action_foup_idx, lm_action_wafer_idx])
        action_wafer_next_step = torch.ones_like(action_wafer_recipe)
        next_stage = self.recipe_table.get('flow')[lm_action_batch_idx, action_wafer_recipe, action_wafer_next_step]

        next_stage_pm = self.loc.stage[lm_batch_idx] == next_stage[:, None].repeat(1, self.loc.num_loc)
        unloaded_status = torch.Tensor([self.loc.status_dict['unload'],
                                        self.loc.status_dict['purge']])
        empty_pm = torch.isin(self.loc.status, unloaded_status)
        action_locs = torch.logical_and(next_stage_pm, empty_pm[lm_batch_idx,:])

        if self.purge_constraint:
            pm_purge_time = torch.full_like(self.loc.purge_end_time[lm_batch_idx,:], 1e10)
            pm_purge_time[action_locs] = self.loc.purge_end_time[lm_batch_idx,:][action_locs]

            min_remain_purge_time = pm_purge_time.min(dim=-1)[0]
            min_remain_purge_loc = self.loc.purge_end_time[lm_batch_idx, :] ==\
                  min_remain_purge_time[:, None].repeat(1, self.loc.num_loc)

            action_loc = torch.logical_and(action_locs, min_remain_purge_loc)

        else:
            # 여러 PM이 있을 경우 idle time이 긴 PM 선택
            pm_idle_time = torch.full_like(self.loc.process_end_time[lm_batch_idx], 1e10)
            pm_idle_time[action_locs] = self.loc.process_end_time[lm_batch_idx,:][action_locs]
            max_idle_time = pm_idle_time.min(dim=-1)[0]
            max_idle_loc = self.loc.process_end_time[lm_batch_idx] == max_idle_time[:, None].repeat(1, self.loc.num_loc)
            action_loc = torch.logical_and(action_locs, max_idle_loc)


        # tie break by index order
        descending = torch.arange(self.loc.num_loc, 0, -1)
        lm_action_load_loc = torch.argmax(action_loc * descending, dim=-1)

        action.foup_idx[lm_batch_idx] = lm_action_foup_idx
        action.wafer_idx[lm_batch_idx] = lm_action_wafer_idx
        action.unload_loc[lm_batch_idx] = 0 # LL
        action.load_loc[lm_batch_idx] = lm_action_load_loc

    def get_pm_unload_action_mapping(self, action: object):
        # ================================== unload from current PM  ============================================
        # get the wafer id of the PM
        action_idx = action.idx
        pm_batch_idx = action_idx >= self.num_lot_type
        unload_pm = action_idx[pm_batch_idx] - self.num_lot_type + 1
        unload_pm = unload_pm[:, None, None].repeat(1, self.num_foup, self.foup_size)
        unload_wafer_idx = (self.wafer.loc[pm_batch_idx] == unload_pm).nonzero()

        wafer_batch_idx = self.batch_idx[pm_batch_idx]
        wafer_foup_idx = unload_wafer_idx[:, 1]
        wafer_order_idx = unload_wafer_idx[:, 2]

        action.foup_idx[pm_batch_idx] = wafer_foup_idx
        #action.wafer_idx[~ll_action] = wafer_foup_idx % self.loadport_capacity + wafer_order_idx
        action.wafer_idx[pm_batch_idx] = wafer_order_idx
        action.unload_loc[pm_batch_idx] = self.wafer.loc[wafer_batch_idx, wafer_foup_idx, wafer_order_idx]

        # ================================== load next stage PM ================================================
        # select the PM of the next stage
        action_wafer_receipe = self.wafer.get_recipe(
            self.wafer.name[wafer_batch_idx, wafer_foup_idx, wafer_order_idx])
        action_wafer_next_step = self.wafer.name[wafer_batch_idx, wafer_foup_idx, wafer_order_idx] % self.wafer.STEP_MLR + 1
        next_stage = self.recipe_table.get('flow')[wafer_batch_idx, action_wafer_receipe, action_wafer_next_step]

        next_stage_pm = self.loc.stage[pm_batch_idx] == next_stage[:, None].repeat(1, self.loc.num_loc)

        unloaded_status = torch.Tensor([self.loc.status_dict['unload'],
                                        self.loc.status_dict['purge']])
        empty_pm = torch.isin(self.loc.status, unloaded_status)

        action_locs = torch.logical_and(next_stage_pm, empty_pm[pm_batch_idx,:])
        #assert action_locs[~ll_action].any(dim=-1).all(), \
        #    'There is no available PM for the next stage.'

        if self.purge_constraint:
            pm_purge_time = torch.full_like(self.loc.purge_end_time[pm_batch_idx,:], 1e10)
            pm_purge_time[action_locs] = self.loc.purge_end_time[pm_batch_idx,:][action_locs]

            min_remain_purge_time = pm_purge_time.min(dim=-1)[0]
            min_remain_purge_loc = self.loc.purge_end_time[pm_batch_idx, :] ==\
                  min_remain_purge_time[:, None].repeat(1, self.loc.num_loc)

            action_loc = torch.logical_and(action_locs, min_remain_purge_loc)

        else:
            # 여러 PM이 있을 경우 idle time이 긴 PM 선택
            pm_idle_time = torch.full_like(self.loc.process_end_time[pm_batch_idx], 1e10)
            pm_idle_time[action_locs] = self.loc.process_end_time[pm_batch_idx,:][action_locs]
            max_idle_time = pm_idle_time.min(dim=-1)[0]
            max_idle_loc = self.loc.process_end_time[pm_batch_idx] == max_idle_time[:, None].repeat(1, self.loc.num_loc)
            action_loc = torch.logical_and(action_locs, max_idle_loc)

        # tie break by index order
        descending = torch.arange(self.loc.num_loc, 0, -1)
        action_loc = torch.argmax(action_loc * descending, dim=-1)
        action.load_loc[pm_batch_idx] = action_loc

    "========================================  render function ============================================"""
    def render_gantt_chart(self, render_batch_idx, **kwargs):
        pass

    def render(self, action, render_batch_idx, **kwargs):
        render_schedule(self, action, render_batch_idx)
