import random
import argparse
import copy
from typing import List
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class State:
    clock: torch.Tensor = None
    done: torch.Tensor = None
    action_mask : torch.Tensor = None
    stage: torch.Tensor = None

    loc_id:torch.Tensor=None
    loc_stage:torch.Tensor=None
    loc_status:torch.Tensor=None
    loc_hold_wafer:torch.Tensor=None
    loc_process_start_time:torch.Tensor=None
    loc_process_end_time:torch.Tensor=None

    robot_arm1_loc:torch.Tensor=None
    robot_arm2_loc:torch.Tensor=None
    robot_arm1_hold_wafer_recipe:torch.Tensor=None
    robot_arm2_hold_wafer_recipe:torch.Tensor=None
    robot_arm1_hold_wafer_next_stage:torch.Tensor=None
    robot_arm2_hold_wafer_next_stage:torch.Tensor=None

    recipe_flow: torch.Tensor = None
    recipe_time: torch.Tensor = None

    loadlock1_wafer_in: torch.Tensor = None
    loadlock2_wafer_in: torch.Tensor = None
    loadlock1_wafer_recipe: torch.Tensor = None
    loadlock2_wafer_recipe: torch.Tensor = None

    def to(self, device):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[attr] = copy.deepcopy(value).to(device)

    def batch_size(self):
        return self.clock.size(0)

    def device(self):
        return self.clock.device


@dataclass
class Recipe:
    id: int = None
    group: int = None
    flow: np.array = None
    time: np.array = None

@dataclass
class Wafer:
    # static
    id: int = None
    recipe: int = None
    group: int = None

    # dynamic
    loc: int = None
    unloaded_stage: int = None
    remain_prs_time: int = None

@dataclass
class Pm:
    # static
    id: int = None
    stage: int = None
    group: int = None

    # dynamic
    status: int = None
    hold_wafer: Wafer = None
    prs_start_time: int = None
    prs_end_time: int = None

    # func
    def unload(self):
        self.hold_wafer.loc = -1 # pm -> arm
        self.hold_wafer.remain_prs_time = 0 # reset remain_prs_time
        self.hold_wafer.unloaded_stage = self.stage
        self.hold_wafer = None
        self.status = 'idle'
        self.prs_start_time = None
        self.prs_end_time = None


    def load(self, env, wafer):
        self.hold_wafer = wafer
        self.process(env, wafer)

    def process(self, env, wafer):
        self.status = 'processing'
        self.prs_start_time = env.robot.load_end_time
        self.prs_end_time = self.prs_start_time + env.recipes[wafer.recipe].time[self.stage]
        wafer.remain_prs_time = self.prs_end_time - self.prs_start_time

    def purge(self, env, control):
        NotImplementedError

@dataclass
class Robot:
    # static
    load_time: int = None
    unload_time: int = None
    move_time: int = None

    # dynamic
    loc: List[int] = None
    hold_wafer: List[Wafer] = None

    pkup_start_time: int = None
    pkup_end_time: int = None
    unload_start_time: int = None
    unload_end_time: int = None
    move_start_time: int = None
    move_end_time: int = None
    load_start_time: int = None
    load_end_time: int = None

    # func
    def pkup(self, env: object, loc_id: int, arm_id: int):
        self.pkup_start_time = env.clock
        self.pkup_end_time = self.pkup_start_time + self.move_time

        self.loc[arm_id] = loc_id
        self.loc[1 - arm_id] = self._set_arm_loc(env, loc_id)

    def move(self, env: object, loc_id: int, arm_id: int):
        self.move_start_time = env.clock
        self.move_end_time = self.move_start_time + self.move_time

        self.loc[arm_id] = loc_id
        self.loc[1 - arm_id] = self._set_arm_loc(env, loc_id)

    def unload(self, env: object, loc_id: int, group: int, arm_id: int):
        # pkup wafer
        if self.loc[arm_id] != loc_id:
            self.pkup(env, loc_id, arm_id)
        else:
            self.pkup_start_time = env.clock
            self.pkup_end_time = self.pkup_start_time

        # unload
        self.unload_start_time = max(self.pkup_end_time, env.pms[loc_id-1].prs_end_time) if loc_id != 0 else env.clock
        self.unload_end_time = self.unload_start_time + self.unload_time

        if loc_id == 0:  # unload from loadlock
            self._unload_from_loadlock(env, group, arm_id)
        else:  # unload from pm
            self._unload_from_pm(env, loc_id, arm_id)

    def _unload_from_loadlock(self, env: object, group: int, arm_id: int):
        loadlock = env.loadlock_1 if group == 0 else env.loadlock_2
        no_wafer = True
        for foup in loadlock:
            if len(foup.wafers_in) > 0:
                self.hold_wafer[arm_id] = foup.wafers_in.pop(0)
                self.hold_wafer[arm_id].loc = -1
                self.hold_wafer[arm_id].unloaded_stage = 0
                no_wafer = False
                break

        if no_wafer:
            print('Error: Loadlock is empty')

    def _unload_from_pm(self, env: object, loc_id: int, arm_id: int):
        self.hold_wafer[arm_id] = env.pms[loc_id - 1].hold_wafer
        self.hold_wafer[arm_id].loc = -1
        env.pms[loc_id - 1].unload()

    def load(self, env: object, loc_id: int, arm_id: int):
        if self.loc[arm_id] != loc_id:
            self.move(env, loc_id, arm_id)
        else:
            self.move_start_time = env.clock
            self.move_end_time = self.move_start_time

        # load
        self.load_start_time = self.move_end_time  # TODO purge time
        self.load_end_time = self.load_start_time + self.load_time

        if loc_id == env.outloadlock:
            self._load_to_loadlock(env, arm_id)
        else:  # load to pm
            self._load_to_pm(env, loc_id, arm_id)

        self.hold_wafer[arm_id].loc = loc_id
        self.hold_wafer[arm_id] = None

    def _load_to_loadlock(self, env: object, arm_id: int):
        loadlock = env.loadlock_1 if self.hold_wafer[arm_id].group == 0 else env.loadlock_2
        for foup in loadlock:
            if foup.recipe == self.hold_wafer[arm_id].recipe:
                foup.wafers_out.append(self.hold_wafer)
                break
            else:
                print('Error: Loadlock recipe mismatch')

    def _load_to_pm(self, env: object, loc_id: int, arm_id: int):
        if env.pms[loc_id - 1].status == 'idle':
            env.pms[loc_id - 1].load(env, self.hold_wafer[arm_id])
        else:
            print('Error: PM is not idle')

    def _set_arm_loc(self, env, master_arm_loc):
        """
        set opposite arm location (master-slave relationship)
        """
        num_loc = sum(env.stage)+2
        if  num_loc % 2:
            mid = int((num_loc-1)/2)
            if master_arm_loc < mid:
                slave_arm_loc = master_arm_loc + (mid)
            else:
                slave_arm_loc = master_arm_loc - (mid)
        else:
            slave_arm_loc = int((master_arm_loc + num_loc/2) % num_loc)

        return slave_arm_loc


@dataclass
class Foup:
    id: int = None
    recipe: int = None
    group: int = None
    wafers_in: list = None
    wafers_out: list = None

    def get_num_wafers_in(self):
        return len(self.wafers_in)

    def get_num_wafers_out(self):
        return len(self.wafers_out)


class ddcfEnv:
    """
    # dual-armed cluster tool for (dedicated) concurrent flow

    For example:
        6 PMs, with m = [1,1,1,1,1,1]
        Type 1: inloadlock -> stage 1 -> stage 2 -> stage 3 -> outloadlock
        Type 2: inloadlock -> stage 4 -> stage 5 -> stage 6 -> outloadlock

    """

    def __init__(self, foup_size: int,
                 group1_stage: List[int], group1_min_prs_time: int, group1_max_prs_time: int,
                 group2_stage: List[int], group2_min_prs_time: int, group2_max_prs_time: int,
                 prod_quantity: int, done_quantity: int, num_lot_type: int):
        # Initialize the environment with the given parameters

        self.robot_type = 'dual'
        self.foup_size = foup_size
        self.inloadlock = 0
        self.outloadlock = sum(group1_stage) + sum(group2_stage) + 1
        self.pm_status = {'idle': 0, 'processing': 1, 'finished': 2}
        self.stage = group1_stage + group2_stage
        self.num_lot_type = num_lot_type

        # Type 1 configuration
        self.group1_stage = group1_stage
        self.group1_min_prs_time = group1_min_prs_time
        self.group1_max_prs_time = group1_max_prs_time
        self.queue_1_quantity = prod_quantity
        self.queue_1 = []  # Type 1 queued lots
        self.loadlock_1 = []  # Type 1 current lot
        self.exit_1 = []  # Type 1 exit lots (finished)

        # Group 2 configuration
        self.group2_stage = group2_stage
        self.group2_min_prs_time = group2_min_prs_time
        self.group2_max_prs_time = group2_max_prs_time
        self.queue_2_quantity = prod_quantity
        self.queue_2 = []  # Group 2 queued lots
        self.loadlock_2 = []  # Group 2 current lot
        self.exit_2 = []  # Group 2 exit lots (finished)

        # Schedule
        self.clock = 0
        self.done_quantity = done_quantity

    def step(self, action: int) -> np.array:
        """
        Perform the given action and update the environment state.

        Args:
            action (int): The action to perform.

        Returns:
            np.array: The updated state of the environment.
        """

        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(self.pms)
        num_load_action = len(self.pms) + 1

        if action < num_ll_unload_action + num_pm_unload_action:
            unload = True
            # select robot arm id: located at the action location firstly or select empty arm index order
            action_loc = 0 if action in [0, 1] else action -1
            is_arm_faced = [i for i,j in enumerate(self.robot.loc) if j == action_loc]
            if is_arm_faced:
                arm_id = is_arm_faced[0]
            else:
                for arm_id, hold in enumerate(self.robot.hold_wafer):
                    if hold is None:
                        break

            if action == 0:  # Unload from LL group 1
                self.robot.unload(self, loc_id=0, group=0, arm_id=arm_id)

            elif action == 1:  # Unload from LL group 2
                self.robot.unload(self, loc_id=0, group=1, arm_id=arm_id)

            else:  # Unload from PM
                unload_loc_id = action - 1
                unload_wafer_group = self.pms[unload_loc_id - 1].group
                self.robot.unload(self, loc_id=unload_loc_id, group=unload_wafer_group, arm_id=arm_id)

        else:
            unload = False
            if action < num_ll_unload_action + num_pm_unload_action + num_load_action:
                arm_id = 0
                load_loc_id = action - num_ll_unload_action - num_pm_unload_action + 1
                self.robot.load(self, loc_id=load_loc_id, arm_id=arm_id)

            else:
                arm_id =1
                load_loc_id = action - num_ll_unload_action - num_pm_unload_action - num_load_action + 1
                self.robot.load(self, loc_id=load_loc_id, arm_id=arm_id)

        # Transition the environment state
        self._transition(unload)
        self.action_mask = self.get_action_mask()
        self.done = self.get_done()
        self.state = self.get_state()

        return self.state

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self._init_recipes()  # Initialize recipes
        self._init_wafers_foups_queues()  # Initialize wafers, FOUPs, and queues
        self._init_robot()  # Initialize the robot arm
        self._init_pms()  # Initialize the processing machines (PMs)
        self._init_initial_state()  # Set the initial state of the environment

        self.action_mask = self.get_action_mask()  # Get the initial action mask
        self.done = self.get_done()  # Check if the environment is done
        self.state = self.get_state()  # Get the initial state

        return self.state

    def _init_recipes(self):
        """
        Initialize the recipes for the environment.
        """
        self.recipes = []

        self._create_recipe(0, 0)
        self._create_recipe(1, 1)

    def _create_recipe(self, recipe, group):

        # flow
        flow = np.ones(len(self.stage))
        if group == 0:
            flow[len(self.group1_stage):] = 0
        elif group == 1:
            flow[:len(self.group1_stage)] = 0
        flow = np.insert(flow, 0, 1)  # Add inloadlock to the flow
        flow = np.append(flow, 1)  # Add outloadlock to the flow


        # Random processing time
        min_prs_time = self.group1_min_prs_time if group == 0 else self.group2_min_prs_time
        max_prs_time = self.group1_max_prs_time if group == 0 else self.group2_max_prs_time

        time = np.array([random.randint(min_prs_time, max_prs_time) for _ in range(len(self.stage))])
        if group == 0:
            time[len(self.group1_stage):] = 0
        elif group == 1:
            time[:len(self.group1_stage)] = 0

        time = np.insert(time, 0, 0)  # Add inloadlock time
        time = np.append(time, 0)  # Add outloadlock time

        self.recipes.append(Recipe(id=recipe, group=group, flow=flow, time=time))  # Append the recipe to the list

    def _init_wafers_foups_queues(self):
        """
        Initialize the wafers, FOUPs, and queues for the environment.
        """
        self.queue_1 = []  # Type 1 queued lots
        self.loadlock_1 = []  # Type 1 current lot
        self.exit_1 = []  # Type 1 exit lots (finished)

        self.queue_2 = []  # Group 2 queued lots
        self.loadlock_2 = []  # Group 2 current lot
        self.exit_2 = []  # Group 2 exit lots (finished)

        self.queue_1 = self._create_foups_queue(self.queue_1_quantity, 0, 0)  # Create queue for type 1
        self.loadlock_1 = [self.queue_1.pop(0)]  # Initialize loadlock for type 1

        self.queue_2 = self._create_foups_queue(self.queue_2_quantity, 1, 1)  # Create queue for type 2
        self.loadlock_2 = [self.queue_2.pop(0)]  # Initialize loadlock for type 2

    def _create_foups_queue(self, quantity, group, recipe):
        """
        Create a queue of FOUPs with wafers.

        Args:
            quantity (int): The number of FOUPs to create.
            recipe_id (int): The recipe ID for the wafers.

        Returns:
            List[Foup]: A list of FOUPs.
        """
        queue = []
        for i in range(quantity):
            wafers = [Wafer(id=j, recipe=recipe, group=group, loc=0, remain_prs_time=0) for j in range(self.foup_size)]
            foup = Foup(id=i, group=group, recipe=recipe, wafers_in=wafers, wafers_out=[])
            queue.append(foup)
        return queue

    def _init_robot(self):
        """
        Initialize the robot arm.
        """
        self.robot = Robot(load_time=3, unload_time=3, move_time=3, loc=[self.inloadlock, None], hold_wafer=[None, None])
        self.robot.loc[1] = self.robot._set_arm_loc(self, self.inloadlock)

    def _init_pms(self):
        """
        Initialize the processing machines (PMs).
        """
        self.pms = []
        cnt_pm = 1

        def _init_pms_for_stages(stages, start_stage, group):
            """
            Initialize PMs for the given stages and recipe.

            Args:
                stages (List[int]): The stages for the PMs.
                start_stage (int): The starting stage number.
                recipe (int): The recipe ID for the PMs.
            """
            nonlocal cnt_pm
            for s, m in enumerate(stages, start=start_stage):
                for _ in range(m):
                    pm = Pm(id=cnt_pm, stage=s, group=group, status='idle',
                            hold_wafer=None, prs_start_time=None, prs_end_time=None)
                    self.pms.append(pm)
                    cnt_pm += 1

        _init_pms_for_stages(self.group1_stage, start_stage=1, group=0)  # Initialize PMs for type 1
        _init_pms_for_stages(self.group2_stage, start_stage=len(self.group1_stage)+1, group=1)  # Initialize PMs for type 2


    def _init_initial_state(self):

        def _load_wafer_to_pm(pm, loadlock):
            # Pop a wafer from the loadlock and assign it to the PM
            wafer = loadlock[0].wafers_in.pop(0)
            wafer.loc = pm.id
            wafer_recipe = wafer.recipe
            recipe_time = self.recipes[wafer_recipe].time

            # Calculate remaining processing time with a random weight
            # stage_weight = 1/pm.stage
            stage_weight = 1
            wafer.remain_prs_time = random.randint(10, int(recipe_time[pm.stage] * stage_weight))

            # Update PM status and processing times based on the wafer's remaining processing time
            pm.status = 'processing' if wafer.remain_prs_time > 0 else 'finished'
            pm.hold_wafer = wafer
            pm.prs_start_time = 0 if wafer.remain_prs_time > 0 else None
            pm.prs_end_time = wafer.remain_prs_time if wafer.remain_prs_time > 0 else None

        self.clock = 0 # Reset the clock

        # Initialize each PM with a wafer from the corresponding loadlock
        for pm in self.pms:
            if pm.group == 0:
                _load_wafer_to_pm(pm, self.loadlock_1)
            elif pm.group == 1:
                _load_wafer_to_pm(pm, self.loadlock_2)

    def _transition(self, unload):
        # 시간 전이
        action_elapsed_time = self.robot.unload_end_time - self.clock if unload \
            else self.robot.load_end_time - self.clock

        for pm in self.pms:
            if pm.status == 'processing':
                pm.hold_wafer.remain_prs_time -= action_elapsed_time
                if pm.hold_wafer.remain_prs_time <= 0:
                    pm.hold_wafer.remain_prs_time = 0
                    pm.status = 'finished'

            elif pm.status == 'purging':
                NotImplementedError

        # loadlock에 wafer_in이 없으면, queue에서 wafer를 로드
        no_input_group1_wafer = len(sum([i.wafers_in for i in self.loadlock_1], [])) == 0
        if no_input_group1_wafer and len(self.queue_1) > 0:
            self.loadlock_1.append(self.queue_1.pop(0))

        no_input_group2_wafer = len(sum([i.wafers_in for i in self.loadlock_2], [])) == 0
        if no_input_group2_wafer and len(self.queue_2) > 0:
            self.loadlock_2.append(self.queue_2.pop(0))

        # foup의 모든 wafer가 처리된 경우(=#wafers_out == foup_size), exit으로 이동
        if self.loadlock_1[0].get_num_wafers_out() == self.foup_size:
            self.exit_1.append(self.loadlock_1.pop(0))

        if self.loadlock_2[0].get_num_wafers_out() == self.foup_size:
            self.exit_2.append(self.loadlock_2.pop(0))

        # 시간 단계 종료
        self.clock = self.robot.load_end_time if not unload else self.robot.unload_end_time

    def get_state(self) -> np.array:
        state = State(
            clock=torch.tensor([self.clock], dtype=torch.float),
            done=torch.tensor([self.done], dtype=torch.bool),
            action_mask=torch.tensor(self.action_mask, dtype=torch.bool),
            stage=torch.tensor(self.stage, dtype=torch.int64),

            loc_id=torch.tensor([i.id for i in self.pms], dtype=torch.int64),
            loc_stage=torch.tensor([i.stage for i in self.pms], dtype=torch.int64),
            loc_status=torch.tensor([self.pm_status[i.status] for i in self.pms], dtype=torch.int64),
            loc_hold_wafer=torch.tensor([i.hold_wafer.recipe if i.hold_wafer is not None else -1 for i in self.pms], dtype=torch.int64),
            loc_process_start_time=torch.tensor([i.prs_start_time if i.prs_start_time is not None else -1 for i in self.pms], dtype=torch.float),
            loc_process_end_time=torch.tensor([i.prs_end_time if i.prs_end_time is not None else -1 for i in self.pms], dtype=torch.float),

            robot_arm1_loc=torch.tensor([self.robot.loc[0]], dtype=torch.int64),
            robot_arm2_loc=torch.tensor([self.robot.loc[1]], dtype=torch.int64),
            robot_arm1_hold_wafer_recipe=torch.tensor([self.robot.hold_wafer[0].recipe if self.robot.hold_wafer[0] is not None else -1], dtype=torch.int64),
            robot_arm2_hold_wafer_recipe=torch.tensor([self.robot.hold_wafer[1].recipe if self.robot.hold_wafer[1] is not None else -1], dtype=torch.int64),
            robot_arm1_hold_wafer_next_stage=torch.tensor([self._get_next_stage_of_wafer(self.robot.hold_wafer[0]) if self.robot.hold_wafer[0] is not None else -1], dtype=torch.int64),
            robot_arm2_hold_wafer_next_stage=torch.tensor([self._get_next_stage_of_wafer(self.robot.hold_wafer[1]) if self.robot.hold_wafer[1] is not None else -1], dtype=torch.int64),

            recipe_flow=torch.tensor([i.flow for i in self.recipes], dtype=torch.int64),
            recipe_time=torch.tensor([i.time for i in self.recipes], dtype=torch.float),

            loadlock1_wafer_in=torch.tensor([foup.get_num_wafers_in() for foup in self.loadlock_1 if foup.wafers_in], dtype=torch.int64),
            loadlock2_wafer_in=torch.tensor([foup.get_num_wafers_in() for foup in self.loadlock_2 if foup.wafers_in], dtype=torch.int64),
            loadlock1_wafer_recipe=torch.tensor([foup.recipe for foup in self.loadlock_1 if foup.wafers_in], dtype=torch.int64),
            loadlock2_wafer_recipe=torch.tensor([foup.recipe for foup in self.loadlock_2 if foup.wafers_in], dtype=torch.int64)
        )

        return state

    """
    def get_action_mask(self) -> np.array:


        # Initialize action mask with zeros (0: infeasible action, 1: feasible action)
        num_ll_unload_action = 2 # group 1, group 2
        num_pm_unload_action = len(self.pms)
        num_load_action = len(self.pms)+1

        arm1_load_action_start_idx = num_ll_unload_action+num_pm_unload_action
        arm2_load_action_start_idx = num_ll_unload_action+num_pm_unload_action+num_load_action
        action_mask = np.zeros(num_ll_unload_action+num_pm_unload_action+num_load_action*2)

        def _can_unload_from_loadlock(loadlock):
            has_wafers_in_loadlock = any(len(foup.wafers_in) > 0 for foup in loadlock)
            return has_wafers_in_loadlock

        def _can_unload_from_pm(pm):
            return pm.hold_wafer is not None

        # Check if loading to each PM and out loadlock is possible
        def _can_load_to_pm(pm, arm_id):
            if self.robot.hold_wafer[arm_id] is None:
                return False
            else:
                next_stage = self._get_next_stage(self.robot.hold_wafer[arm_id])
                return pm.status == 'idle' and pm.stage == next_stage

        def _can_load_to_loadlock(arm_id):
            hold_wafer = self.robot.hold_wafer[arm_id]
            if hold_wafer is None:
                return False
            next_stage = self._get_next_stage(hold_wafer)
            return next_stage == len(self.stage) + 1

        def _is_next_stage_avail(next_stage):
            return any([pm.status == 'idle' and pm.stage == next_stage for pm in self.pms])

        ##################
        # masking process
        ##################
        num_idle_arm = sum([1 for i in self.robot.hold_wafer if i is None])

        # unload action mask
        ###################################
        # [case 1] two arm idle
        if num_idle_arm == 2:
            if _can_unload_from_loadlock(self.loadlock_1):
                action_mask[0] = 1

            # Check if unloading from loadlock type 2 is possible
            if _can_unload_from_loadlock(self.loadlock_2):
                action_mask[1] = 1

            # Check if unloading from each PM is possible
            for i, pm in enumerate(self.pms, start=num_ll_unload_action):
                if _can_unload_from_pm(pm):
                    action_mask[i] = 1

        elif num_idle_arm == 1:
            hold_arm_id = [i for i, v in enumerate(self.robot.hold_wafer) if v is not None][0]
            hold_wafer_next_stage = self._get_next_stage(self.robot.hold_wafer[hold_arm_id])
            hold_wafer_next_stage_avail = any([pm.status == 'idle' and
                                               pm.stage == hold_wafer_next_stage
                                               for pm in self.pms])
            # [case 2]
            if hold_wafer_next_stage_avail:
                if _can_unload_from_loadlock(self.loadlock_1):
                    action_mask[0] = 1

                # Check if unloading from loadlock type 2 is possible
                if _can_unload_from_loadlock(self.loadlock_2):
                    action_mask[1] = 1

                # Check if unloading from each PM is possible
                for i, pm in enumerate(self.pms, start=num_ll_unload_action):
                    if _can_unload_from_pm(pm):
                        action_mask[i] = 1

            # [case 3]
            else:
                # Check if unloading from each PM is possible
                for i, pm in enumerate(self.pms, start=num_ll_unload_action):
                    if _can_unload_from_pm(pm):
                        if _is_next_stage_avail(pm.stage+1) or hold_wafer_next_stage == pm.stage:
                            action_mask[i] = 1

        # load action mask
        ###################################
        for i, pm in enumerate(self.pms, start=arm1_load_action_start_idx):
            if _can_load_to_pm(pm, 0):
                action_mask[i] = 1

        for i, pm in enumerate(self.pms, start=arm2_load_action_start_idx):
            if _can_load_to_pm(pm, 1):
                action_mask[i] = 1

        if _can_load_to_loadlock(arm_id=0):
            arm1_ll_load_action_idx = arm2_load_action_start_idx-1
            action_mask[arm1_ll_load_action_idx] = 1

        if _can_load_to_loadlock(arm_id=1):
            arm2_load_action_idx = -1
            action_mask[arm2_load_action_idx] = 1

        return action_mask
        """

    def get_action_mask(self) -> np.array:
        """
        0: unload from LL type 1
        1: unload from LL type 2
        2~n+1: unload from PMs(#n)

        n+2 ~ 2n+2: load to PMs(#n) or outLL by arm 1
        2n+3 ~ 3n+3: load to PMs(#n) or outLL by arm 2

        unload from LL: current loadlock has wafer & at least one pm of the next stage is idle
        unload from PM: current loc is occupied & at least one pm of the next stage is idle
        unload arm is selected by (empty arm & nearest loc) rule

        unload action mask
        # case1: hold wafer is zero.
        # -> all pm holding wafer can be the unloaded

        # case2: hold wafer is one & hold wafer의 next stage가 available 할 때
        # -> all pm holding wafer can be unloaded

        # case3: hold wafer is one & hold wafer의 next stage가 unavailable 할 때
        # -> 1. unload wafer의 next stage가 available 하거나
        # -> 2. unload wafer의 현재 stage가 hold wafer의 next stage 일때

        # case4: hold wafer is two
        # -> 모든 unload action unavailable
        """

        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(self.pms)
        num_load_action = len(self.pms) + 1

        arm1_load_action_start_idx = num_ll_unload_action + num_pm_unload_action
        arm2_load_action_start_idx = num_ll_unload_action + num_pm_unload_action + num_load_action
        action_mask = np.zeros(num_ll_unload_action + num_pm_unload_action + num_load_action * 2)

        def _can_unload_from_loadlock(loadlock):
            return any(len(foup.wafers_in) > 0 for foup in loadlock)

        def _can_unload_from_pm(pm):
            return pm.hold_wafer is not None

        def _can_load_to_pm(pm, arm_id):
            hold_wafer = self.robot.hold_wafer[arm_id]
            if hold_wafer is None:
                return False
            next_stage = self._get_next_stage_of_wafer(hold_wafer)
            return pm.status == 'idle' and pm.stage == next_stage

        def _can_load_to_loadlock(arm_id):
            hold_wafer = self.robot.hold_wafer[arm_id]
            if hold_wafer is None:
                return False
            next_stage = self._get_next_stage_of_wafer(hold_wafer)
            return next_stage == self.outloadlock

        def _is_next_stage_avail(next_stage):
            if next_stage == self.outloadlock:
                return True
            else:
                return any(pm.status == 'idle' and pm.stage == next_stage for pm in self.pms)

        def _set_unload_actions():
            if _can_unload_from_loadlock(self.loadlock_1):
                action_mask[0] = 1
            if _can_unload_from_loadlock(self.loadlock_2):
                action_mask[1] = 1
            for i, pm in enumerate(self.pms, start=num_ll_unload_action):
                if _can_unload_from_pm(pm):
                    action_mask[i] = 1

        def _set_load_actions():
            for i, pm in enumerate(self.pms, start=arm1_load_action_start_idx):
                if _can_load_to_pm(pm, 0):
                    action_mask[i] = 1

            for i, pm in enumerate(self.pms, start=arm2_load_action_start_idx):
                if _can_load_to_pm(pm, 1):
                    action_mask[i] = 1

            if _can_load_to_loadlock(arm_id=0):
                action_mask[arm2_load_action_start_idx - 1] = 1

            if _can_load_to_loadlock(arm_id=1):
                action_mask[-1] = 1

        # unload action mask
        ##########################
        num_idle_arm = sum(1 for i in self.robot.hold_wafer if i is None)
        if num_idle_arm == 2:  # [case 1]
            _set_unload_actions()

        elif num_idle_arm == 1:
            hold_arm_id = next(i for i, v in enumerate(self.robot.hold_wafer) if v is not None)
            hold_wafer_next_stage = self._get_next_stage_of_wafer(self.robot.hold_wafer[hold_arm_id])
            hold_wafer_next_stage_avail = _is_next_stage_avail(hold_wafer_next_stage)

            # [case 2]
            if hold_wafer_next_stage_avail:
                _set_unload_actions()

            # [case 3]
            else:
                for i, pm in enumerate(self.pms, start=num_ll_unload_action):
                    if _can_unload_from_pm(pm):
                        next_stage = self._get_next_stage_of_pm(pm)
                        if _is_next_stage_avail(next_stage) or hold_wafer_next_stage == pm.stage:
                            action_mask[i] = 1

        # load action mask
        ##########################
        _set_load_actions()

        return action_mask

    def _get_next_stage_of_wafer(self, wafer: Wafer) -> int:
        if wafer.group == 0:
            return self.outloadlock if wafer.unloaded_stage == len(self.group1_stage) else wafer.unloaded_stage + 1
        elif wafer.group == 1:
            return len(self.group1_stage) + 1 if wafer.unloaded_stage == 0 else wafer.unloaded_stage + 1

    def _get_next_stage_of_pm(self, pm: Pm) -> int:
        if pm.group == 0:
            return self.outloadlock if pm.stage == len(self.group1_stage) else pm.stage + 1
        elif pm.group == 1:
            return len(self.group1_stage) + 1 if pm.stage == 0 else pm.stage + 1

    def get_reward(self):
        return self.clock

    def get_done(self):
        num_fin_wafers = sum(foup.get_num_wafers_out() for foup in self.loadlock_1) + \
                         sum(foup.get_num_wafers_out() for foup in self.loadlock_2)
        return num_fin_wafers >= self.done_quantity

    def show_state(self):
        print_line_size = 90
        print("=" * print_line_size)
        print(f"Clock: {self.clock} seconds")
        print(f"Arm Location: {self.robot.loc}")
        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(self.pms)
        num_load_action = len(self.pms) + 1

        unload_action_mask = self.action_mask[2:num_ll_unload_action + num_pm_unload_action]
        arm1_load_action_mask = self.action_mask[num_ll_unload_action + num_pm_unload_action:num_ll_unload_action + num_pm_unload_action + num_load_action]
        arm2_load_action_mask = self.action_mask[num_ll_unload_action + num_pm_unload_action + num_load_action:]


        print(f"{'Unload LL Action Mask:':<20} {self.action_mask[:2]}")
        print(f"{'Unload Action     Mask:'} {unload_action_mask}")
        print(f"{'Arm 1 Load Action Mask:'} {arm1_load_action_mask}")
        print(f"{'Arm 2 Load Action Mask:'} {arm2_load_action_mask}")
        print("-" * print_line_size)

        for arm_id, wafer in enumerate(self.robot.hold_wafer):
            wafer_id = wafer.id if wafer else -1
            wafer_next_stage = self._get_next_stage_of_wafer(wafer) if wafer else -1
            wafer_group = wafer.group if wafer else -1
            wafer_recipe = wafer.recipe if wafer else -1
            print(f"Arm {arm_id:<3} | Holding Wafer: {wafer_id:<4} | Next Stage: {wafer_next_stage:<4} | Group: {wafer_group:<4} | Recipe: {wafer_recipe:<4}")
        print("-" * print_line_size)

        for pm in self.pms:
            if pm.group == 1 and pm.stage == 1:
                print()  # Add a blank line between type1 and type2 PMs
            hold_wafer_id = pm.hold_wafer.id if pm.hold_wafer else -1
            remain_prs_time = pm.hold_wafer.remain_prs_time if pm.hold_wafer else 0
            print(f"PM {pm.id:<3} | Stage: {pm.stage:<3} | Status: {pm.status:<10} | Holding Wafer: {hold_wafer_id:<4} | Remaining Process Time: {remain_prs_time:<4}")

        print("-" * print_line_size)

        for i, loadlock in enumerate([self.loadlock_1, self.loadlock_2], start=1):
            for foup in loadlock:
                print(f"Loadlock {i:<3} | FOUP ID: {foup.id:<3} | Wafers In: {foup.get_num_wafers_in():<4} | Wafers Out: {foup.get_num_wafers_out():<4}")

        print("-" * print_line_size)

        for i, queue in enumerate([self.queue_1, self.queue_2], start=1):
            print(f"Queue {i:<6} | Number of FOUPs: {len(queue):<3}")

        print("-" * print_line_size)

        for i, exits in enumerate([self.exit_1, self.exit_2], start=1):
            print(f"Exit {i:<6} | Number of FOUPs: {len(exits):<3}")

        print("=" * print_line_size)

def main():
    SEED = 2024
    random.seed(SEED)
    np.random.seed(SEED)

    # Example configuration
    parser = argparse.ArgumentParser(description="Dual-armed cluster tool for concurrent flow")
    parser.add_argument('--foup_size', type=int, default=60, help='Size of the foup')
    parser.add_argument('--group1_stage', type=int, default=1, help='Stages for type 1')
    parser.add_argument('--group1_min_prs_time', type=int, default=5, help='Minimum processing time for type 1')
    parser.add_argument('--group1_max_prs_time', type=int, default=300, help='Maximum processing time for type 1')
    parser.add_argument('--group2_stage', type=int, default=1, help='Stages for type 2')
    parser.add_argument('--group2_min_prs_time', type=int, default=15, help='Minimum processing time for type 2')
    parser.add_argument('--group2_max_prs_time', type=int, default=250, help='Maximum processing time for type 2')
    parser.add_argument('--prod_quantity', type=int, default=10, help='Production quantity (Unit: FOUP)')
    parser.add_argument('--done_quantity', type=int, default=30, help='Done Production quantity (Unit: FOUP)')
    parser.add_argument('--num_lot_type', type=int, default=2, help='Total number of lot types')

    args = parser.parse_args()

    # Convert Namespace to dictionary
    args_dict = vars(args)

    # Initialize environment
    env = ddcfEnv(**args_dict)


    # Reset environment to initial state
    i = 0
    state = env.reset()
    from envs.algorithms.css import ConcurrentSwapSequence
    policy = ConcurrentSwapSequence(env, strategy='floor')
    while not env.done:
        #action = np.random.choice(np.where(env.action_mask == 1)[0])
        action = policy(env)
        state = env.step(action)
        i += 1

        # print the state
        print("=" * 90)
        num_ll_unload_action = 2  # group 1, group 2
        num_pm_unload_action = len(env.pms)
        num_load_action = len(env.pms) + 1
        if action == 0:
            print(f"{i}th, Selected Action: Unload from LL group 1")
        elif action == 1:
            print(f"{i}th, Selected Action: Unload from LL group 2")
        elif action < num_ll_unload_action + num_pm_unload_action:
            print(f"{i}th, Selected Action: Unload from PM {action - num_ll_unload_action + 1}")
        elif action < num_ll_unload_action + num_pm_unload_action + num_load_action:
            print(f"{i}th, Selected Action: Load to PM {action - num_ll_unload_action - num_pm_unload_action + 1} with Arm 1")
        else:
            print(f"{i}th, Selected Action: Load to PM {action - num_ll_unload_action - num_pm_unload_action - num_load_action + 1} with Arm 2")

        env.show_state()
        print("----")


    print("Environment initialized and reset successfully.")
    env.show_state()


if __name__ == "__main__":
    main()
