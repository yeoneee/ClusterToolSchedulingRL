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

    robot_loc:torch.Tensor=None

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
    loc: int = None
    hold_wafer: Wafer = None

    pkup_start_time: int = None
    pkup_end_time: int = None
    unload_start_time: int = None
    unload_end_time: int = None
    move_start_time: int = None
    move_end_time: int = None
    load_start_time: int = None
    load_end_time: int = None

    # func
    def pkup(self, env: object, loc_id: int):
        self.pkup_start_time = env.clock
        self.pkup_end_time = self.pkup_start_time + self.move_time
        self.loc = loc_id

    def move(self, env: object, loc_id: int):
        self.move_start_time = self.unload_end_time
        self.move_end_time = self.move_start_time + self.move_time
        self.loc = loc_id

    def unload(self, env: object, loc_id: int, group: int):
        # pkup wafer
        if self.loc != loc_id:
            self.pkup(env, loc_id)
        else:
            self.pkup_start_time = env.clock
            self.pkup_end_time = self.pkup_start_time

        # unload
        self.unload_start_time = max(self.pkup_end_time, env.pms[loc_id-1].prs_end_time) if loc_id != 0 else self.pkup_end_time
        self.unload_end_time = self.unload_start_time + self.unload_time

        if loc_id == 0:  # unload from loadlock
            self._unload_from_loadlock(env, group)
        else:  # unload from pm
            self._unload_from_pm(env, loc_id)

    def _unload_from_loadlock(self, env: object, group: int):
        loadlock = env.loadlock_1 if group == 0 else env.loadlock_2
        no_wafer = True
        for foup in loadlock:
            if len(foup.wafers_in) > 0:
                self.hold_wafer = foup.wafers_in.pop(0)
                self.hold_wafer.loc = -1
                no_wafer = False
                break

        if no_wafer:
            print('Error: Loadlock is empty')

    def _unload_from_pm(self, env: object, loc_id: int):
        self.hold_wafer = env.pms[loc_id - 1].hold_wafer
        self.hold_wafer.loc = -1
        env.pms[loc_id - 1].unload()

    def load(self, env: object, loc_id: int):
        if self.loc != loc_id:
            self.move(env, loc_id)

        # load
        self.load_start_time = self.move_end_time  # TODO purge time
        self.load_end_time = self.load_start_time + self.load_time

        if loc_id == env.outloadlock:
            self._load_to_loadlock(env)
        else:  # load to pm
            self._load_to_pm(env, loc_id)

        self.hold_wafer.loc = loc_id
        self.hold_wafer = None

    def _load_to_loadlock(self, env: object):
        loadlock = env.loadlock_1 if self.hold_wafer.group == 0 else env.loadlock_2
        for foup in loadlock:
            if foup.recipe == self.hold_wafer.recipe:
                foup.wafers_out.append(self.hold_wafer)
                break
            else:
                print('Error: Loadlock recipe mismatch')

    def _load_to_pm(self, env: object, loc_id: int):
        if env.pms[loc_id - 1].status == 'idle':
            env.pms[loc_id - 1].load(env, self.hold_wafer)
        else:
            print('Error: PM is not idle')

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


class sscfEnv:
    """
    # single-armed cluster tool for (dedicated) concurrent flow

    For example:
        6 PMs, with m = [1,1,1,1,1,1]
        Type 1: inloadlock -> stage 1 -> stage 2 -> stage 3 -> outloadlock
        Type 2: inloadlock -> stage 4 -> stage 5 -> stage 6 -> outloadlock

    """

    def __init__(self, foup_size: int, stage: List[int],
                 group1_stage: List[int], group1_min_prs_time: int, group1_max_prs_time: int,
                 group2_stage: List[int], group2_min_prs_time: int, group2_max_prs_time: int,
                 shared_min_prs_time: int, shared_max_prs_time: int,
                 prod_quantity: int, done_quantity: int, num_lot_type: int):
        # Initialize the environment with the given parameters
        self.robot_type = 'single'
        self.foup_size = foup_size
        self.inloadlock = 0
        self.outloadlock = len(stage) + 1
        self.pm_status = {'idle': 0, 'processing': 1, 'finished': 2}
        self.stage = stage
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

        # shared
        self.shared_min_prs_time = shared_min_prs_time
        self.shared_max_prs_time = shared_max_prs_time

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
        def _get_next_loc(group, next_stage):
            """
            Get the next location for the wafer based on the group and next stage.

            Args:
            group (int): The group (recipe) of the wafer.
            next_stage (int): The next stage for the wafer.

            Returns:
            int: The ID of the next location.
            """
            for pm in self.pms:
                if pm.stage == next_stage and pm.status == 'idle' and (pm.group == group or pm.group == -1):
                    return pm.id
            return None

        if action == 0:  # Unload from LL group 1
            self.robot.unload(self, loc_id=0, group=0)
            next_stage = self._get_next_stage(group=0, stage=0) # NOTE
            next_loc_id = _get_next_loc(0, next_stage)
            self.robot.load(self, loc_id=next_loc_id)

        elif action == 1:  # Unload from LL group 2
            self.robot.unload(self, loc_id=0, group=1)
            next_stage = self._get_next_stage(group=1, stage=0) # NOTE
            next_loc_id = _get_next_loc(1, next_stage)
            self.robot.load(self, loc_id=next_loc_id)

        else:  # Unload from PM
            unload_loc_id = action - 1
            unload_loc_stage = self.pms[unload_loc_id - 1].stage
            unload_wafer_group = self.pms[unload_loc_id - 1].hold_wafer.group

            self.robot.unload(self, loc_id=unload_loc_id, group=unload_wafer_group)

            if unload_wafer_group == 0:
                if sum(self.group1_stage[:unload_loc_stage]) == sum(self.group1_stage) and\
                      self.group1_stage[unload_loc_stage-1] != 0:
                    next_loc_id = self.outloadlock
                else:
                    next_stage = self._get_next_stage(unload_wafer_group, unload_loc_stage)
                    next_loc_id = _get_next_loc(unload_wafer_group, next_stage)

            elif unload_wafer_group == 1:
                if sum(self.group2_stage[:unload_loc_stage]) == sum(self.group2_stage) and\
                      self.group2_stage[unload_loc_stage-1] != 0:
                    next_loc_id = self.outloadlock
                else:
                    next_stage = self._get_next_stage(unload_wafer_group, unload_loc_stage)
                    next_loc_id = _get_next_loc(unload_wafer_group, next_stage)

            self.robot.load(self, loc_id=next_loc_id)

        # Transition the environment state
        self._transition()
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

        if self.num_lot_type > 2:
            recipe_group = 0
            for i in range(self.num_lot_type - 2):
                self._create_recipe(2+i, recipe_group)
                recipe_group = 1 - recipe_group

        # shared stage has the same process time
        self.shared_stage = [i for i in range(1, len(self.stage)+1) if self.group1_stage[i-1] == 1 and self.group2_stage[i-1] == 1]

        for stage in self.shared_stage:
            shared_prs_time = random.randint(self.shared_min_prs_time, self.shared_max_prs_time)
            for recipe in self.recipes:
                recipe.time[stage] =  shared_prs_time


    def _create_recipe(self, recipe, group):

        # Determine the flow based on the group
        flow = np.array(self.group1_stage if group == 0 else self.group2_stage)
        flow = np.insert(flow, 0, 1)  # Add inloadlock to the flow
        flow = np.append(flow, 1)  # Add outloadlock to the flow

        # Random processing time
        min_prs_time = self.group1_min_prs_time if group == 0 else self.group2_min_prs_time
        max_prs_time = self.group1_max_prs_time if group == 0 else self.group2_max_prs_time

        time = np.zeros(len(flow))
        for i in range(len(flow)):
            if flow[i] == 1:
                time[i] = random.randint(min_prs_time, max_prs_time)
        time[0] = 0  # Set inloadlock processing time to 0
        time[-1] = 0  # Set outloadlock processing time to 0

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
        self.robot = Robot(load_time=3, unload_time=3, move_time=3, loc=self.inloadlock)

    def _init_pms(self):
        """
        Initialize the processing machines (PMs).
        """
        self.pms = []
        cnt_pm = 1

        for s, m in enumerate(self.stage, start=1):
            for i in range(m):
                if s in self.shared_stage:
                    group = -1
                else:
                    group = 0 if self.group1_stage[s-1] == 1 else 1

                self.pms.append(Pm(id=cnt_pm, stage=s, group=group,
                                   status='idle', hold_wafer=None,
                                   prs_start_time=None, prs_end_time=None))
                cnt_pm += 1

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
            else: # shared PM
                if len(self.shared_stage) == 1:
                    _load_wafer_to_pm(pm, self.loadlock_1)

                elif len(self.shared_stage) > 1:
                    if pm.id % 2 == 0:
                        _load_wafer_to_pm(pm, self.loadlock_1)
                    else:
                        _load_wafer_to_pm(pm, self.loadlock_2)

    def _transition(self):
        # 시간 전이
        action_elapsed_time = self.robot.load_end_time - self.clock

        for pm in self.pms:
            if pm.status == 'processing':
                pm.hold_wafer.remain_prs_time -= action_elapsed_time
                if pm.hold_wafer.remain_prs_time <= 0:
                    pm.hold_wafer.remain_prs_time = 0
                    pm.status = 'finished'

            elif pm.status == 'purging':
                NotImplementedError

        # 로봇 위치 전이, inloadlock == outloadlock
        #self.arm.loc = self.inloadlock if self.arm.loc == self.outloadlock else self.arm.loc

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
        self.clock = self.robot.load_end_time

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

            robot_loc=torch.tensor([self.robot.loc], dtype=torch.int64),

            recipe_flow=torch.tensor(np.array([i.flow for i in self.recipes]), dtype=torch.int64),
            recipe_time=torch.tensor(np.array([i.time for i in self.recipes]), dtype=torch.float),

            loadlock1_wafer_in=torch.tensor([foup.get_num_wafers_in() for foup in self.loadlock_1 if foup.wafers_in], dtype=torch.int64),
            loadlock2_wafer_in=torch.tensor([foup.get_num_wafers_in() for foup in self.loadlock_2 if foup.wafers_in], dtype=torch.int64),
            loadlock1_wafer_recipe=torch.tensor([foup.recipe for foup in self.loadlock_1 if foup.wafers_in], dtype=torch.int64),
            loadlock2_wafer_recipe=torch.tensor([foup.recipe for foup in self.loadlock_2 if foup.wafers_in], dtype=torch.int64),
        )

        return state

    def get_action_mask(self) -> np.array:
        """
        0: unload from LL type 1
        1: unload from LL type 2
        2~n: unload from PM

        unload from LL: current loadlock has wafer & at least one pm of the next stage is idle
        unload from PM: current loc is occupied & at least one pm of the next stage is idle
        """

        # Initialize action mask with zeros (0: infeasible action, 1: feasible action)
        action_mask = np.zeros(len(self.pms) + 2)

        def _can_unload_from_loadlock(loadlock, group):
            """
            Check if unloading from loadlock is possible.
            Conditions:
            - Loadlock has wafers in it.
            - At least one PM of the next stage is idle.
            """
            has_wafers_in_loadlock = any(len(foup.wafers_in) > 0 for foup in loadlock)

            next_stage = self._get_next_stage(group, stage=0)
            has_idle_pm_in_next_stage = any(
            #pm.stage == next_stage and pm.group == group and pm.status == 'idle'
            pm.stage == next_stage and pm.status == 'idle'
            for pm in self.pms
            )

            return has_wafers_in_loadlock and has_idle_pm_in_next_stage

        def _can_unload_from_pm(pm):
            """
            Check if unloading from PM is possible.
            Conditions:
            - If PM is at the last stage, it must hold a wafer.
            - If PM is not at the last stage, it must hold a wafer and at least one PM of the next stage must be idle.
            """
            has_wafers_in_pm = pm.hold_wafer is not None

            if has_wafers_in_pm:
                next_stage = self._get_next_stage(pm.group, pm.stage, pm)
                if next_stage == self.outloadlock:
                    return True  # Last stage

                else:
                    for next_pm in self.pms:
                        if (next_pm.stage == next_stage and
                            next_pm.group == pm.hold_wafer.group and
                            next_pm.status == 'idle'):
                            return True

                        if next_pm.group == -1: # shared pm
                            if next_pm.stage == next_stage and next_pm.status == 'idle':
                                return True

            else:
                return False

        # Check if unloading from loadlock type 1 is possible
        if _can_unload_from_loadlock(self.loadlock_1, 0):
            action_mask[0] = 1

        # Check if unloading from loadlock type 2 is possible
        if _can_unload_from_loadlock(self.loadlock_2, 1):
            action_mask[1] = 1

        # Check if unloading from each PM is possible
        for i, pm in enumerate(self.pms, start=2):
            if _can_unload_from_pm(pm):
                action_mask[i] = 1

        return action_mask

    def get_reward(self):
        return self.clock

    def get_done(self) -> np.array:
        """
        if len(self.exit_1) + len(self.exit_2) >= self.done_quantity:
            return True
        else:
            return False
        """
        if sum([len(i.wafers_out) for i in self.loadlock_1]) + \
            sum([len(i.wafers_out) for i in self.loadlock_2]) >= self.done_quantity:
            return True
        else:
            return False

    def _get_next_stage(self, group, stage, pm=None):
        if group == 0:
            next_stage = next((i for i, s in enumerate(self.group1_stage[stage:], start=stage+1)
                               if s == 1), self.outloadlock)
        elif group == 1:
            next_stage = next((i for i, s in enumerate(self.group2_stage[stage:], start=stage+1)
                               if s == 1), self.outloadlock)
        else:  # shared
            if pm and pm.hold_wafer.recipe == 0:
                next_stage = self._get_next_stage(0, stage)
            else:
                next_stage = self._get_next_stage(1, stage)

        return next_stage



    def show_state(self):
        print_line_size = 90
        print("=" * print_line_size)
        print(f"Clock: {self.clock} seconds")
        print(f"Arm Location: {self.robot.loc}")
        print(f"Action Mask: {self.action_mask}")
        print("-" * print_line_size)

        for pm in self.pms:
            if pm.group == 1 and pm.stage == 1:
                print()  # Add a blank line between type1 and type2 PMs
            hold_wafer_id = pm.hold_wafer.id if pm.hold_wafer else -1
            remain_prs_time = pm.hold_wafer.remain_prs_time if pm.hold_wafer else 0
            hold_wafer_recipe = pm.hold_wafer.recipe if pm.hold_wafer else -1
            print(f"PM {pm.id:<3} | Stage: {pm.stage:<3} | Status: {pm.status:<10} | Holding Wafer: {hold_wafer_id:<4} | Remaining Process Time: {remain_prs_time:<4} | Wafer Recipe: {hold_wafer_recipe:<4}")

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
    parser = argparse.ArgumentParser(description="Single-armed cluster tool for concurrent flow")
    parser.add_argument('--foup_size', type=int, default=25, help='Size of the foup')
    parser.add_argument('--stage', type=int, nargs='+', default=[1,1,1,1,1], help='stage (m)')
    parser.add_argument('--group1_stage', type=int, nargs='+', default=[1,1,1,1,0], help='Stages for type 1, [1,0,1,0,1], [1,1,1,1,0]')
    #parser.add_argument('--group1_stage', type=int, nargs='+', default=[1,0,1,0,1], help='Stages for type 1, [1,0,1,0,1], [1,1,1,1,0]')
    parser.add_argument('--group1_min_prs_time', type=int, default=5, help='Minimum processing time for type 1')
    parser.add_argument('--group1_max_prs_time', type=int, default=300, help='Maximum processing time for type 1')
    parser.add_argument('--group2_stage', type=int, nargs='+', default=[0,0,1,1,1], help='Stages for type 2, [0,1,1,1,0], [0,0,1,1,1]')
    #parser.add_argument('--group2_stage', type=int, nargs='+', default=[0,1,1,1,0], help='Stages for type 2, [0,1,1,1,0], [0,0,1,1,1]')
    parser.add_argument('--group2_min_prs_time', type=int, default=15, help='Minimum processing time for type 2')
    parser.add_argument('--group2_max_prs_time', type=int, default=250, help='Maximum processing time for type 2')
    parser.add_argument('--prod_quantity', type=int, default=10, help='Production quantity (Unit: FOUP)')
    parser.add_argument('--done_quantity', type=int, default=20, help='Done Production quantity (Unit: FOUP)')
    parser.add_argument('--num_lot_type', type=int, default=2, help='Total number of lot types')
    parser.add_argument('--shared_min_prs_time', type=int, default=10, help='Minimum processing time for shared stage')
    parser.add_argument('--shared_max_prs_time', type=int, default=200, help='Maximum processing time for shared stage')

    args = parser.parse_args()

    # Convert Namespace to dictionary
    args_dict = vars(args)

    # Initialize environment
    env = sscfEnv(**args_dict)


    # Reset environment to initial state
    i = 0
    state = env.reset()
    from envs.algorithms.abs import AlternatingBackwardSequence
    policy = AlternatingBackwardSequence(env)
    while not env.done:
        #action = np.random.choice(np.where(env.action_mask == 1)[0])
        action = policy(env)
        state = env.step(action)
        i += 1

        if action == 0:
            print(f"{i}th, Selected Action: Unload from LL group 1")
        elif action == 1:
            print(f"{i}th, Selected Action: Unload from LL group 2")
        else:
            print(f"{i}th, Selected Action: Unload from PM {action-1}")
        env.show_state()


    print("Environment initialized and reset successfully.")


if __name__ == "__main__":
    main()
