

# Deprecated
##########################################################################################
from dataclasses import dataclass
import torch
import copy

@dataclass
class IndArmedRobot:
    # static
    num_arm = 2
    move_time = 3
    load_time = 3
    unload_time = 3

    # dynamic
    loc: torch.Tensor = None
    hold_wafer: torch.Tensor = None
    pkup_start_time: torch.Tensor = None
    pkup_end_time: torch.Tensor = None
    unload_start_time: torch.Tensor = None
    unload_end_time: torch.Tensor = None
    move_start_time: torch.Tensor = None
    move_end_time: torch.Tensor = None
    load_start_time: torch.Tensor = None
    load_end_time: torch.Tensor = None

    def __init__(self, env):
        self.loc = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.int64,
        )

        self.hold_wafer = -torch.ones(
            size=(*env.batch_size,2),
            dtype=torch.int64,
        )
        self.pkup_start_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.pkup_end_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.unload_start_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.unload_end_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.move_start_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.move_end_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.load_start_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )
        self.load_end_time = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.float,
        )

        if env.kwargs.get('norm_time', True):
            self.move_time = (self.move_time - 2) / (300 - 2)
            self.load_time = (self.load_time - 2) / (300 - 2)
            self.unload_time = (self.unload_time - 2) / (300 - 2)

    def next_clock(self, env):
        pass


    def pkup(self, env, action):
        """
        unload action을 수행하는 batch 중 arm이 움직여야하는 경우 수행
        """

        # check the unload batch index
        unload_batch_idx = action.is_load == False & ~env.done
        unload_loc_idx = action.unload_loc
        unload_arm_idx = action.robot_idx

        # check the pick up need
        need_pkup = self.loc[env.batch_idx, unload_arm_idx] != unload_loc_idx

        # set pick up batch index
        pkup_batch_idx = unload_batch_idx & need_pkup

        # arm loc update
        self.loc[pkup_batch_idx, unload_arm_idx[pkup_batch_idx]] = unload_loc_idx[pkup_batch_idx]

        # move time update
        self.pkup_start_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]] = env.clock[unload_batch_idx]
        self.pkup_end_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]] = env.clock[unload_batch_idx]

        self.pkup_end_time[pkup_batch_idx, unload_arm_idx[pkup_batch_idx]] += self.move_time



    def move(self, env, action):
        """
        load action을 수행하는 batch 에 대해서 수행
        """
        load_batch_idx = action.is_load == True & ~env.done
        load_loc_idx = action.load_loc
        load_foup_idx = action.foup_idx
        load_wafer_idx = action.wafer_idx
        load_arm_idx = action.robot_idx

        # check the move need
        move_batch_idx = torch.logical_and(self.loc[env.batch_idx, load_arm_idx] != load_loc_idx, load_batch_idx)

        # arm loc update
        self.loc[move_batch_idx, load_arm_idx[move_batch_idx]] = load_loc_idx[move_batch_idx]

        # move time update
        self.move_start_time[load_batch_idx, load_arm_idx[load_batch_idx]] = env.clock[load_batch_idx]
        self.move_end_time[load_batch_idx, load_arm_idx[load_batch_idx]] = env.clock[load_batch_idx]

        self.move_end_time[move_batch_idx, load_arm_idx[move_batch_idx]] += self.move_time


    def unload(self, env, action):
        """
        # case: unload the wafer from the chamber: unload(3s)
        # case: unload the wafer from the loadlock & other arm unloaded firstly: z-move(3s) -> unload(3s)
        # case: unload the wafer from the loadlock & no other arm unloaded firstly: unload(3s)
        """


        # 1. pick up if needed
        self.pkup(env, action)

        # 2. get the wafer name & unload ready time
        unload_batch_idx = env.batch_idx[action.is_load == False & ~env.done]
        foup_idx = action.foup_idx[action.is_load == False & ~env.done]
        wafer_idx = action.wafer_idx[action.is_load == False & ~env.done]
        wafer_name = env.wafer.name[unload_batch_idx, foup_idx, wafer_idx]
        wafer_ready_time = env.wafer.ready_time[unload_batch_idx, foup_idx, wafer_idx]
        unload_arm_idx = action.robot_idx

        # 3. unload start/end time update
        self.unload_start_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]] = torch.max(
            torch.stack([self.pkup_end_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]],
                         wafer_ready_time]), dim=0
        ).values

        self.unload_end_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]] =\
              self.unload_start_time[unload_batch_idx, unload_arm_idx[unload_batch_idx]] + self.unload_time

        # 4. load wafer to the arm
        self.hold_wafer[unload_batch_idx, unload_arm_idx[unload_batch_idx]] = wafer_name

        # 5. unload from the PM
        env.loc.unload(env, action)

        # 6. unload the wafer & update
        env.wafer.unload(env, action)

        # 7. purge the PM
        if env.purge_constraint:
            env.loc.purge(env, action)

    def _unload(self, env, td):
        """
        # case: unload the wafer from the chamber: unload(3s)
        # case: unload the wafer from the loadlock & other arm unloaded firstly: z-move(3s) -> unload(3s)
        # case: unload the wafer from the loadlock & no other arm unloaded firstly: unload(3s)
        """
        unload_idx = env.control.is_load==0

        # get the unload ready time(=process end time)
        wafer_ready_time = td['wafer_ready_time']\
            [torch.arange(*td.shape), env.control.tar_lot, env.control.wafer]

        # unload start/end time update
        self.unload_start_time = copy.deepcopy(self.pkup_end_time[:, self.unload_tar_arm])
        self.unload_end_time = copy.deepcopy(self.unload_start_time)

        lm_unload_idx = self.loc[unload_idx, self.unload_tar_arm] == 0
        # 같은 위치에 있는 arm이 unload를 먼저 같은 시점에 수행한 경우 wait time 추가
        other_arm_just_unload_lm_idx = (self.loc[unload_idx, 1-self.unload_tar_arm] ==
                                     self.loc[unload_idx, self.unload_tar_arm]) &\
                                     self.hold_wafer[unload_idx, 1-self.unload_tar_arm] != -1 &\
                                     self.unload_start_time[unload_idx, 1-self.unload_tar_arm] ==\
                                        self.unload_start_time[unload_idx, self.unload_tar_arm]


        z_wait_time = torch.zeros_like(self.unload_start_time)
        z_wait_time[unload_idx & lm_unload_idx & other_arm_just_unload_lm_idx] = self.move_time
        # (batch, 2)

        self.unload_start_time[unload_idx, self.unload_tar_arm] = z_wait_time + torch.max(
            torch.stack([self.pkup_end_time[unload_idx, self.unload_tar_arm],
                         wafer_ready_time[unload_idx]]), dim=0).values

        self.unload_end_time[unload_idx, self.unload_tar_arm] =\
              self.unload_start_time[unload_idx, self.unload_tar_arm] + self.unload_time

        # load batch case -> -1 (not unload)
        self.unload_start_time[~unload_idx, self.unload_tar_arm] = -1
        self.unload_end_time[~unload_idx, self.unload_tar_arm] = -1

        # unload from the chamber
        env.loc.unload(env, td)
        env.loc.purge(env, td)

        # unload wafer
        env.wafer.unload(env, td)

        self.hold_wafer[unload_idx[:, None].repeat(1,2) & self.unload_tar_arm] =\
              env.control.wafer[unload_idx]


    def load(self, env, action, delay_time=None):
        load_batch_idx = action.is_load == True & ~env.done
        load_loc_idx = action.load_loc[load_batch_idx]
        load_arm_idx = action.robot_idx

        # 1. move to the next PM
        self.move(env, action)

        # 2. get the load ready time
        # 3. load start / end time
        if env.purge_constraint:
            loc_ready_time = env.loc.purge_end_time[load_batch_idx, load_loc_idx]

            self.load_start_time[load_batch_idx, load_arm_idx[load_batch_idx]] = torch.max(
                torch.stack([self.move_end_time[load_batch_idx, load_arm_idx[load_batch_idx]], loc_ready_time]), dim=0
            ).values

        else:
            loc_ready_time = torch.zeros_like(load_batch_idx, dtype=torch.float)

            self.load_start_time[load_batch_idx, load_arm_idx[load_batch_idx]] = torch.max(
                torch.stack([self.move_end_time[load_batch_idx, load_arm_idx[load_batch_idx]],
                             loc_ready_time[load_batch_idx]]), dim=0
            ).values

        self.load_end_time[load_batch_idx, load_arm_idx[load_batch_idx]] =\
              self.load_start_time[load_batch_idx, load_arm_idx[load_batch_idx]] + self.load_time

        # 4. unload from the arm
        self.hold_wafer[load_batch_idx, load_arm_idx[load_batch_idx]] = -1

        # 5. load wafer & loc
        env.wafer.load(env, action)
        env.loc.load(env, action)

        # 6. process wafer & loc
        env.loc.process(env, action)
        env.wafer.process(env, action)


    def _load(self, env, td):
        """
        # case: load the wafer to the chamber & other arm unloaded firstly: z-move(3s) -> load(3s)
        # case: load the wafer to the loadlock & other arm loaded firstly: z-move(3s) -> load(3s)
        # case: load the wafer to the loadlock & no other arm loaded firstly: load(3s)
        """
        load_idx = env.control.is_load==1

        # get the load ready time
        loc_ready_time = td['loc_purge_end_time']\
            [torch.arange(*td.shape), env.control.loc]

        # load start/end time
        self.load_start_time = copy.deepcopy(self.move_end_time)
        self.load_end_time = copy.deepcopy(self.load_start_time)

        pm_load_idx = self.loc[load_idx, self.load_tar_arm] != env.loc.count_stage()+1
        lm_load_idx = self.loc[load_idx, self.load_tar_arm] == env.loc.count_stage()+1

        other_arm_just_unload_pm_idx = (self.loc[load_idx, 1-self.load_tar_arm] ==
                                             self.loc[load_idx, self.load_tar_arm]) &\
                                    self.hold_wafer[load_idx, 1-self.load_tar_arm] != -1 &\
                                    self.load_start_time[load_idx, 1-self.load_tar_arm] ==\
                                    self.unload_start_time[load_idx, self.load_tar_arm]

        other_arm_just_load_lm_idx = (self.loc[load_idx, 1-self.load_tar_arm] ==
                                                self.loc[load_idx, self.load_tar_arm]) &\
                                        self.hold_wafer[load_idx, 1-self.load_tar_arm] == -1 &\
                                        self.load_start_time[load_idx, 1-self.load_tar_arm] ==\
                                        self.load_start_time[load_idx, self.load_tar_arm]

        z_wait_time = torch.zeros_like(self.load_start_time)
        z_wait_time[load_idx & pm_load_idx & other_arm_just_unload_pm_idx] = self.move_time
        z_wait_time[load_idx & lm_load_idx & other_arm_just_load_lm_idx] = self.move_time


        self.load_start_time[load_idx] = z_wait_time + torch.max(
            torch.stack([self.move_end_time[load_idx],
                         loc_ready_time[load_idx]]), dim=0).values

        self.load_end_time[load_idx] =\
                self.load_start_time[load_idx] + self.load_time

        # unload action
        self.load_start_time[~load_idx] = -1
        self.load_end_time[~load_idx] = -1

        # load to the chamber
        env.loc.load(env, td)
        env.loc.process(env, td)


        # load wafer
        env.wafer.load(env, td)

        # (batch, 2)
        self.hold_wafer[load_idx[:, None].repeat(1,2) & self.load_tar_arm] = -1


    def valid_load_action(self, env):
        """
        load action 중 avail 한 action mask
        robot arm이 들고 있는 wafer의 다음 stage에 비어있는 PM이 존재할 때 valid load
        TODO: 다른 arm이 하러 가고 있는 경우에 대한 처리 필요
        """
        # (batch size, num_arm, num_pm+1)
        action_mask = torch.zeros(size=(*env.batch_size, self.num_arm, env.loc.num_pm+1), dtype=torch.bool)

        for arm in range(self.num_arm):
            arm_hold_batch_idx = (self.hold_wafer[:, arm] != -1).nonzero()[:, 0]

            arm_hold_wafer = self.hold_wafer[arm_hold_batch_idx, arm]
            arm_wafer_recipe = env.wafer.get_recipe(arm_hold_wafer)
            arm_wafer_step = env.wafer.get_step(arm_hold_wafer)
            arm_wafer_next_step = arm_wafer_step + 1
            arm_wafer_next_stage = env.recipe_table.get('flow')\
                [arm_hold_batch_idx, arm_wafer_recipe, arm_wafer_next_step]

            arm_wafer_next_stage_loc = env.loc.stage[arm_hold_batch_idx, :]==\
                arm_wafer_next_stage[:, None].repeat(1, env.loc.num_loc)
            arm_wafer_next_stage_avail_loc = torch.logical_and(
                arm_wafer_next_stage_loc,
                env.loc.get_unloaded_pm()[arm_hold_batch_idx, :]
            )

            action_mask[arm_hold_batch_idx, arm, :] = arm_wafer_next_stage_avail_loc[:, 1:]

        return action_mask

    def valid_unload_action(self, env):
        # --------------------------------------------------------------------------------
        # case1: hold wafer is zero.
        # -> all pm holding wafer can be the unloaded

        # case2: hold wafer is one & hold wafer의 next stage가 available 할 때
        # -> all pm holding wafer can be unloaded

        # case3: hold wafer is one & hold wafer의 next stage가 unavailable 할 때
        # -> 1. unload wafer의 next stage가 available 하거나
        # -> 2. unload wafer의 현재 stage가 hold wafer의 next stage 일때

        # case4: hold wafer is two
        # -> 모든 unload action unavailable
        # --------------------------------------------------------------------------------
        loaded_loc = env.loc.get_loaded_loc()
        stage_avail = env.loc.get_avail_stage(env)

        two_arm_free_batch_idx = (self.hold_wafer == -1).count_nonzero(dim=-1) == 2 #(batch,)
        one_arm_free_batch_idx = (self.hold_wafer == -1).count_nonzero(dim=-1) == 1

        hold_arm_idx = torch.zeros(size=(*env.batch_size,), dtype=torch.int64)
        hold_arm_idx[one_arm_free_batch_idx] = (self.hold_wafer[one_arm_free_batch_idx] != -1).nonzero()[:, 1]

        hold_wafer_name = self.hold_wafer[env.batch_idx, hold_arm_idx] # (batch)
        hold_wafer_recipe = torch.clip(env.wafer.get_recipe(hold_wafer_name), max=env.num_lot_type-1)
        hold_wafer_step = env.wafer.get_step(hold_wafer_name)
        hold_wafer_next_step = torch.clip(hold_wafer_step + 1, max=env.num_step-1)
        # clipping
        hold_wafer_next_stage = env.recipe_table.get('flow')[env.batch_idx, hold_wafer_recipe, hold_wafer_next_step]
        hold_wafer_next_stage_avail = stage_avail[env.batch_idx, hold_wafer_next_stage]

        # unload loadlock action mask
        # --------------------------------------------------------------------------------
        # 1. check the earliest foup wafer still in loadlock
        lm_wafer_idx = (env.wafer.status == env.wafer.status_dict.get('inloadport')) # (batch, num_foup, foup_size)
        wafer_foup_idx = env.wafer.get_foup(env.wafer.name)
        earlier_foup_idx = (torch.where(lm_wafer_idx, wafer_foup_idx, 1e9)
                         .reshape(*env.batch_size, -1)
                         .min(dim=-1)[0]
                         .to(torch.int64))

        earlier_foup_idx = earlier_foup_idx[:, None, None].repeat(1, env.num_foup, env.wafer.foup_size)
        earlier_foup_wafer_idx = wafer_foup_idx == earlier_foup_idx       # (batch, num_foup, foup_size)

        is_earlier_foup_wafer_in_lm = (torch.logical_and(lm_wafer_idx, earlier_foup_wafer_idx)
                                       .reshape(*env.batch_size, -1).any(dim=-1))      # (batch, )

        # 2.loadlock wafer available
        avail_lm_wafer_idx = torch.zeros_like(env.wafer.name, dtype=torch.bool)

        # 2.1 earlier FOUP wafers are selected first
        avail_lm_wafer_idx[is_earlier_foup_wafer_in_lm] =\
            torch.logical_and(lm_wafer_idx, earlier_foup_wafer_idx)[is_earlier_foup_wafer_in_lm]

        # 2.2 otherwise, other FOUP wafer can be selected
        avail_lm_wafer_idx[~is_earlier_foup_wafer_in_lm] = lm_wafer_idx[~is_earlier_foup_wafer_in_lm]

        # 3. loadlock lot type is available (exist that lot type wafers in loadlock)
        avail_lm_wafer_idx = avail_lm_wafer_idx.nonzero()
        lm_wafer_batch_idx = avail_lm_wafer_idx[:, 0]
        lm_wafer_foup_idx = avail_lm_wafer_idx[:, 1]
        lm_wafer_idx = avail_lm_wafer_idx[:, 2]
        lm_wafer_recipe_idx =env.wafer.get_recipe(env.wafer.name[lm_wafer_batch_idx,lm_wafer_foup_idx,lm_wafer_idx])

        is_lot_type_in_lm = torch.zeros(size=(*env.batch_size, env.num_lot_type), dtype=torch.int64)
        is_lot_type_in_lm[lm_wafer_batch_idx, lm_wafer_recipe_idx] = 1
        is_lot_type_avail = is_lot_type_in_lm > 0

        # 4. wafers' next load stage available
        lm_wafer_next_stage_idx = env.recipe_table.get('flow')[:, :, 1] # (batch, num_lot_type)
        is_next_stage_avail = torch.gather(stage_avail, -1, lm_wafer_next_stage_idx)

        ll_unload_action_mask = torch.zeros(size=(*env.batch_size, self.num_arm, env.num_lot_type), dtype=torch.bool)

        # 5. case 1. hold wafer is zero. -> all ll wafer can be the unloaded
        ll_unload_action_mask[two_arm_free_batch_idx, :, :] =\
              is_lot_type_avail[:,None, :].repeat(1, self.num_arm, 1)[two_arm_free_batch_idx]

        # 5. case 2. hold wafer is one & hold wafer의 next stage가 available 할 때
        case2 = one_arm_free_batch_idx & hold_wafer_next_stage_avail
        ll_unload_action_mask[case2,  (1-hold_arm_idx)[case2]] = is_lot_type_avail[case2]

        # 5. case 3. hold wafer is one & hold wafer의 next stage가 unavailable 할 때
        # -> 1. unload wafer의 next stage가 available 할 때
        # -> 2. unload wafer의 현재 stage가 hold wafer의 next stage 일때 -> loadlock의 경우 case 없음
        case3 = one_arm_free_batch_idx & ~hold_wafer_next_stage_avail
        ll_unload_action_mask[case3, (1-hold_arm_idx)[case3]] = (is_lot_type_avail & is_next_stage_avail)[case3]


        # unload PM action mask
        # --------------------------------------------------------------------------------
        pm_unload_action_mask = torch.zeros(size=(*env.batch_size, self.num_arm, env.loc.num_pm), dtype=torch.bool)

        # case 1: 두 arm이 모두 free한 경우 -> 모든 pm에서 wafer unload 가능
        loaded_pm = loaded_loc[:, 1:-1] # (batch, num_pm)
        pm_unload_action_mask[two_arm_free_batch_idx, :, :] = loaded_pm[:, None, :].repeat(1, self.num_arm, 1)

        # case 2: arm 하나만 free하고 들고있는 wafer의 next stage가 available한 경우(배치 조건). 모든 pm에서 wafer unload 가능
        case2 = one_arm_free_batch_idx & hold_wafer_next_stage_avail
        pm_unload_action_mask[case2, (1-hold_arm_idx)[case2]] = loaded_pm[case2]

        # case 3: arm 하나만 free하고 들고있는 wafer의 next stage가 unavailable한 경우.
        # (1) unload wafer의 next stage가 available하거나
        # (2) unload wafer의 stage가 hold wafer의 next stage인 경우 ->  pm에서 wafer unload 가능
        case3 = one_arm_free_batch_idx & ~hold_wafer_next_stage_avail

        # case 3-1
        pm_wafer_name = env.loc.hold_wafer[:, 1:-1]
        pm_wafer_recipe = torch.clip(env.wafer.get_recipe(pm_wafer_name), max=env.num_lot_type-1)
        pm_wafer_step = env.wafer.get_step(pm_wafer_name)
        pm_wafer_next_step = torch.clip(pm_wafer_step + 1, max=env.num_step-1)
        pm_wafer_next_stage = env.recipe_table.get('flow')[env.batch_idx[:, None].repeat(1, env.loc.num_pm), pm_wafer_recipe, pm_wafer_next_step]
        loaded_wafer_next_stage_avail = stage_avail[env.batch_idx[:, None].repeat(1, env.loc.num_pm), pm_wafer_next_stage]

        pm_unload_action_mask[case3, (1-hold_arm_idx)[case3]] = (loaded_pm & loaded_wafer_next_stage_avail)[case3]

        # case 3-2
        is_hold_wafer_next_stage_pm = env.loc.stage[:, 1:-1] == hold_wafer_next_stage[:, None].repeat(1, env.loc.num_pm)
        pm_unload_action_mask[case3, (1-hold_arm_idx)[case3]] = (loaded_pm & is_hold_wafer_next_stage_pm)[case3]

        # concat input action + move action
        unload_action_mask = torch.cat([ll_unload_action_mask, pm_unload_action_mask], dim=-1)


        return unload_action_mask
