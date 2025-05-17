import torch
from dataclasses import dataclass

@dataclass
class DualArmedRobot:
    # static
    num_arm = 2
    move_time = 3
    load_time = 3
    unload_time = 3

    # dynamic
    loc:torch.Tensor  = None                  # (batch, 2) dual arm 1,2
    hold_wafer:torch.Tensor  = None           # (batch, 2) dual arm 1,2
    pkup_start_time:torch.Tensor  = None      # (batch, 1)
    pkup_end_time:torch.Tensor  = None        # (batch, 1)
    unload_start_time:torch.Tensor  = None    # (batch, 1)
    unload_end_time:torch.Tensor  = None      # (batch, 1)
    move_start_time:torch.Tensor  = None      # (batch, 1)
    move_end_time:torch.Tensor  = None        # (batch, 1)
    load_start_time:torch.Tensor  = None      # (batch, 1)
    load_end_time:torch.Tensor  = None        # (batch, 1)


    def __init__(self, env):
        self.loc = torch.zeros(
            size=(*env.batch_size,2),
            dtype=torch.int64,
        )
        self.loc[:,1] = self.get_opposite_arm_loc(env, self.loc[:,0])

        self.hold_wafer = -torch.ones(
            size=(*env.batch_size,2),
            dtype=torch.int64,
        )
        self.pkup_start_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.pkup_end_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.unload_start_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.unload_end_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.move_start_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.move_end_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.load_start_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )
        self.load_end_time = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.float,
        )

        if env.kwargs.get('norm_time', True):
            min_time_unit = env.kwargs.get('min_time_unit', 2)
            max_time_unit = env.kwargs.get('max_process_time', 100)

            self.move_time = (self.move_time - min_time_unit) / (max_time_unit - min_time_unit)
            self.load_time = (self.load_time - min_time_unit) / (max_time_unit - min_time_unit)
            self.unload_time = (self.unload_time - min_time_unit) / (max_time_unit - min_time_unit)

    def pkup(self, env, action):
        """
        Perform when the arm needs to move among the batch performing the unload action
        """
        # check the unload batch index
        unload_batch_idx = action.is_load == False & ~env.done
        unload_loc_idx = action.unload_loc                 # check the wafer loc

        # check the pick up need
        faced_arm = self.loc == unload_loc_idx[:, None].repeat(1, 2)
        empty_arm = self.hold_wafer == -1
        need_pkup = ~((faced_arm & empty_arm).any(dim=-1)) & (~env.done)

        # select the target arm
        tar_arm = -torch.ones(*env.batch_size, dtype=torch.int64)

        # case1: arm faced the PM of the wafer -> no pick up move
        arm_idx = faced_arm[unload_batch_idx & ~need_pkup].nonzero(as_tuple=True)[1]
        tar_arm[unload_batch_idx & ~need_pkup] = arm_idx

        # case2: arm not faced the PM of the wafer -> pick up move
        arm1_is_closer = torch.abs(self.loc - unload_loc_idx[:, None].repeat(1,2)).argmin(dim=-1) == 0
        arm1_is_empty = empty_arm[:, 0] == True
        arm2_is_empty = empty_arm[:, 1] == True

        # set target arm
        pkup_batch_idx = unload_batch_idx & need_pkup

        # one arm free
        tar_arm[pkup_batch_idx & arm1_is_empty & ~arm2_is_empty] = 0
        tar_arm[pkup_batch_idx & ~arm1_is_empty & arm2_is_empty] = 1

        # two arm free
        tar_arm[pkup_batch_idx & arm1_is_empty & arm2_is_empty & arm1_is_closer] = 0
        tar_arm[pkup_batch_idx & arm1_is_empty & arm2_is_empty & ~arm1_is_closer] = 1

        # (batch, 2)
        # arm loc update
        self.loc[pkup_batch_idx, tar_arm[pkup_batch_idx]] = unload_loc_idx[pkup_batch_idx]
        self.loc[pkup_batch_idx, (1-tar_arm)[pkup_batch_idx]] =\
              self.get_opposite_arm_loc(env, self.loc[pkup_batch_idx, tar_arm[pkup_batch_idx]])

        # move time update
        self.pkup_start_time[unload_batch_idx] = env.clock[unload_batch_idx]
        self.pkup_end_time[unload_batch_idx & ~need_pkup] = self.pkup_start_time[unload_batch_idx & ~need_pkup]
        self.pkup_end_time[unload_batch_idx & need_pkup] = self.pkup_start_time[unload_batch_idx & need_pkup] + self.move_time

        return tar_arm

    def move(self, env, action):
        """
        load할 wafer를 들고 있는 arm이 load할 PM에 있는지
        """
        load_batch_idx = action.is_load == True & ~env.done
        load_loc_idx = action.load_loc
        load_foup_idx = action.foup_idx
        load_wafer_idx = action.wafer_idx


        # check the move need
        faced_arm = self.loc == load_loc_idx[:, None].repeat(1, 2)
        wafer_hold_arm = self.hold_wafer ==\
              env.wafer.name[env.batch_idx,
                             load_foup_idx,
                             load_wafer_idx
                             ][:, None].repeat(1, 2)

        need_move_batch_idx = torch.logical_and(~(faced_arm & wafer_hold_arm).any(dim=-1), load_batch_idx)
        # (batch, )

        # arm loc update
        tar_arm_idx = wafer_hold_arm[need_move_batch_idx].nonzero()[:,1]
        self.loc[need_move_batch_idx, tar_arm_idx] = load_loc_idx[need_move_batch_idx]
        self.loc[need_move_batch_idx, (1-tar_arm_idx)] =\
                self.get_opposite_arm_loc(env, self.loc[need_move_batch_idx, tar_arm_idx])

        # move time update
        self.move_start_time[load_batch_idx] = env.clock[load_batch_idx]
        self.move_end_time[load_batch_idx] = self.move_start_time[load_batch_idx]

        self.move_end_time[need_move_batch_idx] += self.move_time


    def unload(self, env, action):
        # 1. pick up if needed
        tar_arm = self.pkup(env, action)

        # 2. get the wafer name & unload ready time (=process end time)
        unload_batch_idx = env.batch_idx[action.is_load == False & ~env.done]
        foup_idx = action.foup_idx[action.is_load == False & ~env.done]
        wafer_idx = action.wafer_idx[action.is_load == False & ~env.done]

        wafer_name = env.wafer.name[unload_batch_idx, foup_idx, wafer_idx]
        wafer_ready_time = env.wafer.ready_time[unload_batch_idx, foup_idx, wafer_idx]

        # 3. unload start/end time update
        self.unload_start_time[unload_batch_idx] = torch.max(
            torch.stack([self.pkup_end_time[unload_batch_idx],wafer_ready_time]), dim=0
        ).values

        self.unload_end_time[unload_batch_idx] =\
              self.unload_start_time[unload_batch_idx] + self.unload_time

        # 4. load wafer to the arm
        self.hold_wafer[unload_batch_idx, tar_arm[unload_batch_idx]] = wafer_name

        # 5. unload from the PM
        env.loc.unload(env, action)

        # 6. unload the wafer & update
        env.wafer.unload(env, action)

        # 7. additional) purge operation
        if env.purge_constraint:
            env.loc.purge(env, action)


    def load(self, env, action, delay_time):
        load_batch_idx = action.is_load == True & ~env.done
        load_loc_idx = action.load_loc[load_batch_idx]

        # 1. move to the next PM
        self.move(env, action)

        # 2. get the load ready time
        if env.purge_constraint:
            loc_ready_time = env.loc.purge_end_time[load_batch_idx, load_loc_idx]

            # 3. load start/end time
            self.load_start_time[load_batch_idx] = torch.max(
                torch.stack([self.move_end_time[load_batch_idx], loc_ready_time]), dim=0
            ).values

        else:
            loc_ready_time = torch.zeros_like(load_batch_idx, dtype=torch.float)

            # 3. load start/end time
            self.load_start_time[load_batch_idx] = torch.max(
                torch.stack([self.move_end_time[load_batch_idx], loc_ready_time[load_batch_idx]]), dim=0
            ).values

        if delay_time is not None:
            self.load_start_time[load_batch_idx] += delay_time[load_batch_idx]

        self.load_end_time[load_batch_idx] = self.load_start_time[load_batch_idx] + self.load_time

        # 4. unload from the arm
        load_arm_idx = (self.loc[load_batch_idx, :] == load_loc_idx[:, None].repeat(1, 2)).nonzero()[:, 1]
        self.hold_wafer[load_batch_idx, load_arm_idx] = -1

        # 5. load wafer & loc
        env.wafer.load(env, action)
        env.loc.load(env, action)

        # 6. process wafer & loc
        env.loc.process(env, action)
        env.wafer.process(env, action)

    def get_opposite_arm_loc(self, env, master_arm:torch.Tensor)->torch.Tensor:
        if (env.loc.num_loc) % 2:
            mid = int(((env.loc.num_loc) - 1)/2)
            slave_arm = -torch.ones_like(master_arm, dtype=torch.int64)
            slave_arm[master_arm < mid] = master_arm[master_arm < mid] + (mid + 1)
            slave_arm[master_arm > mid] = master_arm[master_arm > mid] - (mid + 1)

        else:
            slave_arm = (master_arm +
                         torch.full_like(master_arm,(env.loc.num_loc)/2)) % (env.loc.num_loc)

        return slave_arm

    def valid_load_action(self, env):
        """
        load action 중 avail한 action mask
        robot arm이 들고 있는 wafer의 다음 stage에 비어있는 PM들이 있을 때 valid load
        """

        # arm1 load action space + arm2 load action space (size: 2* (num_pm + out ll))
        action_mask = torch.zeros(size=(*env.batch_size, 2*(env.loc.num_pm+1)), dtype=torch.bool)

        # arm1
        arm1_hold_batch_idx = (self.hold_wafer[:,0] != -1).nonzero()[:,0]
        arm1_hold_wafer = self.hold_wafer[arm1_hold_batch_idx,0]

        arm1_wafer_recipe = env.wafer.get_recipe(arm1_hold_wafer)
        arm1_wafer_step = env.wafer.get_step(arm1_hold_wafer)
        arm1_wafer_next_step = arm1_wafer_step + 1
        arm1_wafer_next_stage = env.recipe_table.get('flow')\
            [arm1_hold_batch_idx, arm1_wafer_recipe, arm1_wafer_next_step]

        arm1_wafer_next_stage_loc = env.loc.stage[arm1_hold_batch_idx, :] ==\
              arm1_wafer_next_stage[:, None].repeat(1, env.loc.num_loc)
        arm1_wafer_next_stage_avail_loc = torch.logical_and(arm1_wafer_next_stage_loc,
                                                            env.loc.get_unloaded_pm()[arm1_hold_batch_idx, :])

        action_mask[arm1_hold_batch_idx, :env.loc.num_pm+1] = arm1_wafer_next_stage_avail_loc[:, 1:]

        # arm2
        arm2_hold_batch_idx = (self.hold_wafer[:,1] != -1).nonzero()[:,0]
        arm2_hold_wafer = self.hold_wafer[arm2_hold_batch_idx, 1]

        arm2_wafer_recipe = env.wafer.get_recipe(arm2_hold_wafer)
        arm2_wafer_step = env.wafer.get_step(arm2_hold_wafer)
        arm2_wafer_next_step = arm2_wafer_step + 1
        arm2_wafer_next_stage = env.recipe_table.get('flow')\
            [arm2_hold_batch_idx, arm2_wafer_recipe, arm2_wafer_next_step]

        arm2_wafer_next_stage_loc = env.loc.stage[arm2_hold_batch_idx, :] ==\
            arm2_wafer_next_stage[:, None].repeat(1, env.loc.num_loc)
        arm2_wafer_next_stage_avail_loc = torch.logical_and(arm2_wafer_next_stage_loc,
                                                            env.loc.get_unloaded_pm()[arm2_hold_batch_idx, :])

        action_mask[arm2_hold_batch_idx, env.loc.num_pm+1:] = arm2_wafer_next_stage_avail_loc[:, 1:]

        # FIFO mask 
        # 두 arm이 모두 wafer를 들고 있을 때, FIFO FOUP의 wafer를 먼저 load 해야한다. 
        arm_hold_wafer = env.wafer.get_step(self.hold_wafer)
        both_arm_hold_same_step_wafer_batch_idx = torch.logical_and((self.hold_wafer != -1).all(-1), arm_hold_wafer[:, 0] == arm_hold_wafer[:,1]) 
        arm_wafer_foup = env.wafer.get_foup(self.hold_wafer)
        
        # arm1이 fifo wafer를 hold하고 있을 때 -> arm2 load mask
        arm1_hold_fifo_wafer_batch_idx = torch.logical_and(both_arm_hold_same_step_wafer_batch_idx, 
                                                           arm_wafer_foup[:, 0] < arm_wafer_foup[:, 1])
        action_mask[arm1_hold_fifo_wafer_batch_idx, env.loc.num_pm+1:] =  False 
        
        # arm2가 fifo wafer를 hold하고 있을 때 -> arm1 load mask
        arm2_hold_fifo_wafer_batch_idx = torch.logical_and(both_arm_hold_same_step_wafer_batch_idx, 
                                                           arm_wafer_foup[:, 0] > arm_wafer_foup[:, 1]) 
        action_mask[arm2_hold_fifo_wafer_batch_idx, :env.loc.num_pm+1] =  False 
        
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
        loaded_loc = env.loc.get_loaded_loc()        # (batch, num_loc)
        stage_avail = env.loc.get_avail_stage(env) # (batch, num_stage)

        two_arm_free_batch_idx = (self.hold_wafer == -1).count_nonzero(dim=-1) == 2 #(batch,)
        one_arm_free_batch_idx = (self.hold_wafer == -1).count_nonzero(dim=-1) == 1

        arm_idx = torch.zeros(size=(*env.batch_size,), dtype=torch.int64)
        arm_idx[one_arm_free_batch_idx] = (self.hold_wafer[one_arm_free_batch_idx] != -1).nonzero()[:, 1]

        hold_wafer_name = self.hold_wafer[env.batch_idx, arm_idx] # (batch)
        hold_wafer_recipe = torch.clip(env.wafer.get_recipe(hold_wafer_name), max=env.num_lot_type-1)
        hold_wafer_step = env.wafer.get_step(hold_wafer_name)
        hold_wafer_next_step = torch.clip(hold_wafer_step + 1, max=env.num_step-1)
        # clipping
        hold_wafer_next_stage = env.recipe_table.get('flow')[env.batch_idx,hold_wafer_recipe,hold_wafer_next_step]
        hold_wafer_next_stage_avail = stage_avail[env.batch_idx, hold_wafer_next_stage]


        # unload loadlock action mask
        # --------------------------------------------------------------------------------
        # 1. check the earliest foup wafer still in loadlock
        #wip_wafer_idx = env.wafer.get_wip_wafer()
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

        ll_unload_action_mask = torch.zeros(size=(*env.batch_size, env.num_lot_type), dtype=torch.bool)
        # 5. case 1. hold wafer is zero. -> all ll wafer can be the unloaded
        case1 = two_arm_free_batch_idx[:, None].repeat(1, env.num_lot_type) & is_lot_type_avail
        ll_unload_action_mask[case1] = True

        # 5. case 2. hold wafer is one & hold wafer의 next stage가 available 할 때
        case2 = (one_arm_free_batch_idx & hold_wafer_next_stage_avail)[:, None].repeat(1, env.num_lot_type) & is_lot_type_avail
        ll_unload_action_mask[case2] = True

        # 5. case 3. hold wafer is one & hold wafer의 next stage가 unavailable 할 때
        # -> 1. unload wafer의 next stage가 available 할 때
        # -> 2. unload wafer의 현재 stage가 hold wafer의 next stage 일때 -> loadlock의 경우 case 없음
        case3 = (one_arm_free_batch_idx & ~hold_wafer_next_stage_avail)[:, None].repeat(1, env.num_lot_type) & is_lot_type_avail & is_next_stage_avail
        ll_unload_action_mask[case3] = True


        # unload PM action mask
        # --------------------------------------------------------------------------------
        # pm unload action mask
        pm_unload_action_mask = torch.zeros(size=(*env.batch_size, env.loc.num_pm), dtype=torch.bool)
        
        # stage 내에 여러 loaded wafer가 있을 경우 FIFO FOUP wafer 우선 unload 
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1): 
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid, 
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))
        FIFO_pm = FIFO_FOUP_loaded_pm[:, 1:-1]

        # case 1: 두 arm이 모두 free한 경우 -> 모든 pm에서 wafer unload 가능
        loaded_pm = loaded_loc[:, 1:-1] & FIFO_pm # (batch, num_pm)
        pm_unload_action_mask[two_arm_free_batch_idx[:, None].repeat(1, env.loc.num_pm) & loaded_pm] = True

        # case 2: arm 하나만 free하고 들고있는 wafer의 next stage가 available한 경우(배치 조건). 모든 pm에서 wafer unload 가능

        case2 = (one_arm_free_batch_idx & hold_wafer_next_stage_avail)[:, None].repeat(1, env.loc.num_pm)
        pm_unload_action_mask[case2 & loaded_pm] = True

        # case 3: arm 하나만 free하고 들고있는 wafer의 next stage가 unavailable한 경우.
        # (1) unload wafer의 next stage가 available하거나
        # (2) unload wafer의 stage가 hold wafer의 next stage인 경우 ->  pm에서 wafer unload 가능

        case3 = (one_arm_free_batch_idx & ~hold_wafer_next_stage_avail)[:, None].repeat(1, env.loc.num_pm)

        # case 3-1
        pm_wafer_name = env.loc.hold_wafer[:, 1:-1]
        pm_wafer_recipe = torch.clip(env.wafer.get_recipe(pm_wafer_name), max=env.num_lot_type-1)
        pm_wafer_step = env.wafer.get_step(pm_wafer_name)
        pm_wafer_next_step = torch.clip(pm_wafer_step + 1, max=env.num_step-1)
        pm_wafer_next_stage = env.recipe_table.get('flow')[env.batch_idx[:, None].repeat(1, env.loc.num_pm),
                                                           pm_wafer_recipe, pm_wafer_next_step]
        loaded_wafer_next_stage_avail = stage_avail[env.batch_idx[:, None].repeat(1, env.loc.num_pm),
                                                    pm_wafer_next_stage]
        pm_unload_action_mask[case3 & loaded_pm & loaded_wafer_next_stage_avail] = True

        # case 3-2
        is_hold_wafer_next_stage_pm = env.loc.stage[:, 1:-1] == hold_wafer_next_stage[:, None].repeat(1, env.loc.num_pm)
        pm_unload_action_mask[case3 & loaded_pm & is_hold_wafer_next_stage_pm] = True

        # concat input action + move action
        unload_action_mask = torch.cat([ll_unload_action_mask, pm_unload_action_mask], dim=-1)

        return unload_action_mask

    """
    def valid_pm_unload_action(self, env):
        # check the stage available(=empty)
        loaded_pm = env.loc.get_loaded_pm()        # (batch, num_loc)
        unloaded_pm = ~loaded_pm                   # (batch, num_loc)
        stage_avail = env.loc.get_avail_stage(env) # (batch, num_stage)

        # find the next stage of the loaded wafer
        loaded_wafer_name = env.loc.hold_wafer[~unloaded_pm]
        loaded_wafer_batch_idx = (~unloaded_pm).nonzero()[:, 0]        # flatten
        loaded_wafer_recipe = env.wafer.get_recipe(loaded_wafer_name)
        loaded_wafer_step = env.wafer.get_step(loaded_wafer_name)
        loaded_wafer_next_step = loaded_wafer_step + 1

        loaded_wafer_next_stage = env.recipe_table.get('flow')\
            [loaded_wafer_batch_idx, loaded_wafer_recipe, loaded_wafer_next_step]

        next_stage_avail = stage_avail[loaded_wafer_batch_idx, loaded_wafer_next_stage]
        loaded_pm_id = env.loc.id[loaded_pm]-1

        # pm unload action mask
        fifo_foup_loc = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for stage_id in range(1, env.num_stage+1):
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==stage_id, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            fifo_foup_loc += torch.logical_and(env.loc.stage == stage_id,
                                         env.wafer.get_foup(env.loc.hold_wafer)==\
                                            FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))

        fifo_foup_avail = fifo_foup_loc[:, 1:-1][loaded_wafer_batch_idx, loaded_pm_id]
        # pm unload action mask


        # check the robot available
        robot_avail = (self.hold_wafer == -1).any(dim=-1)   # (batch)
        robot_avail_idx = robot_avail[loaded_wafer_batch_idx]

        pm_action_mask = torch.zeros(size=(*env.batch_size, env.loc.num_pm), dtype=torch.bool)
        pm_action_mask[loaded_wafer_batch_idx[robot_avail_idx], loaded_pm_id[robot_avail_idx]] =\
              next_stage_avail[robot_avail_idx] & fifo_foup_avail[robot_avail_idx]

        return pm_action_mask

    def valid_lm_unload_action(self, env):

        stage_avail = env.loc.get_avail_stage(env)
        intool_wafer_idx = env.wafer.get_intool_wafer()
        lm_wafer_idx = (env.wafer.status == env.wafer.status_dict.get('inloadport')) # (batch, num_foup, foup_size)
        wafer_foup_idx = env.wafer.get_foup(env.wafer.name)
        fi_foup_idx = (torch.where(intool_wafer_idx, wafer_foup_idx, 1e9)
                         .reshape(*env.batch_size, -1)
                         .min(dim=-1)[0]
                         .to(torch.int64))

        fi_foup_idx = fi_foup_idx[:, None, None].repeat(1, env.num_foup, env.wafer.foup_size)
        fi_foup_wafer_idx = wafer_foup_idx == fi_foup_idx       # (batch, num_foup, foup_size)

        # check the earliest income foup wafer still in loadlock
        is_fi_foup_wafer_in_lm = (torch.logical_and(lm_wafer_idx, fi_foup_wafer_idx)
                            .reshape(*env.batch_size, -1)
                            .any(dim=-1))      # (batch, )

        avail_lm_wafer_idx = torch.zeros_like(env.wafer.name, dtype=torch.bool)

        # first income FOUP priority
        avail_lm_wafer_idx[is_fi_foup_wafer_in_lm] =\
            torch.logical_and(lm_wafer_idx, fi_foup_wafer_idx)[is_fi_foup_wafer_in_lm]
        # otherwise, other FOUP wafer can be selected
        avail_lm_wafer_idx[~is_fi_foup_wafer_in_lm] = lm_wafer_idx[~is_fi_foup_wafer_in_lm]
        avail_lm_wafer_idx = avail_lm_wafer_idx.nonzero()

        lm_wafer_batch_idx = avail_lm_wafer_idx[:, 0]
        lm_wafer_foup_idx = avail_lm_wafer_idx[:, 1]
        lm_wafer_idx = avail_lm_wafer_idx[:, 2]
        lm_wafer_recipe_idx =env.wafer.get_recipe(
            env.wafer.name[lm_wafer_batch_idx,lm_wafer_foup_idx,lm_wafer_idx])

        is_recipe_in_lm = torch.zeros(size=(*env.batch_size, env.num_lot_type), dtype=torch.int64)
        is_recipe_in_lm[lm_wafer_batch_idx, lm_wafer_recipe_idx] = 1
        is_recipe_avail = is_recipe_in_lm > 0

        # load stage available
        #lm_wafer_next_stage_idx = env.recipe_table.get('flow')[:, 1][None, :].repeat(*env.batch_size, 1)
        lm_wafer_next_stage_idx = env.recipe_table.get('flow')[:, :, 1] # (batch, num_lot_type)
        is_next_stage_avail = torch.gather(stage_avail, -1, lm_wafer_next_stage_idx)

        # input action mask
        lm_action_mask = torch.logical_and(is_recipe_avail, is_next_stage_avail)

        # dual arm robot mask
        # check the robot available
        robot_avail = (self.hold_wafer == -1).any(dim=-1)   # (batch)
        robot_avail = robot_avail[:, None].repeat(1, env.num_lot_type)    #(bathc, num_loc)

        lm_action_mask = torch.logical_and(lm_action_mask, robot_avail)

        return lm_action_mask
    """

