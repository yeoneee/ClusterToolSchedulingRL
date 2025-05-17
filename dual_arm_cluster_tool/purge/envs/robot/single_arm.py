import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SingleArmedRobot:
    # static
    num_arm = 1
    move_time = 3
    load_time = 3
    unload_time = 3
    # dynamic
    loc:torch.Tensor = None                  # (batch, )
    hold_wafer:torch.Tensor = None           # (batch, )
    pkup_start_time:torch.Tensor = None      # (batch, )
    pkup_end_time:torch.Tensor = None        # (batch, )
    unload_start_time:torch.Tensor = None    # (batch, )
    unload_end_time:torch.Tensor = None      # (batch, )
    move_start_time:torch.Tensor = None      # (batch, )
    move_end_time:torch.Tensor = None        # (batch, )
    load_start_time:torch.Tensor = None      # (batch, )
    load_end_time:torch.Tensor = None        # (batch, )


    def __init__(self, env):

        self.loc = torch.zeros(
            size=(*env.batch_size,),
            dtype=torch.int64,
        )
        self.hold_wafer = -torch.ones(
            size=(*env.batch_size,),
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
            max_time_unit = env.kwargs.get('max_process_time', 300)

            self.move_time = (self.move_time - min_time_unit) / (max_time_unit - min_time_unit)
            self.load_time = (self.load_time - min_time_unit) / (max_time_unit - min_time_unit)
            self.unload_time = (self.unload_time - min_time_unit) / (max_time_unit - min_time_unit)

    def pkup(self, env: object, action: object):
        """
        unload action을 수행하는 batch 중 arm이 움직여야하는 경우 수행
        """
        # get the target wafer idx
        not_done = ~env.done
        loc_idx = action.unload_loc
        need_pkup = (loc_idx != self.loc) & not_done

        # arm loc update
        self.loc[need_pkup] = loc_idx[need_pkup]

        # move time update
        self.pkup_start_time[not_done] = env.clock[not_done]

        self.pkup_end_time[~need_pkup] = self.pkup_start_time[~need_pkup]
        self.pkup_end_time[need_pkup] = self.pkup_start_time[need_pkup] + self.move_time

    def move(self, env: object, action:object):
        not_done = ~env.done

        # arm loc update
        self.loc[not_done] = action.load_loc[not_done]

        # move time update
        self.move_start_time[not_done] = self.unload_end_time[not_done]
        self.move_end_time[not_done] = self.move_start_time[not_done] + self.move_time

    def unload(self, env: object, action: object):
        self.pkup(env, action)

        not_done = ~env.done
        foup_idx = action.foup_idx
        wafer_idx = action.wafer_idx

        # get the unload ready time(=process end time)
        wafer_ready_time = env.wafer.ready_time[env.batch_idx, foup_idx, wafer_idx]
        wafer_name = env.wafer.name[env.batch_idx, foup_idx, wafer_idx]

        # unload start/end time update
        self.unload_start_time[not_done] = torch.max(
            torch.stack([self.pkup_end_time, wafer_ready_time]),
            dim=0).values[not_done]

        self.unload_end_time[not_done] =\
              self.unload_start_time[not_done] + self.unload_time

        # load wafer to the arm
        self.hold_wafer[not_done] = wafer_name[not_done]

        # unload from the chamber, wafer status update
        env.loc.unload(env, action)

        if env.purge_constraint:
            env.loc.purge(env, action)

        env.wafer.unload(env, action)

    def load(self, env: object, action: object, delay_time: Optional[torch.Tensor] = None):
        self.move(env, action)

        not_done = ~env.done
        loc_idx = action.load_loc

        # get the load ready time
        if env.purge_constraint:
            loc_ready_time = env.loc.purge_end_time[env.batch_idx, loc_idx]
        else:
            loc_ready_time = torch.zeros_like(action.idx, dtype=torch.float)

        self.load_start_time[not_done] = torch.max(
            torch.stack([self.move_end_time, loc_ready_time])
            ,dim=0).values[not_done]

        if delay_time is not None:
            self.load_start_time[not_done] += delay_time[not_done]

        self.load_end_time[not_done] = self.load_start_time[not_done] + self.load_time

        # unload from the arm
        self.hold_wafer[not_done] = -torch.ones_like(self.hold_wafer, dtype=torch.int64)[not_done]

        env.wafer.load(env, action)
        env.loc.load(env, action)

        env.loc.process(env, action)
        env.wafer.process(env, action)

    def valid_pm_unload_action(self, env: object):
        # check the stage available(=empty)
        loaded_loc = env.loc.get_loaded_loc()
        stage_avail = env.loc.get_avail_stage(env)

        # find the next stage of the loaded wafer
        is_idle_pm = env.loc.hold_wafer == -1
        loaded_wafer_name = env.loc.hold_wafer[~is_idle_pm]
        loaded_wafer_batch_idx = (~is_idle_pm).nonzero()[:, 0]
        loaded_wafer_recipe = env.wafer.get_recipe(loaded_wafer_name)
        loaded_wafer_step = env.wafer.get_step(loaded_wafer_name)
        loaded_wafer_next_step = loaded_wafer_step + 1

        loaded_wafer_next_stage = env.recipe_table.get('flow')\
            [loaded_wafer_batch_idx, loaded_wafer_recipe, loaded_wafer_next_step]

        next_stage_avail = stage_avail[loaded_wafer_batch_idx, loaded_wafer_next_stage]
        loaded_pm_id = env.loc.id[loaded_loc]-1
        
        
        # stage 내에 여러 loaded wafer가 있을 경우 FIFO FOUP wafer 우선 unload 
        FIFO_FOUP_loaded_pm = torch.zeros(size=(*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for sid in range(1, env.num_stage+1): 
            loc_stage_holding_wafer = torch.logical_and(env.loc.stage==sid, env.loc.hold_wafer != -1)
            FIFO_FOUP, _ = torch.where(loc_stage_holding_wafer, env.wafer.get_foup(env.loc.hold_wafer), 1e10).min(dim=-1)
            FIFO_FOUP_loaded_pm += torch.logical_and(env.loc.stage==sid, 
                                                     env.wafer.get_foup(env.loc.hold_wafer)==\
                                                         FIFO_FOUP[:, None].repeat(1, env.loc.stage.size(-1)))
        FIFO_pm_avail = FIFO_FOUP_loaded_pm[:, 1:-1][loaded_wafer_batch_idx, loaded_pm_id]
        
        # pm unload action mask
        pm_action_mask = torch.zeros(size=(*env.batch_size, env.loc.num_pm), dtype=torch.bool)
        pm_action_mask[loaded_wafer_batch_idx, loaded_pm_id] = next_stage_avail & FIFO_pm_avail

        return pm_action_mask

    def valid_lm_unload_action(self, env: object):

        stage_avail = env.loc.get_avail_stage(env)
        lm_wafer_idx = (env.wafer.status == env.wafer.status_dict.get('inloadport')) # (batch, num_foup, foup_size)
        wafer_foup_idx = env.wafer.get_foup(env.wafer.name)
        earlier_foup_idx = (torch.where(lm_wafer_idx, wafer_foup_idx, 1e9)
                         .reshape(*env.batch_size, -1)
                         .min(dim=-1)[0]
                         .to(torch.int64))

        earlier_foup_idx = earlier_foup_idx[:, None, None].repeat(1, env.num_foup, env.wafer.foup_size)
        fi_foup_wafer_idx = wafer_foup_idx == earlier_foup_idx       # (batch, num_foup, foup_size)

        # check the earliest income foup wafer still in loadlock
        is_earlier_foup_wafer_in_lm = (torch.logical_and(lm_wafer_idx, fi_foup_wafer_idx)
                                        .reshape(*env.batch_size, -1)
                                        .any(dim=-1))      # (batch, )

        avail_lm_wafer_idx = torch.zeros_like(env.wafer.name, dtype=torch.bool)

        # first income FOUP priority
        avail_lm_wafer_idx[is_earlier_foup_wafer_in_lm] =\
            torch.logical_and(lm_wafer_idx, fi_foup_wafer_idx)[is_earlier_foup_wafer_in_lm]
        # otherwise, other FOUP wafer can be selected
        avail_lm_wafer_idx[~is_earlier_foup_wafer_in_lm] = lm_wafer_idx[~is_earlier_foup_wafer_in_lm]
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

        return lm_action_mask

    def valid_unload_action(self, env: object):
        """
        not finished wafer action(current loc is not outputloadlock)
        next stage loc action
        next stage loc is empty

        new wafer input action (N: number of recipe) + unload PM action
        """
        lm_action_mask = self.valid_lm_unload_action(env)
        pm_action_mask = self.valid_pm_unload_action(env)

        # concat input action + move action
        action_mask = torch.cat([lm_action_mask, pm_action_mask], dim=-1)

        return action_mask

    """
    def not_reenter_deadlock_action(self, env, td):
        action_mask = torch.ones(
            (*td.batch_size, env.num_action),
            dtype=torch.bool,
            device=td.device
        )

        reenter_stage = env.recipe.reenter_stage
        wafer_stage = td['wafer_stage']
        wafer_next_stage = td['recipe_stage_flow']\
            [0, td['wafer_recipe'], td['wafer_step']+1] # [b, r]

        # 1. For each batch, check whether there is a wafer whose next step is the reenter stage.
        reenter_batch_idx = (wafer_stage > wafer_next_stage).any(dim=-1)
        action_reenter_batch_idx = reenter_batch_idx[:, None].repeat(1, env.num_action)

        # 2. check wheter the reenter stage is lack(available PM is not larger than 1)
        reenter_stage_loc = td['loc_stage'] == reenter_stage
        is_empty_loc = ((td['loc_status'] == env.loc.status_dict.get('unload')) |
                        (td['loc_status'] == env.loc.status_dict.get('purge')))
        lack_reenter_stage = (reenter_stage_loc & is_empty_loc).count_nonzero(dim=-1) <= 1
        action_lack_reenter_stage = lack_reenter_stage[:, None].repeat(1, env.num_action)

        # 3. check the forward direction wafers whose next stage is the reenter stage
        action_forward_wafer = (wafer_stage < wafer_next_stage).repeat_interleave(env.loc.num_loc, dim=-1)
        action_next_stage_is_reenter_stage_wafer = (wafer_next_stage == reenter_stage).repeat_interleave(env.loc.num_loc, dim=-1)
        action_forward_reenter_stage_wafer = action_forward_wafer & action_next_stage_is_reenter_stage_wafer

        # 4. check the next stage of the reenter stage is slack stage(available PM is larger than 1)
        reenter_next_stage = reenter_stage+1
        no_slack_reenter_next_stage = ((td['loc_stage'] == reenter_next_stage) & is_empty_loc).count_nonzero(dim=-1) == 0
        action_no_slack_reenter_next_stage = no_slack_reenter_next_stage[:, None].repeat(1, env.num_action)

        action_forward_lack_reenter_stage_wafer = (action_reenter_batch_idx &
                                                   action_lack_reenter_stage &
                                                   action_forward_reenter_stage_wafer &
                                                   action_no_slack_reenter_next_stage)

        action_mask[action_forward_lack_reenter_stage_wafer] = False

        # 5. mask the other wafer action that makes the next stage of the reenter stage is not slack stage
        action_skip_flow_wafer_reenter_next_stage = ((wafer_next_stage == reenter_next_stage) &
                                                     (wafer_stage +1 != wafer_next_stage)
                                                     ).repeat_interleave(env.loc.num_loc, dim=-1)

        action_next_stage_is_reenter_next_stage_wafer = (action_skip_flow_wafer_reenter_next_stage &
                                                         action_reenter_batch_idx &
                                                         action_lack_reenter_stage
                                                         )

        action_mask[action_next_stage_is_reenter_next_stage_wafer] = False


        return action_mask
    """