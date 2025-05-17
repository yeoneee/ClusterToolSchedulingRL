import torch
from dataclasses import dataclass
import copy 

@dataclass
class Module:
    status_dict = {
        'unload':0,      # empty & no operation
        'purge': 1,      # empty & purge operation
        'load': 2,       # occupied & no operation
        'process': 3,    # occupied & process operation
        'finish': 4,     # occupied & no operation & process finished
    }

    # static
    id:torch.Tensor = None
    stage:torch.Tensor = None            # (batch. num_pm + num_lm)

    # dynamic
    status:torch.Tensor = None
    hold_wafer:torch.Tensor = None
    process_start_time:torch.Tensor = None
    process_end_time:torch.Tensor = None

    # additional constraints
    purge_time: torch.Tensor = None
    purge_start_time:torch.Tensor = None
    purge_end_time:torch.Tensor = None

    down_machine_idx: torch.Tensor = None


    def __init__(self, env: object):
        self.purge_constraint = env.purge_constraint
        self.delay_constraint = env.delay_constraint
        self.arm_type = env.arm_type

        self.num_lm = 2                             # loadlock(ll) module count
        self.num_pm = sum(env.stage)                # process module count
        self.num_loc = self.num_pm + self.num_lm    # module count
        self.num_stage = len(env.stage)             # stage count
        self.num_all_stage = self.num_stage + self.num_lm # stage count including loadlock

        # include loadlock stage
        full_stage = torch.tensor([1] + env.stage + [1])
        self.stage = (
            torch.arange(len(full_stage))
            .repeat_interleave(full_stage)
            .repeat(*env.batch_size, 1)
        ) # (batch_size, num lm + pm)

        self.id = torch.arange(self.num_loc)\
            [None, :].repeat(*env.batch_size, 1)

        self.status = torch.zeros(
            size=(*env.batch_size, self.num_loc),
            dtype=torch.int64,
        )

        self.hold_wafer = -torch.ones(
            size=(*env.batch_size, self.num_loc),
            dtype=torch.int64,
        )

        self.process_start_time = -torch.ones(
            size=(*env.batch_size, self.num_loc),
            dtype=torch.float,
        )

        self.process_end_time = -torch.ones(
            size=(*env.batch_size, self.num_loc),
            dtype=torch.float,
        )

        if env.purge_constraint:
            self.purge_start_time = -torch.ones(
                size=(*env.batch_size, self.num_loc),
                dtype=torch.float,
            )
            self.purge_end_time = -torch.ones(
                size=(*env.batch_size, self.num_loc),
                dtype=torch.float,
            )

    def set_init_status(self, env):
        # loc status, hold wafer, process start time, end time, purge start time, end time
        for w in range(env.wafer.num_init_loading_wafer):
            loc = env.wafer.loc[env.batch_idx, 0, w]
            self.hold_wafer[env.batch_idx, loc] = env.wafer.name[env.batch_idx, 0, w]
            self.status[env.batch_idx, loc] = self.status_dict['process']
            self.process_start_time[env.batch_idx, loc] = 0
            self.process_end_time[env.batch_idx, loc] = env.wafer.ready_time[env.batch_idx, 0, w]

        # Purge
        if self.purge_constraint:
            # generate stage purge time
            self.purge_time = self.gen_stage_purge_time(env)
            # (batch, num_stage + num_ll)

            # assn init remain purge time to each empty loc
            purge_batch_idx, purge_loc_idx = (self.hold_wafer == -1).nonzero(as_tuple=True)
            purge_stage_idx = self.stage[purge_batch_idx, purge_loc_idx]

            scaling_factor = torch.rand(len(purge_stage_idx))
            self.status[purge_batch_idx, purge_loc_idx] = self.status_dict['purge']
            self.purge_start_time[purge_batch_idx, purge_loc_idx] = 0
            self.purge_end_time[purge_batch_idx, purge_loc_idx] =\
                self.purge_time[purge_batch_idx, purge_stage_idx] * scaling_factor

            # loadlock is no purge operation
            self.status[:, [0, -1]] = self.status_dict['unload']
            self.purge_start_time[:, [0, -1]] = -1
            self.purge_end_time[:, [0, -1]] = -1



    def unload(self, env: object, action: object):
        is_unload = (action.is_load == False) & (~env.done)
        batch_idx = env.batch_idx[is_unload]
        foup_idx = action.foup_idx[is_unload]
        wafer_idx = action.wafer_idx[is_unload]
        loc_idx = action.unload_loc[is_unload]

        # status, hold wafer update
        self.status[batch_idx, loc_idx] = self.status_dict.get("unload")
        self.hold_wafer[batch_idx, loc_idx] = -1


    def load(self, env: object, action: object):
        is_load = (action.is_load == True) & (~env.done)
        batch_idx = env.batch_idx[is_load]
        loc_idx = action.load_loc[is_load]
        foup_idx = action.foup_idx[is_load]
        wafer_idx = action.wafer_idx[is_load]
        wafer_name = env.wafer.name[batch_idx, foup_idx, wafer_idx]

        # status, hold wafer update
        self.status[batch_idx, loc_idx] = self.status_dict.get("load")
        self.hold_wafer[batch_idx, loc_idx] = wafer_name


    def process(self, env:object, action:object):
        # Check if the action is load
        is_load = (action.is_load == True) & (~env.done)
        batch_idx = env.batch_idx[is_load]
        loc_idx = action.load_loc[is_load]
        foup_idx = action.foup_idx[is_load]
        wafer_idx = action.wafer_idx[is_load]
        load_arm_idx = action.robot_idx[is_load]

        # status update
        self.status[batch_idx, loc_idx] = self.status_dict.get("process")

        # process start/end time
        wafer_recipe = env.wafer.get_recipe(env.wafer.name[is_load, foup_idx, wafer_idx])
        wafer_stage = env.wafer.stage[is_load, foup_idx, wafer_idx]
        process_time = env.recipe_table['process_time'][batch_idx, wafer_recipe, wafer_stage]

        if self.arm_type == 'single':
            self.process_start_time[batch_idx, loc_idx] =\
                env.robot.load_end_time[batch_idx]

        elif self.arm_type == 'dual':
            self.process_start_time[batch_idx, loc_idx] =\
                env.robot.load_end_time[batch_idx]

        elif self.arm_type == 'ind':
            self.process_start_time[batch_idx, loc_idx] =\
                env.robot.load_end_time[batch_idx, load_arm_idx]

        self.process_end_time[batch_idx, loc_idx] =\
            self.process_start_time[batch_idx, loc_idx] + process_time

    def purge(self, env:object, action:object):
        is_unload = (action.is_load == False) & (~env.done)
        batch_idx = env.batch_idx[is_unload]
        foup_idx = action.foup_idx[is_unload]
        wafer_idx = action.wafer_idx[is_unload]
        unload_arm_idx = action.robot_idx[is_unload]


        if self.arm_type == 'single':
            unload_loc = env.wafer.loc[batch_idx, foup_idx, wafer_idx]
        else:
            unload_loc = action.unload_loc[is_unload]

        # status update
        self.status[batch_idx, unload_loc] = self.status_dict.get("purge")

        # purge start/end time
        loc_stage = self.stage[batch_idx, unload_loc]
        purge_time = self.purge_time[batch_idx, loc_stage]

        if self.arm_type == 'single':
            self.purge_start_time[batch_idx, unload_loc] =\
                env.robot.unload_end_time[batch_idx]

        elif self.arm_type == 'dual':
            self.purge_start_time[batch_idx, unload_loc] =\
                env.robot.unload_end_time[batch_idx]

        elif self.arm_type == 'ind':
            self.purge_start_time[batch_idx, unload_loc] =\
                env.robot.unload_end_time[batch_idx, unload_arm_idx]


        # breakdown case. larger purge(=maintenance) time
        consider_breakdown = env.kwargs.get('breakdown', False)
        env.breakdown = consider_breakdown
        if consider_breakdown:
            break_down_prob = torch.rand(batch_idx.size(), )
            maintenance_threshold = 0.03

            # maintenance time is constant...
            min_process_time = env.kwargs.get('min_process_time', 10)
            max_process_time = env.kwargs.get('max_process_time', 100)
            min_time_unit = env.kwargs.get('min_time_unit', 2)
            max_time_unit = max_process_time

            maintenance_time = 600 # larger purge time
            if env.kwargs.get('norm_time', True):
                maintenance_time =\
                      (maintenance_time - min_time_unit) / (max_time_unit - min_time_unit)
            purge_time[break_down_prob <= maintenance_threshold] = maintenance_time

        # loadlock postprocessing
        unload_ll_batch_idx = unload_loc == 0
        purge_time[unload_ll_batch_idx] = 0.

        self.purge_end_time[batch_idx, unload_loc] =\
            self.purge_start_time[batch_idx, unload_loc] + purge_time



    def update(self, env: object):
        #==========================Process module update===================================
        not_done_expand_loc = ~env.done[:, None].repeat(1, self.num_loc)
        clock_expand_loc = env.clock[:, None].repeat(1, self.num_loc)
        # (batch, num_loc)

        # update process -> finish
        processing = self.status == self.status_dict.get("process")
        finished = clock_expand_loc >= self.process_end_time
        self.status[not_done_expand_loc & processing & finished] =\
            self.status_dict.get("finish")

        # update purge -> finish
        if self.purge_constraint:
            purging = self.status == self.status_dict.get("purge")
            finished = clock_expand_loc >= self.purge_end_time
            self.status[not_done_expand_loc & purging & finished] =\
                self.status_dict.get("unload")

        #=============================Loadlock module update==================================
        # update loadlock. load is always idle, and no process
        self.status[:, [0,-1]] = self.status_dict.get("unload")

        # update loadlock other info
        self.hold_wafer[:, [0,-1]] = -1
        self.process_start_time[:, [0,-1]] = -1
        self.process_end_time[:, [0,-1]] = -1

        if self.purge_constraint:
            self.purge_start_time[:, [0,-1]] = -1
            self.purge_end_time[:, [0,-1]] = -1


    def get_loaded_loc(self):
        # loaded PM
        loaded_status = torch.Tensor([
            self.status_dict.get('load'),
            self.status_dict.get('process'),
            self.status_dict.get('finish')
        ])
        loaded_loc_id = torch.isin(self.status, loaded_status)

        return loaded_loc_id

    def get_unloaded_pm(self):
        # unloaded PM
        unloaded_status = torch.Tensor([
            self.status_dict.get('unload'),
            self.status_dict.get('purge')
        ])
        unloaded_loc_id = torch.isin(self.status, unloaded_status)

        return unloaded_loc_id

    def get_avail_stage(self, env):
        loaded_loc_id = self.get_loaded_loc()
        stage_avail = torch.ones(size=(*env.batch_size, self.num_all_stage), dtype=torch.bool)

        for stage in range(1, self.num_stage+1):
            next_stage_avail = torch.logical_and(self.stage==stage, ~loaded_loc_id).any(dim=-1)
            stage_avail[:, stage] = next_stage_avail

        return stage_avail


    def gen_stage_purge_time(self, env):
        min_purge_time = env.kwargs.get('min_purge_time', 10)
        max_purge_time = env.kwargs.get('max_purge_time', 100)
        min_time_unit = env.kwargs.get('min_time_unit', 2)
        max_time_unit = env.kwargs.get('max_process_time', 100)
        purge_type = env.kwargs.get('purge_type', 'long')

        if purge_type == 'long':
            if str(env.stage) == '[1, 2, 1]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 200, (*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)

            elif str(env.stage) == '[1, 3, 2]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 200, (*env.batch_size, 1)),
                    torch.randint(0, 150, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)

            elif str(env.stage) == '[1, 2, 2, 1]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 200, (*env.batch_size, 1)),
                    torch.randint(0, 200, (*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)

        elif purge_type == 'short':
            if str(env.stage) == '[1, 2, 1]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 50, (*env.batch_size, 1)),
                    torch.randint(0, 190, (*env.batch_size, 1)),
                    torch.randint(0, 50, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)

            elif str(env.stage) == '[1, 3, 2]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 50, (*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 70, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)

            elif str(env.stage) == '[1, 2, 2, 1]':
                purge_time = torch.concat([
                    torch.zeros(size=(*env.batch_size, 1)),
                    torch.randint(0, 50, (*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 100, (*env.batch_size, 1)),
                    torch.randint(0, 50, (*env.batch_size, 1)),
                    torch.zeros(size=(*env.batch_size, 1)),
                ], dim=-1)
        else:
            purge_time = torch.randint(min_purge_time, max_purge_time, (*env.batch_size, env.num_stage+2))

        # Min-max normalization
        self.cleaning_time = copy.deepcopy(purge_time)
        if env.kwargs.get('norm_time', True):
            purge_time = (purge_time - min_time_unit) / (max_time_unit - min_time_unit)

        # loadlock purge time is 0.
        purge_time[:, [0, -1]] = 0

        return purge_time


    def gen_stage_delay_limit_time(self, env):
        min_delay_time = env.kwargs.get('min_delay_time', 10)
        max_delay_time = env.kwargs.get('max_delay_time', 100)

        min_time_unit = env.kwargs.get('min_time_unit', 2)
        max_time_unit = env.kwargs.get('max_process_time', 100)

        delay_time = torch.randint(min_delay_time, max_delay_time, (*env.batch_size, env.num_stage+2))

        # Min-max normalization
        if env.kwargs.get('norm_time', True):
            delay_time = (delay_time - min_time_unit) / (max_time_unit - min_time_unit)

        # loadlock delay time is 0.
        delay_time[:, [0, -1]] = 0

        return delay_time