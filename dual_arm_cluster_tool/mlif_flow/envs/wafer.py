import random
import itertools
import torch
from dataclasses import dataclass

@dataclass
class Wafer:
    status_dict = {
        'queue': -1,            # Registered in tool queue
        'inloadport': 0,        # In loadport
        'process': 1,           # Processing in tool
        'finish': 2,            # Process completed in tool
        'outloadport': 3,       # Exported to loadport
        'exit': 4,              # Exported from loadport
        'robot': 5
    }
    name:torch.Tensor = None           # (batch, num_foup, foup_size)
    recipe:torch.Tensor = None       # (batch, num_foup, foup_size)

    status:torch.Tensor = None       # (batch, num_foup, foup_size)
    stage:torch.Tensor = None        # (batch, num_foup, foup_size)
    loc:torch.Tensor = None          # (batch, num_foup, foup_size)

    ready_time:torch.Tensor = None   # (batch, num_foup, foup_size)
    residency_time:torch.Tensor = None   # (batch, num_foup, foup_size, ll+stage num, 2) Time the wafer stayed at the stage

    exit_foup:torch.Tensor = None

    # id multiplier = foup_id * 100000 + recipe_id*1000 + wafer_id*10 + step
    FOUP_MLR = 100000
    RECIPE_MLR = 1000
    STEP_MLR = 10

    def __init__(self, env: object):
        self.num_foup = env.num_foup
        self.loadport_capacity = env.loadport_capacity
        self.foup_size = env.foup_size

        # consider z
        # ------------------------------
        self.loading_stage = [i-j for i,j in zip(env.stage, env.init_partial_loading)]

        if env.purge_constraint:
            if 'stage_z' not in env.kwargs:
                z_ranges = [(0, ) if x == 1 else tuple(range(x)) for x in self.loading_stage]
                self.z_list = list(itertools.product(*z_ranges))
                self.z = self.z_list[random.randint(0, len(self.z_list)-1)]
            else:
                self.z = env.kwargs['stage_z']
        else:
            self.z = [0 for _ in self.loading_stage]

        self.num_init_loading_wafer = sum(self.loading_stage) - sum(self.z) # the number of empty parallel chambers for process step i

        # no consider z
        # ------------------------------
        #self.loading_stage = [i-j for i,j in zip(env.stage, env.init_partial_loading)]
        #self.num_init_loading_wafer = sum(self.loading_stage)

        self.status = torch.full((*env.batch_size, self.num_foup, self.foup_size), self.status_dict.get('queue'), dtype=torch.int64)
        self.recipe = self.set_lot_recipe(env)

        # id = foup_id * 100000 + recipe_id*1000 + wafer_id*10 + step
        self.name = torch.zeros((*env.batch_size, self.num_foup, self.foup_size), dtype=torch.int64)
        for foup_id in range(self.num_foup):
            self.name[:, foup_id] = torch.arange(
                start=(foup_id+1)*self.FOUP_MLR,
                end=(foup_id+1)*self.FOUP_MLR + self.foup_size * self.STEP_MLR,
                step=self.STEP_MLR)
        self.name += self.recipe * self.RECIPE_MLR

        self.stage = -torch.ones((*env.batch_size, self.num_foup, self.foup_size), dtype=torch.int64)
        self.loc = -torch.ones((*env.batch_size, self.num_foup, self.foup_size), dtype=torch.int64)
        #self.step = torch.zeros((*env.batch_size, self.num_foup, self.foup_size), dtype=torch.int64)
        self.ready_time = -torch.ones((*env.batch_size, self.num_foup, self.foup_size), dtype=torch.float)
        self.residency_time = -torch.ones((*env.batch_size, self.num_foup, self.foup_size, env.loc.num_loc, 2), dtype=torch.float)

        self.exit_foup = torch.ones((*env.batch_size,), dtype=torch.int64) # 0st loadport lot should be exit


    def get_recipe(self, wafer_name:torch.Tensor):
        recipe = (wafer_name - (wafer_name // self.FOUP_MLR)* self.FOUP_MLR)\
                    // self.RECIPE_MLR
        return recipe

    def get_foup(self, wafer_name:torch.Tensor):
        return wafer_name // self.FOUP_MLR

    def get_step(self, wafer_name:torch.Tensor):
        return wafer_name % self.STEP_MLR

    def get_wip_wafer(self):
        # per recipe exist wafer in loadlock
        intool_status = torch.Tensor([
            self.status_dict.get('inloadport'),
            self.status_dict.get('process'),
            self.status_dict.get('finish'),
            self.status_dict.get('outloadport'),
            self.status_dict.get('robot')
        ])
        intool_wafer_idx = torch.isin(self.status, intool_status)

        return intool_wafer_idx

    def load(self, env: object, action: object):
        is_load = (action.is_load == True) & (~env.done)
        batch_idx = env.batch_idx[is_load]
        load_loc = action.load_loc[is_load]
        foup_idx = action.foup_idx[is_load]
        wafer_idx = action.wafer_idx[is_load]

        # stage update
        #action_wafer_recipe = (self.name[batch_idx, foup_idx, wafer_idx] -
        #                       (self.name[batch_idx, foup_idx, wafer_idx] // self.FOUP_MLR) * self.FOUP_MLR) // self.RECIPE_MLR
        wafer_recipe_idx = self.get_recipe(self.name[batch_idx, foup_idx, wafer_idx])
        wafer_next_step_idx = self.get_step(self.name[batch_idx, foup_idx, wafer_idx]) + 1

        self.stage[batch_idx, foup_idx, wafer_idx] =\
              env.recipe_table['flow'][batch_idx, wafer_recipe_idx, wafer_next_step_idx]

        # wafer id(step) update
        self.name[batch_idx, foup_idx, wafer_idx] += 1

        # loc update
        self.loc[batch_idx, foup_idx, wafer_idx] = load_loc

        # residency time update
        #self.residency_time[batch_idx, foup_idx, wafer_idx, load_loc, 0] = env.robot.load_end_time[batch_idx]

        # wafer status
        process_idx = load_loc != env.loc.num_pm + 1
        out_idx = load_loc == env.loc.num_pm + 1

        self.status[batch_idx[process_idx],
                    foup_idx[process_idx],
                    wafer_idx[process_idx]] = self.status_dict['process']

        self.status[batch_idx[out_idx],
                    foup_idx[out_idx],
                    wafer_idx[out_idx]] = self.status_dict['outloadport']

    def unload(self, env: object, action: object):
        is_unload = (action.is_load == False) & (~env.done)
        batch_idx = env.batch_idx[is_unload]
        foup_idx = action.foup_idx[is_unload]
        wafer_idx = action.wafer_idx[is_unload]
        unload_loc = action.unload_loc[is_unload]
        unload_arm = action.robot_idx[is_unload]

        # loc update
        self.loc[batch_idx, foup_idx, wafer_idx] = env.loc.num_loc # robot

        # wafer status
        self.status[batch_idx, foup_idx, wafer_idx] = self.status_dict['robot']

        # residency time update
        #self.residency_time[batch_idx, foup_idx, wafer_idx, unload_loc, 1] =\
        #    env.robot.unload_end_time[batch_idx, unload_arm]

    def process(self, env:object, action:object):
        is_load = (action.is_load == True) & (~env.done)
        batch_idx = env.batch_idx[is_load]
        load_loc = action.load_loc[is_load]
        foup_idx = action.foup_idx[is_load]
        wafer_idx = action.wafer_idx[is_load]

        # ready time update
        self.ready_time[batch_idx, foup_idx, wafer_idx] = env.loc.process_end_time[batch_idx, load_loc]

    def set_init_status(self, env:object):
        """loaded FOUP (init loadport wafers) status"""
        # loaded FOUP status, stage, loc, step, ready_time

        # loading wafers
        self.status[:, 0, :self.num_init_loading_wafer] = self.status_dict['process']

         # consider z
        self.stage[:, 0, :self.num_init_loading_wafer] = torch.arange(1, env.loc.num_stage+1)\
            .repeat_interleave(torch.tensor(self.loading_stage) - torch.tensor(self.z))

        # no consider z
        #self.stage[:, 0, :self.num_init_loading_wafer] = (torch.arange(1, env.loc.num_stage+1).repeat_interleave(torch.tensor(self.loading_stage)))

        # other wafers
        self.status[:, 0, self.num_init_loading_wafer:] = self.status_dict['finish']
        self.stage[:, 0, self.num_init_loading_wafer:] = env.loc.num_stage + 1

        # loaded FOUP wafers are serial flow, so stage == step
        self.name[:, 0, :] += self.stage[:, 0, :]

        # loc
        pm_assigned = torch.zeros((*env.batch_size, env.loc.num_loc), dtype=torch.bool)
        for w in range(self.num_init_loading_wafer):
            wafer_stage = self.stage[:, 0, w][:, None].repeat(1, env.loc.num_loc)
            wafer_stage_pm = (env.loc.stage == wafer_stage)
            not_assn_pm = ~pm_assigned
            pm = (wafer_stage_pm & not_assn_pm).nonzero()[0,1]
            self.loc[:, 0, w] = pm
            pm_assigned[:, pm] = True
        self.loc[:, 0, self.num_init_loading_wafer:] = env.loc.num_pm + 1

        # remain process time = (= wafer ready time)
        # NOTE init process time
        init_wafer_ready_time_is_diverse = False
        if init_wafer_ready_time_is_diverse:
            self.ready_time[:, 0, :self.num_init_loading_wafer] = torch.randint(
                low = env.kwargs['min_process_time'],
                high = env.kwargs['max_process_time'],
                size=(*env.batch_size, self.num_init_loading_wafer),
                dtype=torch.int64,
            )

        else:
            self.ready_time[:, 0, :self.num_init_loading_wafer] = torch.randint(
                low = env.kwargs['min_process_time'],
                high = env.kwargs['max_process_time'],
                size=(self.num_init_loading_wafer,),
                dtype=torch.int64,
            )[None, :].repeat(*env.batch_size, 1)

        # Min-max normalization
        if env.kwargs.get('norm_time', True):
            min_time_unit = env.kwargs.get('min_time_unit', 2)
            max_time_unit = env.kwargs.get('max_process_time', 100)

            self.ready_time[:, 0, :self.num_init_loading_wafer] =\
                (self.ready_time[:, 0, :self.num_init_loading_wafer] - min_time_unit) / (max_time_unit - min_time_unit)

        """first FOUP status -> enters the loadport from the queue"""
        self.status[:, 1, :] = self.status_dict['inloadport']
        self.stage[:, 1, :] = 0 # loadport
        self.loc[:, 1, :] = 0
        self.ready_time[:, 1, :] = 0


    def set_lot_recipe(self, env):
        # wafer lot type setting
        lot_dist = env.kwargs.get('lot_dist', 'multi_lot_imbalanced')
        consider_lot_type = env.kwargs.get('consider_lot_type', 5)

        if lot_dist == 'single_lot':
            # single lot FOUP 1 (type 2) -> FOUP 2 (type 1) -> ...
            lot_per_foup = []
            for i in range(env.num_foup):
                lot_type = torch.randint(consider_lot_type, (*env.batch_size,))[:, None].repeat(1, env.foup_size)
                lot_per_foup.append(lot_type)
            lot_per_foup = torch.stack(lot_per_foup, dim=1)

        elif lot_dist == 'uniform_lot':
            # single lot FOUP 1 (type 1) -> FOUP 2 (type 1) -> ...
            lot_per_foup = torch.zeros(*env.batch_size, env.num_foup, env.foup_size, dtype=torch.int64)

        elif lot_dist == 'multi_lot_balanced':
            # balanced foup processing
            quotient, remainder = divmod(env.foup_size, consider_lot_type)
            lot_sizes = [quotient + (1 if i < remainder else 0) for i in range(consider_lot_type)]
            lots = []
            for i,j in enumerate(lot_sizes):
                lots.append(
                    torch.tensor([i])[None, None, :].repeat(*env.batch_size, env.num_foup, j)
                )
            lot_per_foup = torch.cat(lots, dim=-1)

        elif lot_dist == 'multi_lot_imbalanced':
            #min_wafer_per_lot = torch.arange(consider_lot_type)[None, None, :].repeat(*env.batch_size, env.num_foup, 1)
            #add_wafer_per_lot = torch.randint(consider_lot_type, (*env.batch_size, env.num_foup, env.foup_size-consider_lot_type))
            #lot_per_foup = torch.concat([min_wafer_per_lot, add_wafer_per_lot], dim=-1)
            lot_per_foup = torch.randint(consider_lot_type, (*env.batch_size, env.num_foup, env.foup_size))

        elif lot_dist == 'multi_lot_mixed':
            # balanced foup & imbalanced foup processing
            # balanced foup 1 -> imbalanced foup 1 -> balanced foup 2 -> imbalanced foup 2 -> ...
            foup = []
            for i in range(env.num_foup):
                if i % 2 == 0: # balanced lot
                    quotient, remainder = divmod(env.foup_size, consider_lot_type)
                    lot_sizes = [quotient + (1 if i < remainder else 0) for i in range(consider_lot_type)]

                    lots = []
                    for i,j in enumerate(lot_sizes):
                        lots.append(
                            torch.tensor([i])[None, :].repeat(*env.batch_size, j)
                        )
                    lots_in_foup = torch.cat(lots, dim=-1)
                    foup.append(lots_in_foup)

                else: # imbalance lot
                    foup.append(torch.randint(consider_lot_type, (*env.batch_size, env.foup_size)))

            lot_per_foup = torch.stack(foup, dim=1)

        elif lot_dist == 'single_lot_multi_lot_balanced':
            foup = []
            for i in range(env.num_foup):
                if i % 2 == 0: # balanced lot
                    quotient, remainder = divmod(env.foup_size, consider_lot_type)
                    lot_sizes = [quotient + (1 if i < remainder else 0) for i in range(consider_lot_type)]

                    lots = []
                    for i,j in enumerate(lot_sizes):
                        lots.append(
                            torch.tensor([i])[None, :].repeat(*env.batch_size, j)
                        )
                    lots_in_foup = torch.cat(lots, dim=-1)
                    foup.append(lots_in_foup)

                else: # single lot
                    lot_type = torch.randint(consider_lot_type, (*env.batch_size,))[:, None].repeat(1, env.foup_size)
                    foup.append(lot_type)

            lot_per_foup = torch.stack(foup, dim=1)

        elif lot_dist == 'single_lot_multi_lot_imbalanced':
            foup = []
            for i in range(env.num_foup):
                if i % 2 == 0: # imbalanced lot
                    foup.append(torch.randint(consider_lot_type, (*env.batch_size, env.foup_size)))
                else: # single lot
                    foup.append(torch.randint(consider_lot_type, (*env.batch_size,))[:, None].repeat(1, env.foup_size))

            lot_per_foup = torch.stack(foup, dim=1)



        return lot_per_foup


    def update(self, env):
        # finished wafer update
        clock_per_wafer = env.clock[:, None, None].repeat(1, self.num_foup, self.foup_size)
        processing = self.status == self.status_dict.get("process")
        ready = clock_per_wafer >= self.ready_time
        processed = torch.logical_and(processing, ready)
        self.status[processed] = self.status_dict.get("finish")

        # exit from LL wafer update
        out_lm_wafers = self.loc == env.loc.num_pm + 1
        wafer_foup = self.get_foup(self.name)
        exit_foup = self.exit_foup[:, None, None].repeat(1, self.num_foup, self.foup_size)
        fi_foup_wafers = wafer_foup == exit_foup

        fi_foup_wafers_in_lm = torch.logical_and(out_lm_wafers, fi_foup_wafers).reshape(*env.batch_size, -1)
        exit_batch_idx = fi_foup_wafers_in_lm.count_nonzero(dim=-1) == self.foup_size   # FOUP의 모든 wafer가 LL에 있을 때
        is_queue_empty = (self.exit_foup == self.num_foup-1)
        exit_batch_idx = exit_batch_idx & ~is_queue_empty

        # tool exit foup status update 
        self.status[exit_batch_idx, self.exit_foup[exit_batch_idx]-1, :] = self.status_dict.get("exit")
        self.stage[exit_batch_idx, self.exit_foup[exit_batch_idx]-1, :] = -1
        self.loc[exit_batch_idx, self.exit_foup[exit_batch_idx]-1, :] = -1

        # tool next enter foup status update 
        self.status[exit_batch_idx, self.exit_foup[exit_batch_idx]+1, :] = self.status_dict.get("inloadport")
        self.stage[exit_batch_idx, self.exit_foup[exit_batch_idx]+1, :] = 0
        self.loc[exit_batch_idx, self.exit_foup[exit_batch_idx]+1, :] = 0
        self.ready_time[exit_batch_idx, self.exit_foup[exit_batch_idx]+1, :] = 0


        # if all the exit foup wafers are entered to the tool, next FOUP wafers can be inserted.
        exit_foup = torch.clamp(self.exit_foup, max=env.num_foup-1)
        self.exit_foup = exit_foup

        # not in in-loadloc or robot arm
        #fin_input_exit_foup = torch.logical_and(self.loc[env.batch_idx, exit_foup]  != 0, self.loc[env.batch_idx, exit_foup]  !=  env.loc.num_loc).all(-1)
        fin_input_exit_foup = (self.status[env.batch_idx, exit_foup] == self.status_dict['outloadport']).all(dim=-1)
        self.exit_foup[fin_input_exit_foup] += 1



