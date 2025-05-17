import torch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tensordict import TensorDict

color_dict = list(mcolors.CSS4_COLORS.keys())
color_dict = [
    'dodgerblue','blueviolet','sienna','darkgreen','crimson','darkorange','mediumspringgreen','lightblue',
    'chartreuse','rebeccapurple','aliceblue','aquamarine','darkmagenta','darkkhaki','palegoldenrod','skyblue','mintcream',
    'darkturquoise','midnightblue','steelblue','firebrick','darkblue','lightsteelblue','peachpuff','lightseagreen','burlywood','chocolate',
    'ghostwhite','papayawhip','cadetblue','royalblue','blanchedalmond','beige','whitesmoke','indigo','darkred','thistle','cornflowerblue',
    'yellow','antiquewhite','peru','cornsilk','lightgoldenrodyellow','lightpink','palevioletred','deepskyblue','darkslateblue','darkseagreen',
    'lime','lightcoral','silver','gold','moccasin','oldlace','mediumblue','mediumslateblue','azure','paleturquoise','cyan','maroon','olive',
    'lavenderblush','navajowhite','olivedrab','mediumorchid','fuchsia','lavender','ivory','magenta','gainsboro','mediumturquoise','linen','snow',
    'palegreen','orangered','lemonchiffon','red','seashell','turquoise','brown','slateblue','darkgoldenrod','lightyellow',
    'hotpink','khaki','salmon','lightgreen','lightsalmon','aqua','lawngreen','bisque','green','indianred','limegreen','pink','purple','darkviolet',
    'darkcyan','white','plum','teal','goldenrod','powderblue','tomato','springgreen','mediumpurple','forestgreen','blue','floralwhite','rosybrown',
    'darkolivegreen','sandybrown','coral','honeydew','mediumseagreen','violet','tan','lightskyblue','navy','yellowgreen',
    'darkorchid','orchid','orange','mediumvioletred','mistyrose','mediumaquamarine','deeppink','darksalmon','saddlebrown','greenyellow',]

color_dict = ['black' for _ in range(100)]


def render_gantt_chart(env, td: TensorDict, batch_idx=0, save_path=None):
    pm_schedule = torch.stack(env.schedule.process_module).permute(1,0,2)[batch_idx, :, :]
    robot_schedule = torch.stack(env.schedule.transport_module).permute(1,0,2)[batch_idx, :, :]
    # render gantt chart

    # wafer color
    wafer_color = {i:color_dict[i] for i in range(100)}

    # figure shape
    fig, gnt = plt.subplots(figsize=(100,20))
    title = f"Gantt {td['clock'][batch_idx].item()}"
    gnt.set_title(title, fontsize=20)
    gnt.set_xlabel('t')
    init_wafer_num = env.lot[0][-1]
    queue_lot_wafer_start_time = robot_schedule[robot_schedule[:, 0]>=init_wafer_num][0,3]
    gnt.vlines(x=queue_lot_wafer_start_time,  ymin=0, ymax=(env.PMs.num_pm+1)*20, colors='red', linestyles='dashed', linewidth=2, label='queue lot start')

    # bar patch
    patch_list =[]
    for i in range(env.wafers.num_wafer):
        if i < init_wafer_num:
            patch = mpatches.Patch(color=wafer_color[i], label=f'(init) w_{i}')
        else:
            patch = mpatches.Patch(color=wafer_color[i], label=f'w_{i}')

        patch_list.append(patch)
    gnt.legend(handles=patch_list, loc='upper right', fontsize=15)


    # draw PM process gantt
    for pm_id in range(1, env.PMs.num_pm+1):
        schedule = pm_schedule[pm_schedule[:, 0]==pm_id]

        # init wafer process gantt
        barhs = [[i[2].item(), (i[3] - i[2]).item()] for i in schedule]
        facecolors = [wafer_color[i[1].item()] for i in schedule]
        if td_horizon[batch_idx,0]['pm_hold_wafer_id'][pm_id] != -1:
            barhs.insert(0, [0, td_horizon[batch_idx,0]['pm_process_end_time'][pm_id].item()])
            facecolors.insert(0, wafer_color[td_horizon[batch_idx,0]['pm_hold_wafer_id'][pm_id].item()])

        # draw queue wafer process gantt
        gnt.broken_barh(
            barhs,
            (pm_id*20+10, 9),
            edgecolor='dimgrey',
            facecolors=facecolors,
        ),

    # draw residency gantt
    for pm_id in range(1, env.PMs.num_pm+1):
        schedule = pm_schedule[pm_schedule[:, 4]==pm_id]
        barhs = [[i[5].item(), (i[6] - i[5]).item()] for i in schedule]
        facecolors = [wafer_color[i[1].item()] for i in schedule]

        gnt.broken_barh(
            barhs,
            (pm_id*20+10, 9),
            edgecolor='dimgrey',
            facecolors=facecolors,
            alpha=0.5,
            hatch='s',
        ),

    if env.need_purge:
        # draw PM purge gantt
        for pm_id in range(1, env.PMs.num_pm+1):
                schedule = pm_schedule[pm_schedule[:, 4]==pm_id]
                barhs = [[i[7].item(), (i[8] - i[7]).item()] for i in schedule]
                gnt.broken_barh(
                barhs,
                (pm_id*20+10, 9),
                edgecolor='dimgrey',
                facecolors='dimgrey',
                alpha=0.5,
                ),

    # m2p: dynamic
    gnt.broken_barh(
        [[i[3].item(), (i[4]-i[3]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        hatch='\\',
    )
    # (m2p->unload) wait: dynamic
    gnt.broken_barh(
        [[i[4].item(), (i[5]-i[4]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        alpha=0.8,
        hatch='X',
    )

    # unload: static
    gnt.broken_barh(
        [[i[5].item(), (i[6]-i[5]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        hatch='/',
    )

    # move: static
    gnt.broken_barh(
        [[i[7].item(), (i[8]-i[7]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
    )

    # (m2d -> load) wait: dynamic
    gnt.broken_barh(
        [[i[8].item(), (i[9]-i[8]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        alpha=0.8,
        hatch='O'
    )

    # load: static
    gnt.broken_barh(
        [[i[9].item(), (i[10]-i[9]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
    )

    gnt.set_xticks([i for i in range(0, int(td['clock'][batch_idx]), 10)])
    gnt.set_yticks([20*(i)+15 for i in range(env.PMs.num_pm+2)]) # 2 = robot, out
    gnt.set_yticklabels(['R']+[f'PM{i+1}' for i in range(env.PMs.num_pm)]+['Out'])
    plt.xticks(rotation=45)
    plt.grid()

    # save plot to file if save_path is provided
    if save_path is not None:
        #log.info(f"save gantt chart...<{save_path}>")
        plt.savefig(save_path)

    plt.show()


def render_gantt_horizon(env, td: TensorDict, td_horizon, batch_idx=0, save_path=None):
    pm_schedule = torch.stack(env.schedule.process_module).permute(1,0,2)[batch_idx, :, :]
    robot_schedule = torch.stack(env.schedule.transport_module).permute(1,0,2)[batch_idx, :, :]
    # render gantt chart

    # wafer color
    wafer_color = {i:color_dict[i] for i in range(100)}

    # figure shape
    fig, gnt = plt.subplots(figsize=(100,20))
    title = f"Gantt {td['clock'][batch_idx].item()}"
    gnt.set_title(title, fontsize=20)
    gnt.set_xlabel('t')
    init_wafer_num = env.lot[0][-1]
    queue_lot_wafer_start_time = robot_schedule[robot_schedule[:, 0]>=init_wafer_num][0,3]
    gnt.vlines(x=queue_lot_wafer_start_time,  ymin=0, ymax=(env.PMs.num_pm+1)*20, colors='red', linestyles='dashed', linewidth=2, label='queue lot start')

    # bar patch
    patch_list =[]
    for i in range(env.wafers.num_wafer):
        if i < init_wafer_num:
            patch = mpatches.Patch(color=wafer_color[i], label=f'(init) w_{i}')
        else:
            patch = mpatches.Patch(color=wafer_color[i], label=f'w_{i}')

        patch_list.append(patch)
    gnt.legend(handles=patch_list, loc='upper right', fontsize=15)


    # draw PM process gantt
    for pm_id in range(1, env.PMs.num_pm+1):
        schedule = pm_schedule[pm_schedule[:, 0]==pm_id]

        # init wafer process gantt
        barhs = [[i[2].item(), (i[3] - i[2]).item()] for i in schedule]
        facecolors = [wafer_color[i[1].item()] for i in schedule]
        if td_horizon[batch_idx,0]['pm_hold_wafer_id'][pm_id] != -1:
            barhs.insert(0, [0, td_horizon[batch_idx,0]['pm_process_end_time'][pm_id].item()])
            facecolors.insert(0, wafer_color[td_horizon[batch_idx,0]['pm_hold_wafer_id'][pm_id].item()])

        # draw queue wafer process gantt
        gnt.broken_barh(
            barhs,
            (pm_id*20+10, 9),
            edgecolor='dimgrey',
            facecolors=facecolors,
        ),

    # draw residency gantt
    for pm_id in range(1, env.PMs.num_pm+1):
        schedule = pm_schedule[pm_schedule[:, 4]==pm_id]
        barhs = [[i[5].item(), (i[6] - i[5]).item()] for i in schedule]
        facecolors = [wafer_color[i[1].item()] for i in schedule]

        gnt.broken_barh(
            barhs,
            (pm_id*20+10, 9),
            edgecolor='dimgrey',
            facecolors=facecolors,
            alpha=0.5,
            hatch='s',
        ),

    if env.need_purge:
        # draw PM purge gantt
        for pm_id in range(1, env.PMs.num_pm+1):
                schedule = pm_schedule[pm_schedule[:, 4]==pm_id]
                barhs = [[i[7].item(), (i[8] - i[7]).item()] for i in schedule]
                gnt.broken_barh(
                barhs,
                (pm_id*20+10, 9),
                edgecolor='dimgrey',
                facecolors='dimgrey',
                alpha=0.5,
                ),

    # m2p: dynamic
    gnt.broken_barh(
        [[i[3].item(), (i[4]-i[3]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        hatch='\\',
    )
    # (m2p->unload) wait: dynamic
    gnt.broken_barh(
        [[i[4].item(), (i[5]-i[4]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        alpha=0.8,
        hatch='X',
    )

    # unload: static
    gnt.broken_barh(
        [[i[5].item(), (i[6]-i[5]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        hatch='/',
    )

    # move: static
    gnt.broken_barh(
        [[i[7].item(), (i[8]-i[7]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
    )

    # (m2d -> load) wait: dynamic
    gnt.broken_barh(
        [[i[8].item(), (i[9]-i[8]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
        alpha=0.8,
        hatch='O'
    )

    # load: static
    gnt.broken_barh(
        [[i[9].item(), (i[10]-i[9]).item()] for i in robot_schedule],
        (10,9),
        edgecolor='dimgrey',
        facecolors=[wafer_color[i[0].item()] for i in robot_schedule],
    )

    gnt.set_xticks([i for i in range(0, int(td['clock'][batch_idx]), 10)])
    gnt.set_yticks([20*(i)+15 for i in range(env.PMs.num_pm+2)]) # 2 = robot, out
    gnt.set_yticklabels(['R']+[f'PM{i+1}' for i in range(env.PMs.num_pm)]+['Out'])
    plt.xticks(rotation=45)
    plt.grid()

    # save plot to file if save_path is provided
    if save_path is not None:
        #log.info(f"save gantt chart...<{save_path}>")
        plt.savefig(save_path)

    plt.show()

"""
def render_schedule_table(env, td, bidx):
    action_wafer = td['wafer_id'][td['batch_idx'], td['action_wafer_index']][bidx]
    action_curr_loc = td['action_unload_loc'][bidx]
    action_next_loc = td['action_load_loc'][bidx]

    # action print
    line_length = 70
    # action print
    print("="*line_length)
    if action_next_loc == env.loc.num_pm+1:
        print(f"{td['i'][bidx].item()}th"+
                f" | unload: w_{action_wafer.item():<2}"+
                f" | from PM_{action_curr_loc.item():<2}"+
                f" -> LL"+
                f" | clock: {td['clock'][bidx].item():<2}s"
                )
    else:
        print(f"{td['i'][bidx].item()}th"+
                f" | unload: w_{action_wafer.item():<2}"+
                f" | from PM_{action_curr_loc.item():<2}"+
                f" -> PM_{action_next_loc.item():<2}"+
                f" | clock: {td['clock'][bidx].item():<2}s"
                )
    print(f"-"*line_length)

    # arm loc
    if env.arm_type == 'dual':
        arm1_loc = td['robot_loc'][bidx][0]
        print(f"arm: | PM{arm1_loc.item():<2}")

    print(f"-"*line_length)
    locs = ["LL"] + ["PM"+str(i+1) for i in range(env.loc.num_pm)] + ["LL"]
    locs = [f"{i:<10}" for i in locs]
    inverse_status_dict = {v: k for k, v in env.loc.status_dict.items()}

    loc_status = [inverse_status_dict[i.item()] for i in td['loc_status'][bidx]]
    loc_status = [f"{i:<10}" for i in loc_status]

    loc_wafers = [f"w_{i}" if i!=-1 else "" for i in td['loc_hold_wafer'][bidx]]
    loc_wafers = [f"{i:<10}" for i in loc_wafers]

    if env.purge_constraint:
        loc_start_time = torch.max(td['loc_process_start_time'], td['loc_purge_start_time'])
        loc_end_time = torch.max(td['loc_process_end_time'], td['loc_purge_end_time'])
    else:
        loc_start_time = td['loc_process_start_time']
        loc_end_time = td['loc_process_end_time']

    loc_start_time = [f"{i:<10}" for i in loc_start_time[bidx]]
    loc_end_time = [f"{i:<10}" for i in loc_end_time[bidx]]

    print("".join(locs) + "\n" + "".join(loc_status) + "\n" +\
            "".join(loc_wafers) + "\n" + "".join(loc_start_time) + "\n" + "".join(loc_end_time))
    print(f"-"*line_length)

    def _print_elements(elements, num_per_line):
        for i in range(0, len(elements), num_per_line):
            print(' '.join(str(elements[j]) for j in range(i, min(i + num_per_line, len(elements)))))

    # in loadport wafers
    inloadport_wafers = (td['wafer_loc'] == 0).nonzero()
    inloadport_wafers = inloadport_wafers[inloadport_wafers[:, 0]==bidx][:, 1].tolist()
    in_LL_wafer_ids = td['wafer_id'][bidx, inloadport_wafers]
    print(f"in Loadport wafers")
    _print_elements(in_LL_wafer_ids.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # out loadport wafers
    outloadport_wafers = (td['wafer_loc'] == env.loc.num_pm+1).nonzero()
    outloadport_wafers = outloadport_wafers[outloadport_wafers[:, 0]==bidx][:, 1].tolist()
    out_LL_wafer_ids = td['wafer_id'][bidx, outloadport_wafers]
    print(f"out Loadport wafers")
    _print_elements(out_LL_wafer_ids.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # queue wafers
    queue_wafers = env.wafer.status[bidx] == env.wafer.status_dict['queue']
    print(f"queue wafers")
    _print_elements(env.wafer.id[bidx, queue_wafers].tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # exit wafers
    exit_wafers = env.wafer.status[bidx] == env.wafer.status_dict['exit']
    print(f"exit wafers")
    _print_elements(env.wafer.id[bidx, exit_wafers].tolist(), env.wafer.foup_size)
    print(f"-"*line_length)
    print("")
    print("")
"""

def render_schedule(env:object, action:object, rid:int):
    action_wafer = env.wafer.name[env.batch_idx, action.foup_idx, action.wafer_idx][rid]
    action_curr_loc = action.unload_loc[rid]
    action_next_loc = action.load_loc[rid]

    # norm factor
    w = (300 - 2)
    # action print
    line_length = 70
    # action print
    print("="*line_length)
    if env.arm_type == 'single':
        if action_next_loc == env.loc.num_pm+1:
            print(f"{env.i[rid].item()}th"+
                    f" | unload: w_{action_wafer.item():<2}"+
                    f" | from PM_{action_curr_loc.item():<2}"+
                    f" -> LL"+
                    f" | clock: {w*env.clock[rid].item():<2.3f}s"
                    )
        else:
            print(f"{env.i[rid].item()}th"+
                    f" | unload: w_{action_wafer.item():<2}"+
                    f" | from PM_{action_curr_loc.item():<2}"+
                    f" -> PM_{action_next_loc.item():<2}"+
                    f" | clock: {w*env.clock[rid].item():<2.3f}s"
                    )
    elif env.arm_type == 'dual':
        action_type = 'load' if action.is_load[rid] else 'unload'
        action_loc = action.load_loc[rid] if action.is_load[rid] else action.unload_loc[rid]
        print(f"{env.i[rid].item()}th"+
              f" | {action_type}: w_{action_wafer.item():<2}"+
              f" | PM_{action_loc.item():<2}"+
              f" | clock: {w*env.clock[rid].item():<2.3f}s"
              )

    print(f"-"*line_length)

    # arm loc
    if env.arm_type == 'dual':
        arm1_loc = env.robot.loc[rid,0]
        arm2_loc = env.robot.loc[rid,1]
        arm1_hold_wafer = env.robot.hold_wafer[rid,0]
        arm2_hold_wafer = env.robot.hold_wafer[rid,1]
        print(f"arm1: PM{arm1_loc.item():<3} | hold: w_{arm1_hold_wafer.item()}")
        print(f"arm2: PM{arm2_loc.item():<3} | hold: w_{arm2_hold_wafer.item()}")

    print(f"-"*line_length)
    locs = ["LL"] + ["PM"+str(i+1) for i in range(env.loc.num_pm)] + ["LL"]
    locs = [f"{i:<10}" for i in locs]
    inverse_status_dict = {v: k for k, v in env.loc.status_dict.items()}

    loc_status = [inverse_status_dict[i.item()] for i in env.loc.status[rid]]
    loc_status = [f"{i:<10}" for i in loc_status]

    loc_wafers = [f"w_{i}" if i!=-1 else "" for i in env.loc.hold_wafer[rid]]
    loc_wafers = [f"{i:<10}" for i in loc_wafers]

    if env.purge_constraint:
        loc_start_time = torch.max(env.loc.process_start_time, env.loc.purge_start_time)
        loc_end_time = torch.max(env.loc.process_end_time, env.loc.purge_end_time)
    else:
        loc_start_time = env.loc.process_start_time
        loc_end_time = env.loc.process_end_time

    loc_start_time = [f"{w*i:<10.3f}" for i in loc_start_time[rid]]
    loc_end_time = [f"{w*i:<10.3f}" for i in loc_end_time[rid]]

    print("".join(locs) + "\n" + "".join(loc_status) + "\n" +\
            "".join(loc_wafers) + "\n" + "".join(loc_start_time) + "\n" + "".join(loc_end_time))
    print(f"-"*line_length)

    def _print_elements(elements, num_per_line):
        for i in range(0, len(elements), num_per_line):
            print(' '.join(str(elements[j]) for j in range(i, min(i + num_per_line, len(elements)))))

    # in loadport wafers
    in_LL_wafer_ids = env.wafer.name[rid][env.wafer.loc[rid] == 0]
    print(f"in Loadport wafers")
    _print_elements(in_LL_wafer_ids.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # out loadport wafers
    out_LL_wafer_ids = env.wafer.name[rid][env.wafer.loc[rid] == env.loc.num_pm+1]
    print(f"out Loadport wafers")
    _print_elements(out_LL_wafer_ids.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # queue wafers
    queue_wafers = env.wafer.name[rid][env.wafer.status[rid] == env.wafer.status_dict['queue']]
    print(f"queue wafers")
    _print_elements(queue_wafers.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)

    # exit wafers
    exit_wafers = env.wafer.name[rid][env.wafer.status[rid] == env.wafer.status_dict['exit']]
    print(f"exit wafers")
    _print_elements(exit_wafers.tolist(), env.wafer.foup_size)
    print(f"-"*line_length)
    print("")
    print("")
