import os
import pandas as pd
import numpy as np
import argparse
import json

import re
import matplotlib.pyplot as plt 
import matplotlib
import glob

def parse_directory(results_dir, num_epochs, steps_per_epoch, plot_keys):
    df_idx = pd.MultiIndex.from_product([['IIWA', 'Sawyer', 'Panda', 'UR5e', 'Jaco', 'Kinova3'],
                                  ['PickPlace', 'Push', 'Trashcan', 'Shelf'],
                                  ['dumbbell', 'plate', 'hollowbox', 'bread'],
                                  ['None', 'GoalWall', 'GoalDoor', 'ObjectWall', 'ObjectDoor'],
                                  np.arange(1, num_epochs + 1) * steps_per_epoch],
                                 names=['Robot', 'Task', 'Object', 'Obstacle', 'epoch'])

    df = pd.DataFrame(columns=plot_keys, index=df_idx, dtype=float)
    cnt = 0
    for i, subdir in enumerate(glob.iglob(results_dir + '/t:*')):
        with open(os.path.join(subdir, 'config.json')) as f:
            config = json.load(f)
        exp_name = config['exp_name']
        exp_args = exp_name.split('_')
        for arg in exp_args:
            k,v = arg.split(':')
            if k == 'robot': robot = v
            elif k == 'task': task = v
            elif k == 'object': obj = v
            elif k == 'obstacle': obstacle = v
        tmp_df = pd.read_csv(os.path.join(subdir, 'progress.txt'), sep='\t', usecols=plot_keys)[plot_keys]
        try:
            df.loc[(robot, task, obj, obstacle)] = tmp_df.values[:num_epochs]
        except ValueError:
            cnt += 1
            pass  # eventually should at least raise a warning
    print('number of tasks that dint finish:', cnt)
    return df

def plot_avg_curve(results_df, num_epochs, steps_per_epoch, plot_keys, plot_format='pdf'):
    avg_perf = df.groupby('epoch').mean()
    avg_perf.plot()
    plt.savefig('avg_curve.' + plot_format)

def plot_barchart(results_df, num_epochs, steps_per_epoch, deavg_key, plot_format='pdf'):
    # final_df = results_df.xs(tuple(np.arange(num_epochs-10, num_epochs) * steps_per_epoch), level='epoch')
    final_df = results_df.loc[(slice(None), slice(None), slice(None), slice(None), np.arange(num_epochs-10, num_epochs) * steps_per_epoch)]

    final_df = final_df.dropna()
    final_df = final_df.groupby(level=(0, 1, 2, 3)).mean()
    # print(final_df.loc[final_df.index.get_level_values(level=1) == 'PickPlace'])
    print(final_df.groupby(level=(0,2)).mean())
    deavg_df = pd.DataFrame(columns=final_df.columns)
    for key in deavg_key:   # deavg_key = ['Robot', 'Object', 'Obstacle']
        tmp_df = final_df.groupby(key).mean()
        deavg_df = deavg_df.append(tmp_df)
    print(deavg_df)

def plot_histogram(results_df, num_epochs, steps_per_epoch, deavg_key, plot_format='pdf'):
    final_df = results_df.xs(num_epochs * steps_per_epoch, level='epoch')
    # for level in range(4):
    #     for key, new_df in final_df.groupby(level=level):
    #         new_df.hist(bins=len(new_df))
    #         plt.title(key)
    #         plt.show()

    for level in range(4):
        final_df.unstack(level=level).plot(kind='hist', sharey=True, subplots=True)#)
        plt.show()

    # final_df.hist(bins=50)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Spnningup results')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training or update steps')
    parser.add_argument('--steps-per-epoch', type=int, default=8000, help='Number of environment steps per epoch')
    parser.add_argument('--results-dir', type=str, default='tmp/', help='Directory where results are logged')
    parser.add_argument('--plot-style', type=str, default=['avg-curve'], help='List of the types of plots to generate',
                            choices=['avg-curve', 'print-table', 'barchart', 'histogram'], nargs='+')
    parser.add_argument('--plot-key', type=str, default=['Success'], help='List of keys to plot',
                            choices=['Success', 'StepsAtGoal', 'AverageEpRet', 'Entropy', 'SuccessLift', 'SuccessGrasp', 'SuccessHover'], nargs='+')
    parser.add_argument('--deavg-key', type=str, default=None, help='List of keys to deavg by',
                            choices=['Robot', 'Task', 'Object', 'Obstacle'], nargs='+')
    parser.add_argument('--table-format', type=str, default='markdown', help='Format for printing results table',
                            choices=['markdown', 'latex'])
    parser.add_argument('--plot-format', type=str, default='pdf', help='Format for saving figures',
                            choices=['pdf', 'svg', 'png'])
    args = parser.parse_args()

    df = parse_directory(args.results_dir, args.num_epochs, args.steps_per_epoch, args.plot_key)
    for style in args.plot_style:
        if style == 'avg-curve':
            plot_avg_curve(df, args.num_epochs, args.steps_per_epoch, args.plot_key, args.plot_format)
        elif style == 'print-table':
            pass
        elif style == 'barchart':
            plot_barchart(df, args.num_epochs, args.steps_per_epoch, args.deavg_key, args.plot_format)
        elif style == 'histogram':
            plot_histogram(df, args.num_epochs, args.steps_per_epoch, args.deavg_key, args.plot_format)



# # algos = ['comp-ppo', 'comp-ppo-fixed', 'stl-ppo', 'pandc-ewc-ppo']
# algos = ['comp-ppo-tanh', 'stl-ppo-tanh']
# num_seeds = 3
# num_tasks = 25
# num_epochs = 150
# steps_per_epoch = 8000

# rewards_results_df = pd.DataFrame(columns=['Algorithm', 'Zero-shot', 'Online', 'Off-line', 'Final'])
# success_results_df = pd.DataFrame(columns=['Algorithm', 'Zero-shot', 'Online', 'Off-line', 'Final'])
# rewards_errors_df = pd.DataFrame(columns=['Algorithm', 'Zero-shot', 'Online', 'Off-line', 'Final'])
# success_errors_df = pd.DataFrame(columns=['Algorithm', 'Zero-shot', 'Online', 'Off-line', 'Final'])


# for algo in algos:
#     if 'comp-ppo' in algo:
#         success_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         reward_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         entropy_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         success_accommodation = np.full((num_seeds, num_tasks, num_tasks), np.nan)
#         reward_accommodation = np.full((num_seeds, num_tasks, num_tasks), np.nan)
#         for seed_i in range(num_seeds):
#             f_name = os.path.join('results_mini', experiment_name[algo], inner_dir_map[algo], f'{inner_dir_map[algo]}_s{seed_i}', 'progress.txt')
#             accommodation_df = pd.read_csv(f_name, sep='\t')
#             for task_i in range(num_tasks):
#                 f_name = os.path.join('results_mini', experiment_name[algo], inner_dir_map[algo], f'{inner_dir_map[algo]}_s{seed_i}', f'task_{task_i}', 'progress.txt')
#                 df = pd.read_csv(f_name, sep='\t', usecols=[f'Success', 'AverageEpRet', 'Entropy'])
#                 start_epoch = max(0, num_epochs - len(df['Success']))              # [not working for first case] if I'm plotting fewer epochs than happened, 0; elif df is shorter (because of gridsearch), num_grid_search_steps
#                 success_curve[seed_i, task_i, start_epoch:] = df['Success']             # truncate in case we want to plot fewer epochs     
#                 reward_curve[seed_i, task_i, start_epoch:] = df['AverageEpRet']         # truncate in case we want to plot fewer epochs
#                 entropy_curve[seed_i, task_i, start_epoch:] = df['Entropy'] 

#                 # success_curve[seed_i, task_i, :] = df['Success'][:num_epochs]             # truncate in case we want to plot fewer epochs     
#                 # reward_curve[seed_i, task_i, :] = df['AverageEpRet'] [:num_epochs]        # truncate in case we want to plot fewer epochs 
#                 # entropy_curve[seed_i, task_i, :] = df['Entropy'][:num_epochs]

#                 success_accommodation[seed_i, task_i, :task_i + 1] = accommodation_df.loc[accommodation_df['train task'] == task_i, 'success_mean']
#                 reward_accommodation[seed_i, task_i, :task_i + 1] = accommodation_df.loc[accommodation_df['train task'] == task_i, 'reward_mean']

#         # print('online')
#         # print('\t',success_curve[:, :, -1].mean())
#         # print('\t',reward_curve[:, :, -1].mean())
#         # print('Off-line')
#         # print('\t',success_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean())
#         # print('\t',reward_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean())
#         # print('final')
#         # print('\t',success_accommodation[:, -1].mean())
#         # print('\t',reward_accommodation[:, -1].mean())
#         if algo == 'comp-ppo':  # "jumpstart" for comp-ppo is not time 0 but later
#             idx = np.isnan(reward_curve).sum(axis=-1)
#             reward_jumpstart_results_tmp = reward_curve[np.arange(idx.shape[0]).reshape(-1, 1), np.arange(idx.shape[1]).reshape(1, -1), idx]
#             reward_jumpstart_results_mean = np.mean(reward_jumpstart_results_tmp)
#             reward_jumpstart_results_err = reward_jumpstart_results_tmp.mean(axis=1).std() / np.sqrt(num_seeds)
#         else:
#             reward_jumpstart_results_mean = np.nanmean(reward_curve[:, :, 0])
#             reward_jumpstart_results_err = reward_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)

#         reward_assimilation_results_mean = np.nanmean(reward_curve[:, :, -1])
#         reward_assimilation_results_err = reward_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_accommodation_results_mean = np.nanmean(reward_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)])
#         reward_accommodation_results_err = reward_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_final_results_mean = np.nanmean(reward_accommodation[:, -1, :])
#         reward_final_results_err = reward_accommodation[:, -1, :].mean(axis=1).std() / np.sqrt(num_seeds)

#         if algo == 'comp-ppo':  # "jumpstart" for comp-ppo is not time 0 but later
#             idx = np.isnan(success_curve).sum(axis=-1)
#             success_jumpstart_results_tmp = success_curve[np.arange(idx.shape[0]).reshape(-1, 1), np.arange(idx.shape[1]).reshape(1, -1), idx]
#             success_jumpstart_results_mean = np.mean(success_jumpstart_results_tmp)
#             success_jumpstart_results_err = success_jumpstart_results_tmp.mean(axis=1).std() / np.sqrt(num_seeds)
#         else:
#             success_jumpstart_results_mean = np.nanmean(success_curve[:, :, 0])
#             success_jumpstart_results_err = success_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)

#         success_assimilation_results_mean = np.nanmean(success_curve[:, :, -1])
#         success_assimilation_results_err = success_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_accommodation_results_mean = np.nanmean(success_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)])
#         success_accommodation_results_err = success_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_final_results_mean = np.nanmean(success_accommodation[:, -1, :])
#         success_final_results_err = success_accommodation[:, -1, :].mean(axis=1).std() / np.sqrt(num_seeds)
        
        
#         legend = legend_map[algo]
#     elif algo == 'pandc-replay-ppo' or algo == 'pandc-ewc-ppo':
#         success_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         reward_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         entropy_curve = np.full((num_seeds, num_tasks, num_epochs), np.nan)
#         success_accommodation = np.full((num_seeds, num_tasks, num_tasks), np.nan)
#         reward_accommodation = np.full((num_seeds, num_tasks, num_tasks), np.nan)
#         for seed_i in range(num_seeds):
#             f_name = os.path.join('results_mini', experiment_name[algo], inner_dir_map[algo], f'{inner_dir_map[algo]}_s{seed_i}', 'progress.txt')
#             accommodation_df = pd.read_csv(f_name, sep='\t')
#             for task_i in range(num_tasks):
#                 f_name = os.path.join('results_mini', experiment_name[algo], inner_dir_map[algo], f'{inner_dir_map[algo]}_s{seed_i}', f'task_{task_i}', 'progress.txt')
#                 df = pd.read_csv(f_name, sep='\t', usecols=[f'Success', 'AverageEpRet', 'Entropy'])
#                 start_epoch = max(0, num_epochs - len(df['Success']))              # [not working for first case] if I'm plotting fewer epochs than happened, 0; elif df is shorter (because of gridsearch), num_grid_search_steps
#                 success_curve[seed_i, task_i, start_epoch:] = df['Success']             # truncate in case we want to plot fewer epochs     
#                 reward_curve[seed_i, task_i, start_epoch:] = df['AverageEpRet']         # truncate in case we want to plot fewer epochs
#                 entropy_curve[seed_i, task_i, start_epoch:] = df['Entropy'] 

#                 success_accommodation[seed_i, task_i, :task_i + 1] = accommodation_df.loc[accommodation_df['train task'] == task_i, 'success_mean']
#                 reward_accommodation[seed_i, task_i, :task_i + 1] = accommodation_df.loc[accommodation_df['train task'] == task_i, 'reward_mean']

#         reward_jumpstart_results_mean = np.nanmean(reward_curve[:, :, 0])
#         reward_jumpstart_results_err = reward_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)

#         reward_assimilation_results_mean = np.nanmean(reward_curve[:, :, -1])
#         reward_assimilation_results_err = reward_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_accommodation_results_mean = np.nanmean(reward_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)])
#         reward_accommodation_results_err = reward_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_final_results_mean = np.nanmean(reward_accommodation[:, -1, :])
#         reward_final_results_err = reward_accommodation[:, -1, :].mean(axis=1).std() / np.sqrt(num_seeds)

#         success_jumpstart_results_mean = np.nanmean(success_curve[:, :, 0])
#         success_jumpstart_results_err = success_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)

#         success_assimilation_results_mean = np.nanmean(success_curve[:, :, -1])
#         success_assimilation_results_err = success_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_accommodation_results_mean = np.nanmean(success_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)])
#         success_accommodation_results_err = success_accommodation[:, np.arange(num_tasks), np.arange(num_tasks)].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_final_results_mean = np.nanmean(success_accommodation[:, -1, :])
#         success_final_results_err = success_accommodation[:, -1, :].mean(axis=1).std() / np.sqrt(num_seeds)
        
#         legend = legend_map[algo]

#     elif algo == 'mtl-ppo':
#         for num_tasks in num_tasks:
#             success_curve = np.empty((num_seeds, num_tasks, num_epochs))
#             reward_curve = np.empty((num_seeds, num_tasks, num_epochs))
#             for seed_i in range(num_seeds):
#                 f_name = os.path.join('results_mini','mtl', f'mtl-ppo-{num_tasks}-tasks',f'mtl-ppo-{num_tasks}-tasks_s{seed_i}','progress.txt')
#                 df = pd.read_csv(f_name, sep='\t', usecols=[f'Success_{task_i}' for task_i in range(num_tasks)] + [f'AverageEpRet_{task_i}' for task_i in range(num_tasks)])
#                 for task_i in range(num_tasks):
#                     success_curve[seed_i, task_i, :] = df[f'Success_{task_i}'][:num_epochs]             # truncate in case we want to plot fewer epochs     
#                     reward_curve[seed_i, task_i, :] = df[f'AverageEpRet_{task_i}'][:num_epochs]         # truncate in case we want to plot fewer epochs 

#             legend = f'MTL-{num_tasks}'

#     elif algo == 'stl-ppo' or algo == 'stl-ppo-tanh':
# ################
#         success_curve = np.empty((num_seeds, num_tasks, num_epochs))
#         reward_curve = np.empty((num_seeds, num_tasks, num_epochs))
#         entropy_curve = np.empty((num_seeds, num_tasks, num_epochs))
#         for seed_i in range(num_seeds):
#             for task_i in range(num_tasks):
#                 f_name = os.path.join('results_mini',experiment_name[algo], inner_dir_map[algo],f'{inner_dir_map[algo]}_s{seed_i}', f'task_{task_i}', 'progress.txt')
#                 # try:
#                 df = pd.read_csv(f_name, sep='\t', usecols=[f'Success', 'AverageEpRet', 'Entropy'])
#                 # except:
#                     # print(seed_i, task_i)
#                 success_curve[seed_i, task_i, :] = df['Success'][:num_epochs]             # truncate in case we want to plot fewer epochs     
#                 reward_curve[seed_i, task_i, :] = df['AverageEpRet'][:num_epochs]         # truncate in case we want to plot fewer epochs 
#                 entropy_curve[seed_i, task_i, :] = df['Entropy'][:num_epochs]         # truncate in case we want to plot fewer epochs 
#         print(algo)
#         print('\t',success_curve[:, :, -1].mean())
#         print('\t',reward_curve[:, :, -1].mean())
#         print('\t', (success_curve.sum(axis=2) == 0).mean(axis=0).sum())
#         legend = legend_map[algo]
            
#         reward_jumpstart_results_mean = np.nanmean(reward_curve[:, :, 0])
#         reward_jumpstart_results_err = reward_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_assimilation_results_mean = np.nanmean(reward_curve[:, :, -1])
#         reward_assimilation_results_err = reward_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         reward_accommodation_results_mean = 0#reward_assimilation_results_mean
#         reward_accommodation_results_err = np.nan#reward_assimilation_results_err
#         reward_final_results_mean = 0#reward_assimilation_results_mean
#         reward_final_results_err = np.nan#reward_assimilation_results_err

#         success_jumpstart_results_mean = np.nanmean(success_curve[:, :, 0])
#         success_jumpstart_results_err = success_curve[:, :, 0].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_assimilation_results_mean = np.nanmean(success_curve[:, :, -1])
#         success_assimilation_results_err = success_curve[:, :, -1].mean(axis=1).std() / np.sqrt(num_seeds)
#         success_accommodation_results_mean = 0#success_assimilation_results_mean
#         success_accommodation_results_err = np.nan#success_assimilation_results_err
#         success_final_results_mean = 0#success_assimilation_results_mean
#         success_final_results_err = np.nan#success_assimilation_results_err

#     rewards_results_df = rewards_results_df.append(
#     {
#         'Algorithm': tick_map[algo], 
#         'Zero-shot': reward_jumpstart_results_mean,
#         'Online': reward_assimilation_results_mean,
#         'Off-line': reward_accommodation_results_mean,
#         'Final': reward_final_results_mean
#     }, 
#     ignore_index=True)

#     success_results_df = success_results_df.append(
#     {
#         'Algorithm': tick_map[algo], 
#         'Zero-shot': success_jumpstart_results_mean,
#         'Online': success_assimilation_results_mean,
#         'Off-line': success_accommodation_results_mean,
#         'Final': success_final_results_mean
#     }, 
#     ignore_index=True)

#     rewards_errors_df = rewards_errors_df.append(
#     {
#         'Algorithm': tick_map[algo], 
#         'Zero-shot': reward_jumpstart_results_err,
#         'Online': reward_assimilation_results_err,
#         'Off-line': reward_accommodation_results_err,
#         'Final': reward_final_results_err
#     }, 
#     ignore_index=True)

#     success_errors_df = success_errors_df.append(
#     {
#         'Algorithm': tick_map[algo], 
#         'Zero-shot': success_jumpstart_results_err,
#         'Online': success_assimilation_results_err,
#         'Off-line': success_accommodation_results_err,
#         'Final': success_final_results_err
#     }, 
#     ignore_index=True)

#     x_axis = np.arange(num_epochs) * steps_per_epoch / 1e6

#     plt.figure(0)
#     success_mean = success_curve.mean(axis=(0,1))
#     success_err = success_curve.mean(axis=1).std(axis=0) / np.sqrt(num_seeds)
#     plt.plot(x_axis, success_mean, linestyle=linestyle_map[algo], color=color_map[algo], linewidth=2, label=legend)
#     # plt.plot(x_axis, success_curve.mean(axis=1).T, label=legend, alpha=0.5)
#     plt.fill_between(x_axis, success_mean - success_err, success_mean + success_err, color=color_map[algo], alpha=0.2)
#     if algo == 'comp-ppo':
#         start_idx = np.isnan(success_mean).sum()
#         ax = plt.gca()
#         ax.axvspan(0, x_axis[start_idx], alpha=.1, color='black')
#         arrow_xy = (x_axis[start_idx // 4], 0.6)
#         text_xy = (0.15, 0.65)
#         plt.annotate('Data used for discrete search\nin ' + legend_map[algo], 
#             xy=arrow_xy, xytext=text_xy, 
#             arrowprops={'arrowstyle': '<-', 'connectionstyle': 'arc3,rad=0.2', 'relpos': (0,0.5)},
#             fontsize=14,color='black')

#     plt.figure(1)
#     reward_mean = reward_curve.mean(axis=(0,1))
#     reward_err = reward_curve.mean(axis=1).std(axis=0) / np.sqrt(num_seeds)
#     plt.plot(x_axis, reward_mean, linestyle=linestyle_map[algo], color=color_map[algo], linewidth=2, label=legend)
#     # plt.plot(x_axis, reward_curve.mean(axis=1).T, label=legend, alpha=0.5)
#     plt.fill_between(x_axis, reward_mean - reward_err, reward_mean + reward_err, color=color_map[algo], alpha=0.2)

#     plt.figure(2)
#     entropy_mean = entropy_curve.mean(axis=(0,1))
#     entropy_err = entropy_curve.mean(axis=1).std(axis=0) / np.sqrt(num_seeds)
#     plt.plot(x_axis, entropy_mean, linestyle=linestyle_map[algo], color=color_map[algo], linewidth=2, label=legend)
#     # plt.plot(x_axis, entropy_curve.mean(axis=1).T, label=legend, alpha=0.5)
#     plt.fill_between(x_axis, entropy_mean - entropy_err, entropy_mean + entropy_err, color=color_map[algo], alpha=0.2)

# plt.figure(0)
# # plt.legend()
# plt.ylabel('avg success',fontsize=24)
# plt.xlabel('# 1M steps per task',fontsize=24)
# plt.tick_params(labelsize=24)
# plt.tight_layout()
# plt.savefig('learning_curves_tanh_success.pdf')
# plt.close()

# plt.figure(1)
# # plt.legend()
# plt.ylabel('avg return',fontsize=24)
# plt.xlabel('# 1M steps per task',fontsize=24)
# plt.tick_params(labelsize=24)
# plt.tight_layout()
# plt.savefig('learning_curves_tanh_reward.pdf')
# plt.close()

# plt.figure(2)
# # plt.legend()
# plt.ylabel('avg entropy',fontsize=24)
# plt.xlabel('# 1M steps per task',fontsize=24)
# plt.tick_params(labelsize=24)
# plt.tight_layout()
# plt.savefig('learning_curves_tanh_entropy.pdf')
# # plt.close()

# ax = plt.gca()
# legend = ax.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.5, -0.4),
#                     fancybox=False, shadow=False, ncol=2)
# fig = legend.figure
# fig.canvas.draw()
# bbox = legend.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('learning_curves_tanh_legend.pdf', dpi="figure", bbox_inches=bbox)
# plt.close()

# rewards_results_df.set_index('Algorithm', inplace=True)
# rewards_errors_df.set_index('Algorithm', inplace=True)

# plt.figure(3)
# ax = rewards_results_df.plot(kind='bar', yerr=rewards_errors_df, capsize=4, legend=False)
# cols = ["_" + col for col in rewards_results_df.columns]    # this will make the second barchart (for the single hatch on comp-ppo) not to show up on the legend
# rewards_results_df.columns = cols
# rewards_results_df.plot(kind='bar', ax=ax, legend=False, facecolor='none')
# if algos[0] == 'comp-ppo':
#     ax.axvline(0.5, color='black', linestyle='--')

#     bars = [rect for rect in ax.patches if isinstance(rect, matplotlib.patches.Rectangle)]
#     bars[0].set_alpha(0.5)
#     bars[len(bars) // 2].set_hatch('//')
# plt.xlabel('',fontsize=24)
# plt.ylabel('avg return',fontsize=24)
# plt.tick_params(labelsize=24)    
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('reward_barchart_tanh.pdf')

# if algos[0] == 'comp-ppo':
#     bars[0].set_alpha(1)
# legend = ax.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.5, -0.8),
#                     fancybox=False, shadow=False, ncol=2)
# fig = legend.figure
# fig.canvas.draw()
# bbox = legend.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('reward_barchart_tanh_legend.pdf', dpi="figure", bbox_inches=bbox)

# plt.close()

# success_results_df.set_index('Algorithm', inplace=True)
# success_errors_df.set_index('Algorithm', inplace=True)

# print(success_results_df.to_markdown())
# print(success_errors_df.to_markdown())

# plt.figure(4)
# ax = success_results_df.plot(kind='bar', yerr=success_errors_df, capsize=4, legend=False)
# cols = ["_" + col for col in success_results_df.columns]    # this will make the second barchart (for the single hatch on comp-ppo) not to show up on the legend
# success_results_df.columns = cols
# success_results_df.plot(kind='bar', ax=ax, legend=False, facecolor='none')
# if algos[0] == 'comp-ppo':
#     ax.axvline(0.5, color='black', linestyle='--')

#     bars = [rect for rect in ax.patches if isinstance(rect, matplotlib.patches.Rectangle)]
#     bars[0].set_alpha(0.5)
#     bars[len(bars) // 2].set_hatch('//')
# plt.xlabel('',fontsize=24)
# plt.ylabel('avg success',fontsize=24)
# plt.tick_params(labelsize=24)    
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('success_barchart_tanh.pdf')

# if algos[0] == 'comp-ppo':
#     bars[0].set_alpha(1)
# legend = ax.legend(fontsize=24, loc='upper center', bbox_to_anchor=(0.5, -0.8),
#                     fancybox=False, shadow=False, ncol=2)
# fig = legend.figure
# fig.canvas.draw()
# bbox = legend.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('success_barchart_tanh_legend.pdf', dpi="figure", bbox_inches=bbox)

# plt.close()
