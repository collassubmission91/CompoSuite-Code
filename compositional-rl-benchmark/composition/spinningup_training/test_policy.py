import argparse
from itertools import product
import json
import os
import glob
import numpy as np
import pandas as pd
from composition.env.main import make
from mujoco_py.generated import const
from PIL import Image

import torch

from spinup import EpochLogger
from spinup.utils.test_policy import load_pytorch_policy
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import proc_id

from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.compositional_core import CompositionalMLPActorCritic


def parse_mt_directory(results_dir, num_tasks=56, experiment_method='default'):
    subdir = glob.glob(results_dir)[0]

    tasks_that_ran = set()
    for j in range(num_tasks):
        with open(os.path.join(subdir, f'args_{j}.json')) as f:
            args = json.load(f)
        robot = args['robot']
        task = args['task']
        obj = args['object']
        obstacle = args['obstacle']
        tasks_that_ran.add((robot, task, obj, obstacle))

    axes = [('IIWA', 'Panda', 'Jaco', 'Kinova3'),
            ('PickPlace', 'Push', 'Trashcan', 'Shelf'),
            ('dumbbell', 'plate', 'hollowbox', 'bread'),
            ('None', 'GoalDoor', 'ObjectWall', 'ObjectDoor')]
    lst = [i for i in product(*axes)]
    all_tasks = set(lst)

    helper_set = all_tasks - tasks_that_ran

    if experiment_method == 'holdout':
        zero_shot_tasks = set()
        for task in helper_set:
            if task[1] == "PickPlace":
                zero_shot_tasks.add(task)
    elif experiment_method == 'smallscale':
        zero_shot_tasks = set()
        for task in helper_set:
            if task[0] == "IIWA":
                zero_shot_tasks.add(task)
    else:
        zero_shot_tasks = helper_set


    return sorted(list(tasks_that_ran)), sorted(list(zero_shot_tasks))


def run_policy(env, policy, num_episodes, max_ep_len=500, logger=None):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    success = 0
    while n < num_episodes:
        a, _, _ = policy.step(torch.from_numpy(o).float())
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if r == 1:
            success = 1

        if d or (ep_len == max_ep_len):
            if logger:
                logger.store(EpRet=ep_ret, EpLen=ep_len, Success=success)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Success %i' %
                  (n, ep_ret, ep_len, success))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
            success = 0

    if logger:
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('Success')
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()


def average_dirs(load_dir):
    subdirs = glob.glob(os.path.join(load_dir, "*"))

    all_sucess = []
    all_return = []

    for s in subdirs:
        df = pd.read_csv(os.path.join(s, "progress.txt"), delimiter='\t')
        all_sucess.append(df['AverageSuccess'].values)
        all_return.append(df['AverageEpRet'].values)

    print("Success:", np.mean(all_sucess))
    print("Return:", np.mean(all_return))


def load_model(path, learner, single_task):
    env = make(robot=single_task[0],
               obj=single_task[2],
               obstacle=single_task[3],
               task=single_task[1],
               use_task_id_obs=True, controller='joint')

    if learner == 'MTL':
        ac_kwargs = dict(hidden_sizes=[256]*2, log_std_init=0)
        policy = MLPActorCritic(
            env.observation_space, env.action_space, env.observation_positions, **ac_kwargs)
        state_dicts = torch.load(os.path.join(
            path, 'pyt_save', 'state_dicts.pt'))
        policy.load_state_dict(state_dicts['model'])
    elif learner == 'Comp':
        hidden_sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
        ac_kwargs = {
            # 'hidden_sizes': [args.hid]*args.l,
            'log_std_init': 0,
            'hidden_sizes': hidden_sizes,
            'module_names': ['obstacle_id', 'object_id', 'subtask_id', 'robot_id'],
            'module_input_names': ['obstacle-state', 'object-state', 'goal-state', 'robot0_proprio-state'],
            'interface_depths': [-1, 1, 2, 3],
            'graph_structure': [[0], [1], [2], [3]],
        }
        policy = CompositionalMLPActorCritic(
            env.observation_space, env.action_space, env.observation_positions, **ac_kwargs)
        state_dicts = torch.load(os.path.join(
            path, 'pyt_save', 'state_dicts.pt'))
        policy.load_state_dict(state_dicts['model'])
    else:
        policy = load_pytorch_policy(path, '')

    return policy


def create_mini_imgs(model_paths, learner=None):
    best_imgs = {}
    for path in model_paths:
        train_tasks, zero_shot_tasks = parse_mt_directory(path)
        all_tasks = train_tasks + zero_shot_tasks
        policy = load_model(path, learner=learner,
                            single_task=zero_shot_tasks[0])

        for task in all_tasks:
            env = make(robot=task[0], obj=task[2], obstacle=task[3], task=task[1],
                       has_offscreen_renderer=True, has_renderer=False,
                       env_horizon=500, use_task_id_obs=True, controller='joint')

            img, best_rew = create_img_from_run(
                env, policy=policy, num_episodes=1)
            if task in best_imgs.keys():
                if best_imgs[task][0] < best_rew:
                    best_imgs[task] = [best_rew, img]
            else:
                best_imgs[task] = [best_rew, img]
            del(env)

    for task in best_imgs.keys():
        img = best_imgs[task][1]
        save_img_dir = os.path.join(
            'spinningup_training/images', str(task))
        os.makedirs(save_img_dir, exist_ok=True)
        img = Image.fromarray(img)
        img.save(save_img_dir + '/best_rew_img.jpeg')


def create_img_from_run(env, policy, num_episodes, max_ep_len=500, ):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    success = 0
    best_reward = 0
    while n < num_episodes:
        a = policy(o)  # this may throw depending on the loaded model
        env.sim._render_context_offscreen.cam.fixedcamid = env.sim.model.camera_name2id(
            'frontview')
        env.sim._render_context_offscreen.cam.type = const.CAMERA_FIXED
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if r == 1:
            success = 1

        if best_reward < r:
            img = np.asarray(env.sim.render(1920, 1080, depth=False)[
                             ::-1, :, :], dtype=np.uint8)

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Success %i' %
                  (n, ep_ret, ep_len, success))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
            success = 0

    return img, best_reward


def run_test(base_path, save_path, load_dir, learner='MTL', experiment_method='default', num_tasks=56):
    path = os.path.join(base_path, save_path, load_dir)
    _, zero_shot_tasks = parse_mt_directory(path, experiment_method=experiment_method, num_tasks=num_tasks)
    # policy = load_pytorch_policy(path, '')
    policy = load_model(path, learner=learner, single_task=zero_shot_tasks[0])

    np.random.seed(proc_id())

    for task in zero_shot_tasks:
        env = make(robot=task[0], obj=task[2], obstacle=task[3], task=task[1],
                   has_offscreen_renderer=False, has_renderer=False,
                   env_horizon=500, use_task_id_obs=True, controller='joint')
        logger_kwargs = setup_logger_kwargs(str(task),
                                            data_dir=os.path.join(
                                                'spinningup_training/zero_shot_results',
                                            save_path))
        logger = EpochLogger(**logger_kwargs)

        run_policy(env, policy=policy, num_episodes=1, logger=logger)
        del(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=int)
    args = parser.parse_args()

    directories = np.array([
        ["comp_experiments/s0/", "MTL_56/"],
        ["comp_experiments/s1/", "MTL_56/"],
        ["comp_experiments/s2/", "MTL_56/"],
        ["mtl_experiments/s0/", "MTL_56/"],
        ["mtl_experiments/s1/", "MTL_56/"],
        ["mtl_experiments/s2/", "MTL_56/"],
        ["smallscale_comp_experiments/s0/", "MTL_32/"],
        ["smallscale_comp_experiments/s1/", "MTL_32/"],
        ["smallscale_comp_experiments/s2/", "MTL_32/"],
        ["missing_smallscale_mtl_experiments/s0/", "MTL_32/"],
        ["smallscale_mtl_experiments/s1/", "MTL_32/"],
        ["smallscale_mtl_experiments/s2/", "MTL_32/"],
        ["holdout_comp_experiments/s0/", "MTL_56/"],
        ["holdout_comp_experiments/s1/", "MTL_56/"],
        ["holdout_comp_experiments/s2/", "MTL_56/"],
        ["holdout_mtl_experiments/s0/", "MTL_56/"],
        ["holdout_mtl_experiments/s1/", "MTL_56/"],
        ["holdout_mtl_experiments/s2/", "MTL_56/"],
    ])

    learner_types = np.array([
        'Comp', 'Comp', 'Comp',
        'MTL', 'MTL', 'MTL',
        'Comp', 'Comp', 'Comp',
        'MTL', 'MTL', 'MTL',
        'Comp', 'Comp', 'Comp',
        'MTL', 'MTL', 'MTL'
    ])

    experiment_methods = np.array([
        'default', 'default', 'default',
        'default', 'default', 'default',
        'smallscale', 'smallscale', 'smallscale',
        'smallscale', 'smallscale', 'smallscale',
        'holdout', 'holdout', 'holdout',
        'holdout', 'holdout', 'holdout'
    ])

    num_tasks_list = np.array([
        56, 56, 56, 
        56, 56, 56, 
        32, 32, 32, 
        32, 32, 32,
        56, 56, 56, 
        56, 56, 56
    ])

    base_path = '.'
    save_path = directories[args.model_id][0]
    load_dir = directories[args.model_id][1]
    learner = learner_types[args.model_id]
    experiment_method = experiment_methods[args.model_id]
    num_tasks = num_tasks_list[args.model_id]

    run_test(base_path=base_path, save_path=save_path, load_dir=load_dir, learner=learner, experiment_method=experiment_method, num_tasks=num_tasks)


if __name__ == '__main__':
    main()
