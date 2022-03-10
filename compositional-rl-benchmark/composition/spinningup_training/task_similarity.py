import argparse
import os
import numpy as np
from composition.env.main import make

import torch

from spinup import EpochLogger
from spinup.utils.test_policy import load_pytorch_policy
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import proc_id

from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.compositional_core import CompositionalMLPActorCritic


def overwrite_obs(o, wrong_task_id, observation_positions):
    subtask_id = observation_positions['subtask_id']
    object_id = observation_positions['object_id']
    obstacle_id = observation_positions['obstacle_id']
    robot_id = observation_positions['robot_id']

    o[robot_id] = 0
    o[robot_id[wrong_task_id[0]]] = 1
    o[subtask_id] = 0
    o[subtask_id[wrong_task_id[1]]] = 1
    o[object_id] = 0
    o[object_id[wrong_task_id[2]]] = 1
    o[obstacle_id] = 0
    if wrong_task_id[3] == 3:
        o[obstacle_id[wrong_task_id[3] + 1]] = 1
    else:
        o[obstacle_id[wrong_task_id[3]]] = 1


    return o


def run_policy(env, policy, num_episodes, max_ep_len=500, logger=None, wrong_task_id=-1):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    o = overwrite_obs(o, wrong_task_id=wrong_task_id,
                      observation_positions=env.observation_positions)
    success = 0
    steps_at_goal = 0
    while n < num_episodes:
        a, _, _ = policy.step(torch.from_numpy(o).float())
        o, r, d, _ = env.step(a)
        o = overwrite_obs(o, wrong_task_id=wrong_task_id,
                          observation_positions=env.observation_positions)

        ep_ret += r
        ep_len += 1
        if r == 1:
            success = 1
            steps_at_goal += 1

        if d or (ep_len == max_ep_len):
            if logger:
                logger.store(EpRet=ep_ret, EpLen=ep_len, Success=success, StepsAtGoal=steps_at_goal)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t Success %i' %
                  (n, ep_ret, ep_len, success))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            o = overwrite_obs(o, wrong_task_id=wrong_task_id,
                              observation_positions=env.observation_positions)
            n += 1
            success = 0
            steps_at_goal = 0

    if logger:
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('Success')
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('StepsAtGoal', average_only=True)
        logger.dump_tabular()


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


def run_test(base_path, save_path, load_dir, learner='MTL', wrong_task_id=-1, task=None):
    path = os.path.join(base_path, save_path, load_dir)
    policy = load_model(path, learner=learner, single_task=task)
    np.random.seed(proc_id())

    env = make(robot=task[0], obj=task[2], obstacle=task[3], task=task[1],
               has_offscreen_renderer=False, has_renderer=False,
               env_horizon=500, use_task_id_obs=True, controller='joint')

    logger_kwargs = setup_logger_kwargs(str(task) + '&' + str(wrong_task_id),
                                        data_dir=os.path.join(
                                            'spinningup_training/task_similarity_2',
                                        save_path))
    logger = EpochLogger(**logger_kwargs)

    run_policy(env, policy=policy, num_episodes=1, logger=logger,
               wrong_task_id=wrong_task_id)
    del(env)


def get_wrong_task_indicator(task):
    _robots = ["IIWA", "Jaco", "Panda", "Kinova3"]
    _objects = ["bread", "dumbbell", "plate", "hollowbox"]
    _objectives = ["PickPlace", "Push", "Shelf", "Trashcan"]
    _obstacles = ["None", "ObjectWall", "ObjectDoor", "GoalDoor"]

    idx = [
        _robots.index(task[0]),
        _objectives.index(task[1]),
        _objects.index(task[2]),
        _obstacles.index(task[3]),
    ]

    return idx


def get_env_config(task_id):
    _robots = ["IIWA", "Jaco", "Panda", "Kinova3"]
    _objects = ["bread", "dumbbell", "plate", "hollowbox"]
    _objectives = ["PickPlace", "Push", "Shelf", "Trashcan"]
    _obstacles = ["None", "ObjectWall", "ObjectDoor", "GoalDoor"]

    idx = np.unravel_index(task_id, (len(_robots), len(
        _objects), len(_objectives), len(_obstacles)))
    robot = _robots[idx[0]]
    obj = _objects[idx[1]]
    task = _objectives[idx[2]]
    obstacle = _obstacles[idx[3]]

    return robot, task, obj, obstacle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--array-id', default=0, type=int)
    args = parser.parse_args()

    indices = np.unravel_index(args.array_id, (3, 256))

    directories = [
        ["comp_experiments/256_tasks_rs_s0/", "MTL_56/"],
        ["comp_experiments/256_tasks_rs_s1/", "MTL_56/"],
        ["comp_experiments/256_tasks_rs_s2/", "MTL_56/"],
        ["mtl_experiments/56_tasks_rs_s0/", "MTL_56/"],
        ["mtl_experiments/56_tasks_rs_s1/", "MTL_56/"],
        ["mtl_experiments/56_tasks_rs_s2/", "MTL_56/"],
    ]
    learner_types = [
        'Comp', 'Comp', 'Comp',
        'MTL', 'MTL', 'MTL',
    ]

    base_path = '.'
    save_path = directories[indices[0]][0]
    load_dir = directories[indices[0]][1]
    learner = learner_types[indices[0]]

    task = get_env_config(task_id=indices[1])

    _robots = ["IIWA", "Jaco", "Panda", "Kinova3"]
    _objects = ["bread", "dumbbell", "plate", "hollowbox"]
    _objectives = ["PickPlace", "Push", "Shelf", "Trashcan"]
    _obstacles = ["None", "ObjectWall", "ObjectDoor", "GoalDoor"]

    for i, axis in enumerate([_robots, _objectives, _objects, _obstacles]):
        for j, elem in enumerate(axis):
            if elem == task[i]:
                continue
            
            wrong_task = list(task)
            wrong_task[i] = elem
            wrong_task_id = get_wrong_task_indicator(wrong_task)

            run_test(base_path=base_path, save_path=save_path,
                     load_dir=load_dir, learner=learner,
                     wrong_task_id=wrong_task_id, task=task)

    run_test(base_path=base_path, save_path=save_path,
            load_dir=load_dir, learner=learner,
            wrong_task_id=get_wrong_task_indicator(task), task=task)

if __name__ == '__main__':
    main()
