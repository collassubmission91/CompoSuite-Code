from spinup.utils.test_policy import run_policy, load_policy_and_env
import argparse
from torch import load, from_numpy
import os

import composition
from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.compositional_core import CompositionalMLPActorCritic


def load_model(path, learner, single_task):
    env = composition.make(robot=single_task[0],
                           obj=single_task[2],
                           obstacle=single_task[3],
                           task=single_task[1],
                           use_task_id_obs=True, controller='joint')

    if learner == 'MTL':
        ac_kwargs = dict(hidden_sizes=[256]*2, log_std_init=0)
        policy = MLPActorCritic(
            env.observation_space, env.action_space, env.observation_positions, **ac_kwargs)
        state_dicts = load(os.path.join(
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
        state_dicts = load(os.path.join(
            path, 'pyt_save', 'state_dicts.pt'))
        policy.load_state_dict(state_dicts['model'])
    else:
        raise NotImplementedError

    return policy


def run_policy(env, policy, num_episodes, max_ep_len=500):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        a, _, _ = policy.step(from_numpy(o).float())
        o, r, d, _ = env.step(a)
        env.render()
        ep_ret += r
        ep_len += 1
        print(r)

        if d or (ep_len == max_ep_len):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--learner_type', type=str)
    parser.add_argument('--len', '-l', type=int, default=500)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')

    parser.add_argument('--controller', type=str, default="joint")
    parser.add_argument('--robot', type=str, default="IIWA")
    parser.add_argument('--object', type=str, default="hollowbox")
    parser.add_argument('--obstacle', type=str, default=None)
    parser.add_argument('--task', type=str, default="PickPlace")
    parser.add_argument('--horizon', type=int, default=250)

    args = parser.parse_args()

    policy = load_model(args.path, args.learner_type,
                        [args.robot, args.task, args.object, args.obstacle])

    env = composition.make(
        args.robot, args.object, args.obstacle, args.task, args.controller, args.horizon, has_renderer=True, use_task_id_obs=True)

    run_policy(env, policy, args.episodes, args.len)
