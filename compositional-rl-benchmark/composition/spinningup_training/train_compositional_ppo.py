from email.policy import default
import numpy as np
import argparse
import composition
import os
import json

import torch

from spinup.algos.pytorch.ppo.compositional_core import CompositionalMLPActorCritic
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import proc_id, num_procs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='spinningup_training/logs')
    parser.add_argument('--load-dir', default=None)

    parser.add_argument('--gridsearch-id', type=int, default=-1)
    parser.add_argument('--task-id', type=int, default=-1)

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--epochs', type=int, default=625)
    parser.add_argument('--exp-name', type=str, default='ppo')
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--pi-lr', type=float, default=1e-4)
    parser.add_argument('--vf-lr', type=float, default=1e-4)
    parser.add_argument('--pi-iters', type=int, default=128)
    parser.add_argument('--vf-iters', type=int, default=128)
    parser.add_argument('--target-kl', type=float, default=0.02)
    parser.add_argument('--ent-coef', type=float, default=0.02)
    parser.add_argument('--log-std-init', type=float, default=0.)

    parser.add_argument('--controller', type=str, default="joint")
    parser.add_argument('--robot', type=str, default="IIWA")
    parser.add_argument('--object', type=str, default="Hollowbox")
    parser.add_argument('--obstacle', type=str, default=None)
    parser.add_argument('--task', type=str, default="PickPlace")
    parser.add_argument('--horizon', type=int, default=500)

    parser.add_argument('--large-net', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    task_list = np.random.choice(256, num_procs(), replace=False)

    args.task_id = int(task_list[proc_id()])
    _robots = ["IIWA", "Jaco", "Kinova3", "Panda"]
    _objects = ["Box", "Dumbbell", "Plate", "Hollowbox"]
    _objectives = ["PickPlace", "Push", "Shelf", "Trashcan"]
    _obstacles = ["None", "GoalWall", "ObjectDoor", "ObjectWall"]

    idx = np.unravel_index(args.task_id, (len(_robots), len(_objects), len(_objectives), len(_obstacles)))
    args.robot = _robots[idx[0]]
    args.object = _objects[idx[1]]
    args.task = _objectives[idx[2]]
    args.obstacle = _obstacles[idx[3]]

    # args.exp_name = "t:" + str(args.task_id) + "_name:" + args.exp_name + "_robot:" + str(args.robot) + "_task:" + str(args.task) + "_object:" + str(args.object) + "_obstacle:" + str(args.obstacle)
    args.exp_name = 'MTL_{}'.format(len(task_list))

    return args


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    args = parse_args()
    os.makedirs(os.path.join(args.data_dir, args.exp_name), exist_ok=True)
    with open(os.path.join(args.data_dir, args.exp_name, 'args_{}.json'.format(proc_id())), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger_kwargs = setup_logger_kwargs(
        args.exp_name, data_dir=args.data_dir)

    if args.large_net:
        hidden_sizes = ((64,), (64, 64), (128, 128, 128), (128, 128, 128))
    else:
        hidden_sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))

    ac_kwargs = {
        # 'hidden_sizes': [args.hid]*args.l, 
        'log_std_init': args.log_std_init,
        'hidden_sizes': hidden_sizes, 
        'module_names': ['obstacle_id', 'object_id', 'subtask_id', 'robot_id'],
        'module_input_names': ['obstacle-state', 'object-state', 'goal-state', 'robot0_proprio-state'],
        'interface_depths': [-1, 1, 2, 3] ,   
        'graph_structure': [[0], [1], [2], [3]],
    }

    checkpoint = None
    if args.load_dir is not None:
        checkpoint = torch.load(os.path.join(args.load_dir, 'pyt_save', 'state_dicts.pt'))

    ppo(lambda: composition.make(
        args.robot, args.object, args.obstacle, args.task, args.controller, args.horizon, use_task_id_obs=True), actor_critic=CompositionalMLPActorCritic,
        ac_kwargs=ac_kwargs, seed=args.seed, gamma=args.gamma, steps_per_epoch=args.steps, epochs=args.epochs, clip_ratio=args.clip,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.pi_iters, train_v_iters=args.vf_iters, target_kl=args.target_kl,
        logger_kwargs=logger_kwargs, max_ep_len=args.horizon, ent_coef=args.ent_coef, log_per_proc=True, checkpoint=checkpoint)


if __name__ == '__main__':
    main()


'''

obs = 14
obj = 14
goal = 17
robot = 32

task-id = 17

(small)
obs:    32 (14 + 1)                                                 = 480
obj:    32 (14 + 1) + 32 (64 + 1)                                   = 2560 
goal:   64 (17 + 1) + 64 (64 + 1)   + 64 (96 + 1)                   = 11520
robot:  64 (32 + 1) + 64 (64 + 1)   + 64 (64 + 1)   + 1 (128 + 1)   = 10561
                                                                    = 25121

(large)
obs:    64 (14 + 1)                                                         = 960
obj:    64 (14 + 1)     + 64 (128 + 1)                                      = 9216 
goal:   128 (17 + 1)    + 128 (128 + 1)   + 128 (192 + 1)                   = 43520
robot:  128 (32 + 1)    + 128 (128 + 1)   + 128 (128 + 1)   + 1 (256 + 1)   = 37505
                                                                            = 91201

original: 256 (94 + 1) + 256 (256 + 1) + 1 (256 + 1) = 90369
'''
