import numpy as np
import argparse
import composition
import os
import json

import torch

from spinup.algos.pytorch.ppo.core import MLPActorCritic
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.run_utils import setup_logger_kwargs


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
    parser.add_argument('--ent-coef', type=float, default=0.)
    parser.add_argument('--log-std-init', type=float, default=0.)

    parser.add_argument('--controller', type=str, default="joint")
    parser.add_argument('--robot', type=str, default="IIWA")
    parser.add_argument('--object', type=str, default="Hollowbox")
    parser.add_argument('--obstacle', type=str, default=None)
    parser.add_argument('--task', type=str, default="PickPlace")
    parser.add_argument('--horizon', type=int, default=500)

    args = parser.parse_args()

    if args.gridsearch_id != -1:
        _lr = (5e-4,)
        _gamma = (0.99,)
        _iters = (32, 64)
        _tkl = (0.05,)
        _ent_coef = (0.01, 0.02)
        _log_std_init = (-0.5, 0)

        idx = np.unravel_index(args.gridsearch_id, (len(_lr), len(
            _gamma), len(_iters), len(_tkl), len(_ent_coef), len(_log_std_init)))

        args.pi_lr = _lr[idx[0]]
        args.vf_lr = _lr[idx[0]]
        args.gamma = _gamma[idx[1]]
        args.pi_iters = _iters[idx[2]]
        args.vf_iters = _iters[idx[2]]
        args.target_kl = _tkl[idx[3]]
        args.ent_coef = _ent_coef[idx[4]]
        args.log_std_init = _log_std_init[idx[5]]
    
    if args.task_id != -1:
        _robots = ["IIWA", "Jaco", "Kinova3", "Panda", "Sawyer", "UR5e"]
        _objects = ["bread", "dumbbell", "plate", "hollowbox"]
        _objectives = ["PickPlace", "Push", "Shelf", "Trashcan"]
        _obstacles = ["None", "GoalWall", "GoalDoor", "ObjectDoor", "ObjectWall"]

        idx = np.unravel_index(args.task_id, (len(_robots), len(_objects), len(_objectives), len(_obstacles)))
        args.robot = _robots[idx[0]]
        args.object = _objects[idx[1]]
        args.task = _objectives[idx[2]]
        args.obstacle = _obstacles[idx[3]]

    args.exp_name = "t:" + str(args.task_id) + "_name:" + args.exp_name + "_robot:" + str(args.robot) + "_task:" + str(args.task) + "_object:" + str(args.object) + "_obstacle:" + str(args.obstacle)

    return args


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    args = parse_args()
    
    logger_kwargs = setup_logger_kwargs(
        args.exp_name, data_dir=args.data_dir)

    os.makedirs(os.path.join(args.data_dir, args.exp_name), exist_ok=True)
    with open(os.path.join(args.data_dir, args.exp_name, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.robot == 'UR5e' or args.robot == 'Sawyer':
        exit()

    checkpoint = None
    if args.load_dir is not None:
        checkpoint = torch.load(os.path.join(args.load_dir, 'pyt_save', 'state_dicts.pt'))

    ppo(lambda: composition.make(
        args.robot, args.object, args.obstacle, args.task, args.controller, args.horizon), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, log_std_init=args.log_std_init), seed=args.seed, gamma=args.gamma, steps_per_epoch=args.steps, epochs=args.epochs, clip_ratio=args.clip,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.pi_iters, train_v_iters=args.vf_iters, target_kl=args.target_kl,
        logger_kwargs=logger_kwargs, max_ep_len=args.horizon, ent_coef=args.ent_coef, checkpoint=checkpoint)


if __name__ == '__main__':
    main()
