# Anonymous Submission #91 to Conference on Lifelong Learning Agents

## Installation

```
conda create --name compositional_env python=3.7 pip

cd spinningup
pip install -e .
cd ..

git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
cd ..

conda install -c anaconda h5py 
```

## Teleoperating Environments

```
cd compositional-rl-benchmark/
python -m composition.demos.demo_compositional_teleop
```

## Running experiments

All of the experiments are started from the compositional-rl-benchmark directory. 

```
cd compositional-rl-benchmark/
```

### Single-Task Experiments

To run a single-task experiment with 16 parallel processes a version of MPI is required.
If you do not have 16 slots in your machine, reduce the number of processes.

To run a specific task, pass in the axes keys that you want. If no keys are passed, 
the default configuration is used.

```
mpirun -np 16 python -m spinningup_training.train_ppo --robot <RobotKey> --object <ObjectKey> --obstacle <ObstacleKey> --task <ObjectiveKey>
```

### Multi-Task and Compositional Experiments

There is three different scripts for the different experiment types reported in the paper.
For the full benchmark setting run (num_tasks=56 or 224 in paper):

MTL:
```
mpirun -np <num_tasks> -m spinningup_training.train_mtl_ppo --exp-name <name> --steps <16000 * num_tasks>
```
Comp:
```
mpirun -np <num_tasks> -m spinningup_training.train_comp_ppo --exp-name <name> --steps <16000 * num_tasks>
```

For the smallscale setting use the axis (num_tasks=32 in paper):


MTL:
```
mpirun -np <num_tasks> -m spinningup_training.train_mtl_ppo_smallscale --exp-name <name> --steps <16000 * num_tasks>
```
Comp:
```
mpirun -np <num_tasks> -m spinningup_training.train_comp_ppo_smallscale --exp-name <name> --steps <16000 * num_tasks>
```


For the holdout setting with a single fixed PickPlace task run (num_tasks=56 in paper):

MTL:
```
mpirun -np <num_tasks> -m spinningup_training.train_mtl_ppo_holdout --exp-name <name> --steps <16000 * num_tasks>
```
Comp:
```
mpirun -np <num_tasks> -m spinningup_training.train_comp_ppo_holdout --exp-name <name> --steps <16000 * num_tasks>
```
