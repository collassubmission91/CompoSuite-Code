import os, sys, subprocess

def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable]
        args += ["-m", "spinningup_training.train_mtl_ppo"]
        for arg in sys.argv:
            if "spinningup_training" not in arg:
                args += [arg]

        subprocess.check_call(args, env=env)
        sys.exit()
