from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from composition.env.gym_wrapper import GymWrapper


AVAILABLE_ROBOTS = ["IIWA", "Jaco", "Kinova3", "Panda"]
AVAILABLE_OBSTACLES = [None, "None", "GoalWall",
                       "GoalDoor", "ObjectDoor", "ObjectWall"]
AVAILABLE_OBJECTS = ["Box", "Dumbbell", "Plate", "Hollowbox"]
AVAILABLE_TASKS = ["PickPlace", "Push", "Shelf", "Trashcan"]

def make(robot="IIWA", obj="milk", obstacle=None, task="PickPlace", controller="osc",
         env_horizon=500, has_renderer=False, has_offscreen_renderer=False, reward_shaping=True,
         ignore_done=False, use_camera_obs=False, control_freq=20, **kwargs) -> GymWrapper:
    """Creates a compositional environment

    Args:
        args (Namespace): Namespace arguments.

    Raises:
        NotImplementedError: Raises in case of controller is misspecified.

    Returns:
        GymWrapper: Environment wrapped into Gym interface.
    """

    assert robot in AVAILABLE_ROBOTS
    assert obstacle in AVAILABLE_OBSTACLES
    assert obj in AVAILABLE_OBJECTS
    assert task in AVAILABLE_TASKS

    if obstacle == "None":
        obstacle = None

    # defined options to create robosuite environment
    options = {}

    if task == "PickPlace":
        options["env_name"] = "PickPlaceSubtask"
    elif task == "Push":
        options["env_name"] = "PushSubtask"
    elif task == "Shelf":
        options["env_name"] = "ShelfSubtask"
    elif task == "Trashcan":
        options["env_name"] = "TrashcanSubtask"
    else:
        raise NotImplementedError

    options["robots"] = robot
    options["obstacle"] = obstacle
    options["object_type"] = obj

    if controller == "osc":
        controller_name = "OSC_POSITION"
    elif controller == "joint":
        controller_name = "JOINT_POSITION"
    elif controller == 'osc_pose':
        controller_name = 'OSC_POSE'
    else:
        print("Controller unknown")
        raise NotImplementedError

    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    env = suite.make(
        **options,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        reward_shaping=reward_shaping,
        ignore_done=ignore_done,
        use_camera_obs=use_camera_obs,
        control_freq=control_freq,
        horizon=env_horizon,
        **kwargs
    )

    env.reset()
    return GymWrapper(env)
