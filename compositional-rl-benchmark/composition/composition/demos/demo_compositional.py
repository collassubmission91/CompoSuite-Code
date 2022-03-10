import composition
from composition.utils.demo_utils import *
from robosuite.utils.input_utils import *

if __name__ == "__main__":
    robot = choose_robots(exclude_bimanual=True)
    task = choose_task()
    obj = choose_object().lower()
    obstacle = choose_obstacle()
    if obstacle == "None": obstacle = None

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = composition.make(
        robot=robot,
        obj=obj,
        obstacle=obstacle,
        task=task,
        has_renderer=True,
        ignore_done=True,
    )
    env.reset()
    env.viewer.set_camera(camera_id=1)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        # for k, v in obs.items():
        #     print(k, v.shape)
        # exit()
        env.render()
