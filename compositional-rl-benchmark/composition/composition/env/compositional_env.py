import numpy as np

from composition.objects.objects import GoalWallObject, ObjectWallObject, ObjectDoorFrameObject
from composition.objects.objects import \
    DumbbellObject, DumbbellVisualObject, \
    PlateObject, PlateVisualObject, \
    HollowBoxObject, HollowBoxVisualObject, \
    CustomBoxObject, CustomBoxVisualObject

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from robosuite.models.tasks import ManipulationTask

from robosuite.models.grippers import GripperModel
from robosuite.models.base import MujocoModel

from collections import OrderedDict


class CompositionalEnv(SingleArmEnv):
    def __init__(
        self,
        robots,       # first compositional axis
        object_type,
        obstacle,
        bin1_pos,
        bin2_pos,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="RethinkGripper",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
        use_goal_obs=True,
        use_obstacle_obs=True,
        use_task_id_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        reward_scale=1.0,
        reward_shaping=False,
    ):
        if gripper_types != "RethinkGripper":
            raise ValueError(
                'We want to keep all robots using the same gripper because otherwise they have different observation spaces')
        # task includes arena, robot, and objects of interest

        self.step_counter = 0

        # task settings
        self.object_type = object_type
        self.obstacle_type = obstacle
        self.robot_name = robots
        self.obj_class_map = OrderedDict({
            "Box": (CustomBoxObject, CustomBoxVisualObject),
            "Dumbbell": (DumbbellObject, DumbbellVisualObject),
            "Plate": (PlateObject, PlateVisualObject),
            "Hollowbox": (HollowBoxObject, HollowBoxVisualObject)})
        self.obstacle_class_map = OrderedDict({
            None: None,
            "ObjectWall": ObjectWallObject,
            "ObjectDoor": ObjectDoorFrameObject,
            "GoalWall": GoalWallObject})
        self.robot_list = ["IIWA", "Jaco", "Panda", "Kinova3"]

        # Needed for Robosuite gym wrapper
        self.use_object_obs = use_object_obs
        self.use_goal_obs = use_goal_obs
        self.use_obstacle_obs = use_obstacle_obs
        self.use_task_id_obs = use_task_id_obs

        # whether to use ground-truth object states
        self.object_sensor_names = None

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # settings for bin position
        self.bin1_pos = bin1_pos
        self.bin2_pos = bin2_pos

        super().__init__(
            robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def staged_reward(self, action=None):
        raise NotImplementedError("Refer to task")

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # Arena always gets set to zero origin

    def _initialize_model(self):
        self.bin1_pos = self.mujoco_arena.bin1_pos
        self.bin2_pos = self.mujoco_arena.bin2_pos

        self.bin1_size = self.mujoco_arena.bin1_size
        self.bin2_size = self.mujoco_arena.bin2_size

        self.mujoco_arena.set_origin([0, 0, 0])

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["bins"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        self.object = self.obj_class_map[self.object_type][0](
            self.object_type.title())
        self.visual_object = self.obj_class_map[self.object_type][1](
            'Visual' + self.object_type.title())
        mujoco_objects = [self.visual_object, self.object]

        if self.obstacle_type is not None:
            self.obstacle = self.obstacle_class_map[self.obstacle_type](
                'Obstacle' + self.obstacle_type)
            mujoco_objects.append(self.obstacle)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[
                robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _setup_references(self):
        super()._setup_references()

        # object-specific ids
        self.obj_body_id = self.sim.model.body_name2id(
            self.object.root_body)
        self.visual_obj_body_id = self.sim.model.body_name2id(
            self.visual_object.root_body)
        # required for setting up observations in parent task
        self.goal_body_id = self.visual_obj_body_id

        if self.obstacle_type is not None:
            self.obstacle_body_id = self.sim.model.body_name2id(
                self.obstacle.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.visual_obj_body_id] = obj_pos
                    self.sim.model.body_quat[self.visual_obj_body_id] = obj_quat
                elif "obstacle" in obj.name.lower():
                    # Set the obstacle body locations
                    self.sim.model.body_pos[self.obstacle_body_id] = obj_pos
                    self.sim.model.body_quat[self.obstacle_body_id] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(
                        obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Make sure to update sensors' active and enabled states
        if self.object_sensor_names is not None:
            for name in self.object_sensor_names:
                # Set all of these sensors to be enabled and active if this is the active object, else False
                self._observables[name].set_enabled(True)
                self._observables[name].set_active(True)

    def _setup_observables(self):
        observables = super()._setup_observables()
        # return self.task._setup_observables(observables)
        pf = self.robots[0].robot_model.naming_prefix

        # for conversion to relative gripper frame
        @sensor(modality="ref")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        enableds = [True]
        actives = [False]

        if self.use_task_id_obs:
            modality = "task"

            @sensor(modality=modality)
            def subtask_id(obs_cache):
                onehot = np.zeros(4)
                onehot[self.subtask_id] = 1
                return onehot

            @sensor(modality=modality)
            def object_id(obs_cache):
                onehot = np.zeros(len(self.obj_class_map))
                obj_id = list(self.obj_class_map.keys()
                              ).index(self.object_type)
                onehot[obj_id] = 1
                return onehot

            @sensor(modality=modality)
            def obstacle_id(obs_cache):
                onehot = np.zeros(len(self.obstacle_class_map))
                obstacle_id = list(self.obstacle_class_map.keys()).index(
                    self.obstacle_type)
                onehot[obstacle_id] = 1
                return onehot

            @sensor(modality=modality)
            def robot_id(obs_cache):
                onehot = np.zeros(len(self.robot_list))
                robot_id = self.robot_list.index(self.robot_name)
                onehot[robot_id] = 1
                return onehot

            sensors += [subtask_id, object_id, obstacle_id, robot_id]
            names += ['subtask_id', 'object_id',
                      'obstacle_id', 'robot_id']

            enableds += [True] * 4
            actives += [True] * 4

        if self.use_object_obs:
            # Get robot prefix and define observables modality
            modality = "object"

            # object-related observables
            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj_body_id])

            @sensor(modality=modality)
            def obj_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id]), to="xyzw")

            @sensor(modality=modality)
            def obj_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["obj_pos", "obj_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                obj_pose = T.pose2mat(
                    (obs_cache["obj_pos"], obs_cache["obj_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(
                    obj_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"obj_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def obj_to_eef_quat(obs_cache):
                return obs_cache[f"obj_to_{pf}eef_quat"] if \
                    f"obj_to_{pf}eef_quat" in obs_cache else np.zeros(4)

            sensors += [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
            names += ['obj_pos', 'obj_quat',
                      'obj_to_eef_pos', 'obj_to_eef_quat']

            enableds += [True] * 4
            actives += [True] * 4

        if self.use_goal_obs:
            modality = "goal"

            # goal-related observables
            @sensor(modality=modality)
            def goal_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.goal_body_id])

            @sensor(modality=modality)
            def goal_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.goal_body_id]), to="xyzw")

            @sensor(modality=modality)
            def goal_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["goal_pos", "goal_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                goal_pose = T.pose2mat(
                    (obs_cache["goal_pos"], obs_cache["goal_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(
                    goal_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"goal_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def goal_to_eef_quat(obs_cache):
                return obs_cache[f"goal_to_{pf}eef_quat"] if \
                    f"goal_to_{pf}eef_quat" in obs_cache else np.zeros(4)

            # in principle, like other things, we could add the quat as well
            @sensor(modality=modality)
            def obj_to_goal(obs_cache):
                return obs_cache["goal_pos"] - obs_cache["obj_pos"] if \
                    "obj_pos" in obs_cache and "goal_pos" in obs_cache else np.zeros(3)

            sensors += [goal_pos, goal_quat, goal_to_eef_pos,
                        goal_to_eef_quat, obj_to_goal]
            names += ['goal_pos', 'goal_quat', 'goal_to_eef_pos',
                      'goal_to_eef_quat', 'obj_to_goal']

            enableds += [True] * 5
            actives += [True] * 5

        if self.use_obstacle_obs:
            modality = "obstacle"

            # goal-related observables
            @sensor(modality=modality)
            def obstacle_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obstacle_body_id]) \
                    if self.obstacle_type is not None else np.zeros(3)

            @sensor(modality=modality)
            def obstacle_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obstacle_body_id]), to="xyzw") \
                    if self.obstacle_type is not None else np.zeros(4)

            @sensor(modality=modality)
            def obstacle_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any([name not in obs_cache for name in
                        ["obstacle_pos", "obstacle_quat", "world_pose_in_gripper"]]):
                    return np.zeros(3)
                obstacle_pose = T.pose2mat(
                    (obs_cache["obstacle_pos"], obs_cache["obstacle_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(
                    obstacle_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"obstacle_to_{pf}eef_quat"] = rel_quat
                return rel_pos

            @sensor(modality=modality)
            def obstacle_to_eef_quat(obs_cache):
                return obs_cache[f"obstacle_to_{pf}eef_quat"] if \
                    f"obstacle_to_{pf}eef_quat" in obs_cache else np.zeros(4)

            sensors += [obstacle_pos, obstacle_quat,
                        obstacle_to_eef_pos, obstacle_to_eef_quat]
            names += ['obstacle_pos', 'obstacle_quat',
                      'obstacle_to_eef_pos', 'obstacle_to_eef_quat']

            enableds += [True] * 4
            actives += [True] * 4

        # Create observables
        for name, s, enabled, active in zip(names, sensors, enableds, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                enabled=enabled,
                active=active
            )
        return observables

    def _check_success(self):
        raise NotImplementedError('Thought this wasnt needed')

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler")

        # Add obstacle
        if self.obstacle_type is not None:
            if 'object' in self.obstacle_type.lower():
                # Place obstacle a quarter of the way into the table, leaving ~3/4 of the table to place the object
                obstacle_pos = [-self.bin1_size[0] /
                                2.2, 0, 0] + self.bin1_pos
                rotation = np.pi / 2

                # sample anywhere beyond the obstacle
                obj_x_min = 0
                obj_x_max = self.bin1_size[0] / 2
                obj_y_min = -(self.bin1_size[1] / 2)
                obj_y_max = -obj_y_min

            elif 'goal' in self.obstacle_type.lower():
                obstacle_pos = [
                    0, self.bin1_size[1], 0] + self.bin1_pos
                rotation = 0.
                obj_x_min = -(self.bin1_size[0] / 2)
                obj_x_max = -obj_x_min
                obj_y_min = -(self.bin1_size[1] / 2)
                obj_y_max = -obj_y_min
            else:
                raise ValueError(
                    'Obstacle must be placed between robot and object for object obstacles.')

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="CollisionObstacleSampler",
                    mujoco_objects=self.obstacle,
                    rotation=rotation,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=obstacle_pos
                )
            )
        else:
            # can sample anywhere in bin
            obj_x_min = -(self.bin1_size[0] / 2)
            obj_x_max = -obj_x_min
            obj_y_min = -(self.bin1_size[1] / 2)
            obj_y_max = -obj_y_min

        # each object should just be sampled in the bounds of the shelf (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.object,
                x_range=[obj_x_min, obj_x_max],
                y_range=[obj_y_min, obj_y_max],
                rotation=0,
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0.02,
            )
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

          - a discrete reward of 1.0 if the object is placed in the shelf

        Un-normalized components if using reward shaping, where the maximum is returned if not solved:

          - Reaching: in [0, 0.1], proportional to the distance between the gripper and the closest object
          - Grasping: in {0, 0.35}, nonzero if the gripper is grasping an object
          - Lifting: in {0, [0.35, 0.5]}, nonzero only if object is grasped; proportional to lifting height
          - Hovering: in {0, [0.5, 0.7]}, nonzero only if object is lifted; proportional to distance from object to shelf

        Note that a successfully completed task (object in shelf) will return 1.0 irregardless of whether the
        environment is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 4.0 (or 1.0 if only a single object is
        being used) as well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        # compute sparse rewards
        reward = float(self._check_success())
        # add in shaped re  wards
        if (reward < 1 or (reward < 1 and reward > 0)) and self.reward_shaping:
            staged_rewards = self.staged_rewards(action)
            reward += max(staged_rewards)
            self.step_counter += 1
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.object.root_body,
                target_type="body",
            )

    def _check_grasp(self, gripper, object_geoms):
        if self.object_type.title() == "Plate":
            return self._check_plate_grasp(gripper, object_geoms)
        return super()._check_grasp(gripper, object_geoms)

    def _check_plate_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.

        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.

        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.

        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(
                object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            right_g_geoms = [gripper.important_geoms["right_finger"],
                             gripper.important_geoms["left_fingerpad"]]
            left_g_geoms = [gripper.important_geoms["left_finger"],
                            gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            right_g_geoms = [[gripper]]
            left_g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            right_g_geoms = [[g_group] if type(
                g_group) is str else g_group for g_group in gripper]
            left_g_geoms = [[g_group] if type(
                g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for right_g_group in right_g_geoms:
            if not self.check_contact(right_g_group, o_geoms):
                for left_g_group in left_g_geoms:
                    if not self.check_contact(left_g_group, o_geoms):
                        return False
        return True
