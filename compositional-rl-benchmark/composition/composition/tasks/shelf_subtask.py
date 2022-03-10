import numpy as np

from composition.arenas.shelf_arena import ShelfArena
from composition.env.compositional_env import CompositionalEnv
from composition.tasks.task_utils import dot_product_angle

import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import  UniformRandomSampler


class ShelfSubtask(CompositionalEnv):
    """
    This class corresponds to the pick place task for a single robot arm.

    Args:
        shelf_pos (3-tuple): Absolute cartesian coordinates of the shelf

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.

    Raises:
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        object_type,
        obstacle,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="RethinkGripper",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
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
        bin1_pos=(0.1, -0.26, 0.8),
        bin2_pos=(0.1, 0.18, 0.8),
        reward_scale=1.0,
        reward_shaping=False,
    ):

        self.subtask_id = 2
        super().__init__(
            robots,
            object_type,
            obstacle,
            bin1_pos,
            bin2_pos,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            use_task_id_obs=use_task_id_obs,
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
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
        )


    def staged_rewards(self, action):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.2
        grasp_mult = 0.3
        lift_mult = 0.5
        align_mult = 0.8
        approach_mult = 0.9

        # reaching reward governed by distance to closest object
        r_reach = 0.
        if not self.object_in_shelf:
            # get reaching reward via minimum distance to a target object
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.object.root_body,
                target_type="body",
                return_distance=True,
            )
            r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = int(self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.object.contact_geoms])
        ) * grasp_mult
        
        object_x_loc = self.sim.data.body_xpos[self.obj_body_id, 0]
        shelf_x_low = self.shelf_pos[0] - self.shelf_x_size 
        shelf_x_high = self.shelf_pos[0] + self.shelf_x_size 
        x_check = shelf_x_low < object_x_loc < shelf_x_high

        object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
        shelf_z_low = self.shelf_z_offset + self.shelf_pos[2]
        shelf_z_high = shelf_z_low + self.shelf_z_height
        z_check = shelf_z_low + self.mujoco_arena.shelf_thickness < object_z_loc < shelf_z_high

        object_infront_shelf = z_check and x_check
        
        # lifting reward for picking up an object
        r_lift = 0.
        if r_grasp > 0.:
            z_target = self.bin2_pos[2] + 0.25
            object_z_loc = self.sim.data.body_xpos[self.obj_body_id, 2]
            dist = np.abs(z_target - object_z_loc)
            r_lift = r_grasp + (1 - np.tanh(5.0 * dist)) * (
                lift_mult - grasp_mult
            )

        r_align = 0
        # align reward
        if object_infront_shelf:
            gripper_quat = T.convert_quat(self.sim.data.get_body_xquat(
                self.robots[0].robot_model.eef_name), to="xyzw")
            gripper_mat = T.quat2mat(gripper_quat)

            gripper_in_world = gripper_mat.dot(np.array([0, 0, 1]))
            target_orientation = np.array([0, 1, 0])
            orientation_dist = dot_product_angle(gripper_in_world, target_orientation)

            r_align = lift_mult + \
                (1 - np.tanh(orientation_dist)) * \
                (align_mult - lift_mult)

        r_approach = 0
        if object_infront_shelf and r_align > 0.6:
            # approach
            y_target = self.shelf_pos[1] - self.shelf_y_size
            y_dist = np.maximum(y_target - self.sim.data.body_xpos[self.obj_body_id, 1], 0.)
            r_approach = align_mult + \
                (1 - np.tanh(5.0 * y_dist)) * \
                (approach_mult - align_mult)

        return r_reach, r_grasp, r_lift, r_align, r_approach


    def not_in_shelf(self, obj_pos):
        # get position of shelf
        shelf_x_low = self.shelf_pos[0] - self.shelf_x_size 
        shelf_x_high = self.shelf_pos[0] + self.shelf_x_size 
        shelf_y_low = self.shelf_pos[1] - (self.shelf_y_size * 2) * 0.75
        shelf_y_high = self.shelf_pos[1]

        res = True
        if (
            shelf_x_low < obj_pos[0] < shelf_x_high
            and shelf_y_low < obj_pos[1] < shelf_y_high
            and self.shelf_z_offset + self.shelf_pos[2] < obj_pos[2] < self.shelf_z_offset + self.shelf_pos[2] + 0.15
        ):
            res = False
        return res

    def _get_placement_initializer(self):
        super()._get_placement_initializer()

        # get limits of shelf
        shelf_x_low = - self.shelf_x_size / 2
        shelf_x_high = self.shelf_x_size  / 2
        shelf_y_low = - self.shelf_y_size * 2
        shelf_y_high = - 0.025

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.visual_object.name}ObjectSampler",
                mujoco_objects=self.visual_object,
                x_range=[shelf_x_low, shelf_x_high],
                y_range=[shelf_y_low, shelf_y_high],
                rotation=0.,
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                z_offset=self.shelf_z_offset,
                reference_pos=self.shelf_pos,
            )
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # load model for table top workspace
        self.mujoco_arena = ShelfArena(
            bin1_pos=self.bin1_pos,
            bin2_pos=self.bin2_pos,
        )

        # Load model propagation
        super()._load_model()

        # Generate placement initializer
        self._initialize_model()
        self._get_placement_initializer()


    def _initialize_model(self):
        super()._initialize_model()
        self.shelf_pos = self.mujoco_arena.shelf_pos
        self.shelf_x_size = self.mujoco_arena.shelf_x_size 
        self.shelf_y_size = self.mujoco_arena.shelf_y_size
        self.shelf_z_offset = self.mujoco_arena.shelf_z_offset
        self.shelf_z_height =self.mujoco_arena.shelf_z_height
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        # keep track of if object is in the shelf
        self.object_in_shelf = False
        super()._setup_references()


    def _check_success(self):
        """
        Check if object has been successfully placed in the shelf.

        Returns:
            bool: True if object is placed correctly
        """
        # remember if object is in the shelf
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]
        self.object_in_shelf = not self.not_in_shelf(obj_pos)

        # returns True if the object is in the shelf
        return self.object_in_shelf
