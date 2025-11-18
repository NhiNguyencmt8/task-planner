import magnum as mn
import numpy as np
import torch

from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill


class HumanAwareNavSkill(OracleNavSkill):
    """
    - Reuses OracleNavSkill for:
        * target entity lookup
        * target pose sampling (embodied_unoccluded_navmesh_snap)
        * navmesh shortest paths
    - Adds:
        * human 3D centroid estimation from segmentation + depth
        * velocity estimation
        * belief tracking (mean + covariance) when human leaves FOV
        * predicted 'human corridor' over short horizon
        * safety filter / velocity shaping to avoid that corridor
    """

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env,
        agent_uid,
    ):
        super().__init__(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            batch_size=batch_size,
            env=env,
            agent_uid=agent_uid,
        )

        # === Parameters for human tracking & prediction =======================
        # 2D (ground-plane) mean and covariance of human position
        self.human_mu_2d = None  # np.array([x, z])
        self.human_Sigma_2d = None  # 2x2 covariance
        self.human_vel_2d = np.zeros(2)  # np.array([vx, vz])

        self.last_seen_step = -1
        self.step_count = 0

        # Fixed dt for propagation (you can replace with actual sim dt if available)
        self.dt = 1.0 / float(self.sim_freq)

        # Process noise for belief propagation
        self.Q = np.diag([0.05, 0.05])  # tweak as needed

        # Corridor prediction horizon (in steps)
        self.corridor_horizon = getattr(config, "corridor_horizon", 10)

        # Base corridor radius around predicted human positions
        self.base_corridor_radius = getattr(config, "base_corridor_radius", 0.6)

        # How fast corridor radius grows with uncertainty
        self.uncertainty_scale = getattr(config, "uncertainty_scale", 1.0)

        # When belief gets too uncertain, we fade back toward oracle-only behavior
        self.max_uncertainty_trace = getattr(config, "max_uncertainty_trace", 4.0)

        # Safety margin for how close robot can approach corridor
        self.safe_margin = getattr(config, "safe_margin", 0.2)

        # Depth camera intrinsics for head_depth / humanoid_detector_sensor
        # You should set these from your config or environment.
        agent_depth_head_sensor = env.env.env.env.unwrapped._env.sim._sensors[
            f"agent_{agent_uid}_head_depth"
        ]
        sensor_state = agent_depth_head_sensor._sensor_object
        hfov = sensor_state.hfov  # horizontal field-of-view, radians
        depth_image_shape = observation_space[f"agent_{agent_uid}_head_depth"].shape
        width = depth_image_shape[0]
        height = depth_image_shape[1]
        fx = width / (2 * np.tan(np.deg2rad(float(hfov)) / 2))
        self.fx = fx
        self.fy = fx
        self.cx = width / 2
        self.cy = height / 2
        # agent_rotation = sim.get_agent(0)._sensors["color_sensor"].node.absolute_transformation()

        # Minimum number of human pixels in segmentation to treat as "visible"
        self.min_human_pixels = getattr(config, "min_human_pixels", 100)

        # Maximum steps since last seen where we still trust prediction
        self.max_invisible_steps = getattr(config, "max_invisible_steps", 150)

    # ======================================================================
    #                         OVERRIDES
    # ======================================================================
    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        # reset human belief
        self.human_mu_2d = None
        self.human_Sigma_2d = None
        self.human_vel_2d = np.zeros(2)
        self.last_seen_step = -1
        self.step_count = 0

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic: bool = False,
    ):
        """
        Wrap OracleNavSkill's _internal_act and then apply a human-aware
        safety filter on the resulting velocities.
        """

        # First, let OracleNavSkill compute nominal action
        action, termination_message = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # If navigation already failed or teleported, do nothing
        if self.failed or self.do_teleport:
            return action, termination_message

        # Update time step
        self.step_count += 1

        # 1) Update human belief using current observations
        self.update_human_belief_from_obs(observations)

        # 2) Build predicted human corridor (if any belief exists)
        corridor = self.build_predicted_human_corridor()

        # 3) If we have a corridor, apply safety filter to velocities
        if corridor is not None:
            lin_idx = self.linear_velocity_index
            ang_idx = self.angular_velocity_index

            # Extract nominal velocity from OracleNav
            v0 = float(action[cur_batch_idx, lin_idx].cpu().numpy())
            w0 = float(action[cur_batch_idx, ang_idx].cpu().numpy())
            nominal_vel = np.array([v0, w0], dtype=np.float32)

            # Current robot pose
            robot_pos = np.array(self.articulated_agent.base_pos)
            base_T = self.articulated_agent.base_transformation

            safe_vel = self.apply_corridor_safety_filter(
                nominal_vel=nominal_vel,
                robot_pos=robot_pos,
                base_T=base_T,
                corridor=corridor,
            )

            # Write back into action tensor
            action[cur_batch_idx, lin_idx] = safe_vel[0]
            action[cur_batch_idx, ang_idx] = safe_vel[1]

        return action, termination_message

    # ======================================================================
    #                  HUMAN BELIEF: UPDATE & PREDICTION
    # ======================================================================
    def update_human_belief_from_obs(self, observations) -> bool:
        """
        Update human mean, velocity, and covariance given current observation.
        If human not visible, propagate belief forward using motion model.

        Returns:
            human_visible (bool): whether human was segmented in this frame.
        """
        # Expect segmentation in observations["humanoid_detector_sensor"]
        # and depth in observations["head_depth"].
        if (
            "humanoid_detector_sensor" not in observations
            or "head_depth" not in observations
        ):
            # Nothing we can do; keep previous belief if any.
            self.propagate_belief_if_invisible()
            return False

        seg = observations["humanoid_detector_sensor"].squeeze()  # shape (H, W)
        depth = observations["head_depth"].squeeze()  # shape (H, W)

        # Convert to numpy if torch
        seg_np = seg.cpu().numpy() if isinstance(seg, torch.Tensor) else np.array(seg)

        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy()
        else:
            depth_np = np.array(depth)

        # Assume human is non-zero label in segmentation
        human_mask = seg_np > 0
        num_pts = int(human_mask.sum())

        if num_pts < self.min_human_pixels:
            # Human not reliably visible; propagate belief if we have one
            self.propagate_belief_if_invisible()
            return False

        # Human visible: compute 3D points for masked pixels
        ys, xs = np.where(human_mask)
        zs = depth_np[ys, xs]  # depth values

        # Filter out invalid depth
        valid = zs > 0
        if valid.sum() < self.min_human_pixels:
            self.propagate_belief_if_invisible()
            return False

        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        # Backproject depth to camera coordinates (assumes pinhole model)
        # x_cam = (u - cx) * z / fx
        # y_cam = (v - cy) * z / fy
        x_cam = (xs - self.cx) * zs / self.fx
        y_cam = (ys - self.cy) * zs / self.fy
        z_cam = zs

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to world coordinates using the camera extrinsics
        # For simplicity, we use the agent's head transformation if available
        # Here we approximate with base_transformation + an offset.
        # TODO: replace with exact camera pose from Habitat if you have it.
        head_T = self.articulated_agent.base_transformation
        pts_world = self.transform_points(head_T, pts_cam)

        # Project to ground plane (x,z)
        pts_xz = pts_world[:, [0, 2]]  # 2D points

        # Compute centroid and covariance
        mu_2d = pts_xz.mean(axis=0)
        centered = pts_xz - mu_2d
        if centered.shape[0] > 1:
            Sigma_2d = centered.T @ centered / float(centered.shape[0] - 1)
        else:
            Sigma_2d = np.eye(2) * 0.1

        # Estimate velocity from previous mean (if available)
        if self.human_mu_2d is not None:
            dt = max(self.dt, 1e-3)
            self.human_vel_2d = (mu_2d - self.human_mu_2d) / dt
        else:
            self.human_vel_2d = np.zeros(2)

        self.human_mu_2d = mu_2d
        # observation covariance should be relatively small
        self.human_Sigma_2d = Sigma_2d + 1e-3 * np.eye(2)
        self.last_seen_step = self.step_count

        return True

    def propagate_belief_if_invisible(self):
        """
        When human is not visible, propagate the belief forward in time using
        a constant-velocity model with growing covariance. If the belief is
        too old or too uncertain, we eventually drop it.
        """
        if self.human_mu_2d is None or self.human_Sigma_2d is None:
            return  # nothing to propagate

        # If it's been too long since we saw the human, gradually forget them
        invisible_steps = self.step_count - self.last_seen_step
        if invisible_steps > self.max_invisible_steps:
            # Drop belief entirely
            self.human_mu_2d = None
            self.human_Sigma_2d = None
            self.human_vel_2d = np.zeros(2)
            return

        # Constant-velocity propagation
        dt = self.dt
        self.human_mu_2d = self.human_mu_2d + dt * self.human_vel_2d
        self.human_Sigma_2d = self.human_Sigma_2d + self.Q

    # ======================================================================
    #                 PREDICTED HUMAN CORRIDOR CONSTRUCTION
    # ======================================================================
    def build_predicted_human_corridor(self):
        """
        Build a short-horizon predicted human motion corridor based on
        human_mu_2d, human_vel_2d, and human_Sigma_2d.

        Returns:
            corridor: list of (center_2d, radius) tuples, or None if no belief.
        """
        if self.human_mu_2d is None or self.human_Sigma_2d is None:
            return None

        # If uncertainty too large, corridor is not very informative; skip
        if np.trace(self.human_Sigma_2d) > self.max_uncertainty_trace:
            return None

        corridor = []
        mu = self.human_mu_2d.copy()
        Sigma = self.human_Sigma_2d.copy()

        for k in range(1, self.corridor_horizon + 1):
            dt_k = k * self.dt
            # predicted mean
            mu_k = mu + dt_k * self.human_vel_2d
            # predicted covariance
            Sigma_k = Sigma + dt_k * self.Q

            # radius grows with uncertainty
            eigvals = np.linalg.eigvals(Sigma_k)
            max_eig = float(np.max(eigvals.real))
            r_k = self.base_corridor_radius + self.uncertainty_scale * np.sqrt(
                max(max_eig, 0.0)
            )

            corridor.append((mu_k, r_k))

        return corridor

    # ======================================================================
    #                  SAFETY FILTER / VELOCITY SHAPING
    # ======================================================================
    def apply_corridor_safety_filter(self, nominal_vel, robot_pos, base_T, corridor):
        """
        Given the nominal velocity from OracleNavSkill and a predicted human
        corridor, adjust the velocity to reduce interference.

        Args:
            nominal_vel: np.array([v, w])
            robot_pos: np.array([x,y,z])
            base_T: mn.Matrix4
            corridor: list of (center_2d, radius)

        Returns:
            safe_vel: np.array([v_safe, w_safe])
        """

        v0, w0 = nominal_vel
        safe_vel = np.array([v0, w0], dtype=np.float32)

        # Robot position projected to 2D
        robot_xz = robot_pos[[0, 2]]

        # Find minimum signed distance to corridor
        min_signed_dist = np.inf
        closest_center = None
        for center_2d, radius in corridor:
            d = np.linalg.norm(robot_xz - center_2d) - radius
            if d < min_signed_dist:
                min_signed_dist = d
                closest_center = center_2d

        # If robot is comfortably outside corridor + margin, keep nominal
        if min_signed_dist > self.safe_margin:
            return safe_vel

        # Otherwise, we are too close or inside predicted corridor.
        # Strategy:
        #   1) Slow down forward motion
        #   2) Turn slightly to move away from center of corridor

        # 1) Slow down based on how deep we are
        # min_signed_dist <= safe_margin <= 0 → clamp
        alpha = np.clip(
            (min_signed_dist + self.safe_margin) / self.safe_margin, 0.0, 1.0
        )
        safe_vel[0] = v0 * alpha  # scale forward speed

        # 2) Compute avoidance steering using gradient direction
        # Direction from corridor center to robot
        if closest_center is not None:
            dir_vec = robot_xz - closest_center
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-4:
                dir_unit = dir_vec / norm
                # Robot forward direction in xz-plane
                forward = np.array([1.0, 0.0, 0.0])
                robot_forward = np.array(base_T.transform_vector(forward))[[0, 2]]
                robot_forward /= np.linalg.norm(robot_forward) + 1e-6

                # If dir_unit aligns with robot_forward, go forward.
                # If dir_unit is perpendicular, adjust angular velocity.
                # Compute signed "cross" to know turn direction
                cross = robot_forward[0] * dir_unit[1] - robot_forward[1] * dir_unit[0]
                # cross > 0 → dir is to left → turn left, etc.
                turn_gain = 0.5
                safe_vel[1] = w0 + turn_gain * cross

        return safe_vel

    # ======================================================================
    #                      SMALL UTILITY FUNCTIONS
    # ======================================================================
    @staticmethod
    def transform_points(T: mn.Matrix4, pts_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from camera frame to world using a Magnum Matrix4.

        Args:
            T: mn.Matrix4
            pts_cam: (N, 3) array

        Returns:
            pts_world: (N, 3) array
        """
        # Convert to homogeneous coords
        N = pts_cam.shape[0]
        pts_h = np.concatenate([pts_cam, np.ones((N, 1))], axis=1)  # (N, 4)
        T_np = np.array(T)  # 4x4
        pts_w_h = pts_h @ T_np.T
        return pts_w_h[:, :3]
