# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Tuple

import magnum as mn
import numpy as np
import torch
import habitat_sim
from habitat.tasks.utils import get_angle

from habitat.datasets.rearrange.navmesh_utils import (
    SimpleVelocityControlEnv,
    compute_turn,
    embodied_unoccluded_navmesh_snap,
)

from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.world_model.entity import Object, Receptacle, Room
from habitat_llm.world_model.entities.floor import Floor
from habitat_llm.world_model.entities.furniture import Furniture


class ReactiveNavSkill(SkillPolicy):
    """
    Navigation skill that behaves like OracleNav but adds a simple reactive
    avoidance controller for robot agents (non-humanoid). When a nearby
    human agent is detected within `avoidance_distance`, the robot will
    compute a short lateral waypoint around the human (an arc) and steer
    around them, then resume heading to the original navigation target.

    Config options (with defaults applied here):
      - dist_thresh (inherited)
      - turn_thresh
      - forward_velocity
      - turn_velocity
      - sim_freq
      - enable_backing_up
      - avoidance_distance (meters) default 1.0
      - avoidance_radius (meters) default 1.2
      - avoidance_release_distance (meters) default 1.4
    """

    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )
        self.env = env

        # Determine motion type as in OracleNav
        if f"agent_{self.agent_uid}_humanoidjoint_action" in action_space.spaces:
            self.motion_type = "human_joints"
        else:
            self.motion_type = "base_velocity"

        # Target pose computed in set_target
        self.target_base_pos: Optional[mn.Vector3] = None
        self.target_base_rot: Optional[float] = None
        self._has_reached_goal = torch.zeros(self._batch_size)

        # Velocity controller params
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.dist_thresh = config.dist_thresh
        self.turn_thresh = config.turn_thresh
        self.forward_velocity = config.forward_velocity
        self.turn_velocity = config.turn_velocity
        self.sim_freq = config.sim_freq

        self.enable_backing_up = config.enable_backing_up

        # Reactive avoidance parameters (with sensible defaults)
        self.avoidance_distance = getattr(config, "avoidance_distance", 1.0)
        self.avoidance_radius = getattr(config, "avoidance_radius", 1.2)
        self.avoidance_release_distance = getattr(
            config, "avoidance_release_distance", self.avoidance_radius + 0.2
        )

        # State for avoidance
        self.in_avoidance_mode = False

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[self.agent_uid].articulated_agent

        # Teleport handling copied from OracleNav (optionally configured)
        self.do_teleport = False
        if "teleport" in config:
            self.do_teleport = config.teleport

        if self.do_teleport:
            target_pos_ends = None
            try:
                from habitat_llm.agent.env.actions import find_action_range

                target_pos_ends = find_action_range(
                    self.action_space, f"agent_{self.agent_uid}_teleport"
                )
            except Exception:
                target_pos_ends = None

            if target_pos_ends is not None:
                self.target_pos_range = range(target_pos_ends[0], target_pos_ends[1])
        else:
            if self.motion_type != "human_joints":
                from habitat_llm.agent.env.actions import find_action_range

                self.action_range = find_action_range(
                    self.action_space, f"agent_{self.agent_uid}_base_velocity"
                )
            else:
                from habitat_llm.agent.env.actions import find_action_range

                self.action_range = find_action_range(
                    self.action_space, f"agent_{self.agent_uid}_humanoid_base_velocity"
                )

            self.linear_velocity_index = self.action_range[0]
            self.angular_velocity_index = self.action_range[1] - 1

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._has_reached_goal = torch.zeros(self._batch_size)
        self.target_is_set = False
        self.target_base_pos = None
        self.target_base_rot = None
        self.in_avoidance_mode = False
        return

    def get_state_description(self):
        room_node = self.env.world_graph[self.agent_uid].get_room_for_entity(
            f"agent_{self.agent_uid}"
        )
        return f"Walking in {room_node.name}"

    def _path_to_point(self, point):
        agent_pos = self.articulated_agent.base_pos
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self.env.sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def get_agent_object_ids(self) -> Tuple[list[int], list[int]]:
        agent_object_ids: list[int] = []
        other_agent_object_ids: list[int] = []
        for articulated_agent in self.env.sim.agents_mgr.articulated_agents_iter:
            agent_object_ids.extend(
                [articulated_agent.sim_obj.object_id]
                + [*articulated_agent.sim_obj.link_object_ids.keys()]
            )
        return agent_object_ids, other_agent_object_ids

    def _nearest_human(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return (position, distance) of nearest other agent in world (2D), or None.
        Uses base_pos of articulated agents. Excludes self."""
        my_pos = np.array(self.articulated_agent.base_pos)
        nearest_pos = None
        nearest_d = float("inf")
        for other in self.env.sim.agents_mgr.articulated_agents_iter:
            try:
                other_pos = np.array(other.base_pos)
            except Exception:
                continue
            # Skip self by identity (base_pos may be same, so also check object id)
            if np.allclose(other_pos, my_pos):
                continue
            d = np.linalg.norm((other_pos - my_pos)[[0, 2]])
            if d < nearest_d:
                nearest_d = d
                nearest_pos = other_pos

        if nearest_pos is None:
            return None
        return nearest_pos, nearest_d

    def set_target(self, target_name: str, env):
        if self.target_is_set:
            return

        entity = self.env.world_graph[self.agent_uid].get_node_from_name(target_name)

        self.target_handle = entity.sim_handle
        self.target_is_set = True

        target_object_ids = []
        agent_object_ids, other_agent_object_ids = self.get_agent_object_ids()

        self.target_pos = None
        furniture_parent_handle = None

        if isinstance(entity, Object):
            sim_obj = None
            try:
                from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle

                sim_obj = get_obj_from_handle(self.env.sim, self.target_handle)
            except Exception:
                sim_obj = None

            if sim_obj is not None:
                target_object_ids.append(sim_obj.object_id)

            self.target_pos = entity.get_property("translation")
            attempts = 0
            obj_to_nav_point_dist = 0
            success = False
            max_obj_to_nav_point_dist = 1.8
            while (obj_to_nav_point_dist > max_obj_to_nav_point_dist or not success) and attempts < 200:
                (
                    self.target_base_pos,
                    self.target_base_rot,
                    success,
                ) = embodied_unoccluded_navmesh_snap(
                    target_position=mn.Vector3(self.target_pos),
                    height=1.3,
                    sim=self.env.sim,
                    target_object_ids=target_object_ids,
                    ignore_object_ids=agent_object_ids,
                    ignore_object_collision_ids=other_agent_object_ids,
                    island_id=self.env.sim._largest_indoor_island_idx,
                    min_sample_dist=0.25,
                    agent_embodiment=self.articulated_agent,
                    orientation_noise=0.1,
                )
                if success:
                    obj_to_nav_point_dist = (
                        mn.Vector3(self.target_base_pos) - mn.Vector3(self.target_pos)
                    ).length()
                attempts += 1
            if success and obj_to_nav_point_dist <= max_obj_to_nav_point_dist:
                self.env.sim.dynamic_target = self.target_base_pos
                return

            # try finding parent furniture (optional)
            if self.env.sim.kinematic_relationship_manager is not None:
                try:
                    from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id

                    furniture_parent_id = self.env.sim.kinematic_relationship_manager.relationship_graph.obj_to_parents.get(
                        sim_obj.object_id, None
                    )
                    if furniture_parent_id is None:
                        self.termination_message = f"Could not find a suitable nav target or parent Furniture for Entity Object '{target_name}'. Possibly inaccessible."
                        self.failed = True
                        return
                    furniture_parent_obj = get_obj_from_id(self.env.sim, furniture_parent_id)
                    furniture_parent_handle = furniture_parent_obj.handle
                except Exception:
                    pass

        elif isinstance(entity, Receptacle):
            # sample center of receptacle
            try:
                from habitat_llm.utils.sim import get_receptacle_index

                hab_rec = self.env.perception.receptacles[
                    get_receptacle_index(entity.sim_handle, self.env.perception.receptacles)
                ]
                self.target_pos = hab_rec.get_global_transform(self.env.sim).transform_point(hab_rec.bounds.center())
                furniture_parent_handle = hab_rec.parent_object_handle
            except Exception:
                self.target_pos = entity.get_property("translation")

        elif isinstance(entity, (Floor, Room)):
            self.target_pos = entity.get_property("translation")
        elif isinstance(entity, Furniture):
            furniture_parent_handle = entity.sim_handle
            if entity.sim_handle is None:
                raise ValueError(
                    f"Target Furniture.sim_handle == None. '{target_name}' could be a Floor node?."
                )
        else:
            raise ValueError(
                f"ReactiveNav accepts Object, Furniture, Receptacle, Floor, Room. Got {entity.__class__.__name__}."
            )

        # fallback snap near furniture if needed
        furniture_parent_object_ids = []
        furniture_aabb = None
        parent_obj = None
        if furniture_parent_handle is not None:
            try:
                from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_handle

                parent_obj = get_obj_from_handle(self.env.sim, furniture_parent_handle)
                if parent_obj is None:
                    raise ValueError(
                        f"Could not find parent object with sim handle '{furniture_parent_handle}'."
                    )
                if self.target_pos is None:
                    from habitat.sims.habitat_simulator.sim_utilities import get_global_keypoints_from_object_id

                    self.target_pos = get_global_keypoints_from_object_id(self.env.sim, parent_obj.object_id)[0]
                    furniture_aabb = parent_obj.aabb
                furniture_parent_object_ids = [parent_obj.object_id]
                if isinstance(parent_obj, habitat_sim.physics.ManagedArticulatedObject):
                    furniture_parent_object_ids.extend([*parent_obj.link_object_ids.keys()])
            except Exception:
                pass

        global_points = [self.target_pos]
        if parent_obj is not None and parent_obj.marker_sets.has_taskset("interaction_surface_points"):
            try:
                interaction_points = parent_obj.marker_sets.get_task_link_markerset_points(
                    "interaction_surface_points", "body", "primary"
                )
                global_points.extend(parent_obj.transform_local_pts_to_world(interaction_points, link_id=-1))
            except Exception:
                pass

        attempts = 0
        max_fur_to_nav_point_dist = 1.8
        fur_to_nav_point_dist = 0
        success = False
        while (not success or fur_to_nav_point_dist > max_fur_to_nav_point_dist) and attempts < 200:
            target_variation = mn.Vector3()
            target_position = self.target_pos
            if attempts < len(global_points):
                target_position = global_points[attempts]
            elif furniture_aabb is not None:
                gaus_samp = (np.random.normal(0, 1.0, size=(2, 1)) / 3.5)
                target_variation = mn.Vector3(gaus_samp[0], 0, gaus_samp[1]) * parent_obj.transformation.transform_vector(
                    furniture_aabb.size() / 2.0
                )
                target_position = mn.Vector3(self.target_pos) + target_variation

            (
                self.target_base_pos,
                self.target_base_rot,
                success,
            ) = embodied_unoccluded_navmesh_snap(
                target_position=mn.Vector3(target_position),
                height=1.3,
                sim=self.env.sim,
                target_object_ids=furniture_parent_object_ids,
                ignore_object_ids=agent_object_ids,
                ignore_object_collision_ids=[],
                island_id=self.env.sim._largest_indoor_island_idx,
                min_sample_dist=0.25,
                agent_embodiment=self.articulated_agent,
                orientation_noise=0.1,
            )
            if success:
                fur_to_nav_point_dist = (mn.Vector3(self.target_base_pos) - mn.Vector3(target_position)).length()
            attempts += 1

        if success and fur_to_nav_point_dist <= max_fur_to_nav_point_dist:
            self.env.sim.dynamic_target = self.target_base_pos
            return

        self.termination_message = f"Could not find a suitable nav target for {target_name}. Possibly inaccessible."
        self.failed = True

    def rotation_collision_check(self, next_pos):
        trans = mn.Matrix4(self.articulated_agent.sim_obj.transformation)
        vc = SimpleVelocityControlEnv(120.0)
        angle = float("inf")
        cur_pos = self.articulated_agent.base_pos
        trans.translation = self.articulated_agent.base_pos
        while abs(angle) > self.turn_thresh:
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
            trans = vc.act(trans, vel)
            cur_pos = trans.translation
            if self.is_collision(trans):
                return True
        return False

    def _check_if_held_target(self) -> bool:
        if self.target_handle is not None:
            target_node = self.env.world_graph[self.agent_uid].get_node_from_sim_handle(self.target_handle)
            if isinstance(target_node, Object):
                dummy_action = torch.zeros(1, 1)
                (
                    dummy_action,
                    self.termination_message,
                    self.failed,
                ) = getattr(self.env, "check_if_the_object_is_held_by_agent", lambda *a, **k: (dummy_action, "", False))(
                    self.env, dummy_action, self.target_handle, self.agent_uid
                )
                if self.failed:
                    return True
        return False

    def _internal_act(self, observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx, deterministic=False):
        action = torch.zeros(prev_actions.shape, device=masks.device)

        if self.failed:
            return action, self.termination_message

        if self._check_if_held_target():
            return action, None

        if self.target_base_rot is None:
            self.failed = True
            return action, None

        obj_targ_pos = np.array(self.target_pos)

        if self.do_teleport:
            action[cur_batch_idx, self.target_pos_range[0]] = 1.0
            action[cur_batch_idx, self.target_pos_range[1:4]] = torch.tensor(
                self.target_base_pos, dtype=torch.float32
            ).to(action.device)
            action[cur_batch_idx, self.target_pos_range[4]] = self.target_base_rot
            self._has_reached_goal[cur_batch_idx] = 1
            return action, None

        base_T = self.articulated_agent.base_transformation
        curr_path_points = self._path_to_point(self.target_base_pos)
        robot_pos = np.array(self.articulated_agent.base_pos)

        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")

        if len(curr_path_points) == 1:
            curr_path_points += curr_path_points

        cur_nav_targ = np.array(curr_path_points[1])
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        rel_targ = cur_nav_targ - robot_pos
        robot_forward = robot_forward[[0, 2]]
        rel_targ = rel_targ[[0, 2]]
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
        dist_to_final_nav_targ = np.linalg.norm((np.array(self.target_base_pos) - robot_pos)[[0, 2]])

        if not self.face_object and dist_to_final_nav_targ < self.dist_thresh:
            rel_pos = np.array([np.cos(self.target_base_rot), -np.sin(self.target_base_rot)])
            rel_targ = rel_pos

        angle_to_target = get_angle(robot_forward, rel_targ)
        angle_to_obj = get_angle(robot_forward, rel_pos)
        at_goal = dist_to_final_nav_targ < self.dist_thresh and angle_to_obj < self.turn_thresh

        # Reactive avoidance only for robot agents (base_velocity)
        nearest = self._nearest_human()
        human_close = False
        human_pos = None
        if nearest is not None:
            human_pos, human_d = nearest
            human_close = human_d < self.avoidance_distance

        # Enter avoidance mode if human is close and agent is robot
        if self.motion_type == "base_velocity" and human_close and not at_goal:
            self.in_avoidance_mode = True

        # If in avoidance mode and motion_type is robot, compute lateral waypoint around human
        if self.motion_type == "base_velocity" and self.in_avoidance_mode:
            try:
                # 2D vectors
                human_pos_2 = np.array([human_pos[0], human_pos[2]])
                target_base_2 = np.array([self.target_base_pos[0], self.target_base_pos[2]])
                vec_ht = target_base_2 - human_pos_2
                if np.linalg.norm(vec_ht) < 1e-6:
                    # fallback: pick a perpendicular to robot->human
                    vec_ht = np.array([robot_pos[0] - human_pos[0], robot_pos[2] - human_pos[2]])
                perp = np.array([-vec_ht[1], vec_ht[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)

                # choose side of arc based on sign of cross product between robot forward and vector to human
                # this is a heuristic to keep consistent turning direction
                to_human = human_pos_2 - np.array([robot_pos[0], robot_pos[2]])
                cross = robot_forward[0] * to_human[1] - robot_forward[1] * to_human[0]
                side = 1.0 if cross >= 0 else -1.0

                avoidance_point_2 = human_pos_2 + perp * (self.avoidance_radius * side)
                avoidance_point = np.array([avoidance_point_2[0], robot_pos[1], avoidance_point_2[1]])

                # replace rel_targ with vector to avoidance point
                rel_targ = avoidance_point[[0, 2]] - robot_pos[[0, 2]]

                # exit avoidance mode when robot is safely past the human
                if np.linalg.norm((np.array([robot_pos[0], robot_pos[2]]) - human_pos_2)) > self.avoidance_release_distance:
                    self.in_avoidance_mode = False
            except Exception:
                # On any error, fall back to normal navigation behavior
                self.in_avoidance_mode = False

        if self.motion_type == "base_velocity":
            need_move_backward = False
            if (
                dist_to_final_nav_targ >= self.dist_thresh
                and angle_to_target >= self.turn_thresh
                and not at_goal
            ):
                need_move_backward = self.rotation_collision_check(cur_nav_targ)

            if need_move_backward and self.enable_backing_up:
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                robot_forward = robot_forward[[0, 2]]
                rel_targ = cur_nav_targ - robot_pos
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                dist_to_final_nav_targ = np.linalg.norm((self.target_base_pos - robot_pos)[[0, 2]])
                at_goal = dist_to_final_nav_targ < self.dist_thresh and angle_to_obj < self.turn_thresh

            if not at_goal:
                if dist_to_final_nav_targ < self.dist_thresh:
                    vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
                elif angle_to_target < self.turn_thresh:
                    vel = [self.forward_velocity, 0]
                else:
                    vel = compute_turn(rel_targ, self.turn_velocity, robot_forward)
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            if need_move_backward:
                vel[0] = -1 * vel[0]

            self.fix_robot_leg()

            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]
        else:
            # Humanoid joints or other motion types: fall back to existing oracle behaviour
            if not at_goal:
                if dist_to_final_nav_targ < self._config.dist_thresh:
                    vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
                elif angle_to_target < self.turn_thresh:
                    vel = [self.forward_velocity, 0]
                else:
                    vel = compute_turn(rel_targ, self.turn_velocity, robot_forward)
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]

        return action, None

    def _is_skill_done(self, observations, rnn_hidden_states, prev_actions, masks, batch_idx) -> torch.BoolTensor:
        return (self._has_reached_goal[batch_idx] > 0.0).to(masks.device)

    @property
    def argument_types(self) -> List[str]:
        return ["NAV_TARGET"]
