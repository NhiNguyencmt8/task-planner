#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
ManualControlSkill: A skill that allows direct control of robot velocities
and actions without relying on learned policies.
"""

import torch
from typing import List, Tuple

from habitat_llm.tools.motor_skills.skill import SkillPolicy


class ManualControlSkill(SkillPolicy):
    """
    Skill for manual control of robot with custom velocity/speed commands.
    Allows direct specification of base velocity, turn velocity, arm actions, etc.
    """

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        batch_size,
        env=None,
        agent_uid=0,
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=False,
            env=env,
            agent_uid=agent_uid,
        )
        
        self.env = env
        
        # Store command parameters
        self.base_velocity = 0.0
        self.turn_velocity = 0.0
        self.arm_action = None
        self.gripper_action = 0.0  # -1 = release, 0 = hold, 1 = grasp
        
        # Maximum values for safety
        self.max_base_velocity = 1.0
        self.max_turn_velocity = 1.0
        
        # Parse action space to find indices for different action components
        # Do this after env is set
        if action_space is not None:
            self._parse_action_space()
        else:
            # Defer parsing until first use
            self.base_vel_start = None
            self.base_vel_end = None
            self.arm_action_start = None
            self.arm_action_end = None
            self._action_space_parsed = False

    def _parse_action_space(self):
        """Parse the action space to find indices for different components"""
        from habitat_llm.agent.env.actions import find_action_range
        
        # Find base velocity action indices
        try:
            self.base_vel_start, self.base_vel_end = find_action_range(
                self.action_space, "base_velocity"
            )
        except:
            self.base_vel_start = None
            self.base_vel_end = None
            
        # Find arm action indices
        try:
            self.arm_action_start, self.arm_action_end = find_action_range(
                self.action_space, "arm_action"
            )
        except:
            self.arm_action_start = None
            self.arm_action_end = None

    @property
    def argument_types(self) -> List[str]:
        """Define what types of arguments this skill accepts"""
        return ["command"]

    def set_target(self, target_string: str, env):
        """
        Parse the command string to set velocities and actions.
        
        Example commands:
        - "forward:0.5" - move forward at 0.5 m/s
        - "turn:0.3" - turn at 0.3 rad/s
        - "stop" - stop all motion
        - "forward:0.5,turn:0.2" - combine movements
        """
        self.env = env
        
        # Reset values
        self.base_velocity = 0.0
        self.turn_velocity = 0.0
        
        # Parse command
        commands = target_string.lower().split(',')
        
        for cmd in commands:
            cmd = cmd.strip()
            
            if ':' in cmd:
                action, value = cmd.split(':')
                action = action.strip()
                
                try:
                    value = float(value.strip())
                except ValueError:
                    raise ValueError(f"Invalid numeric value in command: {cmd}")
                
                if action == "forward" or action == "base":
                    self.base_velocity = max(min(value, self.max_base_velocity), -self.max_base_velocity)
                elif action == "turn" or action == "rotate":
                    self.turn_velocity = max(min(value, self.max_turn_velocity), -self.max_turn_velocity)
                elif action == "speed":
                    # Set both forward and turn to same magnitude
                    self.base_velocity = max(min(value, self.max_base_velocity), -self.max_base_velocity)
                else:
                    raise ValueError(f"Unknown action type: {action}")
            
            elif cmd == "stop":
                self.base_velocity = 0.0
                self.turn_velocity = 0.0
            else:
                raise ValueError(f"Invalid command format: {cmd}. Use 'action:value' or 'stop'")
        
        # Mark target as set
        self.target_is_set = True
        self.finished = False
        self.failed = False

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        """
        Manual control executes for one step then finishes.
        """
        # Execute for one step
        is_done = self._cur_skill_step >= 1
        return is_done.to(self.device)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate low-level action based on the set velocities.
        """
        print(f"\n=== ManualControl._internal_act called ===")
        print(f"Base velocity: {self.base_velocity}, Turn velocity: {self.turn_velocity}")
        print(f"prev_actions shape: {prev_actions.shape}")
        
        # Lazy parse action space if not done yet
        if not hasattr(self, '_action_space_parsed') or not self._action_space_parsed:
            if self.action_space is not None:
                self._parse_action_space()
                self._action_space_parsed = True
        
        # Create action tensor matching prev_actions shape
        # prev_actions already has the right shape for the action space
        action = torch.zeros_like(prev_actions, device=self.device)
        
        # Set base velocity if we found the indices
        if self.base_vel_start is not None and self.base_vel_end is not None:
            # Set forward velocity
            action[0, self.base_vel_start] = self.base_velocity
            # Set turn velocity if there's room
            if self.base_vel_end > self.base_vel_start + 1:
                action[0, self.base_vel_start + 1] = self.turn_velocity
            print(f"Set action[0, {self.base_vel_start}] = {self.base_velocity}")
            print(f"Set action[0, {self.base_vel_start + 1}] = {self.turn_velocity}")
        else:
            # If we couldn't find base_velocity indices, try to set first two elements
            # This is a fallback - ideally we'd properly parse the action space
            if prev_actions.shape[1] >= 2:
                action[0, 0] = self.base_velocity
                action[0, 1] = self.turn_velocity
                print(f"Fallback: Set action[0, 0:2] = [{self.base_velocity}, {self.turn_velocity}]")
        
        print(f"Returning action: {action}")
        return action, rnn_hidden_states

    def get_state_description(self) -> str:
        """Return description of current manual control state"""
        return f"ManualControl(forward={self.base_velocity:.2f}, turn={self.turn_velocity:.2f})"

    def reset(self, batch_idxs: List[int]):
        """Reset the skill state"""
        super().reset(batch_idxs)
        self.base_velocity = 0.0
        self.turn_velocity = 0.0
        self.arm_action = None
        self.gripper_action = 0.0
