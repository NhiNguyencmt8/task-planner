#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
CustomControlPlanner: Extends ScriptedCentralizedPlanner to allow manual control
of robot agent while human agent follows scripted tasks.
"""

from typing import Any, Dict, Tuple
from habitat_llm.planner.scripted_centralized_planner import ScriptedCentralizedPlanner
from habitat_llm.world_model import WorldGraph


class CustomControlPlanner(ScriptedCentralizedPlanner):
    """
    Planner that assigns scripted tasks to human agent (agent_1)
    and accepts manual commands for robot agent (agent_0).
    """

    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        
        # Queue for custom robot commands
        self.robot_command_queue = []
        
        # Flag to enable/disable custom control
        self.custom_control_enabled = True
        
        # Robot agent ID (typically 0)
        self.robot_agent_id = 0
        
        # Human agent ID (typically 1)  
        self.human_agent_id = 1
        
        # Step counter for demo
        self.step_counter = 0

    def set_robot_command(self, command: str):
        """
        Add a custom command for the robot to execute.
        
        :param command: Command string (e.g., "forward:0.5", "turn:0.3", "stop")
        """
        self.robot_command_queue.append(command)

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, WorldGraph],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get next actions for both agents.
        - Human agent: follows scripted plan from DAG
        - Robot agent: executes custom manual commands if available
        """
        
        # Auto-inject demo commands based on step counter
        if self.step_counter < 10:
            self.set_robot_command("forward:0.3")
        elif self.step_counter < 20:
            self.set_robot_command("turn:0.5")
        elif self.step_counter < 30:
            self.set_robot_command("forward:-0.2")
        elif self.step_counter < 40:
            self.set_robot_command("stop")
        # After step 40, queue will be empty and robot uses autonomous actions
        
        self.step_counter += 1
        
        # If custom control is disabled, use original scripted planner for both
        if not self.custom_control_enabled:
            return super().get_next_action(instruction, observations, world_graph)
        
        # Get scripted actions for both agents first
        scripted_actions, planner_info, should_end = super().get_next_action(
            instruction, observations, world_graph
        )
        
        # If we have custom robot commands, override robot actions
        if len(self.robot_command_queue) > 0:
            robot_command = self.robot_command_queue.pop(0)
            
            # Create high-level action for robot using ManualControl tool
            robot_action = (
                "ManualControl",  # Tool name
                robot_command,     # Command parameters
                ""                 # No error message
            )
            
            # Override robot action in the scripted_actions
            if self.robot_agent_id in scripted_actions:
                # Store original for logging
                planner_info[f"robot_original_action"] = scripted_actions[self.robot_agent_id]
                planner_info[f"robot_custom_command"] = robot_command
            
            # Update high-level actions
            self.last_high_level_actions[self.robot_agent_id] = robot_action
            
            # Process the custom action through the normal pipeline
            low_level_actions, responses = self.process_high_level_actions(
                {self.robot_agent_id: robot_action}, observations
            )
            
            # Combine robot's custom action with human's scripted action
            if self.human_agent_id in scripted_actions:
                # Keep human's scripted low-level action
                low_level_actions[self.human_agent_id] = scripted_actions[self.human_agent_id]
                
                # Process human's high-level action for response
                if self.human_agent_id in self.last_high_level_actions:
                    _, human_response = self.process_high_level_actions(
                        {self.human_agent_id: self.last_high_level_actions[self.human_agent_id]}, 
                        observations
                    )
                    responses.update(human_response)
            
            # Update planner info
            planner_info["responses"] = responses
            
            return low_level_actions, planner_info, should_end
        
        # No custom commands, return scripted actions for both agents
        return scripted_actions, planner_info, should_end

    def reset(self):
        """Reset planner state including command queue"""
        super().reset()
        self.robot_command_queue = []
        self.step_counter = 0
