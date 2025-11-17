#!/usr/bin/env python3

"""
Demo script showing how to use custom robot control while human agent follows scripted tasks.

Usage:
    python custom_robot_demo.py
"""

import sys
sys.path.append("..")

from habitat_llm.examples.planner_demo import (
    CollaborationDatasetV0,
    register_sensors,
    register_actions,
    register_measures,
    remove_visual_sensors,
)
from habitat_llm.utils import setup_config, fix_config
from habitat_llm.evaluation import CentralizedEvaluationRunner
from habitat_llm.agent.env import EnvironmentInterface
from omegaconf import OmegaConf
import hydra


@hydra.main(config_path="../conf")
def main(config):
    """
    Run the planner with custom robot control enabled.
    """
    
    # Fix and setup config
    fix_config(config)
    seed = 47668090
    config = setup_config(config, seed)
    
    # Load dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    
    # If you want to run just one episode for testing
    if hasattr(config, 'episode_indices') and config.episode_indices:
        episode_subset = [dataset.episodes[i] for i in config.episode_indices]
        dataset = CollaborationDatasetV0(
            config=config.habitat.dataset, episodes=episode_subset
        )
    
    print("\n" + "="*70)
    print("CUSTOM ROBOT CONTROL DEMO")
    print("="*70)
    print("\nThis demo runs the planner where:")
    print("  - Agent 0 (Robot): Receives MANUAL commands (injected below)")
    print("  - Agent 1 (Human): Follows scripted tasks from evaluation function")
    print("\nManual commands being sent to robot:")
    print("  - Steps 0-10: forward:0.3 (move forward)")
    print("  - Steps 11-20: turn:0.5 (turn right)")
    print("  - Steps 21-30: forward:-0.2 (move backward)")
    print("  - Steps 31-40: stop")
    print("  - Then switches back to autonomous navigation")
    print("="*70 + "\n")
    
    # Setup environment interface (same as planner_demo.py)
    if config.env == "habitat":
        # Remove sensors if we are not saving video
        keep_rgb = False
        if "use_rgb" in config.evaluation:
            keep_rgb = config.evaluation.use_rgb
        if not config.evaluation.save_video and not keep_rgb:
            remove_visual_sensors(config)
        
        # Register sensors, actions, measures
        register_sensors(config)
        register_actions(config)
        register_measures(config)
        
        # Create environment interface
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
        
        try:
            env_interface.initialize_perception_and_world_graph()
        except Exception:
            print("Error initializing the environment")
            raise
    else:
        env_interface = None
    
    # Create evaluation runner (centralized for custom control)
    eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)
    
    # Get reference to the planner (CustomControlPlanner)
    planner = eval_runner.planner
    
    # Pre-populate the robot command queue with manual commands
    # These will be executed instead of scripted actions
    print("🤖 Injecting manual commands into robot control queue...")
    
    # Move forward for ~10 steps
    for _ in range(10):
        planner.set_robot_command("forward:0.3")
    
    # Turn right for ~10 steps  
    for _ in range(10):
        planner.set_robot_command("turn:0.5")
    
    # Move backward for ~10 steps
    for _ in range(10):
        planner.set_robot_command("forward:-0.2")
    
    # Stop for ~10 steps
    for _ in range(10):
        planner.set_robot_command("stop")
    
    print(f"✓ Injected {len(planner.robot_command_queue)} manual commands\n")
    print("Note: After manual commands are exhausted, robot will")
    print("      switch back to autonomous scripted actions.\n")
    
    # Run episodes
    for episode_id, episode in enumerate(dataset.episodes):
        print(f"\n{'='*70}")
        print(f"Running Episode {episode_id}: {episode.episode_id}")
        print(f"{'='*70}\n")
        
        # Run the instruction (environment auto-resets to current episode)
        info = eval_runner.run_instruction(output_name=f"episode_{episode_id}_0")
        
        print(f"\n✓ Episode {episode_id} completed")
        print(f"  Success: {info.get('task_state_success', False)}")
        print(f"  Steps: {info.get('sim_step_count', 0)}")
        
        # Reset environment for next episode (if not last episode)
        if episode_id < len(dataset.episodes) - 1:
            env_interface.reset_environment()
            eval_runner.reset()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
