# /home/hope/Project_package/marl_two_agents/marl/utils/visualizer.py
"""
Wrapper for PettingZoo's simple_tag_v2 environment
Handles environment initialization, observation/action spaces, and agent management
"""
import numpy as np
from pettingzoo.mpe import simple_tag_v2

class MARLEnvWrapper:
    def __init__(self, num_adversaries=1, num_agents=2, max_cycles=500):
        """
        Initialize the environment with specified number of agents
        
        Args:
            num_adversaries: Number of adversary (predator) agents
            num_agents: Total number of agents (adversaries + good agents)
            max_cycles: Maximum steps per episode
        """
        self.env = simple_tag_v2.parallel_env(
            num_adversaries=num_adversaries,
            num_good=num_agents - num_adversaries,
            max_cycles=max_cycles
        )
        self.env.reset()
        
        # Get agent names and action/observation spaces
        self.agent_names = self.env.agents
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agent_names}
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.agent_names}
        
    def reset(self):
        """Reset the environment and return initial observations"""
        observations = self.env.reset()
        return observations
    
    def step(self, actions):
        """
        Execute actions for all agents
        
        Args:
            actions: Dictionary of {agent_name: action}
            
        Returns:
            observations: Next observations for each agent
            rewards: Rewards for each agent
            dones: Whether episode is done for each agent
            infos: Additional info for each agent
        """
        observations, rewards, dones, infos = self.env.step(actions)
        return observations, rewards, dones, infos
    
    def render(self):
        """Render the current environment state"""
        self.env.render()
        
    def close(self):
        """Close the environment"""
        self.env.close()
        
    def get_agent_names(self):
        """Get list of all agent names"""
        return self.agent_names
    
    def get_action_space(self, agent_name):
        """Get action space for a specific agent"""
        return self.action_spaces[agent_name]
    
    def get_observation_space(self, agent_name):
        """Get observation space for a specific agent"""
        return self.observation_spaces[agent_name]