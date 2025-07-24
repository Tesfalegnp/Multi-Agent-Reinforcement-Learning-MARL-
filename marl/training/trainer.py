# /home/hope/Project_package/marl_two_agents/marl/training/trainer.py
"""
Training pipeline for MARL with IQL
Handles agent coordination, experience collection, and training loops
"""
import numpy as np
from collections import defaultdict

class MARLTrainer:
    def __init__(self, env_wrapper, agent_configs):
        """
        Initialize the MARL trainer
        
        Args:
            env_wrapper: Environment wrapper instance
            agent_configs: Dictionary of agent configurations
        """
        self.env = env_wrapper
        self.agent_configs = agent_configs
        
        # Initialize agents
        self.agents = {}
        for agent_name in self.env.get_agent_names():
            action_space = self.env.get_action_space(agent_name)
            observation_space = self.env.get_observation_space(agent_name)
            
            # Get config for this agent (or use default)
            config = agent_configs.get(agent_name, {})
            
            self.agents[agent_name] = IQLAgent(
                agent_name=agent_name,
                action_space=action_space,
                observation_space=observation_space,
                **config
            )
            
        # Tracking
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.episode_exploration = []
        
    def train(self, num_episodes=1000, render_every=100, log_every=10):
        """
        Train the agents for specified number of episodes
        
        Args:
            num_episodes: Number of training episodes
            render_every: Render environment every N episodes
            log_every: Print progress every N episodes
        """
        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_rewards = {agent: 0 for agent in self.agents}
            done = False
            steps = 0
            
            while not done:
                # Get actions from all agents
                actions = {}
                for agent_name, agent in self.agents.items():
                    actions[agent_name] = agent.act(observations[agent_name])
                
                # Execute actions in environment
                next_observations, rewards, dones, _ = self.env.step(actions)
                done = all(dones.values())
                
                # Learn from experience
                for agent_name, agent in self.agents.items():
                    agent.learn(
                        next_observations[agent_name],
                        rewards[agent_name],
                        done
                    )
                    episode_rewards[agent_name] += rewards[agent_name]
                
                observations = next_observations
                steps += 1
                
                # Optional: render for visualization
                if episode % render_every == 0:
                    self.env.render()
            
            # Store episode statistics
            for agent_name, reward in episode_rewards.items():
                self.episode_rewards[agent_name].append(reward)
            self.episode_lengths.append(steps)
            self.episode_exploration.append(
                np.mean([agent.exploration_rate for agent in self.agents.values()]))
            
            # Log progress
            if episode % log_every == 0:
                avg_rewards = {k: np.mean(v[-log_every:]) for k, v in self.episode_rewards.items()}
                print(f"Episode {episode}:")
                print(f"  Avg rewards: {avg_rewards}")
                print(f"  Exploration: {self.episode_exploration[-1]:.2f}")
                print(f"  Steps: {steps}")
                
    def get_training_data(self):
        """Get training statistics for visualization"""
        return {
            'episode_rewards': dict(self.episode_rewards),
            'episode_lengths': self.episode_lengths,
            'exploration_rates': self.episode_exploration
        }
    
    def save_models(self, directory):
        """Save all agent models"""
        import os
        os.makedirs(directory, exist_ok=True)
        for agent_name, agent in self.agents.items():
            agent.save(f"{directory}/{agent_name}.pkl")
            
    def load_models(self, directory):
        """Load all agent models"""
        import os
        for agent_name, agent in self.agents.items():
            if os.path.exists(f"{directory}/{agent_name}.pkl"):
                agent.load(f"{directory}/{agent_name}.pkl")