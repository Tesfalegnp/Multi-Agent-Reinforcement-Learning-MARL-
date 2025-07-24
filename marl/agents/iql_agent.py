#/home/hope/Project_package/marl_two_agents/marl/agents/iql_agent.py
"""
Independent Q-Learning Agent Implementation
Each agent maintains its own Q-table and learns independently
"""
import numpy as np
import random
from collections import defaultdict

class IQLAgent:
    def __init__(self, agent_name, action_space, observation_space, 
                 learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration=0.01):
        """
        Initialize the IQL agent
        
        Args:
            agent_name: Unique identifier for the agent
            action_space: Action space of the agent
            observation_space: Observation space of the agent
            learning_rate: How quickly the agent updates its Q-values (alpha)
            discount_factor: Importance of future rewards (gamma)
            exploration_rate: Initial probability of random action (epsilon)
            exploration_decay: Rate at which exploration decreases
            min_exploration: Minimum exploration probability
        """
        self.agent_name = agent_name
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Q-table using dictionary for sparse states
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Tracking
        self.last_action = None
        self.last_state = None
        
    def act(self, observation, training=True):
        """
        Choose an action based on current observation
        
        Args:
            observation: Current state observation
            training: Whether to use exploration (True during training)
            
        Returns:
            action: Chosen action
        """
        # Convert observation to hashable type (tuple) for Q-table
        state = self._process_observation(observation)
        
        # Exploration: random action
        if training and random.random() < self.exploration_rate:
            action = self.action_space.sample()
        # Exploitation: best known action
        else:
            action = np.argmax(self.q_table[state])
            
        # Save for learning
        self.last_state = state
        self.last_action = action
        
        return action
    
    def learn(self, next_observation, reward, done):
        """
        Update Q-table based on experience
        
        Args:
            next_observation: Observation after taking action
            reward: Reward received
            done: Whether episode is complete
        """
        next_state = self._process_observation(next_observation)
        
        # Current Q-value estimate
        current_q = self.q_table[self.last_state][self.last_action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[self.last_state][self.last_action] = new_q
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(
                self.min_exploration, 
                self.exploration_rate * self.exploration_decay
            )
    
    def _process_observation(self, observation):
        """
        Convert observation to hashable type for Q-table
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            processed_state: Tuple representing discretized state
        """
        # Discretize continuous observations (simplification for beginners)
        # In a real implementation, you might use tile coding or neural networks
        discretized = tuple((observation * 10).astype(int))  # Scale and convert to integers
        return discretized
    
    def save(self, filepath):
        """Save Q-table to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
            
    def load(self, filepath):
        """Load Q-table from file"""
        import pickle
        with open(filepath, 'rb') as f:
            q_table = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n), q_table)