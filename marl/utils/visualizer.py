# visualizer.py
"""
Visualization utilities for MARL training results
"""
import matplotlib.pyplot as plt
import numpy as np

class MARLVisualizer:
    @staticmethod
    def plot_rewards(episode_rewards, window=100):
        """
        Plot smoothed reward curves for all agents
        
        Args:
            episode_rewards: Dictionary of {agent_name: list of rewards}
            window: Smoothing window size
        """
        plt.figure(figsize=(10, 6))
        
        for agent_name, rewards in episode_rewards.items():
            # Smooth rewards with moving average
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=agent_name)
            
        plt.title("Smoothed Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel(f"Reward (avg last {window} episodes)")
        plt.legend()
        plt.grid()
        plt.show()
        
    @staticmethod
    def plot_exploration(exploration_rates):
        """Plot exploration rate decay over time"""
        plt.figure(figsize=(10, 4))
        plt.plot(exploration_rates)
        plt.title("Exploration Rate Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Exploration Rate (epsilon)")
        plt.grid()
        plt.show()
        
    @staticmethod
    def plot_episode_lengths(episode_lengths, window=100):
        """Plot episode lengths over time"""
        plt.figure(figsize=(10, 4))
        smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        plt.plot(smoothed)
        plt.title("Episode Lengths Over Time")
        plt.xlabel("Episode")
        plt.ylabel(f"Steps (avg last {window} episodes)")
        plt.grid()
        plt.show()