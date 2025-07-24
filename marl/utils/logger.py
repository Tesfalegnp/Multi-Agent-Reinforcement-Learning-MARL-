# /home/hope/Project_package/marl_two_agents/marl/utils/visualizer.py
"""
Logging utilities for tracking training progress
"""
import json
import time
from collections import defaultdict

class Logger:
    def __init__(self, log_file=None):
        """
        Initialize logger
        
        Args:
            log_file: Optional file to save logs to
        """
        self.log_file = log_file
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
    def log(self, metrics_dict):
        """
        Log metrics
        
        Args:
            metrics_dict: Dictionary of metrics to log
        """
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
    def get_metrics(self):
        """Get all logged metrics"""
        return dict(self.metrics)