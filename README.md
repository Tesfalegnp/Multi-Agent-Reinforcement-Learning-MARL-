# Multi-Agent Reinforcement Learning (MARL) System
## 2-Agent Tag Environment with Independent Q-Learning (IQL)

![Pygame Visualization](https://pettingzoo.farama.org/_images/simple_tag.gif)  
*Example of the simple_tag_v3 environment from PettingZoo*

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Expected Results](#expected-results)
6. [Troubleshooting](#troubleshooting)
7. [Project Structure](#project-structure)
8. [License](#license)

---

## Features
✅ **2-Agent System**  
- 1 adversarial agents (chasers) 
- 1 good agents (evaders)  

✅ **Algorithm**  
- Independent Q-Learning (IQL) with:  
  - Epsilon-greedy & Boltzmann exploration  
  - Adaptive learning rates  
  - Non-stationarity handling  

✅ **Monitoring**  
- Real-time Pygame rendering  
- Training metrics (rewards, coordination, etc.)  
- Model checkpointing  

✅ **Advanced Metrics**  
- Reward correlation tracking  
- Action diversity analysis  
- Agent specialization visualization  

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/marl-5agent-system.git
cd marl-5agent-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
