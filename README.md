# RL Agent for Taiwanese Driving School Test Simulation

## 🚀 Project Description

This project aims to develop a Reinforcement Learning (RL) agent capable of autonomously learning to pass various driving test maneuvers within a simulated Taiwanese driving school environment. The goal is for the agent to master tasks such as S-curves (forward and reverse), parallel parking, and garage parking, ultimately achieving or exceeding the average human pass rate.

---

## ✨ Features (Planned or Implemented)

* **Realistic Simulation Environment:**
    * [x] 2D Visual Simulation (using [highway-env](https://github.com/Farama-Foundation/HighwayEnv))
    * [ ] Accurate replication of Taiwanese driving school test course layouts and dimensions.
    * [x] Basic vehicle dynamics simulation.
    * [x] Simulation of sensor lines/pressure plates as per test course rules.
* **Reinforcement Learning Agent:**
    * [x] Observation space using Lidar and goal state.
    * [ ] Reward function designed according to official Taiwanese driving test scoring criteria.
    * [x] Implementation of state-of-the-art RL algorithms (e.g., PPO, SAC, DQN).
    * [x] Support for Curriculum Learning strategies.
* **Driving Test Maneuver Support (Phased Implementation):**
    * [ ] Straight Line Driving & Stability
    * [ ] S-Curve (Forward)
    * [ ] S-Curve (Reverse)
    * [x] Garage Parking (Reverse Parking)
    * [x] Parallel Parking
    * [ ] Uphill Start (if applicable to the simulation)
* **Visualization & Analysis:**
    * [x] Training progress visualization (e.g., reward curves).
    * [x] Agent's driving behavior playback/replay.
    * [ ] Simulated test results and deduction point analysis.

---

## 🛠️ Environment Setup & Installation

create environment for python 3.10.12

```
conda create --name drivingclass python=3.10.12 -y
conda activate drivingclass
pip install -r requirements.txt
```

## ▶️ How to Use

### Training the Agent

1. **Basic PPO Training:**
   ```bash
   python PPO_train.py
   ```

2. **PPO with RND (Random Network Distillation):**
   ```bash
   python PPO_train_with_RND.py
   ```

3. **HER-SAC Training:**
   ```bash
   python HER_SAC_train.py
   ```

4. **DDPG with HER:**
   ```bash
   python DDPG_HER.py
   ```

5. **Circle Track Training:**
   ```bash
   python Circle_PPO_train.py
   ```

### Testing and Visualization

**Show Environment:**
```bash
python show.py
```
with desired model load in `show.py` file.

## 💻 Technology Stack

- **Python 3.10.12** - Core programming language
- **[highway-env](https://github.com/Farama-Foundation/HighwayEnv)** - Base simulation environment
- **Stable-baselines3** - Reinforcement learning algorithms (PPO, SAC, DDPG)
- **Gymnasium** - RL environment interface
- **TensorBoard** - Visualization
- **HER (Hindsight Experience Replay)** - Goal-conditioned RL
- **RND (Random Network Distillation)** - Exploration bonus

## 📂 Project Structure

```
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── env/                         # Environment and training code
    ├── driving_class.py         # Main driving class environment
    ├── driving_class_lidaronly.py # Lidar-only observation environment
    ├── racetrack_env.py         # Race track environment
    ├── square_track_env.py      # Square track environment
    ├── PPO_train.py             # PPO training script
    ├── PPO_train_with_RND.py    # PPO with exploration bonus
    ├── HER_SAC_train.py         # HER-SAC training
    ├── DDPG_HER.py              # DDPG with HER training
    ├── Circle_PPO_train.py      # Circle track PPO training
    ├── train.py                 # General training script
    ├── test.py                  # Testing script
    └── show.py                  # Environment visualization
```

## 🎯 Current Implementation Status

- ✅ Multiple RL algorithms implemented (PPO, SAC, DDPG)
- ✅ Goal-conditioned learning with HER
- ✅ Exploration enhancement with RND
- ✅ Multiple environment configurations
- ✅ Lidar-based observation space
- ✅ Basic parking maneuvers (garage and parallel parking)
- 🔄 Working towards S-curve and straight-line driving
- 🔄 Taiwanese driving test scoring system integration

## 📄 License

This project is open source. Please check the license file for details.
