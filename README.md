# 🎮 Game AI: Multi-Game Reinforcement Learning

> End-to-end ML pipeline implementation: data collection → model training → strategy deployment

**Tech Stack:** Python · PyTorch · scikit-learn · MLGame · Pygame
**Implementation:** 7.3K+ LOC · 59 Python modules · 4 complete training pipelines

---

## 📊 Technical Overview

| Game | Algorithm | Feature Space | Training Pipeline | Key Challenge |
|------|-----------|---------------|-------------------|---------------|
| **Pingpong** | KNN (K=1) | 8D state vector | Automated data collection → KNN training | Real-time ball trajectory prediction |
| **Arkanoid** | KNN (K=1) | 6D state vector | Multi-level data aggregation → KNN | Multi-bounce path planning |
| **Swimming Squid** | PPO + Q-Learning | 12D observation | RL episode collection → PPO/Q training | Multi-agent survival dynamics |
| **Proly** | PPO + Rule-Based | 15D observation | Supervised pretraining → RL fine-tuning | 3D navigation with sparse rewards |

---

## 💡 Core Techniques

### 1. KNN-Based Imitation Learning
- **Data Collection:** Automated gameplay logging across multiple difficulty levels
- **Feature Engineering:** Physics-based trajectory prediction + state quantization
- **State Space:** Ball position/velocity (4D) + platform state (2D) + predicted landing (2D)
- **Optimization:** Balanced sampling to prevent action bias (STAY vs MOVE)

### 2. Deep Reinforcement Learning
- **Architecture:** PyTorch Actor-Critic networks with PPO optimization
- **Training Loop:** Episode collection → advantage estimation → policy update
- **Reward Design:** Multi-objective shaping (survival + score + efficiency penalties)
- **Optimization:** GAE for variance reduction + entropy regularization

### 3. Hybrid Approaches
- **Q-Learning:** Tabular RL with discrete state quantization
- **Rule-Based Systems:** Vector field navigation for baseline comparison
- **Transfer Learning:** Supervised behavior cloning → RL fine-tuning pipeline

### 4. Engineering Practices
- **Modular Pipeline:** Separate data collection, training, and inference modules
- **Experiment Tracking:** Version control for models and hyperparameters
- **Reproducibility:** Deterministic seeding + environment configuration management

---

## 🏗️ Project Structure

- **[pingpong/](pingpong/)** - KNN imitation learning for 2-player real-time battle
- **[arkanoid/](arkanoid/)** - KNN trajectory prediction across 24 level configurations
- **[swimming_squid_battle/](swimming_squid_battle/)** - PPO/Q-Learning for multi-agent survival
- **[proly/](proly/)** - Hybrid PPO + rule-based 3D navigation system

---

## 🔧 Training Pipeline

```bash
# Pingpong: Collect data → Train KNN
cd pingpong
python -m mlgame --save-progress ./  # Collect training data
python ml/train_knn.py               # Train KNN classifier

# Arkanoid: Multi-level data collection
cd arkanoid
make run_train  # Automated data collection across 24 levels
make train      # Train KNN model with balanced sampling

# Swimming Squid: RL training loop
cd swimming_squid_battle
python ml/ml_play_rl_rectangle_1.py  # PPO training with episode collection

# Proly: Supervised → RL pipeline
cd proly
python ml/train_supervised.py        # Pretrain with behavior cloning
python ml/ppo_mlplay_template.py     # Fine-tune with PPO
```

---

## 🎯 Game Details

### Pingpong
- **State Space:** Ball (x,y,dx,dy) + Platform x + Landing prediction + Obstacles
- **Action Space:** {LEFT, RIGHT, STAY}
- **Challenge:** Handle spin mechanics + variable ball speeds (5-15 px/frame)

### Arkanoid
- **State Space:** Ball trajectory + Brick layout + Power-ups
- **Action Space:** {LEFT, RIGHT, STAY}
- **Challenge:** Multi-bounce prediction + hard brick mechanics (2-hit destroy)

### Swimming Squid Battle
- **State Space:** Self/opponent position + Level (1-6) + Food/garbage locations
- **Action Space:** {UP, DOWN, LEFT, RIGHT}
- **Challenge:** Multi-agent collision dynamics + level-based reward scaling

### Proly
- **State Space:** 3D position + Velocity + Rotation + Enemy vectors + Terrain grid
- **Action Space:** Movement (4D) + Rotation (2D) + Item usage (N items)
- **Challenge:** Sparse rewards + long-horizon planning + 3D collision avoidance

---

## 🔧 Development

**Requirements:** Python 3.9+ · MLGame ≥10.6.0 · PyTorch ≥1.9.0 · scikit-learn

```bash
# Install dependencies
pip install mlgame pygame scikit-learn torch

# Train new models (example: Arkanoid)
cd arkanoid && make run_train  # Collect data
make train                      # Train KNN model
make test                       # Evaluate performance
```

---

## 🎯 Implementation Achievements

- **KNN Scalability:** Successfully trained classifiers on 500K+ samples with real-time inference
- **RL Convergence:** Implemented stable PPO training with multi-objective reward shaping
- **Hybrid Systems:** Combined rule-based baselines with neural network fine-tuning
- **Multi-Agent:** Handled complex survival dynamics with level-based difficulty scaling

---

## 📝 Implementation Highlights

1. **KNN Optimization:** Balanced dataset sampling to prevent action bias (67% STAY → 33% uniform)
2. **PPO Tuning:** Clipping ratio 0.2 · GAE λ=0.95 · Entropy bonus 0.01
3. **Feature Engineering:** Predicted landing point calculation with physics simulation
4. **Reward Shaping:** Sparse rewards → Dense intermediate rewards for faster convergence

---
