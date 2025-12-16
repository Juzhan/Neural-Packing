# Neural Packing: from Visual Sensing to Reinforcement Learning

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->
[![Paper](https://img.shields.io/badge/Paper-ACM%20TOG-red)](https://doi.org/10.1145/3618354)

This repository is the official implementation of the paper **"Neural Packing: from Visual Sensing to Reinforcement Learning"** (ACM Transactions on Graphics, SIGGRAPH Asia 2023).

We propose **TAP-Net++**, a learning-based framework to solve the 3D Transport-and-Packing (TAP) problem. Unlike previous methods, our approach handles the full pipeline from visual sensing of casually stacked objects to robotic packing, optimizing both object selection and placement location (EMS) simultaneously.

<p align="center">
  <img src="doc/overview.png" alt="Teaser Image" width="800"/>
  <br>
  <em>Overview of the TAP-Net++ pipeline</em>
</p>

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)
- **Blender 3.0+** (for visualization tools in `render/` directory)

### Dependencies
Install required packages:

**Option 1: Using requirements.txt**
```bash
pip install -r requirements.txt
```

**Option 2: Manual installation**
```bash
pip install torch torchvision torchaudio
pip install gymnasium tianshou numpy matplotlib tqdm tensorboard scipy
```

**Note for visualization**: The `render/` directory uses Blender Python API (`bpy`) for 3D visualization. `bpy` comes with Blender installation and is not available via pip. To use visualization tools:
1. Install Blender from [blender.org](https://www.blender.org/download/)
2. Ensure Blender's Python includes the project dependencies
3. Or run visualization scripts within Blender's built-in Python environment

### Clone Repository
```bash
git clone https://github.com/Juzhan/Neural-Packing.git
cd Neural-Packing
```

## 🏃 Quick Start

### Training
To train the TAP-Net++ model with default parameters:

```bash
python train.py
```

### Testing
To test a pre-trained model:

```bash
python test.py
```

## 📖 Usage

### Command Line Arguments

The main training script `tap_train.py` supports numerous configuration options:

```bash
python tap_train.py --task tapnet/TAP-v0 \
                    --model tnpp \
                    --box-num 20 \
                    --container-size 100 100 100 \
                    --box-range 10 80 \
                    --fact-type tap_fake \
                    --prec-type attn \
                    --data-type rand \
                    --rotate-axes x y z \
                    --world-type real \
                    --container-type single \
                    --pack-type last \
                    --stable-predict 1 \
                    --reward-type C \
                    --max-epoch 100 \
                    --step-per-epoch 2000 \
                    --device cuda
```

### Key Parameters
- `--model`: Model architecture (`tnpp`, `tn`, `greedy`)
- `--fact-type`: Problem type (`tap_fake` for precedence-aware, `box` for standard packing)
- `--prec-type`: Precedence encoding (`attn`, `cnn`, `rnn`, `none`)
- `--world-type`: Simulation type (`real` with stability, `ideal` without)
- `--container-type`: `single` or `multi` container packing
- `--stable-predict`: Whether to predict stability (0 or 1)
- `--reward-type`: Reward formulation (`C` for compactness, `E` for step reward, etc.)

## 🧠 Algorithm Overview

### TAP-Net++ Architecture

TAP-Net++ consists of three main components:

1. **Object Encoder**: Encodes box dimensions and precedence relationships
2. **Space Encoder**: Encodes Empty Maximum Spaces (EMS) for placement
3. **Cross-Transformer**: Learns interactions between objects and spaces

### Reinforcement Learning Formulation

- **State**: Box states, EMS, precedence masks, heightmaps
- **Action**: Joint selection of (box, rotation, EMS, corner)
- **Reward**: Compactness ratio, stability penalty, container count

## 📁 Code Structure

```
Neural-Packing/
├── tap_train.py              # Main training script with argument parsing
├── train.py                  # Simplified training script
├── test.py                   # Testing script
├── test.sh                   # Shell script for testing
├── train.sh                  # Shell script for training
├── tapnet/                   # Core package
│   ├── __init__.py
│   ├── gym_tap.py            # Gymnasium environment registration
│   ├── envs/                 # Environment implementation
│   │   ├── __init__.py
│   │   ├── env.py            # Main TAP environment
│   │   ├── container.py      # Container management
│   │   ├── factory.py        # Box generation and management
│   │   ├── ems_tools.py      # EMS computation utilities
│   │   ├── convex_hull.py    # Geometry utilities
│   │   └── space.py          # Space representation
│   └── models/               # Neural network models
│       ├── __init__.py
│       ├── network.py        # Main TAP-Net++ architecture
│       ├── attention.py      # Cross-transformer implementation
│       ├── encoder.py        # Object and space encoders
│       ├── policy.py         # Policy networks
│       ├── greedy.py         # Greedy baseline
│       └── old.py            # Legacy models
├── render/                   # Visualization tools
│   ├── render_scripts.py
│   ├── render_tools.py
│   ├── sim_tools.py
│   ├── tools.py
│   └── results/
│       └── plot.py          # Plotting utilities
├── checkpoints/             # Saved model checkpoints
└── README.md               # This file
```

## 🏋️ Training

### Training Process
1. **Environment Setup**: Creates vectorized environments for parallel training
2. **Data Collection**: Uses Tianshou's `Collector` to gather experience
3. **Policy Optimization**: Applies PPO/A2C updates with advantage estimation
4. **Checkpointing**: Saves best policies and periodic checkpoints

### Monitoring
Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir ./log
```

### Hyperparameters
Default training parameters (can be adjusted in `tap_train.py`):
- Learning rate: 3e-4
- Buffer size: 2048
- Batch size: 128
- PPO clip epsilon: 0.2
- Discount factor (gamma): 0.99
- GAE lambda: 0.95

## 🧪 Testing
<!-- 
### Evaluation Metrics
- **Compactness ratio**: Volume utilization efficiency
- **Container count**: Number of containers used
- **Stability rate**: Percentage of stable placements
- **Completion rate**: Percentage of successfully packed objects -->

### Running Tests
```bash
python tap_train.py --train 0 --resume-path ./checkpoints/policy.pth
```

## 📊 Results


### Visualization
The `render/` directory contains tools for visualizing packing sequences and results.

## 🔗 Citation

If you find our work useful in your research, please cite:

```bibtex
@article{Xu2023NeuralPacking,
  title={Neural Packing: from Visual Sensing to Reinforcement Learning},
  author={Xu, Juzhan and Gong, Minglun and Zhang, Hao and Huang, Hui and Hu, Ruizhen},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={Article 269},
  year={2023},
  publisher={ACM}
}
```

<!-- ## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details. -->

<!-- ## 🙏 Acknowledgments

- Built on [Tianshou](https://github.com/thu-ml/tianshou) RL library -->
<!-- 
## ❓ FAQ

**Q: What's the difference between TAP-Net and TAP-Net++?**
A: TAP-Net++ extends TAP-Net with improved architecture, stability prediction, and multi-container support.

**Q: How do I visualize packing results?**
A: Use the scripts in the `render/` directory to generate visualizations.

**Q: Can I use this for real-world robotic packing?**
A: The framework is designed with real-world constraints in mind, but additional sensor integration may be needed for deployment.

**Q: What are the system requirements?**
A: 8GB+ RAM, GPU with 4GB+ VRAM recommended for training. Inference can run on CPU. -->
