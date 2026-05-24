# FlowEqProp: Training Flow Matching Generative Models with Gradient Equilibrium Propagation

Official code for the ICONS '26 paper:

> **FlowEqProp: Training Flow Matching Generative Models with Gradient Equilibrium Propagation**
> Alex Gower
> International Conference on Neuromorphic Systems (ICONS), 2026

We introduce **Gradient Equilibrium Propagation (GradEP)**, a mechanism that extends Equilibrium Propagation (EP) to train energy gradients rather than energy minima. As a first demonstration, we apply GradEP to flow matching for generative modelling — **FlowEqProp** — training a two-hidden-layer MLP (24,896 parameters) on the Optical Recognition of Handwritten Digits dataset using only local equilibrium measurements and **no backpropagation**.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Reproduce the paper experiment (sklearn digits, spring-clamped EP):

```bash
torchrun --standalone --nproc_per_node=1 experiments/mnist/train_cifar_multigpu.py \
    --model_type=ep_mlp \
    --dataset=sklearn_digits \
    --ep_archi=64,128,128 \
    --ep_act=silu \
    --ep_init_gain=0.5 \
    --ep_spectral_norm=False \
    --ep_T=300 \
    --ep_epsilon=0.1 \
    --ep_T1=300 \
    --ep_T2=300 \
    --ep_beta=0.00125 \
    --ep_thirdphase=True \
    --ep_learning_mode=spring \
    --lambda_spring=15 \
    --output_scale=2 \
    --lr=1e-3 \
    --batch_size=1797 \
    --total_steps=2000 \
    --save_step=100 \
    --gen_mode=energy_gd \
    --grad_clip=1.0 \
    --warmup=0
```

Training takes approximately 1 hour on a single NVIDIA A100 GPU.

## Generation from checkpoint

```bash
python experiments/mnist/generate_from_checkpoint.py \
    --model_type=ep_mlp \
    --dataset=sklearn_digits \
    --ep_archi=64,128,128 \
    --ep_act=silu \
    --ep_init_gain=0.5 \
    --ep_spectral_norm=False \
    --ep_T=300 \
    --ep_epsilon=0.1 \
    --lambda_spring=15 \
    --output_scale=2 \
    --ep_learning_mode=spring \
    --gen_modes=spring,energy_gd \
    --gen_t1=1.0,1.2 \
    --gen_dt=0.01 \
    --ckpt=path/to/checkpoint.pt
```

## Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | 64 → 128 → 128 |
| Activation σ | SiLU |
| Weight init | Xavier normal (gain = 0.5) |
| Convergence steps T | 300 (all phases) |
| Step size ε | 0.1 |
| Spring stiffness λ | 15 |
| Output scale α | 2 |
| Nudge strength β | 1.25 × 10⁻³ |
| Optimizer | Adam (β₁=0.9, β₂=0.95) |
| Learning rate | 10⁻³ |
| Batch size | 1797 (full dataset) |
| Training epochs | 2000 |
| Generation dt | 0.01 |

## Repository structure

```
FlowEqProp/
├── experiments/mnist/
│   ├── train_cifar_multigpu.py    # Training script (spring-clamped EP)
│   ├── generate_from_checkpoint.py # Generation from saved checkpoint
│   ├── network_ep_mlp.py          # EP-compatible MLP energy model
│   └── config_multigpu.py         # Hyperparameter flags
├── utils_cifar_imagenet.py        # Training utilities
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

```bibtex
@inproceedings{gower2026floweqprop,
  title={FlowEqProp: Training Flow Matching Generative Models with Gradient Equilibrium Propagation},
  author={Gower, Alex},
  booktitle={International Conference on Neuromorphic Systems (ICONS)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
