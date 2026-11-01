# VGG6 on CIFAR-10 – CS6886W Assignment 1

**Lakshmi Narasimhan G (CS25M515)**
W&B project: https://wandb.ai/cs25m515-iit-madras-foundation/vgg6-cifar10

Training VGG6 on CIFAR-10 with different activations, optimizers, batch sizes, learning rates
and numbers of epochs, with every run logged to Weights & Biases.

## Files

| File | Purpose |
|---|---|
| `data.py` | CIFAR-10 loading, fixed 45k / 5k / 10k train / validation / test split, normalisation, augmentation |
| `model.py` | VGG6 (course template) with a configurable activation function |
| `train.py` | Trains one configuration, logs every epoch to W&B, saves the best-validation checkpoint |
| `evaluate.py` | `evaluate()` used during training + script that loads a checkpoint and reports val / test top-1 |
| `sweep_activation.yaml`, `sweep_optimizer.yaml`, `sweep_lr.yaml`, `sweep_batch.yaml`, `sweep_epochs.yaml` | W&B sweep definitions |
| `best_model.pt` | Trained weights + configuration of the best model |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
wandb login            # API key from https://wandb.ai/authorize
```

CIFAR-10 is downloaded automatically to `./data` on the first run.

## Best configuration (Question 4)

| Setting | Value |
|---|---|
| Activation | ReLU |
| Optimizer | Nesterov-SGD, momentum 0.9 |
| Learning rate | 0.05, cosine annealing to 0 over the run |
| Batch size | 128 |
| Epochs | 20 |
| Weight decay | 5e-4 |
| Batch norm | Yes |
| Seed | 42 |
| **Best validation top-1** | **83.40 % (epoch 19)** |
| **Test top-1** | **82.56 %** |

Reproduce it:

```bash
python train.py --activation relu --optimizer nesterov --lr 0.05 --batch_size 128 --epochs 20 --seed 42 --save_path best_model.pt
```

Evaluate the committed model without training:

```bash
python evaluate.py --checkpoint best_model.pt
```

## Data and transforms

- Split: 50,000 training images split once with `split_seed = 42` into 45,000 train and 5,000 validation. The 10,000 test images are used only to report the final accuracy of the best-validation checkpoint.
- Training: `RandomCrop(32, padding=4)` → `RandomHorizontalFlip()` → `ToTensor()` → `Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))` → `Cutout(1 hole, 16x16)`
- Validation / test: `ToTensor()` → `Normalize(...)` only.

## Running the sweeps (Questions 2 and 3)

Base configuration: ReLU, Nesterov-SGD, lr 0.05, batch 128, 10 epochs. Each sweep changes one setting.

```bash
wandb sweep sweep_activation.yaml          # prints a sweep ID
wandb agent <entity>/vgg6-cifar10/<sweep_id>
```

Repeat for `sweep_optimizer.yaml`, `sweep_lr.yaml`, `sweep_batch.yaml` and `sweep_epochs.yaml`.
In the optimizer sweep, `lr = -1` means each optimizer uses its default learning rate
(SGD / Nesterov 0.05, Adagrad 0.01, Adam / RMSprop / Nadam 0.001).

A single configuration can also be trained directly, e.g.

```bash
python train.py --activation gelu --optimizer adam --lr 0.001 --batch_size 128 --epochs 10
```

## Seed and reproducibility

`set_seed(42)` in `train.py` seeds Python `random`, NumPy, PyTorch (CPU and CUDA) and sets
`torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
The train / validation split uses a fixed `split_seed = 42`.
The same configuration run in two different sweeps gave identical results at every epoch
(78.26 % validation, 77.59 % test at 10 epochs).

## Environment

- Google Colab, NVIDIA Tesla T4 GPU
- Python 3, PyTorch + torchvision (Colab default build with CUDA), NumPy, wandb (see `requirements.txt`)

## Results summary

| Experiment | Values tried | Best setting | Best val / test (%) |
|---|---|---|---|
| Activation | ReLU, Sigmoid, Tanh, SiLU, GELU | ReLU (GELU 78.14) | 78.26 / 77.59 |
| Optimizer | SGD, Nesterov, Adam, Adagrad, RMSprop, Nadam | Nesterov | 78.26 / 77.59 |
| Learning rate | 0.01, 0.05, 0.1 | 0.1 | 80.66 / 80.01 |
| Batch size | 64, 128, 256 | 64 | 79.94 / 79.71 |
| Epochs | 5, 10, 20 | 20 | 83.40 / 82.56 |

The full analysis is in the submitted PDF report.
