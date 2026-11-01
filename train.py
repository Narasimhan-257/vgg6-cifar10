"""Train VGG6 on CIFAR-10 with one configuration and log every epoch to Weights & Biases.

Example:  python train.py --activation gelu --optimizer adam --lr 0.001 --epochs 10
"""

import argparse, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math

from data import get_loaders
from model import vgg6
from evaluate import evaluate

# A fair optimizer comparison needs each optimizer at a sensible LR:
# SGD-type methods need a much larger step than Adam-type methods.
DEFAULT_LR = {"sgd": 0.05, "nesterov": 0.05, "adam": 1e-3,
              "adagrad": 1e-2, "rmsprop": 1e-3, "nadam": 1e-3}

def set_seed(seed):
    """Seed every random number generator so a run can be reproduced exactly (Q5b)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def build_optimizer(name, params, lr, momentum=0.9, weight_decay=5e-4):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return  optim.SGD(params, lr=lr, momentum=momentum or 0.9, nesterov=True,
                         weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass over the training set. Training accuracy is counted on the batches as
    they are trained, so no second pass over the data is needed (the template did one)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += outputs.argmax(1).eq(y).sum().item()
        seen += bs
        if not math.isfinite(total_loss):
            return float("nan"), 100.0 * correct / max(seen, 1)
    return total_loss / seen, 100.0 * correct / seen

def parse_args():
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--activation", default="relu")
    p.add_argument("--optimizer", default="nesterov")
    p.add_argument("--lr", type=float, default=-1, help="<=0 uses the optimizer's default LR")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--save_path", default="", help="default: checkpoints/<run name>.pt")
    p.add_argument("--project", default="vgg6-cifar10")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()

def main():
    args = parse_args()
    if args.lr <= 0:
        args.lr = DEFAULT_LR[args.optimizer.lower()]
    run_name = f"{args.activation}_{args.optimizer}_lr{args.lr:g}_bs{args.batch_size}_ep{args.epochs}"
    wandb.init(project=args.project, config=vars(args), name=run_name, mode=args.wandb_mode)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.data_dir, args.num_workers)

    model = vgg6(args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                args.momentum, args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_path = args.save_path or f"checkpoints/{run_name}.pt"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best_val_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()                                   # cosine: lower the LR for next epoch

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now})
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.2f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.2f} | lr {lr_now:.5f} | {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:                          # keep the best epoch, judged on VALIDATION
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "epoch": epoch, "val_acc": val_acc}, save_path)

    # Test set is used exactly once: on the best-on-validation checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device)["model_state"])
    _, test_acc = evaluate(model, test_loader, criterion, device)
    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["test_acc"] = test_acc
    print(f"Best val acc {best_val_acc:.2f}% (epoch {best_epoch}) | test acc {test_acc:.2f}% | saved {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()



    

