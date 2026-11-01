"""Evaluation: top-1 accuracy and loss of a model on a data loader."""
import torch

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        total_loss += criterion(outputs, y).item() * y.size(0)
        correct += outputs.argmax(1).eq(y).sum().item()
        seen += y.size(0)
    return total_loss / seen, 100.0 * correct / seen

def main():
    """Load a saved checkpoint and report validation and test top-1 accuracy."""
    import argparse
    import torch.nn as nn
    from data import get_loaders
    from model import vgg6

    p = argparse.ArgumentParser(description="Evaluate a saved VGG6 checkpoint")
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--data_dir", default="./data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]                        # the settings the model was trained with

    model = vgg6(cfg["activation"]).to(device)                                
    model.load_state_dict(ckpt["model_state"])                                        

    _, val_loader, test_loader = get_loaders(cfg["batch_size"], args.data_dir)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("Configuration:", {k: cfg[k] for k in
          ["activation", "optimizer", "lr", "batch_size", "epochs", "momentum", "weight_decay", "seed"]})
    print(f"Saved at epoch {ckpt['epoch']}")
    print(f"Validation top-1: {val_acc:.2f}%  (loss {val_loss:.4f})")
    print(f"Test       top-1: {test_acc:.2f}%  (loss {test_loss:.4f})")


if __name__ == "__main__":
    main()
