"""CIFAR-10 data: train/validation/test split, normalisation and augmentation."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
def get_transforms():
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)])

    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return train_tf, eval_tf

def get_loaders(batch_size, data_dir="./data", num_workers=2, val_size=5000, split_seed=42):
    """Returns train (45k, augmented), val (5k, not augmented) and test (10k) loaders."""
    train_tf, eval_tf = get_transforms()
    # Two copies of the SAME 50,000 training images, each with a different transform
    train_full = datasets.CIFAR10(data_dir, train=True, transform=train_tf, download=True)
    val_full   = datasets.CIFAR10(data_dir, train=True, transform=eval_tf, download=True)
    test_set   = datasets.CIFAR10(data_dir, train=False, transform=eval_tf, download=True)

    # Shuffle indices 0..49999 once with a fixed seed, so every run gets the SAME split
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx   = perm[:val_size]          
    train_idx = perm[val_size:]          

    train_set = Subset(train_full, train_idx)
    val_set   = Subset(val_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader  = DataLoader(val_set,  batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
