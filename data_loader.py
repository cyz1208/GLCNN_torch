import os
import time
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from torchvision.transforms import ToTensor
from typing import Iterator, Sequence


class PixelImageDataset(Dataset):
    """
    custom dataset, raw data stored in data/ dir and id_prop.csv file
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.img_dir, "pixel", self.img_labels.iloc[idx, 0] + '.pkl')
        with open(img_path, "rb") as f:
            image = pkl.load(f)
        image = torch.Tensor(image)

        # load feature
        fea_path = os.path.join(self.img_dir, "descriptor", "fea_" + self.img_labels.iloc[idx, 0].split("-")[0] + '.pkl')
        with open(fea_path, "rb") as f:
            fea = pkl.load(f)
        fea = torch.Tensor(fea)

        # load y
        label = torch.Tensor([self.img_labels.iloc[idx, 1]])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, fea, label


class MySubsetSampler(Sampler[int]):
    r"""Samples elements in sequence.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


def aug_idx(idx, aug=20):
    """
    augment set index, ensure augmented data of one sample exist only in one set (train, val or test set)

    Args:
        idx: set index
        aug: augment times

    Returns:
        augmented set index
    """
    idx_new = []
    for i in idx:
        for j in range(aug):
            idx_new.append(i * aug + j)
    return idx_new


def loader():
    """
    load train, val and test data

    Returns:
        train, val and test loader
    """
    print("LOADING DATA ...")
    root_dir = os.getcwd()

    # split train, val, test set index
    len_data = 3132
    train_ratio, test_ratio = 0.8, 0.2
    raw_idx = np.array(list(range(len_data)))
    train_idx, test_idx = train_test_split(raw_idx, test_size=test_ratio, random_state=2022)
    # val_idx, test_idx = train_test_split(test_idx, test_size=test_ratio / (val_ratio + test_ratio), random_state=123)
    # print(len(train_idx), len(val_idx), len(test_idx))

    # augment set index
    train_idx = aug_idx(train_idx)
    # val_idx = aug_idx(val_idx)
    test_idx = aug_idx(test_idx)
    # print(len(train_idx), len(val_idx), len(test_idx))

    # usage of custom dataset
    dataset = PixelImageDataset(os.path.join(root_dir, "data/id_prop.csv"),
                                os.path.join(root_dir, "data"),
                                # transform=ToTensor(),
                                )

    # create sampler
    train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    # test_sampler = SubsetRandomSampler(test_idx)
    test_sampler = MySubsetSampler(test_idx)

    # create data loader
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    # val_loader = DataLoader(dataset, batch_size=len(val_sampler), shuffle=False, sampler=val_sampler,
    #                         num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=400, shuffle=False, sampler=test_sampler,
                             num_workers=2, pin_memory=True)
    print("DONE.")
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = loader()
    print(f"len of train and test loader: {len(train_loader)}, {len(test_loader)}")
    for i, data in enumerate(test_loader):
        images, fea, labels = data
        print(f"shape of image, descriptor and label: {images.shape}, {fea.shape}, {labels.shape}")
        # print(labels)
        # exit()



