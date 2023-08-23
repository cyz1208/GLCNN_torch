import os
import time
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from torchvision.transforms import ToTensor
from typing import Iterator, Sequence
from graph import Utils


# class PixelImageDataset(Dataset):
#     """
#     custom dataset, raw data stored in data/ dir and id_prop.csv file, not used.
#     """
#
#     def __init__(self, annotations_file, img_dir, to_cuda=True, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         # load image
#         img_path = os.path.join(self.img_dir, "pixel", self.img_labels.iloc[idx, 0] + '.pkl')
#         with open(img_path, "rb") as f:
#             image = pkl.load(f)
#         image = torch.Tensor(image)
#
#         # load feature
#         fea_path = os.path.join(self.img_dir, "descriptor",
#                                 "fea_" + self.img_labels.iloc[idx, 0].split("-")[0] + '.pkl')
#         with open(fea_path, "rb") as f:
#             fea = pkl.load(f)
#         fea = torch.Tensor(fea)
#
#         # load y
#         label = torch.Tensor([self.img_labels.iloc[idx, 1]])
#
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#
#         return image, fea, label


def data_augmentation_y(y, times=20):
    """
    input shape: [batch, ...], return shape: [batch * times, ...]
    """
    y_tmp = []
    for i in y:
        for _ in range(times):
            y_tmp.append(i)
    return np.array(y_tmp)


def data_clean(*filenames):
    """
    read name of unstable structures
    """
    print('data cleaning ...')
    name = []
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.readlines()
        for l in data:
            name.append(l.split('	')[:-1])
    return name


class PixelImageInMemoryDataset(Dataset):
    """
    InMemory dataset, raw data stored in pixels.pkl, graphs.pkl and properties.csv in data/ folder
    """

    def __init__(self, pixel_dir, graph_dir, property_dir, to_cuda=True, transform=None, target_transform=None):
        print("LOADING data from files ...")
        clean_name = data_clean('./structure.log')

        with open(pixel_dir, "rb") as f:
            self.pixels = np.array([pixel for name, pixel in pkl.load(f) if name.split() not in clean_name])
        self.pixels = np.transpose(self.pixels, (0, 3, 1, 2))   # channel last to channel first
        self.pixels = torch.Tensor(self.pixels)
        print(f"total pixels shape: {self.pixels.shape}")

        with open(graph_dir, "rb") as f:
            self.graphs = pkl.load(f)
        graph_total = []
        for g in self.graphs:
            if '' in g.name and g.name.split() not in clean_name:  # developer mode
                graph_total.append(Utils.get_shells(g))
                # graph_total.append(Utils.get_shell_laplacian(g))
        graph_total = np.array(graph_total)
        graph_total = scale(graph_total)
        graph_total = data_augmentation_y(graph_total)
        self.graphs = torch.Tensor(graph_total)
        print(f"total graphs shape: {self.graphs.shape}")

        df = pd.read_csv(property_dir)
        self.properties = []
        for _, datum in df.iterrows():
            # data clean
            if [datum['mesh'], datum['add_N'], datum['sub'], datum['metal']] not in clean_name:
                self.properties.append(datum['property'])
        self.properties = data_augmentation_y(self.properties).reshape((-1, 1))
        self.properties = torch.Tensor(self.properties)
        print(f"total y shape: {self.properties.shape}")

        self.transform = transform
        self.target_transform = target_transform

        if to_cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.pixels, self.graphs, self.properties = (
                self.pixels.to(device), self.graphs.to(device), self.properties.to(device))

    def __len__(self):
        return len(self.properties)

    @property
    def origin_length(self):
        return int(len(self.properties) / 20)

    def __getitem__(self, idx):
        # load image
        image = self.pixels[idx]

        # load graph
        fea = self.graphs[idx]

        # load y
        label = self.properties[idx]

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

    # usage of custom dataset
    # dataset = PixelImageDataset(os.path.join(root_dir, "data/id_prop.csv"),
    #                             os.path.join(root_dir, "data"),
    #                             )

    dataset = PixelImageInMemoryDataset(os.path.join(root_dir, "user_data/pixels.pkl"),
                                        os.path.join(root_dir, "user_data/graphs.pkl"),
                                        os.path.join(root_dir, "user_data/properties.csv"),
                                        )

    # split train, val, test set index
    len_data = dataset.origin_length
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    raw_idx = np.array(list(range(len_data)))
    train_idx, test_idx = train_test_split(raw_idx, test_size=test_ratio, random_state=None)
    val_idx, test_idx = train_test_split(test_idx, test_size=test_ratio / (val_ratio + test_ratio), random_state=None)
    # print(len(train_idx), len(val_idx), len(test_idx))

    # augment set index
    train_idx = aug_idx(train_idx)
    val_idx = aug_idx(val_idx)
    test_idx = aug_idx(test_idx)
    # print(len(train_idx), len(val_idx), len(test_idx))

    # create sampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = MySubsetSampler(val_idx)
    # test_sampler = SubsetRandomSampler(test_idx)
    test_sampler = MySubsetSampler(test_idx)

    # create data loader
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False, sampler=train_sampler,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(dataset, batch_size=len(val_sampler), shuffle=False, sampler=val_sampler,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(dataset, batch_size=len(test_sampler), shuffle=False, sampler=test_sampler,
                             num_workers=0, pin_memory=False)
    print("DONE.")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = loader()
    print(f"len of train, val, test loader: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    for i, data in enumerate(test_loader):
        images, fea, labels = data
        print(f"shape of image, descriptor and label: {images.shape}, {fea.shape}, {labels.shape}")
        print(f"type of image, descriptor and label: {type(images)}, {type(fea)}, {type(labels)}")
        print(labels.flatten()[::20])
        exit()
