import argparse

import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from pytorchtools import EarlyStopping
import torch
from data_loader import loader


class Net(nn.Module):
    """
    multi-inputs model
    """

    def __init__(self):
        super(Net, self).__init__()
        self.Lenet5_conv_part = nn.Sequential(
            nn.Conv2d(6, 6, (5, 5)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6, affine=False),
            nn.ReLU(),
            nn.Conv2d(6, 16, (5, 5)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 120, (5, 5)),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(99, 2000)
        self.fc3 = nn.Linear(2000, 200)
        self.fc4 = nn.Linear(200, 1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x_1, x_2):
        x_1 = self.Lenet5_conv_part(x_1)
        # x = x.view(-1, 16 * 5 * 5)
        x_1 = self.flatten(x_1)
        x_1 = F.relu(self.fc1(x_1))
        x_1 = nn.Dropout(0.2)(x_1)
        x = torch.cat((x_1, x_2), dim=-1)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.2)(x)
        x = F.relu(self.fc3(x))
        x = nn.Dropout(0.2)(x)
        return self.fc4(x)


def train_one_epoch(
        model,
        train_loader,
        loss_fn,
        metric_fn,
        optimizer,
        scheduler):
    """
    train model in one epoch

    Args:
            model:
            train_loader:
            loss_fn:
            metric_fn:
            optimizer:
            scheduler:

    Returns:
            train loss and metric after one epoch and consumed time
    """
    start_time = time.perf_counter()
    # train mode
    model.train(mode=True)

    train_loss = 0.0
    train_metric = 0.0
    for i, data in enumerate(train_loader):
        input_1, input_2, labels = data

        # transfer data to gpu
        # input_1, input_2, labels = input_1.to(device), input_2.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_1, input_2)
        loss = loss_fn(outputs, labels)
        metric = metric_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # sum train loss of all mini-batches for every epoch
        train_loss += loss.item()
        train_metric += metric.item()

    scheduler.step()
    # print(f"current lr: {scheduler.get_last_lr()}")

    end_time = time.perf_counter()
    # return loss and time
    return train_loss / len(train_loader), train_metric / \
        len(train_loader), end_time - start_time


def test(model, test_loader, metric_fn, metric_fn_DA):
    """
    evaluate or test model.
    Args:
            model:
            test_loader:
            metric_fn:
            metric_fn_DA:

    Returns:
    MAE, MAE with DA and consumed time.
    """
    start_time = time.perf_counter()
    # inference mode
    model.eval()

    # model inference
    with torch.no_grad():
        test_metric = 0.0
        test_metric_DA = 0.0   # MAE using DA
        for data in test_loader:
            input_1, input_2, labels = data

            # transfer data to gpu
            # input_1, input_2, labels = input_1.to(device), input_2.to(device), labels.to(device)
            y_origin = labels.flatten()[::20]

            # get predicted values
            outputs = model(input_1, input_2)

            # sum test loss of all mini-batches
            test_metric += metric_fn(outputs, labels).item()
            test_metric_DA += metric_fn_DA(outputs.reshape(-1,
                                           20).mean(axis=-1), y_origin).mean().item()

    end_time = time.perf_counter()
    # return test loss and time
    return test_metric / len(test_loader), test_metric_DA / \
        len(test_loader), end_time - start_time


# class Scheduler(LRScheduler):
# 	"""
# 	define LR scheduler, not completed yet, use LambdaLR.
# 	"""
# 	def __init__(self, optimizer, warmup=20, verbose=False):
# 		self.warmup = warmup
# 		super(Scheduler, self).__init__(optimizer)
#
# 	def get_lr(self) -> float:
# 		if self.last_epoch < self.warmup:
# 			# return 0.0001  # constant warmup
# return 0.0001 + 0.0009 * (epoch + 1) / self.warmup  # Linear increase
# warmup


if __name__ == "__main__":
    WARNUP = 20
    INIT_LR = 0.0001
    MAX_LR = 0.001
    DECAY_LR = -0.015
    EPOCH = 200

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use demo catalysts")
    args = parser.parse_args()

    DEMO = args.demo

    np.set_printoptions(suppress=True)
    # determine if any gpus is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"device: {device}")

    # transfer net to gpu
    net = Net().to(device)

    # define loss object, optimizer and scheduler
    loss_fn = nn.MSELoss()
    metric_fn = nn.L1Loss()
    metric_fn_DA = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=INIT_LR)

    # scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1, verbose=False)
    # scheduler = Scheduler(optimizer, warmup=WARNUP, verbose=False)
    # the outputs of lr_lambda multiply with base_lr. LR firstly increases from INIT_LR to MAX_LR in WARMUP steps,
    # then exponentially decays with DECAY_LR coefficient. LR will always be
    # no less than INIT_LR.
    def lr_lambda(epoch_lr): return (INIT_LR + (MAX_LR - INIT_LR) / WARNUP * (epoch_lr - 1)) / INIT_LR \
        if epoch_lr <= WARNUP else max(MAX_LR * np.exp(DECAY_LR) ** (epoch_lr - 1 - WARNUP) / INIT_LR, 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

    # load train, val and test data
    train_loader, val_loader, test_loader = loader(DEMO)

    # Early stopping initializing
    early_stopping = EarlyStopping(
        patience=7, verbose=True, path="./checkpoint.pt")

    # train model
    best_metric = float("inf")
    for epoch in range(EPOCH):
        train_loss, train_metric, train_time = train_one_epoch(
            net, train_loader, loss_fn, metric_fn, optimizer, scheduler)
        val_metric, val_metric_DA, val_time = test(
            net, val_loader, metric_fn, metric_fn_DA)

        print(
            "epoch: {} train loss: {:.3f}, train_metric: {:.3f}, val metric: {:.3f}, val metric DA: {:.3f},"
            " train time: {:.3f}, val time: {:.3f}".format(
                epoch + 1,
                train_loss,
                train_metric,
                val_metric,
                val_metric_DA,
                train_time,
                val_time))

        if val_metric < best_metric:
            print(
                f"save best model: {np.round(best_metric, decimals=5)} -> {np.round(val_metric, decimals=5)}")
            best_metric = val_metric
            torch.save(net.state_dict(), "./checkpoint.pt")
        else:
            print(f"current best model: {best_metric}")

        # Early stopping and save best model
        # early_stopping(val_metric, net)
        # if early_stopping.early_stop:
        # 	print("Early stopping")
        # 	break

    # load best model
    net.load_state_dict(torch.load("./checkpoint.pt"))

    # test model
    test_metric, test_metric_DA, test_time = test(
        net, test_loader, metric_fn, metric_fn_DA)
    print("test metric: {:.3f}, test metric DA: {:.3f}, "
          "test time: {:.3f}".format(test_metric, test_metric_DA, test_time))

    # check model parameters
    # print(net)
    # for name, param in net.named_parameters():
    # 	print("name: {} param size: {}".format(name, param.shape))
