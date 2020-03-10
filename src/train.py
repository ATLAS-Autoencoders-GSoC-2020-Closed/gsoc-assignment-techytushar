import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import fastai
from fastai import data_block, basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr


class autoencoder(nn.Module):
    def __init__(self, n_features=4):
        super().__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.05)
        self.encoder = nn.Sequential(nn.Linear(n_features, 200),
                                     lrelu,
                                     nn.Linear(200, 100),
                                     lrelu,
                                     nn.Linear(100, 50),
                                     lrelu,
                                     nn.Linear(50, 3))
        self.decoder = nn.Sequential(lrelu,
                                     nn.Linear(3, 50),
                                     lrelu,
                                     nn.Linear(50, 100),
                                     lrelu,
                                     nn.Linear(100, 200),
                                     lrelu,
                                     nn.Linear(200, n_features))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def describe(self):
        return 'autoencoder(200-100-50-3-50-100-200)'


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def prepare_data(X, bs):
    train_ds = TensorDataset(torch.tensor(X), torch.tensor(X))
    valid_ds = TensorDataset(torch.tensor(X_test), torch.tensor(X_test))
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
    db = basic_data.DataBunch(train_dl, valid_dl)
    return db

def train_ae(X, bs=1024, lr=1e-03, wd=1e-04, epochs=100):
    model = autoencoder()
    db = prepare_data(X, bs)
    learn = basic_train.Learner(data=db,
                                model=model,
                                loss_func=nn.MSELoss(),
                                wd=wd,
                                callback_fns=ActivationStats,
                                bn_wd=False,
                                true_wd=True)
    learn.fit(epochs, lr=lr, wd=wd)
    return model
