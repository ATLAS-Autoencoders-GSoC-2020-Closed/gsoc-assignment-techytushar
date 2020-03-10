import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from train import autoencoder
from utils import get_device

def predict(X, path):
    """
    Predicts output from autoencoder

    Parameters:
    X (numpy array): original values
    path (str): path to saved model state dict

    Returns:
    numpy array: predicted values
    """
    assert X.dtype=='float32', "dtype must be float32"
    model = autoencoder()
    model.load_state_dict(torch.load(path))
    device = get_device()
    test_value = torch.tensor(X_test)
    test_value = test_value.to(device)
    pred = model(test_value).detach().cpu().numpy()
    return pred
