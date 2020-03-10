import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(df):
    """
    Custom normalization for each feature

    Parameters:
    df (dataframe): pandas dataframe

    Returns:
    dataframe: dataframe with normalized values
    """
    df_temp = pd.DataFrame()
    df_temp['phi'] = df['phi'] / 3.0
    df_temp['eta'] = df['eta'] / 4.0
    df_temp['m'] = np.log10(df['m']+1)
    df_temp['pt'] = np.log10(df['pt'])
    return df_temp

def inverse_norm(df_norm):
    """
    Inverse normalization for each feature

    Parameters:
    df (dataframe): pandas dataframe

    Returns:
    dataframe: dataframe with rescaled values
    """
    df_temp = pd.DataFrame()
    df_temp['phi'] = df_norm['phi'] * 3.0
    df_temp['eta'] = df_norm['eta'] * 4.0
    df_temp['m'] = (10**df_norm['m'])-1
    df_temp['pt'] = 10**df_norm['pt']
    return df_temp

def plot_hist(X, pred, columns):
    """
    Plot histogram for comparision

    Parameters:
    X (numpy array): original values
    pred (numpy array): predicted values
    columns (list): column names

    Returns:
    ax: matplotlib plot object
    """
    fig, ax = plt.subplots(2,2, figsize=(14,7))

    ax[0][0].hist(X[:,0], bins=50, fc=(0,0,1,0.5), label='Original')
    ax[0][0].hist(pred[:,0], bins=50, fc=(1,0,0,0.5), label='Predicted')
    ax[0][0].set_title(f'{columns[0]}')

    ax[0][1].hist(X[:,1], bins=50, fc=(0,0,1,0.5), label='Original')
    ax[0][1].hist(pred[:,1], bins=50, fc=(1,0,0,0.5), label='Predicted')
    ax[0][1].set_title(f'{columns[1]}')

    ax[1][0].hist(X[:,2], bins=50, fc=(0,0,1,0.5), label='Original')
    ax[1][0].hist(pred[:,2], bins=50, fc=(1,0,0,0.5), label='Predicted')
    ax[1][0].set_title(f'{columns[2]}')

    ax[1][1].hist(X[:,3], bins=50, fc=(0,0,1,0.5), label='Original')
    ax[1][1].hist(pred[:,3], bins=50, fc=(1,0,0,0.5), label='Predicted')
    ax[1][1].set_title(f'{columns[3]}')

    handles, labels = ax[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.suptitle('Histogram of Reconstruction')
    fig.tight_layout()

    return ax
