import numpy as np
import torch

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return np.sqrt(get_MSE(pred, real))

def get_NRMSE(pred, real):
    return np.sqrt(get_MSE(pred, real)) / np.mean(np.abs(real))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))

def get_MAPE(pred, real):
    return get_MAE(pred, real) / np.mean(np.abs(real))

def get_CORR(pred, real):
    pred = pred.reshape([pred.shape[0], -1])
    real = real.reshape([real.shape[0], -1])
    pred = pred - np.mean(pred, axis=1)[:, np.newaxis]
    real = real - np.mean(real, axis=1)[:, np.newaxis]
    return np.mean(np.sum(pred * real, axis=1) / (np.sqrt(np.sum(pred ** 2, axis=1)) * np.sqrt(np.sum(real ** 2, axis=1))))