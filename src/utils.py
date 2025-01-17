# -*- coding: utf-8 -*-
"""
Utility functions.
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression

import constants


def read_files(
    base_path: str, classes: list, real_only: bool = True, test_size: float = 0
) -> tuple[list, list]:
    """
    Read files from path and return list of filenames.

    Args:
        basepath (str): Path of parent folder.
        classes (list): List of classes to read.
        real_only (bool): Only read real instances.
        test_size (float): Split size for test files.

    Returns:
        list: List of train filenames.
        list: List of test filenames.
    """

    train_files = []
    test_files = []

    for c in classes:
        directory = os.path.join(base_path, str(c))
        files = os.listdir(directory)

        if real_only:
            fnames = [file for file in files if "WELL" in file]
            files = fnames

        files = [os.path.join(base_path, str(c), file) for file in files]

        split = int(len(files) * test_size)

        train_files.extend(files[split:])
        test_files.extend(files[:split])

    return train_files, test_files


def create_sequence(data: np.ndarray, steps: int = constants.STEPS) -> np.ndarray:
    """
    Split time series into multiple time windows, given the timestep size.

    Args:
        data (np.ndarray): Time series.
        steps (int): Timesteps to split.

    Returns:
        np.ndarray: Sequences split in timesteps.
    """

    x = []

    for i in range(0, len(data) - steps, steps):
        x.append(data[i : (i + steps)])

    return np.array(x)


def get_features(window: np.ndarray) -> list:
    """
    Calculate features from time window.
    Format: [Mean, Std, Var, Min, Max]

    Args:
        window (np.ndarray): Time series.

    Returns:
        list: Features from time series.
    """

    mean = window.mean(axis=0)
    std = window.std(axis=0)
    # var = window.var(axis=0)
    # min_value = window.min(axis=0)
    # max_value = window.max(axis=0)

    x = np.arange(constants.STEPS).reshape(-1, 1)
    lr = LinearRegression().fit(x, window)

    # return [mean, std, var, min_value, max_value]
    return [mean, std, lr.coef_.flatten()]


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute the Precision Score.

    Precision = TP / (TP + FP)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        pos_label (int): Label of positive class.

    Returns:
        float: Precision Score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.logical_and((y_true == pos_label), (y_pred == pos_label)).sum()
    fp = np.logical_and((y_true != pos_label), (y_pred == pos_label)).sum()

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute the Recall Score.

    Recall = TP / (TP + FN)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        pos_label (int): Label of positive class.

    Returns:
        float: Recall Score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.logical_and((y_true == pos_label), (y_pred == pos_label)).sum()
    fn = np.logical_and((y_true == pos_label), (y_pred != pos_label)).sum()

    return tp / (tp + fn)


def f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute the F1 Score.

    F1 = 2 * Precision * Recall / (Precision + Recall)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        pos_label (int): Label of positive class.

    Returns:
        float: F1 Score.
    """

    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)

    return 2 * p * r / (p + r)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of prediction.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return (y_true == y_pred).mean()
