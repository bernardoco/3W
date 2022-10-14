# -*- coding: utf-8 -*-
"""
Utility functions.
"""

import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import constants

# TODO Define metrics functions without sklearn


def read_files(base_path: str, classes: list, real_only=True) -> list:
    """
    Read files from path and return list of filenames.

    Args:
        basepath (str): Path of parent folder.
        classes (list): List of classes to read.
        real_only (bool): Only read real instances.

    Returns:
        list: List of filenames.
    """

    files = []

    for c in classes:
        for file in os.listdir(os.path.join(base_path, str(c))):
            file_path = os.path.join(base_path, str(c), file)

            if real_only and "WELL" not in file_path:
                continue

            files.append(file_path)

    return files


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


def get_features(window: np.ndarray) -> np.ndarray:
    """
    Calculate features from time window.
    Format: [Mean, Std, Var, Min, Max]

    Args:
        window (np.ndarray): Time series.

    Returns:
        np.ndarray: Features from time series.
    """

    mean = window.mean(axis=0)
    std = window.std(axis=0)
    var = window.var(axis=0)
    min_value = window.min(axis=0)
    max_value = window.max(axis=0)

    return np.array([mean, std, var, min_value, max_value])


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Precision Score.

    Precision = TP / (TP + FP)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Precision Score.
    """

    return precision_score(y_true, y_pred)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Recall Score.

    Recall = TP / (TP + FN)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Recall Score.
    """

    return recall_score(y_true, y_pred)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 Score.

    Recall = 2 * Precision * Recall / (Precision + Recall)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: F1 Score.
    """

    return f1_score(y_true, y_pred)
