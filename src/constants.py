# -*- coding: utf-8 -*-
"""
Constant parameters for the project.
"""

FILE_PATH = "D:/Projects/3W/dataset/"

NORMAL_CLASS = 0

# In this case, we will consider the benchmark proposed by Vargas (2019), with real instances that have windows of more
# than 20 minutes.
ABNORMAL_CLASSES = [1, 2, 5, 6, 7, 8]

# Time step window
STEPS = 60

# Split size of normal samples for train-test
TRAIN_SPLIT = 0.7

# Number of sensors (P-PDG, T-TPT, ...)
N_SENSORS = 6
