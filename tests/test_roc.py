"""
@author: Tziporah
"""
import os

import numpy as np
import pandas as pd
import pytest

from ROC import ROCMetrics


@pytest.fixture
def roc():
    y_true = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
    p_pred = [0.7572399117476334, 0.7735783542845196, 0.777505111257681, 0.7888700032640826, 0.7673247887303871,
              0.771990703411931, 0.7567731295728173, 0.7684176847457296, 0.6976046580667323, 0.76326642236843]
    return ROCMetrics(pd.Series(y_true), pd.Series(p_pred))


@pytest.fixture
def y_pred():
    return pd.Series([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])


@pytest.fixture
def confusion_matrix():
    return np.array([[7, 0], [1, 2]])


def test_conf_matrix(roc, y_pred):
    conf_matrix = roc.conf_matrix(y_pred)
    assert conf_matrix.tolist() == [[7, 0], [1, 2]]


def test_sensitivity(roc, confusion_matrix):
    sensitivity = roc.sensitivity(confusion_matrix)
    assert sensitivity == 0.6666666666666666


def test_specificity(roc, confusion_matrix):
    specificity = roc.specificity(confusion_matrix)
    assert specificity == 1.0


def test_precision(roc, confusion_matrix):
    precision = roc.precision(confusion_matrix)
    assert precision == 1.0


def test_accuracy(roc, y_pred):
    accuracy = roc.accuracy(y_pred)
    assert accuracy == 0.9


def test_f1_score(roc, y_pred):
    f1_score = roc.f1_score(y_pred)
    assert f1_score == 0.8


def test_threshold_matrix(roc):
    thresholds = roc.threshold_matrix(0.1)
    assert thresholds.index.to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'sensitivity', 'specificity', 'precision',
                                          'accuracy', 'f1_score']
    assert thresholds.shape == (15, 10)


def test_roc_plot(roc):
    roc.roc_plot('temp.png')
    try:
        assert 'temp.png' in os.listdir()
    finally:
        os.remove('temp.png')
