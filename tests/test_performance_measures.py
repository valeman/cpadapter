import numpy as np
import pytest
from cpadapter.performance_measures import *
# Tests the performance measures for classification
@pytest.fixture
def class_prediction():
    '''Returns an array of classes'''
    return np.array([0, 2, 2, 1])

@pytest.fixture
def class_interval_prediction():
    '''Returns the predicted confidence interval for class_prediction'''
    return np.array([[True, True, True], [False, False, True], [True, False, False], [False, True, True]])

def test_right_guess(class_prediction, class_interval_prediction):
    y = class_prediction
    pred_interval = class_interval_prediction
    assert right_guess(y, pred_interval) == 0.75

def test_right_guess_float(class_prediction, class_interval_prediction):
    coverage = right_guess(class_prediction, class_interval_prediction)
    assert isinstance(coverage, float)

def test_uncertainty(class_interval_prediction):
    pred_interval = class_interval_prediction
    assert uncertainty(pred_interval) == 0.25

def test_uncertainty_float(class_interval_prediction):
    un = uncertainty(class_interval_prediction)
    assert isinstance(un, float)

def test_width(class_interval_prediction):
    pred_interval = class_interval_prediction
    assert width(pred_interval) == 7/4/3

def test_width_float(class_interval_prediction):
    w = width(class_interval_prediction)
    assert isinstance(w, float)

# Test performance measures for regression cases

@pytest.fixture
def reg_prediction():
    '''Returns an array of predicted target values'''
    return np.array([0.5, 0.7, 0.2, 0.8])

@pytest.fixture
def reg_interval_lower():
    '''Returns the lower bound of the predicted interval for reg_prediction'''
    return np.array([0.5, 0.6, 0.3, 0.6])

@pytest.fixture
def reg_interval_upper():
    '''Returns the upper bound of the predicted interval for reg_prediction'''
    return np.array([0.6, 0.8, 0.6, 0.8])

def test_picp(reg_prediction, reg_interval_lower, reg_interval_upper):
    assert picp(reg_prediction, reg_interval_lower, reg_interval_upper) == 0.75

def test_picp_float(reg_prediction, reg_interval_lower, reg_interval_upper):
    coverage = picp(reg_prediction, reg_interval_lower, reg_interval_upper)
    assert isinstance(coverage, float)

def test_pinaw(reg_prediction, reg_interval_lower, reg_interval_upper):
    assert pinaw(reg_prediction, reg_interval_lower, reg_interval_upper) == 0.2/0.6

def test_pinaw_float(reg_prediction, reg_interval_lower, reg_interval_upper):
    width_norm = pinaw(reg_prediction, reg_interval_lower, reg_interval_upper)
    assert isinstance(width_norm, float)

def test_relative_width(reg_prediction, reg_interval_lower, reg_interval_upper):
    interval = reg_interval_upper - reg_interval_lower
    assert relative_width(reg_prediction, reg_interval_lower, reg_interval_upper) == np.median(interval/reg_prediction)

def test_relative_width_float(reg_prediction, reg_interval_lower, reg_interval_upper):
    rel_width = relative_width(reg_prediction, reg_interval_lower, reg_interval_upper)
    assert isinstance(rel_width, float)

def test_relative_mean_width(reg_prediction, reg_interval_lower, reg_interval_upper):
    width = reg_interval_upper - reg_interval_lower
    assert relative_mean_width(reg_prediction, reg_interval_lower, reg_interval_upper) == width.mean()/0.55

def test_relative_mean_width_float(reg_prediction, reg_interval_lower, reg_interval_upper):
    rel_mean_width = relative_mean_width(reg_prediction, reg_interval_lower, reg_interval_upper)
    assert isinstance(rel_mean_width, float)