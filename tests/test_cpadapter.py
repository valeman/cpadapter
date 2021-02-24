# Test cpadapter classes and functions
import numpy as np
import sklearn
import nonconformist
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pytest
import lightgbm as lgbm
from cpadapter.conformal_adaptor import Adapt_to_CP

# Test Adapt_to_CP class

@pytest.mark.parametrize("model, sk_model, adapt_model", [
    (RandomForestRegressor(), True, sklearn.ensemble.forest.RandomForestRegressor),
    (DecisionTreeClassifier(), True, sklearn.tree.tree.DecisionTreeClassifier),
    (lgbm.LGBMRegressor(), False, lgbm.sklearn.LGBMRegressor)])

def test_Adapt_to_CP_model(model, sk_model, adapt_model):
    adapted_mod = Adapt_to_CP(model, sk_model)
    assert type(adapted_mod.model) == adapt_model

@pytest.mark.parametrize("mod, sk_mod, icp", [
    (RandomForestRegressor(), True, nonconformist.icp.IcpRegressor),
    (DecisionTreeClassifier(), True, nonconformist.icp.IcpClassifier),
    (lgbm.LGBMRegressor(), False, nonconformist.icp.IcpRegressor)])

def test_Adapt_to_CP_icp(mod, sk_mod, icp):
    adapted_mod = Adapt_to_CP(mod, sk_mod)
    assert type(adapted_mod.icp) == icp

@pytest.fixture
def reg_dataset():
    '''Returns a regression toy dataset'''
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    train_per = 0.9
    n_total = X.shape[0]
    n_train = int((1 - train_per)*n_total)
    X_test = X[: n_train, :]
    X_train = X[n_train :, :]
    Y_test = y[: n_train].astype(int)
    Y_train = y[n_train :]
    cal_per = 0.2
    n_total = X_train.shape[0]
    n_train = int((1 - cal_per)*n_total)
    train_data = X_train[: n_train, :]
    cal_data = X_train[n_train :, :]
    train_target = Y_train[: n_train].astype(int)
    cal_target = Y_train[n_train :].astype(int)
    return (train_data, train_target, cal_data, cal_target, X_test, Y_test)

# fit method
def test_fit(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    model = RandomForestRegressor(n_estimators= 10)
    assert Adapt_to_CP(model, True).fit(x_train, y_train) == None

# calibrate method
def test_calibrate(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators= 10), True)
    adapt_model.fit(x_train, y_train)
    assert adapt_model.calibrate(x_cal, y_cal) == None

# predict method (lower bound)
def test_predict_lower(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    adapt_model.calibrate(x_cal, y_cal)
    pred = adapt_model.predict(x_test, 0.8)
    assert type(pred[0]) == np.ndarray

# predict method (target prediction)
def test_predict_pred(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    adapt_model.calibrate(x_cal, y_cal)
    pred = adapt_model.predict(x_test, 0.8)
    assert type(pred[1]) == np.ndarray

# predict method (target prediction)
def test_predict_upper(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    adapt_model.calibrate(x_cal, y_cal)
    pred = adapt_model.predict(x_test, 0.8)
    assert type(pred[2]) == np.ndarray

# calibrate_and_predict method (lower bound)
def test_calibrate_and_predict_lower(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    pred = adapt_model.calibrate_and_predict(x_cal, y_cal, x_test, 0.8)
    assert type(pred[0]) == np.ndarray

# calibrate_and_predict method (target prediction)
def test_calibrate_and_predict_pred(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    pred = adapt_model.calibrate_and_predict(x_cal, y_cal, x_test, 0.8)
    assert type(pred[1]) == np.ndarray

# calibrate_and_predict method (upper bound)
def test_calibrate_and_predict_upper(reg_dataset):
    x_train = reg_dataset[0]
    y_train = reg_dataset[1]
    x_cal = reg_dataset[2]
    y_cal = reg_dataset[3]
    x_test = reg_dataset[4]
    adapt_model = Adapt_to_CP(RandomForestRegressor(n_estimators=10), True)
    adapt_model.fit(x_train, y_train)
    pred = adapt_model.calibrate_and_predict(x_cal, y_cal, x_test, 0.8)
    assert type(pred[2]) == np.ndarray