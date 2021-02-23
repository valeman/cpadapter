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
