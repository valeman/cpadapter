# cpadapter

Getting confidence intervals with the right coverage properties ("we really mean it when we say 95%") is a given in more traditional statistical models, but is usually missing from most of the machine learning applications we've seen.

In our experience, conformal prediction gives the best results for confidence intervals, but is a relatively unknown technique.

This package aims to make it easy to plug your favorite scikitlearn, lightgbm, catboost or xgboost models and get confidence intervals based on conformal prediction. We support both regression and classification models.

We additionaly provide some utility function to visualize the confidence intervals



Uses the noncomformist package (https://github.com/donlnz/nonconformist)

## Installation
Via [PyPi](https://pypi.org/project/cpadapter/):

```python
pip install cpadapter
```

## Usage

```python
from sklearn.ensemble import RandomForestRegressor,
import lightgbm as lgbm
import pandas as pd
import cpadapter
from cpadapter.utils import train_cal_test_split
from cpadapter.visualization import conditional_band_interval_plot


path = "https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv"
categorical_columns = ['sex', 'smoker', 'region']

df_raw = pd.read_csv(path, dtype={cat: 'category' for cat in categorical_columns})

for col in categorical_columns:
    df_raw[col] = df_raw[col].cat.codes

x_train, y_train, x_cal, y_cal, x_test, y_test = train_cal_test_split(df_raw, 'charges', 0.7, 0.2, True)
confidence = 0.8

# 1. Fit a scikit-learn model
##########################
model = RandomForestRegressor(n_estimators=100)
cp_model = cpadapter.Adapt_to_CP(model, True)

cp_model.fit(x_train, y_train)
cp_model.calibrate(x_cal, y_cal)
lower, pred_target, upper = cp_model.predict(x_test, confidence)
#Alternatively, we can use the one-liner cp_model.calibrate_and_predict(x_cal, y_cal, x_test, confidence)

conditional_band_interval_plot(y_test, lower, upper, sort=True)
```

![scikit_ci](https://user-images.githubusercontent.com/3705969/113208400-24584f80-9248-11eb-892e-715e9a02a308.png)

Note that most of the data points are within the confidence interval (we expect around 80% if everything is working out ok).

Now we can use the exact same syntax to obtain the confidence interval for a different model (from lightgbm in this case)

```python
#2. Fit a light-gbm model
#########################
lgbm_model = lgbm.LGBMRegressor(n_estimators=100)
cp_model = cpadapter.Adapt_to_CP(lgbm_model, True)
cp_model.fit(x_train, y_train)

lower, pred_target, upper = cp_model.calibrate_and_predict(x_cal, y_cal, x_test, confidence)
conditional_band_interval_plot(y_test, lower, upper, sort=True)
```

![lightgbm_ci](https://user-images.githubusercontent.com/3705969/113208449-2fab7b00-9248-11eb-8855-83b1165b8425.png)


As expected, the confidence interval plot is similar, though not the same



## Note

This package started from an internship of María Jesús Ugarte (https://github.com/mjesusugarte) at Spike.
