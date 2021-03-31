# cpadapter

Getting confidence intervals with the right coverage properties ("we really mean it when we say 95%") is a given in more tradicional statistical models, but is usually missing from most of the machine learning applications we've seen.

In our experience, conformal prediction gives the best results for confidence intervals, but is a relatively unkown techique.

This package aims to make it easy to plug your favorite scikitlearn, lightgbm, catboost or xgboost models and get confidence intervals based on conformal prediction. We support both regression and classification models.

We additionaly provide some utility function to visualize the confidence intervals



Uses the noncomformist package (https://github.com/donlnz/nonconformist)

## Installation
Via [PyPi](https://pypi.org/project/cpadapter/):

```python
pip install cpadapter
```

## Usage

`TODO`

## Note

This package started from an internship of María Jesús Ugarte (https://github.com/mjesusugarte) at Spike.
