import numpy as np
import pandas as pd

from tabulate import tabulate
from utils_plot import saveFig
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

"""
Import rpart from R
"""
import rpy2
from rpy2.robjects import DataFrame, Formula, pandas2ri
import rpy2.robjects.numpy2ri as npr
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame, Formula
rpart = importr('rpart')
stats = importr('stats')



class rpart_wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, control_cp, control_xval=10, method='class'):
        self.control_cp = control_cp
        self.control_xval = control_xval
        self.method = method

    def fit(self, X, y):
        # Concat X and y into 1 df
        self.dataf_R = X.copy().update(y)
        self.formula = Formula('{} ~.'.format(y.col_name))
        self.tree = rpart.rpart(formula=self.formula, data=self.dataf_R,
            method=self.method,
            control=rpart.rpart_control(cp=0.0, xval=self.control_xval),
            model = stats.model_frame(self.formula, data=self.dataf_R))
        # Argmin of cp_min_idx
        cp_min_idx = self.tree.cptable[]
        self.pruned_tree = rpart.prune(self.tree, )
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        # rpart.predict
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        prediction = rpart.predict(self.tree, X, type="class")
        return prediction

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def get_path_from_rpart(rpart_model):
    rules = rpart.rules(rpart_model)
#     TODO: have to convert back to my original format

if __name__ == "__main__":
    check_estimator()