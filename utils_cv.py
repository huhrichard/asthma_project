### Plotting 
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('ggplot')  # values: {'seaborn', 'ggplot', }
from utils_plot import saveFig, plot_path

"""


Reference 
---------
1. model selection via nested CV 

    a. model selection & performance metric: 

        https://scikit-learn.org/stable/model_selection.html

        + predefined scoring 
            https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    b. CV 

        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

        + GridSearchCV 
           https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html



"""

def run_nested_cv(X, y, model=None, p_grid={}, n_trials=30):
    from matplotlib import pyplot as plt

    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    import numpy as np
    import pandas as pd

    print(__doc__)

    # Number of random trials
    NUM_TRIALS = n_trials

    # Load the dataset
    X, y, features = get_diabetes_data()
    # X, y, features = load_data(input_file='diabetes.csv')

    if len(p_grid): 
        # Set up possible values of parameters to optimize over
        # p_grid = {"C": [1, 10, 100],
        #           "gamma": [.01, .1]}
        p_grid = {"max_depth": [3, 4, 5], 
                  "min_samples_leaf": [1, 0.01]}
    # ... The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches

    if model is None: 
        # We will use a Support Vector Classifier with "rbf" kernel
        # svm = SVC(kernel="rbf")
        model = DecisionTreeClassifier(criterion='entropy', random_state=1)

    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)

    # Loop for each trial
    icv_num = 5
    ocv_num = 5
    best_params = {}
    for i in range(n_trials):
        
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=icv_num, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=ocv_num, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                           iid=False)
        
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='roc_auc')
        nested_scores[i] = nested_score.mean()
        best_params[i] = clf.best_estimator_

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}."
          .format(score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
              x=.5, y=1.1, fontsize="15")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend([difference_plot],
               ["Non-Nested CV - Nested CV Score"],
               bbox_to_anchor=(0, 1, .8, 0))
    plt.ylabel("score difference", fontsize="14")

    plt.show()