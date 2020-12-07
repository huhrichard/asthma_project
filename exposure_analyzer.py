import os, sys
import collections, random

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as neg_bin
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
import os, fnmatch
from io import StringIO
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve, precision_recall_fscore_support
from scipy import stats
import matplotlib.ticker as mtick
import xgboost
# from imblearn.over_sampling import RandomOverSampler
import warnings
import operator
warnings.filterwarnings("ignore")
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from decimal import Decimal
from tabulate import tabulate
# from utils_plot import saveFig
from matplotlib.ticker import PercentFormatter
plt.rcParams.update({'font.size': 12})
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

inequality_operators = {'<': lambda x, y: x < y,
                        '<=': lambda x, y: x <= y,
                        '>': lambda x, y: x > y,
                        '>=': lambda x, y: x >= y}
"""
Import rpart from R
"""
# import rpy2
# from rpy2.robjects import DataFrame, Formula
# import rpy2.robjects.numpy2ri as npr
# rpy2.robjects.numpy2ri.activate()
# from rpy2.robjects.packages import importr
# rpart = importr('rpart')
# stats = importr('stats')

"""


Reference
---------
    1. Decision Tree 
        https://scikit-learn.org/stable/modules/tree.html

    2. Visualization 

        https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

    3. Nested vs non-nested: 

        https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#id2

"""
dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise
# plotDir = os.path.join(os.getcwd(), 'plot')

# plotDir = os.path.join(os.getcwd(), 'plot/nata_diagnose_year')
# output_fn = sys.argv[-1]
# output_folder_name = 'act_score'
# output_folder_name = 'emergency_dept'
# output_folder_name = 'hospitalize_overnight'
# output_folder_name = 'regular_medication'
# output_folder_name = 'regular_asthma_symptoms_past6months'

tree_seed = 0


class Data(object): 
    label = 'label'
    features = []

def f_max(y_true, y_pred):
    y_0 = (y_true == 0).sum()
    y_1 = (y_true == 1).sum()
    if y_1 > y_0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fscores = 2*precision*recall/(precision+recall)
    return np.nanmax(fscores)

def f_max_thres(y_true, y_pred, thres=None, minority=True):
    y_0 = (y_true == 0).sum()
    y_1 = (y_true == 1).sum()

    if (y_1 > y_0 and minority) or ((not minority) and y_1 < y_0):
        y_true = 1 - y_true
        y_pred = 1 - y_pred


    if thres is None:
        precision, recall, thres = precision_recall_curve(y_true, y_pred)
        fscores = 2*precision*recall/(precision+recall)
        return np.nanmax(fscores), thres[np.where(fscores==np.nanmax(fscores))][0]
    else:
        y_pred[y_pred > thres] = 1
        y_pred[y_pred <= thres] = 0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return fmeasure

f_max_score_func = make_scorer(f_max)

def features_to_txt(features, output_fn = 'fmap.txt'):
    output_f = open(output_fn, 'w')
    for idx, f in enumerate(features):
        # output_f.write('{}\t{}\tq'.format(idx, 'dummy_'+str(idx)))
        output_f.write('{}\t{}\tq'.format(idx, f.replace(' ', '_')))
        if idx+1 != len(features):
            output_f.write('\n')

    output_f.close()
    return output_fn


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # print(name)
            if fnmatch.fnmatch(name, pattern):
                # print('matched')
                result.append(name)
    return result

def subscript_like_chemical_name(pollutant_name):
    numbers = [2.5, 2, 4]
    for num in numbers:
        l = len(str(num))
        if str(num) == pollutant_name[-1*l:]:
            pollutant_name = "${}$".format(pollutant_name[:-1*l] + "_{"+ str(num) + "}")
            return pollutant_name
    return pollutant_name

def year_str_for_title(yr):
    if yr == '0':
        return ''
    elif int(yr) > 0:
        return '+' + yr
    else:
        return yr

def plot_histogram(asthma_df, plot_dir, pollutant_name, thres=0, label_plot=['with asthma', 'w/o asthma']):
    df = asthma_df[[pollutant_name, 'label']]

    # pollutant_level_w_asthma = np.log10(df[asthma_df['label'] == 1][pollutant_name])
    # pollutant_level_wo_asthma = np.log10(df[asthma_df['label'] == 0][pollutant_name])
    pollutant_level_w_asthma = df[asthma_df['label'] == 1][pollutant_name]
    pollutant_level_wo_asthma = df[asthma_df['label'] == 0][pollutant_name]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, bins, _ = ax.hist([pollutant_level_w_asthma, pollutant_level_wo_asthma],
            weights=[np.ones(len(pollutant_level_w_asthma))/len(pollutant_level_w_asthma), np.ones(len(pollutant_level_wo_asthma))/len(pollutant_level_wo_asthma)],
            label=label_plot, color=['red', 'green'])



    ax.legend(loc='upper right')
    if thres != 0:
        ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    # ax.set_title(pollutant_name+'_imputed_data')

    pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(subscript_like_chemical_name(pollutant_list[0])))
    ax.set_ylabel('Fraction of children exposed (separated by class)'.format(pollutant_list[0]))
    # y_ticks = ax.get_yticks
    fig.savefig(os.path.join(plot_dir, "histogram_{}.png".format(pollutant_name)), bbox_inches="tight")



def plot_scatter(asthma_df, plot_dir, pollutant_name, thres=0, ylabel=''):
    df = asthma_df[[pollutant_name, 'label']]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df[pollutant_name], df['label'])
    ax.margins(x=0.001)
    if thres != 0:
        ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    # ax.set_title(pollutant_name+'_imputed_data')

    pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(subscript_like_chemical_name(pollutant_list[0])))
    ax.set_ylabel(ylabel)

    # if 'to' in pollutant_list[-1]:
    #     pollutant_start_yr, pollutant_end_yr = pollutant_list[-1].split('to')
    #
    #     title_start_yr = year_str_for_title(pollutant_start_yr)
    #     title_end_yr = year_str_for_title(pollutant_end_yr)
    #
    #     ax.set_title(
    #         'Histogram of {} level from (birth year{}) to (birth year{})'.format(pollutant_list[0], title_start_yr,
    #                                                                              title_end_yr))
    # else:
    #     pollutant_yr = year_str_for_title(pollutant_list[-1])
    #
    #     ax.set_title('Histogram of {} level in (birth year{})'.format(pollutant_list[0], pollutant_yr))
    # # plt.show()

    fig.savefig(os.path.join(plot_dir, "scatter_{}.png".format(pollutant_name)), bbox_inches="tight")

def plot_hist2d(asthma_df, plot_dir, pollutant_name, thres=0, ylabel=''):
    df = asthma_df[[pollutant_name, 'label']]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hist2d = ax.hist2d(df[pollutant_name], df['label'], bins=20, cmap=plt.cm.Greys)
    fig.colorbar(hist2d[-1])
    if thres != 0:
        ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    ax.set_title('2D histogram plot of outcome vs pollutant')

    pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(subscript_like_chemical_name(pollutant_list[0])))
    ax.set_ylabel(ylabel)

    # if 'to' in pollutant_list[-1]:
    #     pollutant_start_yr, pollutant_end_yr = pollutant_list[-1].split('to')
    #
    #     title_start_yr = year_str_for_title(pollutant_start_yr)
    #     title_end_yr = year_str_for_title(pollutant_end_yr)
    #
    #     ax.set_title(
    #         'Histogram of {} level from (birth year{}) to (birth year{})'.format(pollutant_list[0], title_start_yr,
    #                                                                              title_end_yr))
    # else:
    #     pollutant_yr = year_str_for_title(pollutant_list[-1])
    #
    #     ax.set_title('Histogram of {} level in (birth year{})'.format(pollutant_list[0], pollutant_yr))
    # # plt.show()

    fig.savefig(os.path.join(plot_dir, "hisf2d_{}.png".format(pollutant_name)), bbox_inches="tight")

def run_model_selection(X, y, model, p_grid={}, n_trials=30, scoring='roc_auc', output_path='', output_file='', create_dir=True, 
                        index=0, plot_=True, ext='tif', save=False, verbose=False):

    
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
        clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, n_jobs=-1)

        fit_params = {
                    "check_input": False}
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=scoring)
        nested_scores[i] = nested_score.mean()
        best_params[i] = clf.best_params_

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}."
          .format(score_difference.mean(), score_difference.std()))

    if plot_: 
        plt.clf()
        
        # Plot scores on each trial for nested and non-nested CV
        plt.figure()
        plt.subplot(211)
        non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
        nested_line, = plt.plot(nested_scores, color='b')
        plt.ylabel("score", fontsize="14")
        plt.legend([non_nested_scores_line, nested_line],
                   ["Non-Nested CV", "Nested CV"],
                   bbox_to_anchor=(0, .4, .5, 0))
        plt.title("Non-Nested and Nested Cross Validation",
                  x=.5, y=1.1, fontsize="15")

        # Plot bar chart of the difference.
        plt.subplot(212)
        difference_plot = plt.bar(range(n_trials), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend([difference_plot],
                   ["Non-Nested CV - Nested CV Score"],
                   bbox_to_anchor=(0, 1, .8, 0))
        plt.ylabel("score difference", fontsize="14")

        if save: 
            from utils_plot import saveFig
            if not output_path: output_path = os.path.join(os.getcwd(), 'analysis')
            if not os.path.exists(output_path) and create_dir:
                print('(run_model_selection) Creating analysis directory:\n%s\n' % output_path)
                os.mkdir(output_path) 

            if output_file is None: 
                classifier = 'DT'
                name = 'ModelSelect-{}'.format(classifier)
                suffix = n_trials 
                output_file = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix=name, suffix=suffix, index=index, ext=ext)

            output_path = os.path.join(output_path, output_file)  # example path: System.analysisPath

            if verbose: print('(run_model_selection) Saving model-selection-comparison plot at: {path}'.format(path=output_path))
            saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
        else: 
            plt.show()
        
    return best_params, nested_scores

# convenient wrapper for DT classifier 
def classify(X, y, params={}, random_state=0, binary_outcome=True, **kargs):
    assert isinstance(params, dict)

    info_gain_measure = kargs.get('criterion', 'entropy')
    xgb = kargs.get('xgb', False)
    if 'nbt' in suffix:
        balance_training = False
    else:
        balance_training = True


    if binary_outcome:
        # if balance_training:
        #     ros = RandomOverSampler(random_state=random_state)
        #     X, y = ros.fit_resample(X, y)
        if xgb is False:
            model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
        else:
            model = xgboost.XGBClassifier(random_state=random_state)
        scoring = f_max_score_func
        cv_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=tree_seed)
    else:
        if xgb is False:
            model = DecisionTreeRegressor(criterion='mse', random_state=random_state)
        else:
            model = xgboost.XGBRegressor(random_state=random_state)
        scoring = None
        cv_split = KFold(n_splits=10, shuffle=True, random_state=tree_seed)

    if xgb:
        model.fit(X,y)
        return model
    else:
        # print('nan?', X.dtype)
        # if len(params) > 0: model = model.set_params(**params)

        # print(ccp_alphas)
        # return model

        # leaf_search_space = np.append(np.linspace(0.02, 0.4, 5)*y.shape[0], 1).astype(int)[1:]
        leaf = False
        pruning = False
        if leaf:
            leaf_search_space = np.append(np.array([0.05, 0.1, 0.2, 0.3, 0.4])*y.shape[0], 1).astype(int)
        else:
            leaf_search_space = [1]
        params_list = []
        for leaf_search in leaf_search_space:
            params_dict = {'min_samples_leaf': [leaf_search]}
            model = model.set_params(**{'min_samples_leaf': leaf_search})
            model.fit(X, y)

            path = model.cost_complexity_pruning_path(X, y)
            # print(path)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            if pruning:
                params_dict['ccp_alpha'] = ccp_alphas[:-1][ccp_alphas[:-1]>=0]
            else:
                params_dict['ccp_alpha'] = [0.0]
            params_list.append(params_dict)

        if len(ccp_alphas) <= 1:
            print('Only 0/1 ccp_alpha value')
            return model
        else:
            final_tree = GridSearchCV(estimator=model, param_grid=params_list, cv=cv_split,
                                      scoring=scoring)
            final_tree.fit(X, y)
            return final_tree.best_estimator_

def plot_parameters_chosen_histogram(params_name, params_list, outcome, path):
    unique_vals, counts = np.unique(params_list, return_counts=True)
    counts = counts/sum(counts)
    fig1, ax1 = plt.subplots(1,1)
    ax1.bar(unique_vals, counts)
    ax1.set_xlabel('{}({})'.format(params_name, outcome))
    ax1.set_ylabel('Percentage')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig1.savefig('{}/{}_params_distribution_{}.png'.format(path, params_name, outcome))
    # ax1.ylabel('proportion of parameters chosen')

def score_collection(model, binary_outcome, X_train, y_train, X_test, y_test, scores_list):
    if binary_outcome:
        y_pred_train = model.predict_proba(X_train)[:, 1]
        _, thres_minority = f_max_thres(y_train, y_pred_train)
        _, thres_majority = f_max_thres(y_train, y_pred_train, minority=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        f_minority = f_max_thres(y_test, y_pred, thres=thres_minority)
        f_majority = f_max_thres(y_test, y_pred, thres=thres_majority, minority=False)
        scores_list[0].append(f_minority)
        scores_list[1].append(f_majority)
        auc_score = metrics.roc_auc_score(y_test, y_pred)
        scores_list[2].append(auc_score)
        # score_random_shuffle = f_max(y_test, )
        print_str = "{}th Tree Fmax Score: {}"
    else:
        y_pred = model.predict(X_test)
        scoring_func = metrics.r2_score
        print_str = "{}th Tree R2 Score: {}"
        score = scoring_func(y_test, y_pred)
        scores_list[0].append(score)

    return scores_list

def analyze_path(X, y, model=None, p_grid={}, feature_set=[], n_trials=100, n_trials_ms=10, save=False, output_path='', output_file='', 
                             create_dir=True, index=0, binary_outcome=True,  **kargs):
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from utils_tree import visualize, visualize_xgb, count_paths, count_paths2, count_features2, \
        get_feature_threshold_tree, count_paths_with_thres_sign, count_paths_with_thres_sign_from_xgb
    import time
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2)
    verbose = kargs.get('verbose', False)
    merge_labels = kargs.get('merge_labels', True)
    policy_count = kargs.get('policy_counts', 'standard') # options: {'standard', 'sample-based'}
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    validate_tree = kargs.get('validate_tree', True)
    plot_dir = kargs.get('plot_dir', '')
    plot_ext = kargs.get('plot_ext', 'tif')
    to_str = kargs.get('to_str', False)  # if True, decision paths are represented by strings (instead of tuples)
    outcome_dir = os.path.join(plot_dir, kargs.get('outcome_name', ''))
    outcome_name = kargs.get('outcome_name', '')
    xgb = kargs.get('xgb', False)
    fmap_fn = kargs.get('fmap_fn', 'fmap.txt')
    ####################
    
    labels = np.unique(y)
    N, Nd = X.shape
    
    if len(feature_set) == 0: feature_set = ['f%s' % i for i in range(Nd)]
        
    msg = ''
    if verbose: 
        msg += "(analyze_path) dim(X): {} | vars (n={}):\n...{}\n".format(X.shape, len(feature_set), feature_set)
        msg += "... class distribution: {}\n".format(collections.Counter(y))
    print(msg)

    # define model 
    if model is None: model = DecisionTreeClassifier(criterion='entropy', random_state=time.time())

    max_count = 0
    if 'multiple' in suffix:
        multiple_counts = True
    else:
        multiple_counts = False
    # run model selection 
    # if len(p_grid) > 0:
    #     best_params, nested_scores = \
    #       run_model_selection(X, y, model, p_grid=p_grid, n_trials=n_trials_ms, output_path=output_path, ext=plot_ext)
    #     the_index = np.argmax(nested_scores)
    #     the_params = best_params[np.argmax(nested_scores)]
        # print('> type(best_params): {}:\n{}\n'.format(type(best_params), best_params))
        
    # initiate data structures
    # TODO: count do not store
    paths = {}
    paths_threshold = {}
    paths_from = {}
    lookup = {}
    counts = {f: [] for f in feature_set} # maps features to lists of thresholds
    model_list = []
    # build N different decision trees and compute their statistics (e.g. performance measures, decision path counts)
    # auc_scores = []
    test_points = np.random.choice(range(n_trials), 1)
    list_min_sample_leaf = []
    list_ccp_alpha = []
    if binary_outcome:
        stratify = y
    else:
        # nz_y_median = np.median(y[y > 0])
        # y[y < nz_y_median] = 0
        # y[y >= nz_y_median] = 1
        stratify = y
    if binary_outcome:
        scores_list = [[], [], []]
        scores_rand_list = [[], [], []]
    else:
        scores_list = [[]]
        scores_rand_list = [[]]
    for i in range(n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i, stratify=stratify) # 70% training and 30% test
        # print("[{}] dim(X_test): {}".format(i, X_test.shape))

        # [todo]: how to reset a (trained) model?
        # print(y_train.unique())
        model = classify(X_train, y_train, params={}, random_state=i, binary_outcome=binary_outcome, xgb=xgb)
        # print('training:\n',model.get_booster().get_dump(with_stats=True)[50])
        # graph = visualize_xgb(model, feature_names=fmap_fn, labels=labels, file_name="{}th_tree".format(i), ext='tif', outcome_name=outcome_name)
        model_list.append(model)
        """
        Testing
        """
        # y_pred = model.predict(X_test)

        scores_list = score_collection(model=model, binary_outcome=binary_outcome, X_train=X_train,y_train=y_train,
                                       y_test=y_test, X_test=X_test, scores_list=scores_list)
        # print('testing:\n', model.get_booster().get_dump(with_stats=True)[50])
        # shuffle training set
        for shuffle_seed in range(10):
            # np.random.shuffle(y_train)
            y_permutated_train = np.random.permutation(y_train)
            model_rand = classify(X_train, y_permutated_train, params={}, random_state=i, binary_outcome=binary_outcome, xgb=xgb)
            scores_rand_list = score_collection(model=model_rand, binary_outcome=binary_outcome, X_train=X_train, y_train=y_permutated_train,
                                           y_test=y_test, X_test=X_test, scores_list=scores_rand_list)

        if i % 10 == 0:
            print('{}th run result finished!'.format(i))

        if xgb is False:
            list_min_sample_leaf.append(model.min_samples_leaf)
            list_ccp_alpha.append(model.ccp_alpha)
            paths, paths_threshold = count_paths_with_thres_sign(estimator=model, paths=paths,
                                                                 feature_names=feature_set,
                                                                 paths_threshold=paths_threshold)
            # print('paths:\n',paths)
            # print('paths_threshold:\n',paths_threshold)
        else:
            paths, paths_threshold, paths_from = count_paths_with_thres_sign_from_xgb(estimator=model, paths=paths,
                                                                                      paths_from=paths_from, index=i,
                                                                 feature_names=feature_set,
                                                                 paths_threshold=paths_threshold,
                                                                          multiple_counts=multiple_counts)
            if multiple_counts:
                max_count += len(model.get_booster().get_dump())
            else:
                max_count += 1
    # if xgb is False:
    list_params = {
               'min_sample_leaf': list_min_sample_leaf,
               'ccp_alpha': list_ccp_alpha,
                'max_count': max_count
               }
    # for k, v in list_params.items():
    #     plot_parameters_chosen_histogram(params_name=k, params_list=v, outcome=outcome, path=outputDir)
    # print("before summary:", paths)

    sorted_paths = summarize_paths(paths)


    paths_median_threshold = get_median_of_paths_threshold(paths_threshold)
    # topk = len(sorted_paths)
    topk = 10


    topk_profile_str, all_greater_path = topk_profile_with_its_threshold(sorted_paths, paths_median_threshold, topk=topk)
    all_greater_path_fn = os.path.join(plot_dir, '{}_all_greater_paths.csv'.format(outcome_name))
    all_greater_path_df = pd.DataFrame({'profile_name': [k for k, v in all_greater_path.items()],
                                       'count': [v for k, v in all_greater_path.items()]})
    all_greater_path_df.to_csv(all_greater_path_fn, index=False)
    return  {'scores': scores_list,
                                    'scores_random': scores_rand_list}, list_params, \
            topk_profile_str, sorted_paths, paths_median_threshold, \
            {'paths_from': paths_from, 'model_list': model_list}




def load_data(input_file, **kargs):
    """

    Memo
    ----
    1. Example datasets

        a. multivariate imputation applied
            exposures-4yrs-merged-imputed.csv
        b. rows with 'nan' dropped
            exposures-4yrs-merged.csv

    """
    import collections
    import data_processor as dproc

    X, y, features = dproc.load_data(input_path=dataDir, input_file=input_file, sep=',') # other params: input_path/None

    print("(load_data) dim(X): {}, sample_size: {}".format(X.shape, X.shape[0]))

    counts = collections.Counter(y)
    print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))
    print("... variables: {}".format(features))

    return X, y, features

def get_median_of_threshold(counts):
    features_median_threshold = {}
    for feature, threshold_list in counts.items():
        if len(threshold_list) == 0:
            print("{} do not have any unique threshold value".format(feature))
        #     What shd we do with this? If the feature is really included in the profile
        else:
            np_thres_list = np.array(threshold_list)
            features_median_threshold[feature] = np.median(np_thres_list)
            print("{} median threshold value: {}".format(feature, features_median_threshold[feature]))
            print("# threshold value: {}".format(np_thres_list.shape))
    return features_median_threshold

def get_median_of_paths_threshold(paths_thres):
    paths_median_threshold = {}
    for path, threshold_list in paths_thres.items():
        # print(threshold_list)
        paths_median_threshold[path] = np.median(np.array(threshold_list), axis=0)
    return paths_median_threshold

def topk_profile_with_its_threshold(sorted_paths, paths_thres, topk, sep="\t"):
    topk_profile_with_value_str = []
    all_greater_path = {}
    for k, (path, count) in enumerate(sorted_paths):
    # for k, (path, count) in enumerate(sorted_paths):
        greater_ = True
        profile_str = ""
        if count > 10:
            for idx, pollutant in enumerate(path.split(sep)):
                # print()
                profile_str += "{}{}{:.3e}{}".format(sep,pollutant, paths_thres[path][idx], sep)
            # plot_histogram(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])
            # plot_scatter(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])
            # plot_hist2d(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])

            print_str = "{}th paths ({}):{}".format(k, count, profile_str[:-1])
            topk_profile_with_value_str.append(profile_str[1:-1])
            if k < topk:
                print(print_str)
            if profile_str.count('>') > 1 and count > 1 and (not '<' in profile_str):
                # print(print_str)
                all_greater_path[profile_str] = count


        # else:
        #     break
    return topk_profile_with_value_str, all_greater_path
    # print("> Top {} paths (overall):\n{}\n".format(topk, sorted_paths[:topk]))




def profile_indicator_function(path, feature_idx, path_threshold, X, sign_pair, sep='\t'):
    profile_indicator = np.ones((X.shape[0]))
    neg_value = -1
    for n_idx, node_with_sign in enumerate(path.split(sep)):
        node = node_with_sign.replace(sign_pair[0], '').replace(sign_pair[1], '')
        if sign_pair[0] in node_with_sign:
            sign = sign_pair[1]
        else:
            sign = sign_pair[0]
        for x_idx, features in enumerate(X):
            if inequality_operators[sign](features[feature_idx[node]], path_threshold[n_idx]):
                profile_indicator[x_idx] = neg_value
    return profile_indicator
#
# def plot_scores_hist()

def summarize_paths(paths):
    print("> 1. Frequent decision paths overall ...")
    sorted_paths = sorted(paths.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_paths

def runWorkflow(**kargs):
    from utils_tree import visualize, visualize_xgb

    def summarize_vars(X, y):
        counts = collections.Counter(y)
        print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))

    from data_processor import load_data
    from utils_tree import visualize, sort_path
    import operator

    verbose = kargs.get('verbose', True)
    input_file = kargs.get('input_file', '')
    binary_outcome = kargs.get('binary_outcome', True)
    outcome_folder_name = kargs.get('output_folder_name', '')
    p_val_df = kargs.get('p_val_df', pd.DataFrame({}))
    test_score_df = kargs.get('test_score_df', pd.DataFrame({}))
    yr_name = kargs.get('yr_name', '')
    xgb = kargs.get('xgb', False)
    # outcome_name = kargs.get('outcome_name', pd.DataFrame({}))

    outputDir = os.path.join(plotDir, outcome_folder_name)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    possible_results = ['pos_correlate', 'neg_correlate', 'mixed_sign_profile']
    possibleDirs = []
    # result_dir = os.path.join(plotDir, file_prefix)
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    for possible_result in possible_results:
        possibleDirs.append(os.path.join(outputDir, possible_result))
        if not os.path.exists(possibleDirs[-1]):
            os.mkdir(possibleDirs[-1])
    # 1. define input dataset
    if verbose: print("(runWorkflow) 1. Specifying input data ...")
    ######################################################
    file_prefix = input_file.split('.')[0]



    confounding_vars = ['age', 'avg_income',
                       # 'race'
                       'Race_Asian',
                       'Race_Black or African American',
                       'Race_More Than One Race',
                       'Race_Native Hawaiian or Other Pacific Islander',
                       'Race_Unknown / Not Reported',
                       'Race_White',
                       'gender',
                       ]
    # exclude_vars = ['Gender', 'Zip Code'] + confounding_vars
    exclude_vars = confounding_vars + [
                        "ID",
    ]

    """
    Load Data
    """

    X, y, features, confounders_df, whole_df = load_data(input_path=dataDir,
                                               input_file=input_file,
                                               exclude_vars=exclude_vars,
                                               # col_target='Outcome',
                                               confounding_vars=confounding_vars,
                                               verbose=True)

    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    p_grid = {"min_samples_leaf": []}

    if xgb:
        model = xgboost.XGBClassifier(random_state=1)
        sign_pair = ['<', '>=']
    else:
        model = DecisionTreeClassifier(criterion='entropy', random_state=1)
        sign_pair = ['<=', '>']

    # 3. visualize the tree (deferred to analyze_path())
    ######################################################
    test_size = 0.3
    rs = 53
    # topk = 10
    topk_vars = 10
    ######################################################
    labels = [str(l) for l in sorted(np.unique(y))]

    feature_idx_dict = {}
    for idx, feature in enumerate(features):
        feature_idx_dict[feature] = idx

    outcome_dir = os.path.join(plotDir, outcome_folder_name)

    if len(np.unique(y)) > 1:

        fmap_fn = features_to_txt(features)
        scores, list_params, topk_profile_str, sorted_paths, paths_median_threshold, visualize_dict = \
             analyze_path(X, y, model=model, p_grid=p_grid, feature_set=features, n_trials=100, n_trials_ms=30, save=False,
                            merge_labels=False, policy_count='standard', experiment_id=file_prefix,
                               create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False, binary_outcome=binary_outcome,
                          fmap_fn=fmap_fn, plot_dir=plotDir,
                          outcome_name=outcome_folder_name, xgb=xgb)

        """
        Regression with Cofounders
        """

        np.random.seed(0)
        print(len(topk_profile_str))
        profile_counter = 0
        for idx, (profile, profile_occurrence) in enumerate(sorted_paths):
            # print(y)
            print(profile_counter)
            # if profile_counter > (len(sorted_paths)*0.1):
            #     break
            if profile_occurrence > 10:
                binary_profile = profile_indicator_function(path=profile,
                                                            feature_idx=feature_idx_dict,
                                                            path_threshold=paths_median_threshold[profile],
                                                            X=X, sign_pair=sign_pair
                                                            )
                binary_profile = np.array(binary_profile)
                print(profile,' pos_count :', sum(binary_profile==1), 'out of ', binary_profile.shape[0])
                profile_df = pd.DataFrame({topk_profile_str[idx]: binary_profile})
                regression_x_df = pd.concat([profile_df, confounders_df], axis=1)
                all_equal_drop_col = []
                for col in regression_x_df:
                    unique_value = regression_x_df[col].unique()
                    if len(unique_value) == 1:
                        all_equal_drop_col.append(col)
                # print('Column(s) with all equal entries:', all_equal_drop_col)

                regression_x_df.drop(all_equal_drop_col, axis=1, inplace=True)

                try:
                    X_np = np.array(regression_x_df)
                    X_corr = np.corrcoef(X_np, rowvar=0)
                    # print(X_corr)
                    w, v = np.linalg.eig(X_corr)
                    # print('{} eigenvalues: {}'.format(profile, w))
                    # result = regressor_with_confounders.fit(maxiter=500, method='bfgs')
                    regression_x_df['intercept'] = 1.0
                    if binary_outcome:
                        regressor_with_confounders = sm.Logit(y, regression_x_df)
                    else:
                        regressor_with_confounders = sm.OLS(y, regression_x_df)

                    # result = regressor_with_confounders.fit(skip_hessian=True)
                    result = regressor_with_confounders.fit()

                except Exception as inst:

                    regression_x_df['intercept'] = 1.0
                    if binary_outcome:
                        regressor_with_confounders = sm.Logit(y, regression_x_df)
                        result = regressor_with_confounders.fit(method='bfgs')
                    else:
                        regressor_with_confounders = sm.OLS(y, regression_x_df)
                        result = regressor_with_confounders.fit()

                result_summary = result.summary()



                """
                Since pvalue cannot be shown in scientific notation by simply as_csv(),
                addition lines are written
                """
                profile_coef = result.params.values[0]
                p_val = result.pvalues.values[0]

                params = result.params
                conf = result.conf_int()
                conf['Odds Ratio'] = params
                conf.columns = ['5%', '95%', 'Beta_value']
                conf_df = conf.values[0]

                # effect_size_CI = np.exp(conf.values[0])

                if p_val < 0.05:

                    p, count = (profile, profile_occurrence)
                    # if count >= 5:
                    path_from = visualize_dict['paths_from'][p]
                    random.Random(8964).shuffle(path_from)
                    p_name = '_and_'.join(p.split('\t'))



                    for path_loc in path_from[:10]:
                        print('printing XGB Trees')
                        split_idx, booster_idx = path_loc
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                            random_state=split_idx,
                                                                            stratify=y)
                        profile_group = ''
                        profile_str = topk_profile_str[idx]
                        if (profile_str.count('>') > 1) & (profile_str.count('<') == 0) & profile_str.count('\n') > 0:
                            profile_group = 'all_greater'
                        elif (profile_str.count('<') > 1) & (profile_str.count('>') == 0) & profile_str.count('\n') > 0:
                            profile_group = 'all_less'
                        elif profile_str.count('\n') > 0:
                            profile_group = 'mixed_sign_multi_pollutants'
                        elif profile_str.count('\n') == 0:
                            profile_group = 'single_pollutant'
                        tree_sub_dir = os.path.join(outputDir, profile_group)
                        if not os.path.exists(tree_sub_dir):
                            os.mkdir(tree_sub_dir)

                        tree_dir = os.path.join(tree_sub_dir, p_name)
                        if not os.path.exists(tree_dir):
                            os.mkdir(tree_dir)

                        graph = visualize_xgb(visualize_dict['model_list'][split_idx], feature_names=fmap_fn, labels=labels,
                                              outcome_name=outcome_dir,
                                              num_trees=booster_idx,
                                              file_name="split_{}_booster_{}".format(split_idx, booster_idx),
                                              training_data=(X_train, y_train),
                                              tree_dir=tree_dir)


                if (sign_pair[0] in topk_profile_str[idx] and sign_pair[1] in topk_profile_str[idx]) or profile_coef == 0:
                    relation_dir = possibleDirs[-1]
                elif (sign_pair[0] in topk_profile_str[idx] and profile_coef < 0) or (sign_pair[1] in topk_profile_str[idx] and profile_coef > 0):
                    relation_dir = possibleDirs[0]
                elif (sign_pair[0] in topk_profile_str[idx] and profile_coef > 0) or (sign_pair[1] in topk_profile_str[idx] and profile_coef < 0):
                    relation_dir = possibleDirs[1]

                result_dir = os.path.join(relation_dir, file_prefix)



                    # plot_hist2d(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])

                out_path = os.path.join(result_dir, "occur_{}_{}_coef_{:.3e}_pval={:.3e}.csv".format(profile_occurrence,
                                                                                                 topk_profile_str[idx],
                                                                                            profile_coef,
                                                                                         p_val))



                # print(out_path)
                # result_summary.as_csv(out_path)

                # profile_coef
                #

                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)

                opposite_profile = topk_profile_str[idx]

                opposite_profile = opposite_profile.replace(sign_pair[0], 'larger')
                opposite_profile = opposite_profile.replace(sign_pair[1], 'smaller')
                opposite_profile = opposite_profile.replace('larger', sign_pair[1])
                opposite_profile = opposite_profile.replace('smaller', sign_pair[0])

                opposite_files = find('occur_*{}_coef*.csv'.format(opposite_profile), result_dir)
                # opposite_files = [f for f in opposite_files if ' ' not in f]
                # print(topk_profile_str[idx], opposite_profile, opposite_files)
                if (len(opposite_files) == 0 ) or (len(path.split('\t')) > 1):
                    profile_counter += 1
                    cols = p_val_df.columns
                    p_val_df = p_val_df.append({cols[0]: topk_profile_str[idx],
                                     cols[1]: outcome_folder_name,
                                     cols[2]: p_val,
                                     cols[3]: relation_dir.split('/')[-1],
                                     cols[4]: profile_coef,
                                     cols[5]: conf_df[0],
                                     cols[6]: conf_df[1],
                                     cols[7]: profile_occurrence,
                                    cols[8]: sum(binary_profile == 1),
                                    cols[9]: sum(binary_profile == -1),
                                    cols[10]: binary_outcome,
                                    cols[11]: list_params['max_count']
                                    # cols[9]: np.mean(np.array(scores)),
                                    # cols[10]: scipy.stats.mode(np.array(min_number_leaf))[0],
                                    }, ignore_index=True)
                    # print(p_val_df)
                #     for single_pollutant_profile in topk_profile_str[idx].split('\t'):
                #         if sign_pair[0] in single_pollutant_profile:
                #             pollutant_name, thres = single_pollutant_profile.split(sign_pair[0])
                #         elif sign_pair[1] in single_pollutant_profile:
                #             pollutant_name, thres = single_pollutant_profile.split(sign_pair[1])
                #         if binary_outcome:
                #             label_for_hist = ['{}(Yes)'.format(outcome_folder_name),
                #                               '{}(No)'.format(outcome_folder_name)]
                #             plot_histogram(whole_df, result_dir, pollutant_name=pollutant_name,
                #                            label_plot=label_for_hist)
                #         else:
                #             plot_scatter(whole_df, result_dir, pollutant_name=pollutant_name, ylabel=outcome)
                #             plot_hist2d(whole_df, result_dir, pollutant_name=pollutant_name, ylabel=outcome)
                #     if p_val < 0.05:
                #         f = open(out_path, 'w')
                #         for table in result_summary.tables:
                #             #     print(type(table))
                #             html = table.as_html()
                #             df_temp_result = pd.read_html(html, header=0, index_col=0)[0]
                #             pd.options.display.float_format = '{:,.3e}'.format
                #             if 'P>|z|' in df_temp_result.columns:
                #                 # print(type(result.pvalues), type(df_temp_result.loc[:,'P>|z|']))
                #                 # print(result.pvalues, df_temp_result.loc[:, 'P>|z|'])
                #                 df_temp_result.loc[:, 'P>|z|'] = result.pvalues.values
                #                 # print(result.pvalues, df_temp_result.loc[:,'P>|z|'])
                #             csv_buffer = StringIO()
                #             # output_file = df_temp_result.to_csv(csv_buffer, float_format='%.3e') + '\n'
                #             df_temp_result.to_csv(csv_buffer, float_format='%.3e')
                #             # print(csv_buffer.getvalue())
                #             f.write(csv_buffer.getvalue() + '\n')
                #         f.close()

    print('Finished All regressions!')

        # if len(p_val_df['outcome']==output_folder_name) == 0:
    if binary_outcome:
        f_minority = scores['scores'][0]
        f_minority_rand = scores['scores_random'][0]
        f_majority = scores['scores'][1]
        f_majority_rand = scores['scores_random'][1]
        auc = scores['scores'][2]
        auc_rand = scores['scores_random'][2]
        r2 = []
        r2_rand = []
    else:
        f_minority = []
        f_minority_rand = []
        f_majority = []
        f_majority_rand = []
        auc = []
        auc_rand = []
        r2 = scores['scores'][0]
        r2_rand = scores['scores_random'][0]

    if xgb:
        min_sl = 0
    else:
        min_sl = scipy.stats.mode(np.array(list_params['min_sample_leaf']))[0][0]
    test_cols = test_score_df.columns
    test_score_df = test_score_df.append({
                                test_cols[0]: outcome_folder_name,
                                test_cols[1]: binary_outcome,
                                test_cols[2]: min_sl,
                                test_cols[3]: yr_name,
        'mean (std) r2 score from random predictors': '{:.3e}({:.3e})'.format(np.mean(r2_rand), np.std(r2_rand)),
        'mean (std) r2 score': '{:.3e}({:.3e})'.format(np.mean(r2), np.std(r2)),
        'mean (std) f score (minority) from random predictors': '{:.3e}({:.3e})'.format(np.mean(f_minority_rand), np.std(f_minority_rand)),
        'mean (std) f score (minority)': '{:.3e}({:.3e})'.format(np.mean(f_minority), np.std(f_minority)),
        'mean (std) f score (majority) from random predictors': '{:.3e}({:.3e})'.format(np.mean(f_majority_rand), np.std(f_majority_rand)),
        'mean (std) f score (majority)': '{:.3e}({:.3e})'.format(np.mean(f_majority), np.std(f_majority)),
        'mean (std) AUC score from random predictors': '{:.3e}({:.3e})'.format(np.mean(auc_rand), np.std(auc_rand)),
        'mean (std) AUC score': '{:.3e}({:.3e})'.format(np.mean(auc), np.std(auc)),
                                test_cols[-1]: X.shape[0],
                                }, ignore_index=True)

    return p_val_df, test_score_df

if __name__ == "__main__":
    # file_format = 'act_score_7pollutants_no_impute_*.csv'
    # binary_out = False if 'True' != sys.argv[-2] else True
    outcome_binary_dict = {
                            # 'asthma': True,
                            # 'asthma(act_score)':False,
                            # 'age_greaterthan5_diagnosed_asthma': True,
                            # 'age_diagnosed_asthma': False,
                            'daily_controller_past6months': True,
                            'emergency_dept': True,
                            # 'emergency_dept_pastyr_count': False,
                            'hospitalize_overnight': True,
                            # 'hospitalize_overnight_pastyr_count': False,
                            # 'regular_asthma_symptoms_past6months': True,
                           # 'regular_asthma_symptoms_daysCount_pastWeek': False
                           #  'asthma(act_score_lessthan20)': True,
                           #  'emergency_dept_pastyr_count(greaterthan0)': True,
                           #  'emergency_dept_pastyr_count(greaterthan_nz_median)': True,
                           #  'hospitalize_overnight_pastyr_count(greaterthan0)': True,
                           #  'hospitalize_overnight_pastyr_count(greaterthan_nz_median)': True,
                           #  'regular_asthma_symptoms_daysCount_pastWeek(greaterthan0)': True,
                           #  'regular_asthma_symptoms_daysCount_pastWeek(greaterthan_nz_median)': True
                           }

    # suffix = 'nbt_xgb_multiple_counts'
    outcome = sys.argv[-2]
    suffix = sys.argv[-1]
    xgb_predict = True
    plot_predir = './plot_{}'.format(suffix)
    if not os.path.exists(plot_predir):
        os.mkdir(plot_predir)
    # file_format = 'emergency_dept_7pollutants_no_impute_*.csv'
    # file_format = 'hospitalize_overnight_7pollutants_no_impute_*.csv'
    # file_format = 'regular_medication_7pollutants_no_impute_*.csv'
    # file_format = 'regular_asthma_past6months_7pollutants_no_impute_*.csv'
    # output_folder_name = ''
    # file_format = "diabetes.csv"
    # file_format = 'exposure_7pollutants_no_impute_y-1.csv'

    pvalue_df_list = []
    pred_score_df_list = []

    path = 'data'
    yr_dict = {
        'birth_yr': [5],
        # 'diagnose_yr': [-5]
    }
    yr_name = "birth_yr"
    yr_f = 5
    binary_out = outcome_binary_dict[outcome]

    m_Dir = os.path.join(os.getcwd(), 'plot/{}/'.format(suffix))
    if not os.path.exists(m_Dir):
        os.mkdir(m_Dir)
    plotDir = os.path.join(m_Dir, 'nata_{}_{}'.format(yr_name, yr_f))
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
    pvalue_df = pd.DataFrame(columns=['profile', 'outcome', 'p_val', 'relation',
                                      'coef', 'coef_95CI_lower', 'coef_95CI_upper','freq', 'pos_count', 'neg_count', 'binary_outcome',
                                      'max_count'

                                      ])
    pred_score_df = pd.DataFrame(columns=['outcome', 'binary_outcome', 'mode of min_samples_of_leaf', 'year from',
                                          'mean (std) r2 score from random predictors',
                                          'mean (std) r2 score',
                                          'mean (std) f score (minority) from random predictors',
                                          'mean (std) f score (minority)',
                                          'mean (std) f score (majority) from random predictors',
                                          'mean (std) f score (majority)',
                                          'mean (std) AUC score from random predictors',
                                          'mean (std) AUC score',
                                          'num_patients'
                                          ])
    file_format = '{}_NATA_{}_{}.csv'.format(outcome, yr_name, yr_f)
    file_list = find(file_format, path)
    file = file_list[0]
    if yr_f > 0:
        year_name_detail = yr_name + '+' + str(yr_f)
    else:
        year_name_detail = yr_name + str(yr_f)
    pvalue_df, pred_score_df = runWorkflow(input_file=file,
                            # binary_outcome=binary_out,
                            output_folder_name=outcome,
                            p_val_df=pvalue_df,
                            yr_name=year_name_detail,
                            test_score_df = pred_score_df,
                            xgb=xgb_predict
                            # outcome_name = outcome
                            )
    # print('before dropped', pvalue_df.shape)
    pvalue_df.dropna(axis=0, how='any', inplace=True)

    pvalue_df.to_csv('fdr_{}_{}.csv'.format(outcome, suffix), index=False, header=True)

    pred_score_df.to_csv('pred_score_{}_{}.csv'.format(outcome, suffix), index=False, header=True)