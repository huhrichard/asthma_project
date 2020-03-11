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
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import os, fnmatch
from io import StringIO
from statsmodels.stats.multitest import fdrcorrection
from decimal import Decimal
from tabulate import tabulate
# from utils_plot import saveFig
from matplotlib.ticker import PercentFormatter
plt.rcParams.update({'font.size': 12})
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

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
plotDir = os.path.join(os.getcwd(), 'plot')
# output_fn = sys.argv[-1]
# output_folder_name = 'act_score'
# output_folder_name = 'emergency_dept'
# output_folder_name = 'hospitalize_overnight'
# output_folder_name = 'regular_medication'
# output_folder_name = 'regular_asthma_symptoms_past6months'




class Data(object): 
    label = 'label'
    features = []


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

def plot_histogram(asthma_df, plot_dir, pollutant_name, thres, label_plot=['with asthma', 'w/o asthma']):
    df = asthma_df[[pollutant_name, 'label']]

    # pollutant_level_w_asthma = np.log10(df[asthma_df['label'] == 1][pollutant_name])
    # pollutant_level_wo_asthma = np.log10(df[asthma_df['label'] == 0][pollutant_name])
    pollutant_level_w_asthma = df[asthma_df['label'] == 1][pollutant_name]
    pollutant_level_wo_asthma = df[asthma_df['label'] == 0][pollutant_name]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, bins, _ = ax.hist([pollutant_level_w_asthma, pollutant_level_wo_asthma],
            weights=[np.ones(len(pollutant_level_w_asthma))/len(pollutant_level_w_asthma), np.ones(len(pollutant_level_wo_asthma))/len(pollutant_level_wo_asthma)],
            label=label_plot)



    ax.legend(loc='upper right')
    # ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
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



def plot_scatter(asthma_df, plot_dir, pollutant_name, thres):
    df = asthma_df[[pollutant_name, 'label']]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df[pollutant_name], df['label'])
    # ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    # ax.set_title(pollutant_name+'_imputed_data')

    pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(subscript_like_chemical_name(pollutant_list[0])))
    ax.set_ylabel('ACT Score')

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

def plot_hist2d(asthma_df, plot_dir, pollutant_name, thres):
    df = asthma_df[[pollutant_name, 'label']]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hist2d = ax.hist2d(df[pollutant_name], df['label'], bins=20, cmap=plt.cm.Greys)
    fig.colorbar(hist2d[-1])
    # ax.axvline(float(thres), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = ax.get_ylim()
    min_xlim, max_xlim = ax.get_xlim()
    y_pos = 0.75
    # ax.text(float(thres)+(max_xlim-min_xlim)*0.02, (max_ylim*y_pos+(1-y_pos)*min_ylim), 'Threshold = {:.3e}'.format(thres))
    # ax.set_title(pollutant_name+'_imputed_data')

    pollutant_list = pollutant_name.split('y')
    # ax.yaxis.set_major_formatter(PercentFormatter(1))

    # ax.set_xlabel('{} level ({} scale)'.format(subscript_like_chemical_name(pollutant_list[0]), '${log_{10}}$'))
    ax.set_xlabel('{} level'.format(subscript_like_chemical_name(pollutant_list[0])))
    ax.set_ylabel('ACT Score')

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
        clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv)

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
    if binary_outcome:
        model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
    else:
        model = DecisionTreeRegressor(criterion='mse', random_state=random_state)

    # print('nan?', X.dtype)
    if len(params) > 0: model = model.set_params(**params)
    model.fit(X, y)

    path = model.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    cv_split = KFold(n_splits=10, shuffle=True, random_state=random_state)
    final_tree = GridSearchCV(estimator=model, param_grid={'ccp_alpha':ccp_alphas[:-1][ccp_alphas[:-1]>0]}, cv=cv_split)
    final_tree.fit(X, y)


    return final_tree.best_estimator_




def analyze_path(X, y, model=None, p_grid={}, feature_set=[], n_trials=100, n_trials_ms=10, save=False, output_path='', output_file='', 
                             create_dir=True, index=0, binary_outcome=True,  **kargs):
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from utils_tree import visualize, count_paths, count_paths2, count_features2, \
        get_feature_threshold_tree, count_paths_with_thres_sign
    import time
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2)
    verbose = kargs.get('verbose', False)
    merge_labels = kargs.get('merge_labels', True)
    policy_count = kargs.get('policy_counts', 'standard') # options: {'standard', 'sample-based'}
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    validate_tree = kargs.get('validate_tree', True)
    plot_dir = kargs.get('plot_dir', plotDir)
    plot_ext = kargs.get('plot_ext', 'tif')
    to_str = kargs.get('to_str', False)  # if True, decision paths are represented by strings (instead of tuples)
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
    lookup = {}
    counts = {f: [] for f in feature_set} # maps features to lists of thresholds
        
    # build N different decision trees and compute their statistics (e.g. performance measures, decision path counts)
    auc_scores = []
    test_points = np.random.choice(range(n_trials), 1)
    for i in range(n_trials): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i) # 70% training and 30% test
        # print("[{}] dim(X_test): {}".format(i, X_test.shape))

        # [todo]: how to reset a (trained) model? 
        model = classify(X_train, y_train, params={}, random_state=i, binary_outcome=binary_outcome)
        # graph = visualize(model, feature_names=feature_set, labels=labels, file_name="{}th_diabetes_tree".format(i))
        # [test]
        if i in test_points: 
            if verbose:
                print("... building {} versions of the model: {}".format(n_trials, model.get_params()) )
            if validate_tree: 
                fild_prefix = "{id}-{index}".format(id=experiment_id, index=i)
                # graph = visualize(model, feature_set, labels, file_name=file_prefix, ext='tif')
                
                # display the tree in the notebook
                # Image(graph.create_png())  # from IPython.display import Image
        
        y_pred = model.predict(X_test)
        # accuracy = metrics.accuracy_score(y_test, y_pred)
        # auc_score = metrics.roc_auc_score(y_test, y_pred)
        # if i % 10 == 0: print("[{}] Accuracy: {}, AUC: {}".format(i, accuracy, auc_score))
        # auc_scores.append(auc_score)

        if not isinstance(X_test, np.ndarray): X_test = X_test.values   
            
        # --- count paths ---
        #    method A: count number of occurrences of decision paths read off of the decision tree
        #    method B: sample-based path counts
        #              each X_test[i] has its associated decision path => 
        #              in this method, we count the number of decision paths wrt the test examples
        # TODO: Richard: Path = "SO4y4 SO4y4 SO4y4 SO4y4"...?
        # if policy_count.startswith('stand'): # 'standard'
        #     paths, _ = \
        #         count_paths(model, feature_names=feature_set, paths=paths, # count_paths,
        #                     merge_labels=merge_labels, to_str=to_str, verbose=verbose, index=i)
        # else:  # 'full'
        #     paths, _ = \
        #         count_paths2(model, X_test, feature_names=feature_set, paths=paths, # counts=counts,
        #                      merge_labels=merge_labels, to_str=to_str, verbose=verbose)
        #
        # keep track of feature usage in terms of thresholds at splitting points
        # Richard: this is count by
        # counts = count_features2(model, feature_names=feature_set, counts=counts, labels=labels, verbose=True)
        # ... counts: feature -> list of thresholds (used to estimate its median across decision paths)
        paths, paths_threshold = count_paths_with_thres_sign(estimator=model, paths=paths, feature_names=feature_set,
                                                             paths_threshold=paths_threshold)
        # counts = get_feature_threshold_tree(estimator=model, counts=counts, feature_names=feature_set)
        # visualization?
    # print(paths)
    ### end foreach trial
    # paths_tuple = [() for key, items()]
    print("\n(analyze_path) Averaged AUC: {} | n_trials={}".format(np.mean(auc_scores), n_trials))
    # print(counts)
    return paths, paths_threshold

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

def topk_profile_with_its_threshold(sorted_paths, paths_thres, topk, sep=" "):
    topk_profile_with_value_str = []

    for k, (path, count) in enumerate(sorted_paths[:topk]):

        profile_str = ""
        if count > 10:
            for idx, pollutant in enumerate(path.split(sep)):
                profile_str += " {}{:.3e} ".format(pollutant, paths_thres[path][idx])
                # plot_histogram(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])
                # plot_scatter(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])
                # plot_hist2d(asthma_df, result_dir, pollutant_name=pollutant.strip('<=').strip('>'), thres=paths_thres[path][idx])

            print_str = "{}th paths ({}):{}".format(k, count, profile_str[:-1])
            topk_profile_with_value_str.append(profile_str[1:-1])
            print(print_str)
        else:
            break
    return topk_profile_with_value_str
    # print("> Top {} paths (overall):\n{}\n".format(topk, sorted_paths[:topk]))

def profile_indicator_function(path, feature_idx, path_threshold, X, sep=' ', y=None):
    profile_indicator = np.ones((X.shape[0]))
    for n_idx, node_with_sign in enumerate(path.split(sep)):
        larger_than = True
        # '>' in node -> larger_than = True Node: Var > threshold
        # else -> larger_than = False, Node: Var <= threshold
        if '>' in node_with_sign:
            node = node_with_sign[:-1]
        else:
            node = node_with_sign[:-2]
            larger_than = False
        for x_idx, features in enumerate(X):
            if larger_than:
                # print('test larger')
                if features[feature_idx[node]] <= path_threshold[n_idx]:
                    profile_indicator[x_idx] = 0
            else:
                # print('test smaller')
                if features[feature_idx[node]] > path_threshold[n_idx]:
                    profile_indicator[x_idx] = 0
    return profile_indicator

def runWorkflow(**kargs):
    def summarize_paths(paths):
        # labels = np.unique(list(paths.keys()))
        # print('labels', labels)
    
        # print("\n> 1. Frequent decision paths by labels (n={})".format(len(labels)))
        # sorted_paths = sort_path(paths, labels=labels, merge_labels=False, verbose=True)
        # for label in labels:
        #     print("... Top {} paths (@label={}):\n{}\n".format(topk, label, sorted_paths[label][:topk]))
            

        # print("> 2. Frequent decision paths overall ...")
        # sorted_paths = sort_path(paths, labels=labels, merge_labels=True, verbose=True)
        print("> 1. Frequent decision paths overall ...")
        sorted_paths = sorted(paths.items(), key=operator.itemgetter(1), reverse=True)
        # for i in range(3):
        #     path, cnt = sorted_paths[i][0], sorted_paths[i][1]
        #
        #     # counts =
            # for label in labels:
            #     counts.append(paths[label].get(path, 0))
            # print("(sort_path) #[{}] {} | total: {} | label-dep counts: {}\n".format(i, path, cnt, counts))


        return sorted_paths
    def summarize_vars(X, y): 
        counts = collections.Counter(y)
        print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))

    from data_processor import load_data
    from utils_tree import visualize, sort_path
    import operator
    
    verbose = kargs.get('verbose', True)
    input_file = kargs.get('input_file', '')
    binary_outcome = kargs.get('binary_outcome', True)
    output_folder_name = kargs.get('output_folder_name', '')
    p_val_df = kargs.get('p_val_df', pd.DataFrame({}))
    # outcome_name = kargs.get('outcome_name', pd.DataFrame({}))

    outputDir = os.path.join(plotDir, output_folder_name)
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
    # input_file = 'exposures-4yrs-merged-imputed.csv'
    # input_file = 'exposures-4yrs.csv'
    # input_file = 'exposures-4yrs-filtered_na_2.csv'
    # input_file = 'exposures-4yrs-filtered_na_inc_race.csv'
    # input_file = 'exposures-4yrs-filtered_race_in1Col.csv'
    # input_file = 'exposures-4yrs-filtered_0.8_race_in1Col_NNMImpute.csv'
    # input_file = 'exposures-4yrs-merged-imputed_2.csv'
    # input_file = 'exposure_7pollutants_no_impute.csv'
    # input_file = 'exposure_7pollutants_svdImpute.csv'
    file_prefix = input_file.split('.')[0]


    # result_dir = os.path.join(plotDir, file_prefix)
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    ######################################################

    # Richard: Added another output of confounders

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
                        # 'gender',
                      # 'Race_Asian',
                      #  'Race_Black or African American',
                      #  'Race_More Than One Race',
                      #  'Race_Native Hawaiian or Other Pacific Islander',
                      #  'Race_Unknown / Not Reported',
                      #  'Race_White',
    ]
    # confounding_vars = []
    # exclude_vars = []


    X, y, features, confounders_df, whole_df = load_data(input_path=dataDir,
                                               input_file=input_file,
                                               exclude_vars=exclude_vars,
                                               # col_target='Outcome',
                                               confounding_vars=confounding_vars,
                                               verbose=True)

    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    p_grid = {"max_depth": [3, 4, 5, 8, 10, 15], 
              "min_samples_leaf": [1, 5, 10, 15, 20]}
    ######################################################
    # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
    #     A split point at any depth will only be considered if it leaves at least 
    #     min_samples_leaf training samples in each of the left and right branches

    model = DecisionTreeClassifier(criterion='entropy', random_state=1)

    # 3. visualize the tree (deferred to analyze_path())
    ###################################################### 
    test_size = 0.3
    rs = 53
    # topk = 10
    topk_vars = 10
    ######################################################
    labels = [str(l) for l in sorted(np.unique(y))]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs) # 70% training and 30% test

    # params = {'max_depth': 5}  # use model selection to determine the optimal 'max_depth'
    # model = classify(X_train, y_train, params=params, random_state=rs)
    # graph = visualize(model, features, labels=labels, plot_dir=plotDir, file_name=file_prefix, ext='tif')

    # 4. analyze decision paths and keep track of frequent features


    # print(X[np.logical_not(np.isfinite(X))])
    # univariate_path = os.path.join(result_dir, "logistic_reg_concatenated_{}.csv".format(file_prefix.split('y')[-1]))
    # f = open(univariate_path, 'w')
    feature_idx_dict = {}
    for idx, feature in enumerate(features):
        feature_idx_dict[feature] = idx
    #     temp_df = pd.DataFrame()
    #     temp_df[feature] = X[:,idx]
    #     temp_df['intercept'] = 1.0
    #     logistic_regressor = sm.Logit(y, temp_df)
    #     result = logistic_regressor.fit(skip_hessian=True)
    #     result_summary = result.summary()
    #
    #     for table in result_summary.tables:
    #         #     print(type(table))
    #         html = table.as_html()
    #         df_temp_result = pd.read_html(html, header=0, index_col=0)[0]
    #         pd.options.display.float_format = '{:,.3e}'.format
    #         if 'P>|z|' in df_temp_result.columns:
    #             # print(type(result.pvalues), type(df_temp_result.loc[:,'P>|z|']))
    #             # print(result.pvalues, df_temp_result.loc[:, 'P>|z|'])
    #             df_temp_result.loc[:, 'P>|z|'] = result.pvalues.values
    #             # print(result.pvalues, df_temp_result.loc[:,'P>|z|'])
    #         csv_buffer = StringIO()
    #         # output_file = df_temp_result.to_csv(csv_buffer, float_format='%.3e') + '\n'
    #         df_temp_result.to_csv(csv_buffer, float_format='%.3e')
    #         # print(csv_buffer.getvalue())
    #         f.write(csv_buffer.getvalue() + '\n')
    # f.close()


    paths, paths_threshold = \
         analyze_path(X, y, model=model, p_grid=p_grid, feature_set=features, n_trials=100, n_trials_ms=30, save=False,  
                        merge_labels=False, policy_count='standard', experiment_id=file_prefix,
                           create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False, binary_outcome=binary_outcome)
    # print("before summary:", paths)
    sorted_paths = summarize_paths(paths)

    paths_median_threshold = get_median_of_paths_threshold(paths_threshold)
    # topk = len(sorted_paths)
    topk = 10

    topk_profile_str = topk_profile_with_its_threshold(sorted_paths, paths_median_threshold, topk=topk)


    # print("sorted paths: ", sorted_paths)
    confounders_array = np.array(confounders_df)


    for idx, (profile, profile_occurrence) in enumerate(sorted_paths[:len(topk_profile_str)]):
        binary_profile = profile_indicator_function(path=profile,
                                                    feature_idx=feature_idx_dict,
                                                    path_threshold=paths_median_threshold[profile],
                                                    X=X, y=y)
        # print()
        # print(binary_profile)
        # print('profile shape:', binary_profile.shape)
        # print('confounders shape:', confounders_array.shape)
        profile_df = pd.DataFrame({topk_profile_str[idx]: binary_profile})
        regression_x_df = pd.concat([profile_df, confounders_df], axis=1)
        # regression_x_df = profile_df
        # print('')
        # print(metrics.confusion_matrix(binary_profile, np.array(y)))
        all_equal_drop_col = []
        for col in regression_x_df:
            unique_value = regression_x_df[col].unique()
            if len(unique_value) == 1:
                all_equal_drop_col.append(col)
        print('Column(s) with all equal entries:', all_equal_drop_col)
        regression_x_df.drop(all_equal_drop_col, axis=1, inplace=True)

        # regression_x_df = sm.add_constant(regression_x_df)
        # regression_x = np.concatenate([binary_profile, confounders_array], axis=1)
        # regressor_with_confounders = linear_model.LogisticRegression()
        # print(pd.concat([regression_x_df, y], axis=1))
        # plt.clf()

        # plt.hist2d(binary_profile, np.array(y).astype('float'), bins=2)
        # plt.show()

        # regressor_with_confounders = sm.GLM(y, regression_x_df, family=sm.families.NegativeBinomial())
        # regressor_with_confounders = neg_bin(y, regression_x_df, loglike_method='nb1')
        try:
            X_np = np.array(regression_x_df)
            X_corr = np.corrcoef(X_np, rowvar=0)
            # print(X_corr)
            w, v = np.linalg.eig(X_corr)
            print('{} eigenvalues: {}'.format(profile, w))
            # result = regressor_with_confounders.fit(maxiter=500, method='bfgs')
            regression_x_df['intercept'] = 1.0
            if binary_outcome:
                regressor_with_confounders = sm.Logit(y, regression_x_df)
            else:
                regressor_with_confounders = sm.OLS(y, regression_x_df)

            # result = regressor_with_confounders.fit(skip_hessian=True)
            result = regressor_with_confounders.fit()


                    # f.write(output_file)

                    # result_html_0 = result_summary.tables.as_html()
                    # result_html_0 = result_summary.tables[0].as_html()
                    # pd_result_0 = pd.read_html(result_html_0, header=0, index_col=0)[0]
                    # print(pd_result_0)

                    # f.write(result_summary.as_csv())

        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print("Skipped because of singular matrix, Input is not full rank matrix?")
            regression_x_df['intercept'] = 1.0
            if binary_outcome:
                regressor_with_confounders = sm.Logit(y, regression_x_df)
                result = regressor_with_confounders.fit(method='bfgs')
            else:
                regressor_with_confounders = sm.OLS(y, regression_x_df)
                result = regressor_with_confounders.fit()


        print("{}{}{}".format('*' * 10, profile, '*' * 10))
        # print(result.summary(float_format='%.3f'))
        # pd.set_option('display.float_format', '{:.3e}'.format)
        result_summary = result.summary()

        print(result_summary)
        """
        Since pvalue cannot be shown in scientific notation by simply as_csv(), 
        addition lines are written
        """
        profile_coef = result.params.values[0]
        p_val = result.pvalues.values[0]
        if ('<=' in topk_profile_str[idx] and '>' in topk_profile_str[idx]) or profile_coef == 0:
            relation_dir = possibleDirs[-1]
        elif ('<=' in topk_profile_str[idx] and profile_coef < 0) or ('>' in topk_profile_str[idx] and profile_coef > 0):
            relation_dir = possibleDirs[0]
        elif ('<=' in topk_profile_str[idx] and profile_coef > 0) or ('>' in topk_profile_str[idx] and profile_coef < 0):
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

        opposite_profile = opposite_profile.replace('<=', 'larger')
        opposite_profile = opposite_profile.replace('>', 'smaller')
        opposite_profile = opposite_profile.replace('larger', '>')
        opposite_profile = opposite_profile.replace('smaller', '<=')

        opposite_files = find('occur_*{}_coef*.csv'.format(opposite_profile), result_dir)
        opposite_files = [f for f in opposite_files if ' ' not in f]
        print(topk_profile_str[idx], opposite_profile, opposite_files)
        if len(opposite_files) == 0:
            cols = p_val_df.columns
            p_val_df = p_val_df.append({cols[0]: topk_profile_str[idx],
                             cols[1]: output_folder_name,
                             cols[2]: p_val,
                             cols[3]: relation_dir.split('/')[-1],
                             cols[4]: profile_coef}, ignore_index=True)
            print(p_val_df)
            for single_pollutant_profile in topk_profile_str[idx].split(' '):
                if '<=' in single_pollutant_profile:
                    pollutant_name, thres = single_pollutant_profile.split('<=')
                elif '>' in single_pollutant_profile:
                    pollutant_name, thres = single_pollutant_profile.split('>')
                if binary_outcome:
                    label_for_hist = ['{}(Yes)'.format(output_folder_name),
                                      '{}(No)'.format(output_folder_name)]
                    plot_histogram(whole_df, result_dir, pollutant_name=pollutant_name, thres=thres,
                                   label_plot=label_for_hist)
                else:
                    plot_scatter(whole_df, result_dir, pollutant_name=pollutant_name, thres=thres)
                    plot_hist2d(whole_df, result_dir, pollutant_name=pollutant_name, thres=thres)
            if p_val < 0.05:
                f = open(out_path, 'w')
                for table in result_summary.tables:
                    #     print(type(table))
                    html = table.as_html()
                    df_temp_result = pd.read_html(html, header=0, index_col=0)[0]
                    pd.options.display.float_format = '{:,.3e}'.format
                    if 'P>|z|' in df_temp_result.columns:
                        # print(type(result.pvalues), type(df_temp_result.loc[:,'P>|z|']))
                        # print(result.pvalues, df_temp_result.loc[:, 'P>|z|'])
                        df_temp_result.loc[:, 'P>|z|'] = result.pvalues.values
                        # print(result.pvalues, df_temp_result.loc[:,'P>|z|'])
                    csv_buffer = StringIO()
                    # output_file = df_temp_result.to_csv(csv_buffer, float_format='%.3e') + '\n'
                    df_temp_result.to_csv(csv_buffer, float_format='%.3e')
                    # print(csv_buffer.getvalue())
                    f.write(csv_buffer.getvalue() + '\n')
                f.close()


        # for confounder in confounding_var:
        #     x_dropped_confouder = regression_x_df.drop(confounder, axis=1)

        # print(result.params)
        # regressor_with_confounders.fit(X=regression_x, y=y)







        # print("{}:{}".format(profile, binary_profile.shape))



    # for k, ths in counts.items(): 
    #     assert isinstance(ths, list), "{} -> {}".format(k, ths)
    # fcounts = [(k, len(ths)) for k, ths in counts.items()]
    # sorted_features = sorted(fcounts, key=lambda x: x[1], reverse=True)
    # print("> Top {} features:\n{}\n".format(topk_vars, sorted_features[:topk_vars]))
    

    return p_val_df

if __name__ == "__main__":
    # file_format = 'act_score_7pollutants_no_impute_*.csv'
    # binary_out = False if 'True' != sys.argv[-2] else True
    outcome_binary_dict = {
                            'act_score':False,
                            'age_greaterthan5_diagnosed_asthma': True,
                            'age_diagnosed_asthma': False,
                            'daily_controller_past6months': True,
                            'emergency_dept': True,
                            'emergency_dept_pastyr_count': False,
                            'hospitalize_overnight': True,
                            'hospitalize_overnight_pastyr_count': False,
                            'regular_asthma_symptoms_past6months': True,
                           'regular_asthma_symptoms_daysCount_pastWeek': False
                           }


    # file_format = 'emergency_dept_7pollutants_no_impute_*.csv'
    # file_format = 'hospitalize_overnight_7pollutants_no_impute_*.csv'
    # file_format = 'regular_medication_7pollutants_no_impute_*.csv'
    # file_format = 'regular_asthma_past6months_7pollutants_no_impute_*.csv'
    # output_folder_name = ''
    # file_format = "diabetes.csv"
    # file_format = 'exposure_7pollutants_no_impute_y-1.csv'


    pvalue_df = pd.DataFrame(columns=['profile', 'outcome', 'p_val', 'relation', 'coef'])
    path = 'data'

    for outcome, binary_out in outcome_binary_dict.items():
        file_format = '{}_7pollutants_no_impute_*.csv'.format(outcome)
        file_list = find(file_format, path)
        for file in file_list:
            print(file)
            pvalue_df = runWorkflow(input_file=file,
                                    binary_outcome=binary_out,
                                    output_folder_name=outcome,
                                    p_val_df=pvalue_df,
                                    # outcome_name = outcome
                                    )

    p_val_col = pvalue_df['p_val'].values
    rejected, fdr = fdrcorrection(p_val_col)
    print(fdr)
    no_cols = len(pvalue_df.columns)
    pvalue_df.insert(no_cols, "fdr", fdr, True)
    pvalue_df.to_csv('fdr.csv', index=False, header=True)
