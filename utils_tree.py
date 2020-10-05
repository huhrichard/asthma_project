# load all libraries 
import os, sys
import collections
from os.path import exists, abspath, isdir
from os import mkdir
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import xgboost
import matplotlib
import numpy as np
import pandas as pd

from tabulate import tabulate
# from utils_plot import saveFig

inequality_operators = {'<': lambda x, y: x < y,
                        '<=': lambda x, y: x <= y,
                        '>': lambda x, y: x > y,
                        '>=': lambda x, y: x >= y}

"""


Reference
---------
    1. Decision Tree 
        https://scikit-learn.org/stable/modules/tree.html

    2. Visualization 

        https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176


"""
plotDir = os.path.join(os.getcwd(), 'plot')

def check_dir(target_dir):
    if not exists(target_dir):
        mkdir(target_dir)

def visualize(clf, feature_names, labels=['0', '1'], file_name='test', plot_dir='', ext='png', save=True, outcome_name=''):
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    # from IPython.display import Image  
    import pydotplus
    
    if not plot_dir: plot_dir = os.path.join(os.getcwd(), 'plot')

    # ensure that labels are in string format 
    labels = [str(l) for l in sorted(labels)]
    
    # output_path = os.path.join(plot_dir, "{}.{}".format(file_name, ext))
    # output_path = osoutcome_name)
    check_dir(outcome_name)
    output_path = os.path.join(outcome_name, "{}.{}".format(file_name, ext))

    # labels = ['0','1']
    # label_names = {'0': '-', '1': '+'}
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True, # node_ids=True, 
                    special_characters=True, feature_names=feature_names)
    # ... class_names must be of string type

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    if save: 
        graph.write_png(output_path)

    # Image(graph.create_png())

    return graph

def extract_booster_label_to_dict(booster_in_text, x_train, y_train, feature_idx_inv_dict):
    booster_node_list = booster_in_text.replace('\t', '').split('\n')
    label_replace_dict = {}
    b_node_dict = {}
    for bnode in booster_node_list:
        bn_split = bnode.split(':')
        if len(bn_split) == 2:
            b_node_dict[int(bn_split[0])] = bn_split[1]

    n_nodes = len(b_node_dict)
    stack = [0]
    b_node_detail = {0:{'x':x_train, 'y':y_train,
                        '#_of_patients':x_train.shape[0],
                        '#_of_positive_patients': sum(y_train),
                        '#_of_negative_patients': x_train.shape[0]-sum(y_train),
                        # '%_of_positive_patients': '{:.1f}%'.format(sum(y_train)/x_train.shape[0]*100),
                        # '%_of_negative_patients': '{:.1f}%'.format(100-sum(y_train) / x_train.shape[0] * 100),
                        }}
    # is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # with_contradict_sign = np.zeros(shape=n_nodes, dtype=bool)
    # paths_in_this_tree = ['' for i in range(n_nodes)]
    # paths_threshold_in_this_tree = {}
    not_print_list = ['x', 'y']

    while len(stack) > 0:
        node_id = stack.pop()
        split_detail = b_node_dict[node_id]
        split_dict = extract_split_ft_threshold_next_node(split_detail)
        new_node_rule = ''
        if split_dict is None:

            key_replace = split_detail
            # print(key_replace)
        else:
            yes_feat_with_sign, yes_node = split_dict['yes']
            no_feat_with_sign, no_node = split_dict['no']
            thres = split_dict['thres']
            feat = int(split_dict['feature'].replace('f', ''))
            # print(feat, split_dict['feature'])
            # feat = int(feature_idx_inv_dict[split_dict['feature']])
            # b_node_detail[node_id]['node_rule'] = split_dict['node_rule']
            key_replace = split_dict['node_rule']
            new_node_rule = split_dict['node_rule'].replace(split_dict['feature'], feature_idx_inv_dict[feat])
            x_train_in_node = b_node_detail[node_id]['x']
            y_train_in_node = b_node_detail[node_id]['y']


            yes_bool = x_train_in_node[:, feat] < thres
            no_bool = np.logical_not(yes_bool)
            yes_x, yes_y = x_train_in_node[yes_bool], y_train_in_node[yes_bool]
            no_x, no_y = x_train_in_node[no_bool], y_train_in_node[no_bool]


            b_node_detail[yes_node] = {'x':yes_x, 'y':yes_y,
                        '#_of_patients':yes_x.shape[0],
                                       '#_of_positive_patients': sum(yes_y),
                                       '#_of_negative_patients': yes_x.shape[0] - sum(yes_y),
                        # '%_of_positive_patients': '{:.1f}%'.format(sum(yes_y)/yes_x.shape[0]*100),
                        # '%_of_negative_patients': '{:.1f}%'.format(100-sum(yes_y) / yes_x.shape[0] * 100)
                                       }

            b_node_detail[no_node] = {'x': no_x, 'y': no_y,
                                       '#_of_patients': no_x.shape[0],
                                      '#_of_positive_patients': sum(no_y),
                                      '#_of_negative_patients': no_x.shape[0] - sum(no_y),
                                       # '%_of_positive_patients': '{:.1f}%'.format(sum(no_y) / no_x.shape[0] * 100),
                                       # '%_of_negative_patients': '{:.1f}%'.format(100 - sum(no_y) / no_x.shape[0] * 100)
                                      }
            stack.append(yes_node)
            stack.append(no_node)
        if new_node_rule == '':
            replace_string = key_replace
        else:
            replace_string = new_node_rule
        for k, v in b_node_detail[node_id].items():
            if not k in not_print_list:
                replace_string += '\n{}:{}'.format(k, v)
        label_replace_dict[key_replace] = replace_string

    return label_replace_dict


def to_graphviz_custom(booster, training_data, outcome_name, fmap='', num_trees=0, rankdir=None,
                yes_color=None, no_color=None,
                condition_node_params=None, leaf_node_params=None,
                       **kwargs):
    """Convert specified tree to graphviz instance. IPython can automatically plot
    the returned graphiz instance. Otherwise, you should call .render() method
    of the returned graphiz instance.

    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphiz via graph_attr
    yes_color : str, default '#0000FF'
        Edge color when meets the node condition.
    no_color : str, default '#FF0000'
        Edge color when doesn't meet the node condition.
    condition_node_params : dict, optional
        Condition node configuration for for graphviz.  Example:

        .. code-block:: python

            {'shape': 'box',
             'style': 'filled,rounded',
             'fillcolor': '#78bceb'}

    leaf_node_params : dict, optional
        Leaf node configuration for graphviz. Example:

        .. code-block:: python

            {'shape': 'box',
             'style': 'filled',
             'fillcolor': '#e48038'}

    \\*\\*kwargs: dict, optional
        Other keywords passed to graphviz graph_attr, e.g. ``graph [ {key} = {value} ]``

    Returns
    -------
    graph: graphviz.Source

    """
    try:
        from graphviz import Source
        import json
    except ImportError:
        raise ImportError('You must install graphviz to plot tree')
    # if isinstance(booster, XGBModel):
    booster = booster.get_booster()

    # squash everything back into kwargs again for compatibility
    parameters_dot = 'dot'
    parameters_text = 'text'
    extra = {}
    for key, value in kwargs.items():
        extra[key] = value

    if rankdir is not None:
        kwargs['graph_attrs'] = {}
        kwargs['graph_attrs']['rankdir'] = rankdir
    for key, value in extra.items():
        if 'graph_attrs' in kwargs.keys():
            kwargs['graph_attrs'][key] = value
        else:
            kwargs['graph_attrs'] = {}
        del kwargs[key]

    if yes_color is not None or no_color is not None:
        kwargs['edge'] = {}
    if yes_color is not None:
        kwargs['edge']['yes_color'] = yes_color
    if no_color is not None:
        kwargs['edge']['no_color'] = no_color

    if condition_node_params is not None:
        kwargs['condition_node_params'] = condition_node_params
    if leaf_node_params is not None:
        kwargs['leaf_node_params'] = leaf_node_params

    if kwargs:
        parameters_dot += ':'
        parameters_dot += str(kwargs)
        parameters_text += ':'
        parameters_text += str(kwargs)
    tree_dot = booster.get_dump(
        # fmap=fmap,
        dump_format=parameters_dot)[num_trees]
    tree_text = booster.get_dump(
        # fmap=fmap,
        dump_format=parameters_text)[num_trees]
    # print(json.dump(tree_dot.__dict__))
    # print(tree_text_with_stats)
    fmap_df = pd.read_csv(fmap, sep='\t', header=None)
    # print(fmap_df.columns)
    fmap_dict = {}
    for index, row in fmap_df.iterrows():
        fmap_dict[int(row[0])] = row[1]


    label_replace_dict = extract_booster_label_to_dict(tree_text,
                                                       x_train=training_data[0],
                                                       y_train=training_data[1],
                                                       feature_idx_inv_dict = fmap_dict)
    for k, v in label_replace_dict.items():
        tree_dot = tree_dot.replace(k, v)
    tree_dot = tree_dot.replace(', missing', '')
    tree_dot = tree_dot.replace('graph [ rankdir=TB ]',
                                'graph [rankdir=TB, label="{}", labelloc=t, fontsize=30]'.format(outcome_name.split('/')[-1]))
    # tree_dot.replace("digraph {\n")
    # print(tree_dot)
    g = Source(tree_dot)

    # print(json.dumps(g.__dict__, indent=2))
    return g

def plot_tree_tiff_wrapper(booster,training_data,output_path,outcome_name, fmap='', num_trees=0, rankdir=None, ax=None, **kwargs):
    """
    Modified plot tree function
    Plot specified tree.

    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "TB"
        Passed to graphiz via graph_attr
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    kwargs :
        Other keywords passed to to_graphviz

    Returns
    -------
    ax : matplotlib Axes

    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree')
    from io import BytesIO

    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = to_graphviz_custom(booster, training_data, outcome_name=outcome_name,fmap=fmap, num_trees=num_trees, rankdir=rankdir,
                    **kwargs)
    g.render(filename=output_path, cleanup=True)
    # print(type(g))
    # print(g)
    # g.write_png(output_path)
    # # print(g.source)
    #
    # s = BytesIO()
    # s.write(g.pipe(format='svg'))
    # # s.seek(0)
    # img = image.imread(s)
    # #
    # ax.imshow(img)
    # ax.axis('off')
    # return ax

def visualize_xgb(clf, feature_names, training_data, outcome_name, labels=['0', '1'], file_name='test',
                  plot_dir='', ext='png', save=True, num_trees=0,
                  tree_dir=''):
    # from sklearn.tree import export_graphviz
    # from sklearn.externals.six import StringIO
    # # from IPython.display import Image
    # import pydotplus


    """
    Add parameters to the plotting function to control the node shape 
    https://github.com/dmlc/xgboost/issues/3858
    """
    cNodeParams = {'shape': 'box',
                   'style': 'filled,rounded',
                   'fillcolor': '#78bceb'}
    lNodeParams = {'shape': 'box',
                   'style': 'filled',
                   'fillcolor': '#e48038'}

    fig1, ax1 = plt.subplots(1,1, figsize=(15, 8))
    output_path = os.path.join(tree_dir, file_name)
    plot_tree_tiff_wrapper(booster=clf, fmap=feature_names, output_path=output_path, num_trees=num_trees, ax=ax1,
                           training_data=training_data, outcome_name=outcome_name,
                      **{
                          'size': str(5),
                         'condition_node_params': cNodeParams,
                         'leaf_node_params': lNodeParams
                         }
                      # conditionNodeParams=cNodeParams, leafNodeParams=lNodeParams
                      )
    # g = xgboost.to_graphviz(clf, conditionNodeParams=cNodeParams,
                            # leafNodeParams=lNodeParams,
                            # **{'size': str(15)})


    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(, 100)
    # fig.savefig('tree.png')
    # g.write_png(output_path)
    # fig1.savefig(output_path, dpi=400, pil_kwargs={"compression": "tiff_lzw"})


def sort_path(paths, labels=[], merge_labels=False, verbose=True, verify=True):
    import operator
    
    if len(labels) == 0: 
        labels = list(paths.keys())
        if verbose: print("(sort_path) Considering labels: {}".format(labels))
            
    if not merge_labels:
        sorted_paths = {}
        for label in labels: 
            # print("... paths[label]: {}".format(paths[label]))
            sorted_paths[label] = sorted(paths[label].items(), key=operator.itemgetter(1), reverse=True)
    else: # merge and sort
        # merge path counts associated with each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        

        # print("... merged paths: {}".format(paths))
        sorted_paths = sorted(paths2.items(), key=operator.itemgetter(1), reverse=True)
        
        if verify:
            topk = 3 
            for i in range(topk): 
                path, cnt = sorted_paths[i][0], sorted_paths[i][1]
                
                counts = []
                for label in labels: 
                    counts.append(paths[label].get(path, 0))
                print("(sort_path) #[{}] {} | total: {} | label-dep counts: {}\n".format(i, path, cnt, counts))
                
    return sorted_paths
    
def get_lineage(tree, feature_names, mode=0, verbose=False):
    """
    Params
    ------
    mode: {'feature_only'/0, 'full'/1}
    
    
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]    
    # print("> child nodes: {}".format(idx))

    def recurse(child, left, right, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:  # if input child node is among the set of children_left
            parent = np.where(left == child)[0].item() # find the node ID of its parent
            split = 'l'  # left split on the parent node to get to the child
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))
        # path.append(features[parent])

        if parent == 0:
            lineage.reverse()  # reverse order so that the path goes from root to leaf 
            return lineage
        else:
            return recurse(parent, left, right, lineage)

    paths = {}
    if mode in ('full', 1): 
        for child in idx:
            dseq = []  # each child as its corresponding decision path
            for node in recurse(child, left, right):
                if verbose: print(node)
                
                if isinstance(node, tuple): 
                    dseq.append(node)   # 4-tuple: (parent, split, threshold[parent], features[parent])
                    
                else: # a leaf node
                    label_id = np.argmax(tree.tree_.value[node][0])
                    # print('... label: {}'.format(label_id))
                    
                    label = label_id
                    if not label in paths: paths[label] = {}
                    cnt = len(paths[label])
                    paths[label][cnt] = dseq
                
    else: # keep track of the split point only 
        for child in idx:
            dseq = [] # each child as its corresponding decision path
            for node in recurse(child, left, right):
                if verbose: print(node)
                if isinstance(node, tuple): 
                    dseq.append(node[-1]) 
                    
                else: # a leaf node
                    label_id = np.argmax(tree.tree_.value[node][0])
                    # print('... label: {}'.format(label_id))
                    
                    label, dseq = label_id, tuple(dseq)
                    if not label in paths: paths[label] = {}
                    if not dseq in paths[label]: paths[label][dseq] = 0
                    paths[label][dseq] += 1 
                    
    return paths

def count_features2(estimator, feature_names, counts={}, labels = {}, sep=' ', verbose=True):
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    # given a tree, keep track of all its (decision) paths from root to leaves
    dpaths = get_lineage(estimator, feature_names, mode='full', verbose=False)
    
    # collect features and their thresholds
    # paths[label][cnt] = dseq
    
    for label in labels: 
        for index, dpath in dpaths[label].items():
            # index: the ordinal of each decision paths with each element as a 4-tuple: (parent, split, threshold[parent], features[parent])
            assert isinstance(dpath, list), "path value is not a list? {} | value:\n{}\n".format(type(dpath), dpath)
            for entry in dpath: 
                assert isinstance(entry, tuple), "Invalid path value:\n{}\n".format(entry)
                feature, threshold = entry[-1], entry[-2]
                if not feature in counts: counts[feature] = []
                counts[feature].append(threshold)
    
    return counts

def get_feature_threshold_tree(estimator, counts={}, feature_names=[]):
    feature_threshold_count = {}
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    for i in range(n_nodes):
        if not is_leaves[i]:
            # print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            feature_name = feature_names[feature[i]]
            if not feature_name in counts:
                counts[feature_name] = []

            if not feature_name in feature_threshold_count:
                feature_threshold_count[feature_name] = 1
                counts[feature_name].append(threshold[i])
            elif feature_threshold_count[feature_name] == 1:
                # If variable appear more than once in the tree, drop it
                feature_threshold_count[feature_name] += 1
                counts[feature_name].pop()
            # print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
            #       "node %s."
            #       % (node_depth[i] * "\t",
            #          i,
            #          children_left[i],
            #          feature_name,
            #          threshold[i],
            #          children_right[i],
            #          ))
    return counts


def count_features(paths, labels=[], verify=True, sep=' '):
    import collections
    
    # input 'paths' must be a label-dependent path i.e. paths as a dictionary should have labels as keys
    if len(labels) == 0: labels = list(paths.keys())
    if verify: 
        if len(labels) > 0: 
            assert set(paths.keys()) == set(labels), \
                "Inconsistent label set: {} vs {}".format(set(paths.keys()), set(labels))
    
    # merge path counts from each label => label-agnostic path counts
    paths2 = {}
    for label in labels: 
        for dseq, cnt in paths[label].items(): 
            if not dseq in paths2: paths2[dseq] = 0
            paths2[dseq] += cnt
    
    for path in paths2: 
        if isinstance(path, str): path = path.split(sep)
        # policy a. if a node appears more than once in a decision path, count only once?
        # for node in np.unique(path): 
        #     pass

        # policy b. count each occurrence
        for node, cnt in collections.Counter(path).items(): 
            if not node in counts: counts[node] = 0
            counts[node] += cnt
    return counts

def parse_split(list_str, split_delimiter='\t', yes='yes', no='no'):
    root = list_str[0]
    root_rule = root.split('[')[1]
    # reach the leaf
    if len(root_rule) == 0:
        return None
    root_rule = root_rule.split(']')[0]
    yes_node = root.split('yes=')[1].split(',')[0]
    no_node = root.split('no=')[1].split(',')[0]

    yes_node_idx = 0
    no_node_idx = 0
    for str_idx, s in enumerate(list_str):
        a = 0

def extract_split_ft_threshold_next_node(val, xgb_ft_real_dict=None, yes='yes', no='no'):
    if 'leaf' in val:
        return None
    else:
        node_rule = val.split('[')[1].split(']')[0]
        xgb_ft = node_rule.split('<')[0]
        if xgb_ft_real_dict is None:
            node_rule_replaced = node_rule
        else:
            node_rule_replaced = node_rule.replace(xgb_ft, xgb_ft_real_dict[xgb_ft])
        node_splited_by_smaller_sign = node_rule_replaced.split('<')
        node_feat = node_splited_by_smaller_sign[0]
        yes_node = val.split('yes=')[1].split(',')[0]
        no_node = val.split('no=')[1].split(',')[0]
        return {'yes': (node_feat+'<', int(yes_node)),
                'no': (node_feat+'>=', int(no_node)),
                'thres': float(node_splited_by_smaller_sign[-1]),
                'feature': node_feat,
                'node_rule':node_rule}



def count_paths_with_thres_sign_from_xgb(estimator,
                                paths={}, feature_names=[], paths_threshold = {},
                                paths_from = {},
                                labels = {},
                                merge_labels=True, to_str=False,
                                sep=' ', verbose=True, index=0, multiple_counts=True):
    feature_threshold_count = {}
    boosters = estimator.get_booster().get_dump()
    counted_path = set()
    for b_idx, b0 in enumerate(boosters):
        b0_split_n = b0.replace('\t', '').split('\n')
        xgb_ft_to_real_ft = {k: v for k, v in zip(estimator.get_booster().feature_names, feature_names)}

        b0_node_dict = {}
        for bnode in b0_split_n:
            bn_split = bnode.split(':')
            if len(bn_split) == 2:

                b0_node_dict[int(bn_split[0])] = bn_split[1]

        n_nodes = len(b0_node_dict)
        stack = [0]
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        with_contradict_sign = np.zeros(shape=n_nodes, dtype=bool)
        paths_in_this_tree = ['' for i in range(n_nodes)]
        paths_threshold_in_this_tree = {}


        while len(stack) > 0:
            node_id = stack.pop()
            split_detail = b0_node_dict[node_id]
            split_dict = extract_split_ft_threshold_next_node(split_detail, xgb_ft_to_real_ft)
            if split_dict is None:
                is_leaves[node_id] = True
            else:

                if node_id != 0:
                    space = '\t'
                    threshold_list = paths_threshold_in_this_tree[paths_in_this_tree[node_id]] + [split_dict['thres']]
                else:
                    space = ''
                    threshold_list = [split_dict['thres']]

                base_str = paths_in_this_tree[node_id] + space
                yes_feat_with_sign, yes_node = split_dict['yes']
                no_feat_with_sign, no_node = split_dict['no']

                stack.append(yes_node)
                stack.append(no_node)

                yes_string = base_str + yes_feat_with_sign
                no_string = base_str + no_feat_with_sign

                if split_dict['feature'] in paths_in_this_tree[node_id]:
                    with_contradict_sign[yes_node] = True
                    with_contradict_sign[no_node] = True
                else:
                    with_contradict_sign[yes_node] = with_contradict_sign[node_id]
                    with_contradict_sign[no_node] = with_contradict_sign[node_id]

                paths_threshold_in_this_tree[yes_string] = threshold_list
                paths_threshold_in_this_tree[no_string] = threshold_list
                paths_in_this_tree[yes_node] = yes_string
                paths_in_this_tree[no_node] = no_string


        paths_in_this_tree = np.array(paths_in_this_tree)[is_leaves * np.logical_not(with_contradict_sign)]
        # print(paths_in_this_tree)
        for path in paths_in_this_tree:
            # print(path)
            if path != '':
                if not path in paths:
                    paths[path] = 1
                    paths_threshold[path] = [paths_threshold_in_this_tree[path]]
                    counted_path.add(path)
                    paths_from[path] = [(index, b_idx)]
                else:
                    if multiple_counts:
                        paths[path] += 1
                    else:
                        if not path in counted_path:
                            counted_path.add(path)
                            paths[path] += 1
                    paths_from[path].append((index, b_idx))
                    paths_threshold[path].append(paths_threshold_in_this_tree[path])
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    return paths, paths_threshold, paths_from


def count_paths_with_thres_sign(estimator,
                                paths={}, feature_names=[], paths_threshold = {},
                                labels = {},
                                merge_labels=True, to_str=False,
                                sep=' ', verbose=True, index=0):
    feature_threshold_count = {}
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    # print(feature)
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    with_contradict_sign = np.zeros(shape=n_nodes, dtype=bool)
    paths_in_this_tree = ['' for i in range(n_nodes)]
    paths_threshold_in_this_tree = {}



    stack = [(0, -1)]  # seed is the root node id and its parent depth
    # paths_in_this_tree = []
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            if node_id != 0:
                space = '\t'
                threshold_list = paths_threshold_in_this_tree[paths_in_this_tree[node_id]] + [threshold[node_id]]
            else:
                space = ''
                threshold_list = [threshold[node_id]]
                # paths_threshold_in_this_tree[feature_names[feature[node_id]]] = threshold[node_id]
            # Concat string with their parent

            base_str = paths_in_this_tree[node_id] + space + feature_names[feature[node_id]]
            left_string = base_str + "<="
            right_string = base_str + ">"

            # if (feature_names[feature[node_id]] + ">") in base_str:
            #     with_contradict_sign[children_left[node_id]] = True
            #     with_contradict_sign[children_right[node_id]] = with_contradict_sign[node_id]
            # elif (feature_names[feature[node_id]] + "<=") in base_str:
            #     with_contradict_sign[children_right[node_id]] = True
            #     with_contradict_sign[children_left[node_id]] = with_contradict_sign[node_id]
            # else:
            #     with_contradict_sign[children_left[node_id]] = with_contradict_sign[node_id]
            #     with_contradict_sign[children_right[node_id]] = with_contradict_sign[node_id]

            if feature_names[feature[node_id]] in paths_in_this_tree[node_id]:
                with_contradict_sign[children_left[node_id]] = True
                with_contradict_sign[children_right[node_id]] = True
            else:
                with_contradict_sign[children_left[node_id]] = with_contradict_sign[node_id]
                with_contradict_sign[children_right[node_id]] = with_contradict_sign[node_id]

            paths_threshold_in_this_tree[left_string] = threshold_list
            paths_threshold_in_this_tree[right_string] = threshold_list
            paths_in_this_tree[children_left[node_id]] = left_string
            paths_in_this_tree[children_right[node_id]] = right_string

        else:
            # print('leave with contradict sign:', with_contradict_sign[node_id])
            # print('leave path :', paths_in_this_tree[node_id])
            is_leaves[node_id] = True


    paths_in_this_tree = np.array(paths_in_this_tree)[is_leaves*np.logical_not(with_contradict_sign)]
    # print(paths_in_this_tree)
    for path in paths_in_this_tree:
        # print(path)
        if path != '':
            if not path in paths:
                paths[path] = 1
                paths_threshold[path] = [paths_threshold_in_this_tree[path]]
            else:
                paths[path] += 1
                paths_threshold[path].append(paths_threshold_in_this_tree[path])
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    return paths, paths_threshold

def count_paths(estimator, feature_names, paths={}, counts={}, labels = {}, merge_labels=True, to_str=False, 
                sep=' ', verbose=True, index=0):  # cf: use count_paths2() to count on per-instance basis  
    """
    The decision estimator has an attribute called tree_  which stores the entire
    tree structure and allows access to low level attributes. The binary tree
    tree_ is represented as a number of parallel arrays. The i-th element of each
    array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    Some of the arrays only apply to either leaves or split nodes, resp. In this
    case the values of nodes of the other type are arbitrary!

    Among those arrays, we have:
      - left_child, id of the left child of the node
      - right_child, id of the right child of the node
      - feature, feature used for splitting the node
      - threshold, threshold value at the node
      
    """
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    # given a tree, keep track of all its (decision) paths from root to leaves
    paths_prime = get_lineage(estimator, feature_names, mode=0, verbose=verbose)
    # print("... index: {} | paths_prime: {}".format(index, paths_prime))
    if to_str:
        paths_prime2 = {}
        for label in labels: 
            paths_prime2[label] = {}
            for path, cnt in paths_prime[label].items(): 
                assert isinstance(path, tuple), "path value is not a tuple (dtype={})? {}".format(type(path), path)
                path_str = sep.join(path)
                paths_prime2[label][path_str] = cnt
        paths_prime = paths_prime2
    # print("...... index: {} | (to_str)-> paths_prime: {}".format(index, paths_prime)) # sample_dict(paths_prime[label], 5)
        
    # merge new map (of paths) with existing map (of paths)
    for label in labels:
        #assert not to_str or isinstance(next(iter(paths[label].keys())), str), \
        #    "(count_paths) Inconsistent dtype | paths[label]:\n{}\n".format(sample_dict(paths[label], 5))
        if not label in paths: paths[label] = {}
        for dseq, cnt in paths_prime[label].items():
            if to_str: assert isinstance(dseq, str)   
            if not dseq in paths[label]: paths[label][dseq] = 0
            paths[label][dseq] += cnt
                
    # print("(debug) paths[1]: {}".format(paths[1]))
    
    if verbose: 
        for label in labels: 
            print("> Label: {} | (example) decision paths:\n{}\n".format(labels[label], sample_dict(paths[label], 5)))
            
    if merge_labels: 
        # merge path counts from each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        paths = paths2
        
        # count feature usage: how many times was a variable used as a splitting point?
        for path in paths: 
            if isinstance(path, str): path = path.split(sep)
            assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
            # policy a. if a node appears more than once in a decision path, count only once?
            # for node in np.unique(path): 
            #     pass
            
            # policy b. count each occurrence
            for node, cnt in collections.Counter(path).items(): 
                if not node in counts: counts[node] = 0
                counts[node] += cnt
                
    else: 
        # count feature occurrences
        for label in labels: 
            if not label in counts: counts[label] = {} # label-dependent counts
            for path in paths[label].keys(): 
                if isinstance(path, str): path = path.split(sep)
                
                assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
                
                # policy b: count each occurrence
                for node, cnt in collections.Counter(path).items(): 
                    if not node in counts[label]: counts[label][node] = 0
                    counts[label][node] += cnt  
                    
    return paths, counts

def count_paths2(estimator, Xt, feature_names, labels={}, paths={}, counts={}, merge_labels=True, 
                 to_str=False, sep=' ',verbose=True):
    """
    Count decision paths with respect to input data (Xt), where the input data instances 
    are usually the test set from a train-test split: the training split is used to
    build the decision tree, whereas the test split is used to evaluate the performance 
    and count the decision paths (so that we can find out which paths are more popular than 
    the others).
    
    """
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    lookup = {}
    features  = [feature_names[i] for i in estimator.tree_.feature]
    # print("(count_paths2) features: {}".format(features))
    
    assert isinstance(paths, dict), "Invalid dtype for decision paths: {}".format(type(paths))
    for label in labels: 
        if not label in paths: paths[label] = {}
    # if len(counts) == 0: counts = {f: 0 for f in features}
    
    if not isinstance(Xt, np.ndarray): Xt = Xt.values
    N = Xt.shape[0]
    
    node_indicator = estimator.decision_path(Xt)
    feature_index = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # print("> n(feature_index): {}".format(len(feature_index)))

    # the leaves ids reached by each sample.
    leaf_id = estimator.apply(Xt)
    
    # print("(count_path) size(Xt): {}, dim(node_indicator): {} | n(leaf_id): {}".format(N, node_indicator.shape, len(leaf_id)))

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
        
    # [test]
    for i in range(N): 
        #  row i, [indptr[i]:indptr[i+1]] returns the indices of elements to take from data and indices corresponding to row i.
        
        # take the i-th row 
        dseq = []
        for node_id in node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]]: 
            dseq.append(node_id)
        # print("> sample #{} | {}".format(i, dseq)) 


    for i in range(N): 
        sample_id = i
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        
        dseq = [] # path to the leaf for this sample (i)
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                label = label_id = np.argmax(estimator.tree_.value[node_id][0])

                node_descr = "label: {}".format(labels[label_id])
                lookup[node_id] = node_descr
                # print("> final NodeID[{id}: {label}]".format(id=node_id, label=labels[label_id]))
                
                dseq.append(label_id)  # labels[label_id]
                
                continue

            if (Xt[sample_id, feature_index[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # feature value: X[sample_id, feature[node_id]
            node_descr = "{var} {sign} {th}".format(var=features[node_id], sign=threshold_sign, th=threshold[node_id])
            lookup[node_id] = node_descr
            # counts[features[node_id]] += 1
            
            dseq.append(features[node_id])
        ### end foreach node_id ... 
        #   ... node sequence for sample (i) is determined
        
        dseq = tuple(dseq)
        # desc_seq = '> '.join([lookup[node] for node in dseq])
        
        internal_seq, label = dseq[:-1], dseq[-1]
        if not label in paths: paths[label] = {}
        if to_str: internal_seq = sep.join(internal_seq)
            
        if not internal_seq in paths[label]: paths[label][internal_seq] = 0
        paths[label][internal_seq] += 1
        
    ### end foreach test instance
        
    if merge_labels: 
        # merge path counts from each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        paths = paths2
        
        # count feature usage: how many times was a variable used as a splitting point?
        for path in paths: 
            if isinstance(path, str): path = path.split(sep)
            assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
            # policy a. if a node appears more than once in a decision path, count only once?
            # for node in np.unique(path): 
            #     pass
            
            # policy b. count each occurrence
            for node, cnt in collections.Counter(path).items(): 
                if not node in counts: counts[node] = 0
                counts[node] += cnt
    else: 
        # count feature occurrences
        for label in labels: 
            if not label in counts: counts[label] = {} # label-dependent counts
            for path in paths[label].keys(): 
                if isinstance(path, str): path = path.split(sep)
                
                assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
                
                # policy b: count each occurrence
                for node, cnt in collections.Counter(path).items(): 
                    if not node in counts[label]: counts[label][node] = 0
                    counts[label][node] += cnt  
                    
    return paths, counts


def t_vis_tree(dtree):
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus   # use pip install pydotplus

    iris=datasets.load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    y=iris.target

    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())

    return


def t_classification(): 
    from sklearn.tree import DecisionTreeClassifier    # Import decision tree classifier model

    tree = DecisionTreeClassifier(criterion='entropy', # Initialize and fit classifier
        max_depth=4, random_state=1)
    tree.fit(X, y)

    t_vis_tree(tree)

    return

def test(**kargs): 

    load_merge(vars_matrix='exposures-4yrs.csv', label_matrix='nasal_biomarker_asthma1019.csv')

    return

if __name__ == "__main__": 
    test()

