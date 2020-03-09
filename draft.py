from sklearn import tree

"""
Reference
---------
1. Visualize decision tree
   
   a. https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/

2. CV 

   a. http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
   
"""


    #### parameters ####
    test_size = kargs.get('test_size', 0.3)
    verbose = kargs.get('verbose', False)
    plot_ext = kargs.get('ext', 'tif')
    merge_paths = kargs.get('merge_paths', True)
    policy_count = kargs.get('policy_counts', 'standard') # options: {'standard', 'sample-based'}
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    validate_tree = kargs.get('validate_tree', True)
    plot_dir = kargs.get('plot_dir', plotDir)
    plot_ext = kargs.get('plot_ext', 'tif')
    to_str = kargs.get('to_str', False)  # if True, decision paths are represented by strings (instead of tuples)
    ####################


count_features2(estimator, feature_names, counts={}, labels = {}, merge_labels=True, sep=' ', verbose=True)

count_features(paths, labels=[], verify=True, sep=' ')

def visualize(clf, feature_names, class_names, save=True, ext='png'):
    # from sklearn import tree
    
    # Create DOT data
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_names,  
                                    class_names=class_names)

    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)  

    if save: 
        # Create PDF
        fname = "iris-test.{}".format(ext)
        graph.write_pdf(fname)
#         Image(filename=fname)
    else: 
        # Show grhttp://localhost:8888/notebooks/asthma_env/demo-dtree.ipynb#aph
        Image(graph.create_png())
        # Image.open(graph.create_png())


def count_paths(estimator, feature_names, paths={}, counts={}, labels = ['-','+']):  # cf: use count_paths2() to count on per-instance basis  
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
    # feature_names = feature_cols

    # Using those arrays, we can parse the tree structure 
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left  # in terms of node ID
    children_right = estimator.tree_.children_right

    feature_index = estimator.tree_.feature  # feature index
    # print("> n(feature_index): {}".format(len(feature_index)))
    features = [feature_names[i] for i in feature_index]
    print("> variables (from DT): {}".format(features))

    threshold = estimator.tree_.threshold

    ################################################
    
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    dpath = []
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test/internal node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    dseq = []
    for i in range(n_nodes):
        if is_leaves[i]:
            label_id = np.argmax(estimator.tree_.value[i][0])
            label_name = labels[label_id]
            print("%snode=%s leaf node | prediction: %s (%s)" % (node_depth[i] * "\t", i, label_name, label_id))
            
            
        else:
            print("> depth:{}, id:{}, left_child: {}, right_child: {}, feature[i]: {}, threshold: {}".format(
                node_depth[i], i, children_left[i], children_right[i], feature_names[i], threshold[i]))
            print("%snode=%s test node: go to node %s if (%s <= %s) else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     features[i],
                     threshold[i],
                     children_right[i],
                     ))

            if not features[i] in counts: counts[features[i]] = 0
            counts[features[i]] += 1
            
    return paths, counts