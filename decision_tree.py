# decision_tree.py
# Lina Saha (LXS170008)
# Anusha Chandrasekaran (AXC179230)
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import random
import os
#import graphviz

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    dict = {}
    a, c = np.unique(x, return_counts=True)
    for i in a:
        dict[i] = []
        for j in range(len(x)):
            if i == x[j]:
                dict[i].append(j)
    return dict

    raise Exception('Function not yet implemented!')

def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    if w is None:
        w = np.ones((len(y),1), dtype=int)
        
    total=np.sum(w)
    partitionOfy=partition(y)
    ent_y = 0
    
    for i in partitionOfy.keys():
        splitOnw=[]
        for x in partitionOfy[i]:
            splitOnw.append(w[x])
        ent_y += - (np.sum(splitOnw)/total) * np.log2((np.sum(splitOnw)/total))

    return ent_y
        
#    partitionOfy = partition(y) 
#    
#    e, c = np.unique(y, return_counts=True)
#    hy = 0
#    total = np.sum(w)
#    for i in range(len(e)):
#        sum = ((-(np.sum(w[partitionOfy[i]])))/total) * np.log2(np.sum(w[partitionOfy[i]])/total)
#        hy = hy + sum
#
#    
#    print("Calculated entropy------",hy)
#    return hy

    raise Exception('Function not yet implemented!')

def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    # Returns the mutual information: I(x, y) = H(y) - H(y | x)
    # """

    if w is None:
        w = np.ones((len(y),1), dtype=int)
        
    hy = entropy(y, w)
    partitionOfX = partition(x)
    hy_x = 0
    
    partitionOfX = partition(x)
    e, c = np.unique(x, return_counts=True)
    total = np.sum(w)
    for j in partitionOfX:
        sum = (np.sum(w[partitionOfX[j]])/total) * entropy(y[partitionOfX[j]], w[partitionOfX[j]])
        hy_x += sum

    mutual_info = (hy - hy_x)
    #print("Calculated Mutual Information------",mutual_info)
    return mutual_info

    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    if w is None:
        w = np.ones((len(y),1), dtype=int)
    
    tree = {}
    array, count = np.unique(y, return_counts=True)
    infoGain = dict()
    attribute_value = []
    cols = len(x[0])


    if len(array) == 1:
        return array[0]

    if len(array) == 0:
        return 0

    if attribute_value_pairs == None:
        for i in range(cols):
            for val in np.unique(x[:, i]):
                attribute_value.append((i,val))
    else:
            attribute_value = attribute_value_pairs

    if len(attribute_value) == 0 or depth == max_depth:
         return array[np.argmax(count)]

    for i, val in(attribute_value):
        bin = (np.array(x)[:, i] == val).astype(int)
        infoGain[(i,val)] = mutual_information(bin, y, w)

    #print(infoGain)
    attribute, value = max(infoGain, key=infoGain.get)

    #print(attribute, value, depth, max_depth)
    partition_new = partition((np.array(x)[:, attribute] == value).astype(int))


    drop = attribute_value.index((attribute, value))
    copy = attribute_value.copy()
    copy.pop(drop)
    
    
    depth = depth + 1
#    for val, index in partitions.items():
#        split_y=[]
#        split_x=[]
#        for k in index:
#            split_y.append(y[k])
#            split_x.append(np.array(x)[k,:]) 
        
    for split_on, index in partition_new.items():
        split_x = x.take(index, axis=0)
        split_y = y.take(index, axis=0)
        decide_upon = bool(split_on)

        tree[(attribute, value, decide_upon)] = id3(split_x, split_y, attribute_value_pairs=attribute_value, max_depth=max_depth, 
                                                 depth=depth, w=w)

    return tree

    raise Exception('Function not yet implemented!')

def predict(x, tree):
    """
    For prediciting exampls with boosting alogirthm where we multiply the precition with respecte to the alpha and normalize it with
    total alpha vlaues
    Returns the predicted label of x according to tree
    """
    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_trees) is dict:
                label = predict(x, sub_trees)
            else:
                label = sub_trees

            return label

    raise Exception('Function not yet implemented!')

def predict_example(x, h_ens):

    y_pred = []

    for row_x in x:
        pred=[]
        total = 0    
        for i in h_ens.keys():
            alpha, tree = h_ens[i]
            y = predict(row_x, tree)
            pred.append(y*alpha)
            total += alpha
        predictedValue = np.sum(pred)/total        
        if(predictedValue >= 0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return y_pred

    
def compute_error(y_true, y_pred, w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    if w is None:
        w=np.ones((len(y_true),1),dtype=int)
        
    length_y = len(y_true)
    error = [w[i] * (y_true[i] != y_pred[i]) for i in range (length_y)]
    return (np.sum(error)/np.sum(w))

    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


def confusion_matrix(trueLabels, predLabels):

    matrix = np.zeros((2,2))
    for i in range(len(trueLabels)):
        if trueLabels[i] == 0 and predLabels[i] == 0:
            matrix[0][0] += 1
        elif trueLabels[i] == 0 and predLabels[i] == 1:
            matrix[0][1] += 1
        elif trueLabels[i] == 1 and predLabels[i] == 0:
            matrix[1][0] += 1
        elif trueLabels[i] == 1 and predLabels[i] == 1:
            matrix[1][1] += 1

    return matrix    

def subsample(x):
    sample = list()
    n_sample = len(x)
    while len(sample) < n_sample:
        index = random.randint(0, n_sample-1)
        sample.append(index)
    return sample

def bagging(x, y, max_depth, num_trees):
    h_i = {}
    alpha_i = 1
    
    for i in range(0, num_trees):
        indices = subsample(x)
        x_sample = (x[indices])
        y_sample = (y[indices])
            
        decision_tree = id3(x_sample, y_sample, max_depth=max_depth)
        #print("\n",decision_tree)
        h_i[i] = (alpha_i, decision_tree)
        
    return h_i

#def boosting(x, y, max_depth,num_stumps):
#    rows, cols = np.shape(x)
#    h_ens = {}
#    alpha_i = 0
#    y_pred = []
#
#    for stump in range(0, num_stumps):
#        if stump == 0:
#            d = (1/rows)
#            wt = np.full((rows, 1), d)
#        else:
#            weight = wt
#            wt = []
#            for i in range(rows):
#                if y[i] == y_pred[i]:
#                    wt.append(weight[i] * np.exp(-1*alpha_i))
#                else:
#                    wt.append(weight[i] * np.exp(alpha_i))
#            wt = wt/np.sum(wt)
#        decision_tree = id3(x, y, max_depth=max_depth, w=wt)
#
#        y_pred = [predict(row_x, decision_tree) for row_x in x]
#        temp = 0
#        for i in range(rows):
#            if(y_pred[i] != y[i]):
#                temp += wt[i]
#        error = (1/(np.sum(wt))) * temp
#        alpha_i = 0.5 * np.log((1-error)/error)
#        h_ens[stump] = (alpha_i, decision_tree)
#    return h_ens
    
def boosting(x, y, max_depth,num_stumps):

    for stump in range(0, num_stumps):
        rows, cols = np.shape(x)
        d = (1/rows)
        wt = np.full((rows, 1), d)
        h_ens = {}
        alpha_i = 0
        trn_pred = []

        tree = id3(x, y, max_depth=max_depth, w=wt)
        trn_pred = [predict(x_row, tree) for x_row in x]
        error = compute_error(ytst, y_pred, wt)
        alpha_i = 0.5 * np.log((1-error)/error)
        
        weight = wt
        wt = []
        for i in range(rows):
            if y[i] == trn_pred[i]:
                wt.append(weight[i] * np.exp(-1*alpha_i))
            else:
                wt.append(weight[i] * np.exp(alpha_i))
        wt = wt/np.sum(wt)        
        h_ens[stump] = (alpha_i, tree)
    return h_ens

def predict_boosting_example(x, h_ens):
    """
    For prediciting exampls with boosting alogirthm where we multiply the precition with respecte to the alpha and normalize it with
    total alpha vlaues
    Returns the predicted label of x according to tree
    """
    y_pred = []
    total = 0

    for i in h_ens:
        alpha, tree = h_ens[i]
        y = predict(x, tree)
        y_pred.append(y*alpha)
        total += alpha
    predictValue = np.sum(y_pred)/total

    if(predictValue >= 0.5):
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    #Compute test error for bagging
#    for max_depth in (3,5):
#        for num_trees in (5,10):
#            h_i=bagging(Xtrn, ytrn, max_depth, num_trees)
#            print('Bagging with maxdepth ', max_depth, ' bagsize ', num_trees)
#            y_pred=predict_example(Xtst,h_i)
#            tst_err = compute_error(ytst, y_pred)
#            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
#            print('Confusion Matrix:') 
#            print(confusion_matrix(ytst, y_pred))
#            print('\n')

    #Compute test error for boosting
    for max_depth in (1,2):
        for num_trees in (5,10):
            h_i=boosting(Xtrn, ytrn, max_depth, num_trees)
            print('Boosting with maxdepth ', max_depth, ' bagsize ', num_trees)
            y_pred=predict_example(Xtst,h_i)
            tst_err = compute_error(ytst, y_pred)
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
            print('Confusion Matrix:') 
            print(confusion_matrix(ytst, y_pred))
            print('\n')

    