import math
import numpy as np
from graphviz import Digraph
def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    bucket = np.array(bucket)
    entro = 0
    bucket_size = np.sum(np.array(bucket))

    for i in range(bucket.shape[0]):
        if(bucket_size == 0):
            continue

        probs = bucket[i]/bucket_size

        if(probs == 0):
            continue
        entro = entro-(probs)*math.log(probs,2)
    return entro

def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    r_size = np.sum(np.array(right_bucket))
    l_size = np.sum(np.array(left_bucket))

    left_entro = entropy(left_bucket)
    right_entro = entropy(right_bucket)
    parent_entro = entropy(parent_bucket)

    info = ((left_entro*l_size+right_entro*r_size)/
        (l_size+r_size))
    gain = parent_entro-info

    return gain

def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    gini_sum = 0
    b_size = np.sum(np.array(bucket))
    

    for i in range(len(bucket)):
        if(b_size == 0):
            continue

        probs = bucket[i]/b_size
        gini_sum += (probs*probs)

    return 1-gini_sum

def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    l_size = np.sum(np.array(left_bucket))
    r_size = np.sum(np.array(right_bucket))
    t_size = l_size+r_size
    average_gini = (l_size*gini(left_bucket)/t_size)+(r_size*gini(right_bucket)/t_size)

    return average_gini

def helper_class(labels,num_classes):
    tmp = np.zeros((num_classes))
    for i in range(len(labels)):
        tmp[labels[i]]+=1
    return tmp

def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    attributes = data[:,attr_index]
    splitted = np.zeros((data.shape[0]-1,2))

    s_indices = np.argsort(attributes)

    sorted_attr = np.array(attributes)[s_indices]
    labels = np.array(labels)[s_indices]

    parent = helper_class(labels,num_classes)

    for i in range(data.shape[0]-1):
        splitted[i,0] = (sorted_attr[i]+sorted_attr[i+1])/2
        heur_val = 0

        left_labels = labels[sorted_attr<splitted[i,0]]
        right_labels = labels[sorted_attr>=splitted[i,0]]

        if heuristic_name=="info_gain":
            heur_val = info_gain(parent,helper_class(left_labels,num_classes),helper_class(right_labels,num_classes))

        elif heuristic_name=="avg_gini_index":
            heur_val = avg_gini_index(helper_class(left_labels,num_classes),helper_class(right_labels,num_classes))
        
        splitted[i,1]=heur_val
        #print("------")
        #print(labels)
        #print(splitted[i,0])
        #print(splitted[i,1])
        #print(data)
        #print("******")
#    print("------")
#    print(splitted)
#    print("******")
    return splitted

def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    observed = np.array((left_bucket,right_bucket))
    expected = np.zeros(observed.shape)
    
    table = np.zeros((observed.shape[0]+1,observed.shape[1]+1))

    x_size = 0
    for i in range(table.shape[0]-1):
        table[i][-1] = np.sum(observed[i])
        if table[i][-1]!=0:
            x_size+=1

    y_size = 0    
    for i in range(table.shape[1]-1):
        table[-1][i] = np.sum(observed[:,i])
        if table[-1][i]!=0:
            y_size+=1        

    table[-1][-1]=np.sum(table[-1])

    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            expected[i][j] = table[i][-1]*table[-1][j]/table[-1][-1]

    chi = 0
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            top = (observed[i][j]-expected[i][j])
            if expected[i][j]==0:
                continue
            chi+=(top*top/expected[i][j])

    return chi,(x_size-1)*(y_size-1)