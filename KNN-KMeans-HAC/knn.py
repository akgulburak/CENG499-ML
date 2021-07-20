import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_distances(train_data, test_datum):
    """
    Calculates euclidean distances between test_datum and every train_data
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param test_datum: A (D, ) shaped numpy array
    :return: An (N, ) shaped numpy array that contains distances
    """
    new_array = np.empty(train_data.shape[0])
    for i in range(train_data.shape[0]):
        eucDistance=0
        for j in range(train_data.shape[1]):
            distance = train_data[i][j]-test_datum[j]
            eucDistance+=(distance*distance)
        new_array[i]=math.sqrt(eucDistance)

    return new_array

def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """

    sort_indices = np.argsort(distances,0)
    sorted_labels = np.array(labels)[sort_indices]
    #sorted_distances = np.array(distances)[sort_indices]
    votes = np.zeros((distances.shape))
    for i in range(k):
        votes[sorted_labels[i]]+=1
    
    return votes.argmax()

def knn(train_data, train_labels, test_data, test_labels, k):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: A float. The calculated accuracy.
    """
    found_labels = np.zeros((test_labels.shape))
    for i in range(test_data.shape[0]):
        distances = calculate_distances(train_data,test_data[i])
        found_labels[i] = majority_voting(distances, train_labels, k)

    correct = 0
    for i in range(test_data.shape[0]):
        if test_labels[i]==found_labels[i]:
            correct+=1

    return correct/test_data.shape[0]

def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    splitted_data = np.split(whole_train_data,k_fold)
    splitted_labels = np.split(whole_train_labels,k_fold)

    mask = np.ones(len(splitted_data), dtype=bool)
    mask[validation_index] = False

    train_data = np.array(splitted_data)[mask].reshape(-1, np.array(splitted_data).shape[-1])
    train_labels = np.array(splitted_labels)[mask].flatten()

    validation_data = np.array(splitted_data)[validation_index].reshape(-1, np.array(splitted_data).shape[-1])
    validation_labels = np.array(splitted_labels)[validation_index].flatten()

    return train_data, train_labels, validation_data, validation_labels

def cross_validation(whole_train_data, whole_train_labels, k, k_fold):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param k_fold: An integer.
    :return: A float. Average accuracy calculated.
    """
    accuracy = 0
    for i in range(0,k_fold):
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data,
     whole_train_labels, i, k_fold)
        accuracy += knn(train_data, train_labels, validation_data, validation_labels, k)

    return accuracy/k_fold

def test(k):
    test_data = np.load("hw2_data/knn/test_data.npy")
    test_labels = np.load("hw2_data/knn/test_labels.npy")
    accuracy = knn(test_data, test_labels, test_data, test_labels,k)
    print("Accuracy on test data is: " , accuracy)
def main(k_limit,k_fold):
    train_data = np.load("hw2_data/knn/train_data.npy")
    train_labels = np.load("hw2_data/knn/train_labels.npy")
    
    accuracies = np.zeros((k_limit,))
    knn = np.zeros((k_limit,))

    for i in range(1,k_limit+1):
        accuracies[i-1] = cross_validation(train_data, train_labels, i, k_fold)
        knn[i-1] =int(i)

    #make_int = range(math.floor(min(knn)), math.ceil(max(knn))+1)
    plt.xticks(range(1,k_limit,k_limit//10))
    plt.plot(knn, accuracies)
    plt.savefig('knn_graph.png')
    plt.close()
    print("Best k is : " ,knn[np.argmax(accuracies)])

#main(200,10)
#test(11)