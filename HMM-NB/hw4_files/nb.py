import math as m
import numpy as np

def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data
    """
    vocab = set()
    for i in range(len(data)):
        for j in data[i]:
            if not j in vocab:
                vocab.add(j)
    return vocab

def estimate_pi(train_labels):
    """
    Estimates the probability of every class label that occurs in train_labels.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :return: pi. pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    probs = {}
    for i in train_labels:
        if i in probs:
            probs[i]+=1    
        else:
            probs[i]=1    
    for i in probs:
        probs[i]/=len(train_labels)
    
    return probs

def estimate_theta(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all the words in vocab and the values are their estimated probabilities given
             the first level class name.
    """
    theta = {}
    tmp = {}
    count = 0
    for j in vocab:
        tmp[j]=1
        count+=1
    tcount=count
    unchanged = tmp

    labels = np.unique(np.array(train_labels))

    for i in labels:
        tmp = unchanged.copy()
        count=tcount
        for j in range(len(train_data)):
            if train_labels[j]==i:
                for k in train_data[j]:
                    tmp[k]+=1
                    count+=1
        for j in tmp:
            tmp[j]/=count

        theta[i]=tmp
    return theta

def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """
    scores = []
    for i in range(len(test_data)):
        scores.append([])
        for k in theta:
            tmp = 0
            for j in test_data[i]:
                if not j in vocab:
                    continue
                tmp += m.log(theta[k][j])
            tmp+=m.log(pi[k])
            scores[-1].append((tmp,k))

    return scores

def readtxt(path,type):
    blacklist = [".",",","#","@","?","(",")","-","*"]
    f = open(path,"r")
    lines = f.readlines()
    for i in range(len(lines)):
        tmp = ""
        lines[i] = lines[i].replace("\n","")
        for j in range(len(lines[i])):
            if lines[i][j] in blacklist:
                continue
            else:
                tmp+=lines[i][j]
            tmp.replace("\\n","")
        if type==0:
            lines[i]=tmp.split(" ")


    return lines

def get_accuracy(results,labels):
    count = 0
    for i in range(len(results)):
        res = max(results[i])[1]    
        if res == labels[i]:
            count+=1
    print("Test accuracy is:" , count/len(labels))
def main():
    train_data = readtxt('hw4_data/sentiment/train_data.txt',0)
    train_labels = readtxt('hw4_data/sentiment/train_labels.txt',1)
    test_data = readtxt('hw4_data/sentiment/test_data.txt',0)
    test_labels = readtxt('hw4_data/sentiment/test_labels.txt',1)

    vocab_train = vocabulary(train_data)
    vocab_test = vocabulary(test_data)

    pi_train = estimate_pi(train_labels) 
    pi_test = estimate_pi(test_labels) 

    theta_train = estimate_theta(train_data,train_labels,vocab_train)
    theta_test = estimate_theta(test_data,test_labels,vocab_test)

    results = test(theta_train, pi_train, vocab_train, test_data)
    get_accuracy(results,test_labels)

#main()