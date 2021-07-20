import numpy as np
import math as m
import matplotlib.pyplot as plt

def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    minDistance=-1
    index1=-1
    index2=-1
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            distance=0
            for k in range(c1.shape[1]):
                temp = c1[i][k]-c2[j][k]
                distance+=(temp*temp)
            distance = m.sqrt(distance)
            if distance<minDistance or minDistance==-1:
                minDistance=distance
                index1=i
                index2=j

    return minDistance

def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    maxDistance=-1
    index1=-1
    index2=-1
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            distance=0
            for k in range(c1.shape[1]):
                temp = c1[i][k]-c2[j][k]
                distance+=(temp*temp)
            distance = m.sqrt(distance)
            if distance>maxDistance:
                maxDistance=distance
                index1=i
                index2=j

    return maxDistance


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    totalDistance=0
    index1=-1
    index2=-1
    for i in range(c1.shape[0]):
        distance=0
        for j in range(c2.shape[0]):
            data=0
            for k in range(c1.shape[1]):
                temp = c1[i][k]-c2[j][k]
                data +=(temp*temp)
            distance += m.sqrt(data)
        totalDistance += (distance)
    totalDistance=totalDistance/(c1.shape[0]*c2.shape[0])
    return totalDistance

def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    totalDistance=0
    index1=-1
    index2=-1

    center_i = np.zeros((1,c1.shape[1]))
    center_j = np.zeros((1,c1.shape[1]))

    for i in range(c1.shape[0]):
        for k in range(c1.shape[1]):
            center_i[0][k]+=c1[i][k]
    center_i/=c1.shape[0]
    for j in range(c2.shape[0]):
        for k in range(c2.shape[1]):
            center_j[0][k]+=c2[j][k]
    center_j/=c2.shape[0]

    distance=0 
    for t in range(c1.shape[1]):
        temp = center_i[0][t]-center_j[0][t]
        distance+=(temp*temp)
    distance=m.sqrt(distance)
    return distance

def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    clusters = list(np.expand_dims(data.copy(),axis=1))
    while(len(clusters)>stop_length):
        minDistance=-1
        index1=-1
        index2=-1
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if(i==j):
                    continue
                distance = criterion(np.array(clusters[i]),np.array(clusters[j]))
                if distance<minDistance or minDistance==-1:
                    minDistance=distance
                    index1=i
                    index2=j
        #mask = np.ones((clusters.shape[0]),dtype=bool)
        #mask[index1]=False
        #mask[index2]=False
        new_element =  np.array(list(clusters[index1])+list(clusters[index2]))
        if index1>index2:
            del clusters[index1]
            del clusters[index2]
        else:
            del clusters[index2]
            del clusters[index1]

        clusters.append(new_element)
        #array = np.append(np.expand_dims(clusters[index1],axis=0),
        #    np.expand_dims(clusters[index2],axis=0),axis=1)
        #new_clusters = np.append(new_clusters,array,axis=0)
        #print(new_clusters.shape)
        #clusters=new_clusters
        #print(clusters.shape)
    return clusters

def main():

    data1 = np.load("hw2_data/hac/data1.npy")
    data2 = np.load("hw2_data/hac/data2.npy")
    data3 = np.load("hw2_data/hac/data3.npy")
    data4 = np.load("hw2_data/hac/data4.npy")

    kdata1=kdata2=kdata3=2
    kdata4=4

    kdatas = [2,2,2,4]

    criterions=[single_linkage,complete_linkage,average_linkage,centroid_linkage]
    criterion_names = ["single_linkage","complete_linkage","average_linkage","centroid_linkage"]

    data = [data1,data2,data3,data4]
    data_names = ["data1","data2","data3","data4"]
    #cluster1 = hac(data1, single_linkage, kdata1)
    #colors = ["red","black","blue"]
    #for i in range(len(cluster1)):
    #    for j in range(len(cluster1[i])):
    #        plt.plot(cluster1[i][j][0],cluster1[i][j][1],'o',color=colors[i],markersize=1)
    #plt.show()
    colors = ["red","black","blue","purple"]
    
    for i in range(len(data)):
        for j in range(len(criterions)):
            cluster = hac(data[i], criterions[j], kdatas[i])
            for k in range(len(cluster)):
                for t in range(len(cluster[k])):
                    plt.plot(cluster[k][t][0],cluster[k][t][1],'o',color=colors[k],markersize=1)
            plt.savefig(data_names[i]+"_"+criterion_names[j]+'.png')
            plt.clf()


#main()