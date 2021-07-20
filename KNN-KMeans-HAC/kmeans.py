import numpy as np
import math
import matplotlib.pyplot as plt

def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    value = np.empty((data.shape[0],))
    for i in range(data.shape[0]):
        minDistance = -1
        ith_cluster=0
        for k in range(cluster_centers.shape[0]):
            eucDistance = 0
            for j in range(data.shape[1]):
                distance = cluster_centers[k][j]-data[i][j]
                eucDistance += (distance*distance)
            eucDistance = math.sqrt(eucDistance)
            if eucDistance<minDistance or minDistance==-1:
                minDistance=eucDistance
                ith_cluster=k
        value[i]=ith_cluster
    return value

def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    new_cluster_centers = np.zeros((cluster_centers.shape))
    n_data = np.zeros((cluster_centers.shape[0]))
    for i in range(data.shape[0]):
        n_data[int(assignments[i])]+=1
        for j in range(data.shape[1]):
            new_cluster_centers[int(assignments[i])][j]+=data[i][j]

    for i in range(new_cluster_centers.shape[0]):
        if n_data[i]!=0:
            new_cluster_centers[i]/=n_data[i]
        else:
            new_cluster_centers[i]=cluster_centers[i]

    return new_cluster_centers

def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    old_objective = -1
    assignments = assign_clusters(data, initial_cluster_centers)
    cluster_centers = calculate_cluster_centers(data, assignments, initial_cluster_centers, initial_cluster_centers.shape[0])
    while(True):
        objective_func = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                distance = cluster_centers[int(assignments[i])][j]-data[i][j]
                objective_func += (distance*distance)
        if(old_objective==objective_func):
            break
        old_objective = objective_func
        
        assignments = assign_clusters(data, cluster_centers)
        cluster_centers = calculate_cluster_centers(data, assignments, cluster_centers, cluster_centers.shape[0])
    return cluster_centers, objective_func

def initialize_clusters(data,k):
    clusters = []
    clusters.append(data[np.random.randint(data.shape[0]),:])
    for index in range(0,k-1):
        maxPdistance=-1
        p_index = -1
        c_index=-1
        for i in range(data.shape[0]):
            minDistance=-1
            for j in range(len(clusters)):
                distance=0
                for k in range(data.shape[1]):
                    temp=data[i][k]-clusters[j][k]
                    distance += (temp*temp)
                distance = math.sqrt(distance)
                if distance<minDistance or minDistance==-1:
                    minDistance=distance
                    c_index = j
            if minDistance>maxPdistance:
                p_index=i
                maxPdistance = minDistance
        clusters.append(data[p_index])
    return np.array(clusters)

def color_cluster(ith,data,ktimes):
    initial_clusters = initialize_clusters(data,ktimes)
    cluster_centers,objective_func = kmeans(data,initial_clusters)
    assignments = assign_clusters(data, cluster_centers)
    colors = ["red","black","blue","orange","purple","brown","pink","cyan","olive"]

    for i in range(data.shape[0]):
        plt.plot(data[i,0],data[i,1],'o',color=colors[int(assignments[i])],markersize=1)
    for i in range(cluster_centers.shape[0]):
        plt.plot(cluster_centers[i,0],cluster_centers[i,1],'^',color="olive",markersize=20)
    plt.savefig(str(ith)+'th_kmeans_cluster.png')    
    plt.clf()

def main(ntimes):
    clustering1 = np.load("hw2_data/kmeans/clustering1.npy")
    clustering2 = np.load("hw2_data/kmeans/clustering2.npy")
    clustering3 = np.load("hw2_data/kmeans/clustering3.npy")
    clustering4 = np.load("hw2_data/kmeans/clustering4.npy")

    clusters = [clustering1,clustering2,clustering3,clustering4]
    
    for i in range(len(clusters)):
        results = []
        ks = []
        for k in range(1,11):
            minFunc = -1
            for j in range(ntimes):
                initial_clusters = initialize_clusters(clusters[i],k)
                cluster_centers,objective_func = kmeans(clusters[i],initial_clusters)    
                assignments = assign_clusters(clusters[i], cluster_centers)
                if objective_func<minFunc or minFunc==-1:
                    minFunc = objective_func
            results.append(minFunc)
            ks.append(k)
        plt.plot(ks, results)
        plt.savefig(str(int(i+1))+'th_kmeans_graph.png')
        plt.clf()
        #colors = ["red","black","blue"]

        #for i in range(clustering2.shape[0]):
        #    plt.plot(clustering2[i,0],clustering2[i,1],'o',color=colors[int(assignments2[i])],markersize=1)
        #plt.show()

def plots():
    data1 = np.load("hw2_data/kmeans/clustering1.npy")
    data2 = np.load("hw2_data/kmeans/clustering2.npy")
    data3 = np.load("hw2_data/kmeans/clustering3.npy")
    data4 = np.load("hw2_data/kmeans/clustering4.npy")
    color_cluster(0,data1,2)
    color_cluster(1,data2,3)
    color_cluster(2,data3,4)
    color_cluster(3,data4,5)    

#main(10)
#plots()
