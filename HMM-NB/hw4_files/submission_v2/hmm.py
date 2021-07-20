import numpy as np

def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    liste = []
    for i in range(A.shape[0]):
        liste.append([])
    for i in range(A.shape[0]):
        tmp = pi[i]*B[i][O[0]]
        liste[i].append(tmp)

    for i in range(1,O.shape[0]):
        for j in range(A.shape[0]):
            prob = 0
            for k in range(A.shape[1]):
                tmp=1
                tmp*=liste[k][i-1]
                tmp=tmp*A[k][j]*B[j][O[i]]
                prob+=tmp
            liste[j].append(prob)
    np_array = np.array(liste)
    return sum(np_array[:,-1]),np_array

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    liste = []
    index_list = []
    for i in range(A.shape[0]):
        liste.append([])
    for i in range(A.shape[0]):
        tmp = pi[i]*B[i][O[0]]
        liste[i].append(tmp)


    for i in range(1,O.shape[0]):
        for j in range(A.shape[0]):
            maximum = -1
            for k in range(A.shape[1]):
                tmp=1
                tmp*=liste[k][i-1]
                tmp=tmp*A[k][j]*B[j][O[i]]
                if tmp>maximum:
                    maximum=tmp
            liste[j].append(maximum)                
    np_array = np.array(liste)
    winning = np.argmax(liste,axis=0)[-1]

    way = []
    way.append(winning)
    for i in range(O.shape[0]-2,-1,-1):
        maximum=-1
        index=0
        for j in range(A.shape[0]):
            if liste[j][i]*A[j][winning]>maximum:
                index = j
                maximum = liste[j][i]*A[j][winning]
        way.append(index) 
    way.reverse()
    return np.array(way),np_array
