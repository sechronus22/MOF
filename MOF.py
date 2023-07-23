import numpy as np
from scipy.spatial import distance_matrix

def euclid_distance(p1,p2):
  return np.sqrt(np.sum(np.square(p1 - p2)))

def Neighborhood(arr):
    """
    Compare the pairwise distances between two elements and count the number

    of neighborhood (points that the distance is lower than or equal to the interesting point)

    including the interesting point

    Ex arr = [2,3,6,1,8] -> nbh = [2,3,4,1,5]

    Parameters
    ----------
    arr : numpy array
        a 1-dimensional array which each element are distance

    Returns
    -------
    nbh : numpy array
        an array which each element is the number of neighborhood of that point
    """
    arr_size = len(arr)
    n_arr = arr.reshape((1,arr_size))
    distance_diff = n_arr-n_arr.T
    nbh = np.sum(np.where(distance_diff>=0,1,0),axis=0)
    return nbh

def NBH_Matrix(data):
    """
    Compute the pairwise distances between two points and find the number of neighborhood

    of point q respect to point p for all pair (p,q)

    Ex arr = [[906 892]
              [870 323]
              [433 480]
              [602 695]
              [569 849]]

      Neighbor_matrix =  [[1 4 5 3 2]
                          [4 1 3 2 5]
                          [5 4 1 2 3]
                          [4 5 3 1 2]
                          [3 5 4 2 1]]

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    Neighbor_matrix : numpy array
        a matrix which element [i,j] represent the number of neighborhood of point i respect to point j
    """
    d_size = len(data)
    dist_matrix = distance_matrix(data,data)
    # dist_matrix = distance_matrix(data,data)
    # print(dist_matrix)
    Neighbor_matrix = np.ones((d_size,d_size))
    Neighbor_matrix = np.apply_along_axis(Neighborhood, 1, dist_matrix)
    return Neighbor_matrix

def MassRatio(data):
    """
    Compute the mass ratio of all pairwise data points

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    Neighbor_matrix : numpy array
        mass ratio of all pairwise data points
    """
    minor_NBH_matrix = NBH_Matrix(data)
    # print(minor_NBH_matrix)
    return minor_NBH_matrix/np.transpose(minor_NBH_matrix)

def MOF_p(pre_arr,i):
  arr = np.delete(pre_arr,i,axis=0)
  return np.var(arr)

def MOF(data):
    """
    Compute the mass ratio variance of all data points

    Parameters
    ----------
    data : numpy array
        a 2-dimensional array which each row is a data point

    Returns
    -------
    MRV_matrix : numpy array
        an array of mass ratio variance of all data points
    """
    MR_Matrix = MassRatio(data)
    d_size = len(data)
    MRV_Matrix = np.zeros(d_size)
    for i in range(d_size):
      MRV_Matrix[i] = MOF_p(MR_Matrix[:,i],i)

    return MRV_Matrix
