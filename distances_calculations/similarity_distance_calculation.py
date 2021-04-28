# Manhattan distance
"""
Manhattan distance is a metric in which the distance between two points is the sum of the absolute differences of their Cartesian coordinates.
In a simple way of saying it is the total sum of the difference between the x-coordinates and y-coordinates.
"""
from scipy.spatial.distance import cityblock
dataset_man1 = [3, 4, 7, 2]
dataset_euc2 = [2, 54, 13, 15]
man_distance = cityblock(dataset_man1, dataset_euc2)
print(man_distance)


# Euclidean distance
"""
The Euclidean distance between two points in either the plane or 3-dimensional space measures the length of a segment connecting the two points. 
It is the most obvious way of representing distance between two points.
"""
from scipy.spatial.distance import euclidean
dataset_euc1 = [3, 4, 7, 2]
dataset_euc2 = [2, 54, 13, 15]
euc_distance = euclidean(dataset_euc2, dataset_euc1)
print(euc_distance)

# Hamming distance
"""
Given two vectors u,v∈ Fn we define the hamming distance between u and v,d(u,v), to be the number of places where u and v differ
"""
from scipy.spatial.distance import hamming
dataset_ham1 = [101101010]
dataset_ham2 = [111011011]
hamming_distance = hamming(dataset_ham1, dataset_ham2)
print(hamming_distance)
