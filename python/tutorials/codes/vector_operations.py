# Import the usual packages
import numpy as np


# Define a function to calculate the dot product of two vectors
def dot_product(x, y):

    # If the vectors are not the same length, return an error
    if len(x) != len(y):
        raise ValueError('Vectors must be the same length')

    # Calculate the dot product of two vectors
    return np.sum(x * y)


# Define a function to find the angle between two vectors
def angle_between(x, y):

    # If the vectors are not the same length, return an error
    if len(x) != len(y):
        raise ValueError('Vectors must be the same length')

    # Calculate the angle between two vectors
    return np.arccos(dot_product(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


a = np.array([1, 2, 3, 4])
b = np.array([4, 5, 6])
# c = dot_product(a, b)
# print(c)

theta = angle_between(a, b)
print(np.round(theta * 180 / np.pi, 3))
