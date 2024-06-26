import numpy as np

#Function to create an identity matrix, that will be used in several other funcitons
def identity_matrix(n):
    id = np.zeros((n, n))
    for i in range(n):
        id[i, i] = 1
    
    return id

#Function to perform gaussian elimination of a matrix
def gaussian_elimination(matrix, vals):
    #matrix is the matrix of variable coefficients
    #vals is the matrix/vector of values
    
    aug_matrix = np.column_stack((matrix, vals)).astype(float)
    n = len(aug_matrix)
    
    #Iterate through the rows
    for i in range(n):
        #Make the pivot 1
        pivot = aug_matrix[i, i]
        aug_matrix[i] = aug_matrix[i] / pivot
        
        #Iterate through the columns
        for j in range(n):
            #Skip the pivot row
            if i == j:
                continue
            #Get the factor to make the element 0
            factor = aug_matrix[j, i]
            #Subtract the pivot row multiplied by the factor
            aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]
    
    answer = aug_matrix[:, n:]
    return answer

#Function to get the inverse of a matrix
def inverse(matrix):
    #Inverse of a matrix = AxA^-1
    n  = len(matrix)
    matrix = np.array(matrix).astype(float)
    id = np.array(identity_matrix(n)).astype(float)
    
    #Use gaussian elimination with matrix and identity matrix
    inverse = np.array(gaussian_elimination(matrix, id))
    return inverse


#Test gausian elimination function
a = np.array([[2, 1, -1], [1, -3, 1], [-3, 1, 1]])
vector = [0, 7, -5]
print("Test for gaussian elimination function: ")
print("Original a matrix",'\n', a, '\n')
print("Gaussian Elimination of a: ", '\n', gaussian_elimination(a, vector), '\n', '\n')

#Test for Inverse of matrix
b = np.array([[3, 6, 2], [5, 4, 8], [9, 7, 4]])
print("Test for inverse of matrix function: ")
print("Original b matrix: ", '\n', b, '\n')
print("Inverse of b: ", '\n', inverse(b))