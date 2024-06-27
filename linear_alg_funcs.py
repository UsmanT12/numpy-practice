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
    
    #Return only the answer matrix/vector
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

def vector_projection(a, b):
    #a is the vector that is projected on vector b
    a = np.array(a)
    b = np.array(b)
    return a.dot(b) / b.dot(b) * b


#Test for identity matrix function
print("Test for identity matrix function: ")
n = int(input("Enter the size of the identity matrix: "))
print("Identity matrix of size ", n, " is: ", '\n', identity_matrix(n), '\n', '\n')

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

#Test for vector projection function
c = np.array([2, 3, 6])
d = np.array([7, 6, 8])
print("Test for vector projection function: ", '\n', vector_projection(c, d), '\n', '\n')


#Test for Eigenvalues and Eigenvectors Functions in numpy
e = np.array([[5, -3], [1, 1]])
val, vector = np.linalg.eig(e)
print("Eigenvalues: ", val, '\n', "Eigenvectors: ", '\n', vector, '\n', '\n')

#Test for linear algebra solve function in numpy
f = np.array([[2, 4, 5], [7, 2, 8], [9, 5, 6]])
f_vals = np.array([8, 4, 3]).astype(float)
print("Test for linear algebra solve function in numpy: ", '\n', np.linalg.solve(f, f_vals), '\n', '\n')
