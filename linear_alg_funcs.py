import numpy as np

#Function to create an identity matrix, that will be used in several other funcitons
def identity_matrix(n):
    id = np.zeros((n, n))
    for i in range(n):
        id[i, i] = 1
    
    return id

def gaussian_elimination(arr, vals):
    #arr is the matrix of variable coefficients
    #vals is the vector of values
    
    aug_matrix = np.column_stack((arr, vals))
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
    
    answer = aug_matrix[:, n].astype(int)
    return answer
    

#Test gausian elimination, must use floats so that division is not integer division
a = [[2.0, 1.0, -1.0], [1.0, -3.0, 1.0], [-3.0, 1.0, 1.0]]
vector = [0, 7, -5]
print(a)
print(gaussian_elimination(a, vector))

#