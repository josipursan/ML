This document will provide an explanation of how the first "full" version of k-means works.  
Why is this necessary?  
Because it is bit of a mess, and before I move on to cleaning it up I'd like as much clarity on each line of code.  
  
Reference commit : https://github.com/josipursan/ML/commit/3540e172baa6bbcfa0b4b579ccf3e04cf76a1c7e    
  
```python
'''
Assume all coordinates in [x,y] format.
_______________________
K - number of clusters

1. randomly initialize K cluster centroids
    -option_1 : randomly choose K points from the dataset and make them the initial cluster centroids
    -option_2 : generate K random points and make them the cluster centroidsđ

2. Assign dataset points to the cluster centroids
    -how do we assign them?
        Assume K = 2
        Value 0 represents cluster_A
        Value 1 represents cluster_B

        pseudocode implementation : 
        
            cluster_centroid_assignments = []   # list positionally paired to indices of points in dataset
            
            # Assign points to cluster centroids
            for datapoint in dataset:
                if (L2_norm(datapoint - cluster_A)) < (L2_norm(datapoint - cluster_B)) : 
                    cluster_centroid_assignments.append(0)  #ie. datapoint is closer to cluster_A
                else:
                    cluster_centroid_assignments.append(1)  #ie. datapoint is closer to cluster_B
                
            # Move the cluster centroids
            #list containing indices only of datapoints assigned to cluster_A
            list_indices_datapoints_cluster_A = [index for index, value in enumerate(cluster_centroid_assignments) if value == 0]
            
            #list containing indices only of datapoints assigned to cluster_B
            list_indices_datapoints_cluster_B = [index for index, value in enumerate(cluster_centroid_assignments) if value == 1]

            # Reposition the cluster centroid by computing the average position of the datapoints previously assigned to it
            cluster_A_centroid = mean(list_indices_datapoints_cluster_A)
            cluster_B_centroid = mean(list_indices_datapoints_cluster_B)
'''

import random
import math
import numpy as np
import matplotlib.pyplot as plt

def generate_random_x_y(number_of_datapoints):
    x = []
    y = []

    for i in range(number_of_datapoints):
        x.append(random.randint(0, 101))
        y.append(random.randint(0, 101))

    return x, y

def L2_norm(x, y, given_centroid):
    X_movement = abs(x - given_centroid[0])
    Y_movement = abs(y - given_centroid[1])
    distance = math.sqrt(X_movement**2 + Y_movement**2)

    #print("x : {}\ty : {}\ngiven_centroid : {}\nX_movement : {}\nY_movement : {}\ndistance : {}\n".format(x, y, given_centroid, X_movement, Y_movement, distance))

    return distance

def main():
    # Number of cluster centroids
    K = 2
    
    #Number of datapoints
    number_of_datapoints = 100
    K_centroid_coords = []
    x, y = generate_random_x_y(number_of_datapoints)
    print("x : {}\ny : {}\n".format(x, y))

    xy = np.vstack((x,y)).T
    x_np_array = np.array(x)
    y_np_array = np.array(y)
    print("Shapes\nx_np_array : {}\ny_np_array : {}".format(x_np_array.shape, y_np_array.shape))

    # Randomly initialize cluster centroids - option_2 (look at the top)
    for i in range(K):
        cluster_centroid = []
        cluster_centroid.append(random.randint(0, 101))
        cluster_centroid.append(random.randint(0, 101))

        K_centroid_coords.append(cluster_centroid)
    print(K_centroid_coords)

    #This list contains distances for all datapoints to all cluster centroids - it will be composed of K sublists : one for each cluster centroid
    # For each centroid we compute distance to each point
    for iteration in range(10):
        print("ITERATION : {}".format(iteration))
        cluster_centroid_assignments = []
        all_distances = []
        for centroid in range(K):
            current_centroid_distances = []
            for i in range(number_of_datapoints):
                current_centroid_distances.append(L2_norm(x[i], y[i], K_centroid_coords[centroid]))
            all_distances.append(current_centroid_distances)

        all_distances_array = np.array(all_distances)
        
        # This code block is used to determine which datapoints are closest to which cluster centroids. It uses all_distances matrix, traversing column per column, and checking which sublist has the smallest value
        # for the current column.
        for col in range(all_distances_array.shape[1]):
            #min_value = np.min(all_distances_array[:, col]) # How to find the smallest value in a column
            cluster_centroid_assignments.append(np.argmin(all_distances_array[:, col]))    # How to get the row index of the smallest value in column
        
        print("cluster_centroid_assignments : {}\n".format(cluster_centroid_assignments))

        per_cluster_datapoint_indices = []  # list made up of K sublists, each containing indices, in x and y, for the points assigned to the K-th centroid
        for centroid in range(K):
            per_cluster_datapoint_indices.append([index for index,value in enumerate(cluster_centroid_assignments) if value == centroid])
        #centroid_A_datapoint_indices = [index for index,value in enumerate(cluster_centroid_assignments) if value == 0]
        #centroid_A_datapoint_indices = [index for index,value in enumerate(cluster_centroid_assignments) if value == 1]

        print("per_cluster_datapoint_indices : \n{}\n".format(per_cluster_datapoint_indices))
        for centroid in range(K):
            print("centroid : {}\n".format(centroid))
            print("Centroid {} (len : {}) : \n{}\n".format(centroid, len(per_cluster_datapoint_indices[centroid]), per_cluster_datapoint_indices[centroid]))
            print("Centroid {}\nX datapoints : \n{}\nY datapoints : \n{}\n".format(centroid, x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]]))
            current_centroid_datapoints = np.vstack((x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]])).T
            print("\n current_centroid_datapoints : \n{}\n".format(current_centroid_datapoints))
            print("New mean : {}\n".format( np.mean(current_centroid_datapoints, axis = 0) ))
            print("BEFORE K_centroid_coords : \n{}\n".format(K_centroid_coords))
            K_centroid_coords[centroid] = list(np.mean(current_centroid_datapoints, axis = 0))
            print("AFTER K_centroid_coords : \n{}\n".format(K_centroid_coords))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for centroid in range(K):
            if centroid == 0:   #If first centroid, use color blue | Dumb hard encoding just for the sake of this example
                ax1.scatter(x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]], c = 'w', edgecolors = 'b', label = 'dp_centroid_' + str(centroid))
                ax1.scatter(K_centroid_coords[centroid][0], K_centroid_coords[centroid][1], c = 'b', label = 'CENTROID_' + str(centroid), linewidths = 5) # Plotting the centroid_0
            else : 
                ax1.scatter(x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]], c = 'w', edgecolors = 'r', label = 'centroid_' + str(centroid))
                ax1.scatter(K_centroid_coords[centroid][0], K_centroid_coords[centroid][1], c = 'r', label = 'CENTROID_' + str(centroid), linewidths = 5) # Plotting the centroid_1

        ax1.set_title("k-means iteration " + str(iteration))
        plt.xlabel("X coord")
        plt.ylabel("Y coord")
        plt.legend(loc = 'upper left')
        plt.savefig("iteration_" + str(iteration) + ".png")
        print("\n=========================================================\n\n")
                    
main()
```  
  
All relevant chunks will be copied from above when explaining them.  
  
## Chunk 1 - the setup
```python
K = 2
number_of_datapoints = 100
K_centroid_coords = []
x, y = generate_random_x_y(number_of_datapoints)
print("x : {}\ny : {}\n".format(x, y))
```  
*K* represents how many clusters, consequently cluster centroids, we want to create for our dataset.  
  
`K_centroid_coords` is a 2D list.  
It has as many sublists as there is cluster centroid.  
E.g. : if there are 3 cluster centroids (K = 3), *K_centroid_coords* will have 3 sublists.  
Each sublist is made up of two values, one representing the x coordinate, and the other representing the y coordinate.  
  
`x` and `y` are lists.  
They are row vectors, their dimension equaling variable *number_of_datapoints*.  
  
## Chunk 2 - cluster centroid starting positions
```python
# Randomly initialize cluster centroids - option_2 (look at the top)
for i in range(K):
    cluster_centroid = []
    cluster_centroid.append(random.randint(0, 101))
    cluster_centroid.append(random.randint(0, 101))

    K_centroid_coords.append(cluster_centroid)
```  
To start the k-means algorithm, *K* cluster centroids needs to be initialized to some starting positions.  
A common way is to choose *K* datapoints from the dataset and make them the starting points.  
Here I've chosen to randomly generate coordinates for the *K* cluster centroids.  
Notice that, as explained in section **Chunk 1**, x and y coordinates for each cluster centroid are grouped into their own, 1D, list, which is then appended to the `K_centroid_coords` matrix.  
  
## Chunk 3 - k_means algorithm iteration  
```python
for iteration in range(10):
    ...
```  
You will notice that the `compute and update` section of the algorithm is in a non-infinite for loop.  
This is only for testing purposes, and will be later changed.  
In order to make this work like it should, each iteration **i** needs to look at iteration **i-1** whether there were any changes in datapoint assignments to cluster centroids, or if the cluster centroid position changed after computing its new position - if there wasn't, algorithm has converged.  

  
## Chunk 4 - computing distances to all centroids
```python
cluster_centroid_assignments = []
all_distances = []
for centroid in range(K):
    current_centroid_distances = []
    for i in range(number_of_datapoints):
        current_centroid_distances.append(L2_norm(x[i], y[i], K_centroid_coords[centroid]))
    all_distances.append(current_centroid_distances)

all_distances_array = np.array(all_distances)
```  
`all_distances` is a list that holds distances for all datapoints to all cluster centroids.  
It has **K** sublists - each sublist holds distances for each datapoint to the **K-th** cluster centroid.  
  
`cluster_centroid_assignments` is a row vector whose value found at index position **i** represents to which cluster centroid datapoint found at index position **i** belongs.  
Its dimension matches `number_of_datapoints`.  
Why?  
Because it is positionally mapped to the datapoints by index.  
e.g. :  
cluster_centroid_assignments[5] = 1 --> datapoint at index **5** belong to cluster **1**  
Using this index we can easily grab this datapoint, datapoint **5**, that we now know belongs to **cluster** 1.  
  
The nested `for` loops are pretty self-explanatory.  
The outter loop iterates over all cluster centroids.  
`current_centroid_distances = []` is an intermediary list used to store distances for each cluster centroid.  
The inner loop is used to iterate over all datapoints, passing one point at a time to `L2_norm` function, computing distance from each datapoint to the cluster centroid selected by the current iteration of the outter loop (denoted by *centroid* variable).  
Once the inner loop has iterated over all datapoints, the intermediary list is appended to the `all_distances` list :  
```python
all_distances.append(current_centroid_distances)
```  
<br><br/>
Last step is transforming `all_distances` to an numpy array :  
```python
all_distances_array = np.array(all_distances)
``` 
Why?  
Take a look at **Chunk 5**.
<br><br/>  
  
## Chunk 5 - assigning datapoints to centroids
```python
for col in range(all_distances_array.shape[1]):
    cluster_centroid_assignments.append(np.argmin(all_distances_array[:, col])
```  
`cluster_centroid_assignments` list was already exaplined in **Chunk 4**.  
  
`all_distances` is a matrix - list of lists.  
First element of each sublist represents the distance of the first datapoint to that sublist's centroid, second element of each sublist represents the distance of the second datapoint to that sublist's centroid, etc.  
This means if we take the first column of `all_distances` values in this column-vector are showing only distances of the first datapoint from the dataset to each centroid.  
By finding what the smallest value is in this column-vector, and determining which row it stems from (row represents each centroid), we can quickly determine to which centroids our datapoints belong.  
All of this is done by this line in the `for` loop :  
```python
cluster_centroid_assignments.append(np.argmin(all_distances_array[:, col])
```  
This is exactly why we change `all_distances` from a python 2D list to a numpy array in **Chunk 4** - `np.argmin()` allows us to easily get index of the row where the smallest value was found in the column-vector.  
<br></br>
Example of how `cluster_centroid_assignments` will look like :  
```terminal
cluster_centroid_assignments : [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
```  
The above shown example is for **K = 2**, hence why `cluster_centroid_assignments` has only values **0** and **1**.  
## Chunk 6 - sorting datapoints to cluster centroids
```python
per_cluster_datapoint_indices = []  
for centroid in range(K):
    per_cluster_datapoint_indices.append([index for index,value in enumerate(cluster_centroid_assignments) if value == centroid])
```  
  
`per_cluster_datapoint_indices` is a list made up of **K** sublists, each containing indices, from lists holding *x* and *y* coordinates, for the points assigned to the *K*-th centroid.  
Indices found in this matrix correspond to the indices of points assigned to each cluster.  
  
A nice list comprehension with enumeration is used in the for loop to sort which datapoints belong to which cluster centroid.  
`cluster_centroid_assignments` is used as the input for this sorting operation as it is a row-vector whose indices represent datapoint indices, and the value at each index represents datapoint's assignment to specific cluster centroid.  

## Chunk 7 - computing new positions of all cluster centroids
```python
for centroid in range(K):
    current_centroid_datapoints = np.vstack((x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]])).T
    K_centroid_coords[centroid] = list(np.mean(current_centroid_datapoints, axis = 0))
```  
Almost everything up to this point was list/array handling to filter out which datapoints are closest to each centroid.  
Our datapoints, before any actions in **Chunk 7** are done, are in two separate python lists, one representing x coordinates, and the other representing y coordinates.  
Together these two numpy lists, that are positionally (index) matched, represent all datapoints.  
Here is an example, during one of the runs, what the output looks like when we print `x_np_array` and `y_np_array` when we print them by passing them `per_cluster_datapoint_indicex[centroid]` (*note that*`per_cluster_datapoint_indicex[centroid]`*is used to easily get all corresponding datapoints for each centroid*) :  
```terminal
Centroid 0
X datapoints :
[ 18  53  25  70  52  72  87  44  32  99  72  72  29  68  64  75  21  46
   2  88  95  98  90  98  40  32  93  98  72  25  98   4  59  55  87  44
  99  80  88  23  73  67  69  92  74  79  74 101  60  63  57  36  38  49]
Y datapoints :
[ 99  69  89  84  57 100  74  89  88  52  76  86  82  84  44  41  99  62
  87  93 100  32  56  48  78  71  79  27  70  95  67  96  84  94  50  65
  68  86  88  71  56  69  88  69  49  37  94  35  49  47  73  80 100  69]
```  
  
To compute the new position of `Centroid 0`, we need to compute the mean position of `Centroid 0` after the latest `update` step, ie. after computing distances to all datapoints, and assigning those closest.  
We can either do this manually, or leverage some library, such as `numpy` in this case.  
  
We will take our 1D arrays `x_np_array` and `y_np_array`, and stack them vertically in our new numpy array `current_centroid_datapoints`, and then transpose this matrix so that we end up with two columns.  
One column represents the x coordinate (this is orignally `x_np_array`), while the other column represents the y coordinate (this is originally `y_np_array`).  
Each row in `current_centroid_datapoints` represents an individual datapoint.  
Example of `current_centroid_datapoints` :  
```terminal
 current_centroid_datapoints :
[[ 18  99]
 [ 53  69]
 [ 25  89]
 [ 70  84]
 [ 52  57]
 [ 72 100]
 [ 87  74]
 [ 44  89]
 [ 32  88]
 ...
 ...
  [ 36  80]
 [ 38 100]
 [ 49  69]]
```  
All of this is done using this line of code :  
```python
current_centroid_datapoints = np.vstack((x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]])).T
```  
<br></br>
Lastly, computing new position for the current centroid is now a piece of cake :  
```python
K_centroid_coords[centroid] = list(np.mean(current_centroid_datapoints, axis = 0))
```  