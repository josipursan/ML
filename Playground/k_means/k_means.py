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
import random
import sys
import time

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
    # Generating K random colors for plotting k-means iterations
    colors = []
    for i in range(K):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    
    #Number of datapoints
    number_of_datapoints = 100

    K_centroid_coords = []
    x, y = generate_random_x_y(number_of_datapoints)
    print("x : {}\ny : {}\n".format(x, y))

    xy = np.vstack((x,y)).T
    x_np_array = np.array(x)
    y_np_array = np.array(y)
    print("Shapes\nx_np_array : {}\ny_np_array : {}".format(x_np_array.shape, y_np_array.shape))

    previous_iteration_cluster_assignments = []

    # Randomly initialize cluster centroids - option_2 (look at the top)
    for i in range(K):
        cluster_centroid = []
        cluster_centroid.append(random.randint(0, 101))
        cluster_centroid.append(random.randint(0, 101))

        K_centroid_coords.append(cluster_centroid)
    print(K_centroid_coords)

    #This list contains distances for all datapoints to all cluster centroids - it will be composed of K sublists : one for each cluster centroid
    # For each centroid we compute distance to each point
    iteration = 0
    while True:
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
        if (len(previous_iteration_cluster_assignments) == 0):
            previous_iteration_cluster_assignments = cluster_centroid_assignments
        else :
            if(previous_iteration_cluster_assignments == cluster_centroid_assignments):
                print("\nNo assignment changes.\nConverged.\nExiting.\n")
                break
            else:
                previous_iteration_cluster_assignments = cluster_centroid_assignments

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
            ax1.scatter(x_np_array[per_cluster_datapoint_indices[centroid]], y_np_array[per_cluster_datapoint_indices[centroid]], c = 'w', edgecolors = colors[centroid], label = 'dp_centroid_' + str(centroid))
            ax1.scatter(K_centroid_coords[centroid][0], K_centroid_coords[centroid][1], c = colors[centroid], label = 'CENTROID_' + str(centroid), linewidths = 5) # Plotting the centroid_0

        ax1.set_title("k-means iteration " + str(iteration))
        plt.xlabel("X coord")
        plt.ylabel("Y coord")
        plt.legend(loc = 'upper left')
        plt.savefig("iteration_" + str(iteration) + ".png")
        print("\n=========================================================\n\n")
        iteration += 1
                    
main()