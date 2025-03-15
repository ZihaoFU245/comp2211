#import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

def calculate_rank(abundance_matrix):
    """
    Calculate the rank of protein abundance for each cell.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape, containing the rank of protein abundance
    for each cell, where the lowest abundance is ranked 0.
    """
    sorted_ = np.argsort(abundance_matrix , axis=1)
    return sorted_

def calculate_mean(abundance_matrix):
    """
    Calculate the mean abundance of each protein across all cells.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 1D numpy array of shape (num_proteins,) containing the mean abundance.
    """
    sorted_ = np.sort(abundance_matrix , axis=1)
    mean = np.mean(sorted_ , axis=0)
    return mean

def substitute_mean(abundance_matrix, mean_values, rank_matrix):    # Fail
    """
    Substitute each value in the abundance matrix with the corresponding mean value based on ranks.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.
    mean_values: A 1D numpy array of shape (num_proteins,) containing the mean abundance.
    rank_matrix: A 2D numpy array of shape (num_cells, num_proteins) representing the ranks
                 of protein abundances in each cell.

    Returns:
    A 2D numpy array of shape (num_cells, num_proteins) where each value has been
    substituted by the corresponding mean value based on the rank.
    """
    substituded = mean_values[rank_matrix]
    return substituded

def quantile_normalization(abundance_matrix):
    """
    Perform quantile normalization on a protein abundance matrix.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape where each value has been substituted
    by the mean value of its corresponding rank across all cells.
    """
    ranking = calculate_rank(abundance_matrix)
    mean = calculate_mean(abundance_matrix)
    normalized = substitute_mean(abundance_matrix , mean , ranking)
    return normalized

def z_score_normalization(abundance_matrix):
    """
    Perform Z-score normalization on a protein abundance matrix.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape where each value has been normalized
    using the Z-score formula: (X - mean) / std.
    """
    mean = np.mean(abundance_matrix , axis=0)
    std = np.std(abundance_matrix , axis=0 )
    normalized = (abundance_matrix - mean) / std
    return normalized

def preprocess_datasets(abundance_matrix):
    """
    Preprocess the protein abundance matrix by applying quantile normalization followed by Z-score normalization.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.

    Returns:
    A 2D numpy array of the same shape, representing the processed dataset after
    quantile normalization and Z-score normalization.
    """
    quantile_normalized = quantile_normalization(abundance_matrix)
    Z_normalized = z_score_normalization(quantile_normalized)
    return Z_normalized

def label_to_integer(label):
    """
    Convert string labels to integer labels.

    You may consider using np.where
    (https://numpy.org/doc/stable/reference/generated/numpy.where.html)

    Parameters:
    labels: A 1D numpy array of shape (num_cells,) containing string labels for each cell.

    Returns:
    A 1D numpy array of the same shape, where each string label has been converted
    to an integer: "Normal" -> 0, "CancerA" -> 1, "CancerB" -> 2.
    """
    mapped_label = np.where(label == "Normal" , 0 , np.where(label == "CancerA" , 1 , 2))
    return mapped_label

def PCA_and_visualization(abundance_matrix, label):
    """
    Perform PCA on the protein abundance matrix and visualize the results in a 2D scatter plot.

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins)
                      representing protein abundance in cells.
    labels: A 1D numpy array of shape (num_cells,) containing integer labels for each cell.

    Returns:
    x: The x-coordinates of the points in the scatter plot.
    y: The y-coordinates of the points in the scatter plot.
    colors: A list of colors corresponding to each label for visualization.
    component_number: The number of components you have kept.
    """
    """    
    Specify the number of components you would like to keep.
    Hint: We would need to visualize our dataset in a 2D scatter plot.
    Hint: You may try different numbers of component number and print out the result.
    """

    """
    You need to understand the meaning of input parameters x, y and c into function plt.scatter().
    Hint: You may check document as well as usage at https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    and https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py.
    Hint: You do not need to care about inputs other than x,y and c.
    Hint: You have obtained an array principal_component of shape (number of cells, component_number), and you would like to visualize it via a 2D scatter plot.
    The 2 values in each row of array principal_component represent the coordinate of a point.
    You may consider using np.where (https://numpy.org/doc/stable/reference/generated/numpy.where.html)
    """
    component_number = 2 

    pca = PCA(n_components=component_number, svd_solver="arpack", random_state=2)
    principal_component = pca.fit_transform(abundance_matrix)

    X = principal_component[: , 0]
    y = principal_component[: , 1]

    color_list = ["r", "b", "g", "c"]

    colors = np.array(color_list)[label.astype(int)]

    return X , y , colors , component_number

def visualize_processed_datasets(X, label):
    x, y, colors, _ = PCA_and_visualization(X, label)
    #plt.scatter(x=x, y=y, c=colors)
    #plt.show()

def calculate_manhattan_distance(feature_train, feature_test):
    """
    Calculate the Manhattan distance between training and testing feature matrices.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.abs https://numpy.org/doc/stable/reference/generated/numpy.abs.html

    Parameters:
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins)
                   representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins)
                  representing protein abundance in the testing set.

    Returns:
    A 2D numpy array of shape (num_cells_test, num_cells_train) representing the
    Manhattan distance between each test cell and each train cell.
    """

    X_TRAIN_3D = feature_train[np.newaxis , : ,:]
    X_TEST_3D = feature_test[: , np.newaxis , :]
    dist = np.sum(np.abs(X_TEST_3D - X_TRAIN_3D) , axis=2)

    return dist

def calculate_euclidean_distance(feature_train, feature_test):
    """
    Calculate the Euclidean distance between training and testing feature matrices.

    Parameters:
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins)
                   representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins)
                  representing protein abundance in the testing set.

    Returns:
    A 2D numpy array of shape (num_cells_test, num_cells_train) representing the
    Euclidean distance between each test cell and each train cell.
    """
    X_TRAIN_3D = feature_train[np.newaxis , : ,:]
    X_TEST_3D = feature_test[: , np.newaxis , :]
    dist = np.sqrt(np.sum(np.square(X_TEST_3D - X_TRAIN_3D) , axis=2))

    return dist

def choose_nearest_neighbors(k, distance_metric, feature_train, feature_test, labels):
    """
    Choose the k nearest neighbors for each test cell based on the specified distance metric.

    Parameters:
    k: The number of nearest neighbors (integer).
    distance_metric: A string that can be either 'manhattan' or 'euclidean' indicating which distance metric to be used.
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins) representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins) representing protein abundance in the testing set.
    labels: A 1D numpy array of shape (num_cells_train,) containing labels of each cell in the training set.

    Returns:
    distance_k: A 2D numpy array of shape (num_cells_test, k) containing distances to the k nearest neighbors.
    top_k_labels: A 2D numpy array of shape (num_cells_test, k) containing labels of the k nearest neighbors.
    """
    labels = np.array(labels)   # ensure it is a numpy array

    if distance_metric == "manhattan":
        dist = calculate_manhattan_distance(feature_train , feature_test)   #(n_cell_test, n_cell_train)
    elif distance_metric == "euclidean":
        dist = calculate_euclidean_distance(feature_train , feature_test)
    else:
        raise ValueError(f"Unsupported metric type: {distance_metric}")
    
    sorted_indices = np.argsort(dist , axis=1)
    index_k = sorted_indices[: , :k]

    distance_k = np.take_along_axis(dist , index_k , axis=1)
    top_k_labels = labels[index_k]

    return distance_k , top_k_labels

def count_neighbor_class(top_k_labels):
    """
    Count the number of neighbors of each class among the k nearest neighbors for each test cell.

    Parameters:
    top_k_labels: A 2D numpy array of shape (num_cells_test, k) containing labels of the k nearest neighbors.

    Returns:
    class_count: A 2D numpy array of shape (num_cells_test, num_classes) representing the count of each class
                 among the k nearest neighbors for each test cell.
    """
    max_ = np.max(top_k_labels)
    identity_matrix = np.eye(max_ + 1)
    count = identity_matrix[top_k_labels]
    class_count = np.sum(count , axis=1)

    return class_count

def predict_labels(class_count):
    """
    Predict the label for each test cell based on the class counts of the k nearest neighbors.

    Parameters:
    class_count: A 2D numpy array of shape (num_cells_test, num_classes) representing the number of
                 data points belonging to each class among the k nearest neighbors.

    Returns:
    predicted_labels: A 1D numpy array of shape (num_cells_test,) containing the predicted label
                      for each test cell.
    """
    pred_label = np.argmax(class_count , axis=1)

    return pred_label

def get_max_voter(class_count):
    """
    Determine which classes are the max voters among the k nearest neighbors for each test cell.

    Parameters:
    class_count: A 2D numpy array of shape (num_cells_test, num_classes) representing the number of
                 data points belonging to each class among the k nearest neighbors.

    Returns:
    max_voter: A 2D numpy array of shape (num_cells_test, num_classes) where max_voter[j][i] = 1
               if class i is a max voter for test point j, otherwise 0.
    """
    max_ = np.max(class_count , axis=1)
    MASK = (class_count == max_[: , None])
    max_voter = MASK.astype(int)

    return max_voter

def get_useful_labels(max_voter, top_k_labels):
    """
    Determine useful labels based on max voters and the labels of k nearest neighbors.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.max https://numpy.org/doc/stable/reference/generated/numpy.max.html
    You may consider using np.arange https://numpy.org/doc/stable/reference/generated/numpy.arange.html

    Parameters:
    max_voter: A 2D numpy array of shape (num_cells_test, num_classes) where max_voter[j][i] = 1
               if class i is a max voter for test point j, otherwise 0.
    top_k_labels: A 2D numpy array of shape (num_cells_test, k) containing labels of the k nearest neighbors.

    Returns:
    useful_labels: A 3D numpy array of shape (num_cells_test, num_classes, k) where useful_labels[m][n][l] = 1
                   if the l-th neighbor of the m-th test point belongs to class n and class n is a max voter
                   for the m-th test point, otherwise 0.
    """
    _ , n_classes = max_voter.shape
    MAX_VOTER_3D = max_voter[: , : , None]  #(n_tests , n_classes , 1)
    class_indices = np.arange(n_classes).reshape(1 , n_classes ,1)
    IS_CLASS_MASK = (top_k_labels[: , None , :] == class_indices) 
    useful_labels = IS_CLASS_MASK & (MAX_VOTER_3D == 1)

    return useful_labels.astype(int)

def predict(distance_k, useful_labels):
    """
    Predict the label for each test cell based on inverse distances and useful labels.

    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.argmax https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    Parameters:
    distance_k: A 2D numpy array of shape (num_cells_test, k) containing distances to the k nearest neighbors.
    useful_labels: A 3D numpy array of shape (num_cells_test, num_classes, k) indicating whether
                   the l-th neighbor of the m-th test point belongs to class n and is a max voter.

    Returns:
    prediction: A 1D numpy array of shape (num_cells_test,) containing the predicted label for each test cell.
    """
    DIST_3D = distance_k[: , None , :]  #(n_tests , 1 , k)
    USEFUL_DIST = DIST_3D  * useful_labels
    WEIGHTED = np.divide(1 , USEFUL_DIST , out=np.zeros_like(USEFUL_DIST , dtype=float) , where=USEFUL_DIST != 0)

    weighted_sum = np.sum(WEIGHTED , axis=2)
    prediction = np.argmax(weighted_sum , axis=1)

    return prediction

def KNN(k, distance_metric, feature_train, feature_test, labels):
    """
    Perform k-Nearest Neighbors classification.

    Parameters:
    k: The number of nearest neighbors (integer).
    distance_metric: A string that can be either 'euclidean' or 'manhattan' indicating which distance metric to be used.
    feature_train: A 2D numpy array of shape (num_cells_train, num_proteins) representing protein abundance in the training set.
    feature_test: A 2D numpy array of shape (num_cells_test, num_proteins) representing protein abundance in the testing set.
    labels: A 1D numpy array of shape (num_cells_train,) containing labels of each cell in the training set.

    Returns:
    prediction: A 1D numpy array of shape (num_cells_test,) containing the predicted label for each test cell.
    """
    distance_k , top_k_labels = choose_nearest_neighbors(k , distance_metric , feature_train , feature_test , labels)
    class_count = count_neighbor_class(top_k_labels)
    max_voter = get_max_voter(class_count)
    
    useful_labels = get_useful_labels(max_voter , top_k_labels)
    pred = predict(distance_k , useful_labels)

    return pred

def get_accuracy(prediction, ground_truth):
    """
    Calculate the accuracy of the KNN classifier.

    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using array.size https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html

    Parameters:
    prediction: A 1D numpy array of shape (num_cells_test,) containing predicted labels for each test cell.
    ground_truth: A 1D numpy array of shape (num_cells_test,) containing the true labels for each test cell.

    Returns:
    accuracy: A float representing the accuracy of the KNN classifier (between 0 and 1).
    """

    true_pred = np.sum((prediction == ground_truth).astype(int))
    length = ground_truth.shape[0]

    accuracy = true_pred / length

    return accuracy

# Below is K MEANS


def initialize_centroids(abundance_matrix, num_clusters, random_seed=100):
    """
    Initializes centroids for clustering.

    You may consider using np.random.choice. https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    Carefully specify input into np.random.choice so that the returned array will NOT have 2 identical rows.
    You will obtain 0 points for this task if your code generates 2 identical rows.    

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins) representing protein abundance in cells.
    num_clusters: An integer representing the number of clusters.
    random_seed: An integer representing the random seed to ensure reproducibility.

    Returns:
    centroids: A 2D numpy array of shape (num_clusters, num_proteins) representing the initialized centroids.
    """    
    np.random.seed(random_seed)

    row_indices = np.random.choice(abundance_matrix.shape[0] , size=num_clusters , replace=False)
    centroids = abundance_matrix[row_indices]

    return centroids

def compute_centroids(abundance_matrix, labels):
    """
    Computes the centroids of clusters based on assigned labels.

    You may consider using np.arange https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.matmul https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

    Parameters:
    abundance_matrix: A 2D numpy array of shape (num_cells, num_proteins) representing protein abundance in cells.
    labels: An 1D numpy array of shape (num_cells,) indicating the cluster assigned to each cell.

    Returns:
    centroids: A 2D numpy array of shape (num_clusters, num_proteins) representing the new centroids of each cluster.
    """
    
    max_label = np.max(labels)
    sums = np.zeros((max_label + 1, abundance_matrix.shape[1]), dtype=abundance_matrix.dtype)
    counts = np.zeros(max_label + 1, dtype=int)
    
    for i in range(abundance_matrix.shape[0]):
        sums[labels[i]] += abundance_matrix[i]
        counts[labels[i]] += 1

    new_centroids = sums / counts[:, None]
    return new_centroids


def cluster_max_frequency(labels):
    """
    Determines which cluster has the highest number of assigned cells for potential splitting.

    You may consider using np.arange https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.max https://numpy.org/doc/stable/reference/generated/numpy.max.html
    You may consider using np.argmax https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    Alternatively, you may consider using np.bincount https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
    and np.argmax https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    Parameters:
    labels: An numpy 1D array of shape (num_cells,) indicating the cluster assigned to each cell.

    Returns:
    cluster_to_be_split: An integer representing the index of the cluster that has the maximum frequency of assigned cells.
    """
    One_Hot_Encoded = np.eye(np.max(labels) + 1)[labels]
    freq = np.sum(One_Hot_Encoded , axis=0)

    return np.argmax(freq)

def cluster_max_inertia(abundance_matrix, centroids, labels):
    """
    Determines which cluster has the highest inertia for potential splitting.

    You may consider using np.arange https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    You may consider using np.expand_dims https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    You may consider using np.sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    You may consider using np.max https://numpy.org/doc/stable/reference/generated/numpy.max.html
    You may consider using np.argmax https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    Parameters:
    abundance_matrix: A numpy 2D array of shape (num_cells, num_proteins) representing protein abundance in cells.
    centroids: A numpy 2D array of shape (num_centroids, num_proteins) representing protein abundance in centroids.
    labels: A numpy 1D array of shape (num_cells,) indicating the cluster assigned to each cell.

    Returns:
    cluster_to_be_split: An integer representing the index of the cluster that will be further split based on inertia.
    """
    X_3D = abundance_matrix[: , None , :]   #(n_cells , 1 , n_proteins)

    max_ = np.max(labels)
    One_Hot_Encoded = np.eye(max_ + 1)[labels]  #(n_cells , n_classes)
    Encoded_3D = One_Hot_Encoded[: , : , None]  #(n_cells , n_classes , 1)

    preprocessed = X_3D * Encoded_3D    #(n_cells , n_classes , n_proteins)
    preprocessed = np.where(preprocessed == 0 , np.nan , preprocessed)

    # TODO:calculate squared distance
    # centroids shape (n_classes , n_proteins)

    CENTROIDS = centroids[None , : , :]     # (1 , n_classes , n_proteins)
    dist = np.nansum((preprocessed - CENTROIDS) ** 2 , axis=2)     # (n_cells , n_classes) distance between points to center
    dist_all = np.nansum(dist , axis=0)    # (n_classes)

    return np.argmax(dist_all)

def k_means_split(abundance_matrix, initial_k=2, max_iterations=50, frequency=False):   # Fail
    """
    Performs k-means clustering and splits the most appropriate cluster.

    Parameters:
    abundance_matrix: A numpy 2D array of shape (num_cells, num_proteins) representing protein abundance in cells.
    initial_k: An integer representing the initial number of clusters to be initialized.
    max_iterations: An integer representing the maximum number of iterations for the k-means algorithm.
    frequency: A boolean, if True, selects the cluster to be split based on frequency; otherwise, uses inertia.

    Returns:
    centroids: A 2D numpy array of shape (num_clusters, num_proteins) representing the new centroids of each cluster.
    labels: An 1D numpy array of shape (num_cells,) indicating the cluster assigned to each cell.
    """

    centroids = initialize_centroids(abundance_matrix, initial_k)
    labels = np.zeros(abundance_matrix.shape[0], dtype=int)

    for i in range(max_iterations):
        distances = calculate_euclidean_distance(abundance_matrix, centroids)   # (n_centroids , n_cells)
        labels = np.argmin(distances, axis=0)
        centroids = compute_centroids(abundance_matrix, labels)
        next_distances = calculate_euclidean_distance(abundance_matrix, centroids)
        next_label = np.argmin(next_distances, axis=0)
        if np.all(next_label == labels):
            break

    if frequency:
        cluster_to_be_split = cluster_max_frequency(labels)
    else:
        cluster_to_be_split = cluster_max_inertia(abundance_matrix, centroids, labels)

    split_cluster_data = abundance_matrix[labels == cluster_to_be_split]    

    new_centroids = initialize_centroids(split_cluster_data, 2)

    centroids = np.delete(centroids, cluster_to_be_split, axis=0)


    for j in range(max_iterations):
        new_distances = calculate_euclidean_distance(split_cluster_data, new_centroids)
        new_labels = np.argmin(new_distances, axis=0)
        new_centroids = compute_centroids(split_cluster_data, new_labels)
        new_next_distances = calculate_euclidean_distance(split_cluster_data, new_centroids)
        new_next_label = np.argmin(new_next_distances, axis=0)
        if np.all(new_next_label == new_labels):
            break
    
    centroids = np.vstack((centroids, new_centroids))  

    # Reasign the labels
    dist = calculate_euclidean_distance(abundance_matrix , centroids)
    labels = np.argmin(dist , axis=0)

    return centroids, labels

if __name__ == "__main__":
    import pandas as pd
    train_feature = pd.read_csv("train_features.csv", index_col=0)
    train_label = pd.read_csv("train_labels.csv", index_col=0)
    test_feature = pd.read_csv("test_features.csv", index_col=0)
    train_feature = train_feature.to_numpy()
    train_label = train_label.to_numpy()
    test_feature = test_feature.to_numpy()
    train_label = train_label.flatten()

    processed_train_feature = preprocess_datasets(abundance_matrix=train_feature)

    result , a = k_means_split(processed_train_feature)
    print(result)
    print(a)

    # You are expected to get [2 2 2 1 2 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1
    # 1 1 1 1 1 1 1 2 2 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0 0 0 0]
    
