import numpy as np






def preprocess_iris(df):
    """ Preprocesses only petal features (more relevant for clustering). """
    X = np.array(df)
    X_mins = X.min(axis=0)
    X_maxs = X.max(axis=0)
    X_scaled = (X - X_mins) / (X_maxs - X_mins)
    
    return X_scaled



# ------------------
# My Tasks
def initialize_centroids_kmeans_pp(X, k):
    """
    Initializes k cluster centroids using the K-means++ algorithm with simplifications.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        k (int): Number of clusters.

    Returns:
        centroids (ndarray): Initialized centroids of shape (k, n_features).
    """    

    centroids = X[0 : 1]  

    for _ in range(1, k):
        X_expanded = X[:, np.newaxis, :]  #shape: (n_samples, 1, n_features)
        centroids_expanded = centroids[np.newaxis, :, :]  #shape: (1, n_centroids, n_features)

        dist = np.sum((X_expanded - centroids_expanded) ** 2, axis=2)  # (n_samples, n_centroids)
        dist_to_nearest = np.min(dist, axis=1)  #(n_samples,)
        
        new_centroid_index = np.argmax(dist_to_nearest)
        new_centroid = X[new_centroid_index:new_centroid_index+1]
        centroids = np.vstack((centroids, new_centroid))

    return centroids



def assign_clusters(X, centroids):
    """
    Assigns each data point to the nearest centroid.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        centroids (ndarray): Current centroids of shape (k, n_features).

    Returns:
        labels (ndarray): Cluster assignments for each data point.
    """

    X_3D = X[: , np.newaxis , :]
    dist = np.sum((X_3D - centroids) ** 2 , axis=2)
    labels = np.argmin(dist , axis=1)
    
    return labels


def update_centroids(X, labels, k):
    """
    Updates cluster centroids based on the mean of assigned points.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        labels (ndarray): Cluster assignments for each data point.
        k (int): Number of clusters.

    Returns:
        new_centroids (ndarray): Updated centroids of shape (k, n_features).
    """
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        mask = (labels == i)
        selected = X[mask]
        if selected.shape[0] == 0:
            new_centroid = new_centroid[i]  
        else:
            new_centroid = np.mean(selected, axis=0)
        new_centroids[i] = new_centroid

    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4):
    """
    Runs the K-means clustering algorithm.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iters (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        final_centroids (ndarray): Final cluster centroids.
        final_labels (ndarray): Final cluster assignments.
    """
    # Step 1: Initialize centroids using K-means++
    centroids = initialize_centroids_kmeans_pp(X, k)

    for _ in range(max_iters):
        # Step 2: Assign points to clusters
        labels = assign_clusters(X, centroids)

        # Step 3: Compute new centroids
        new_centroids = update_centroids(X, labels, k)

        # Step 4: Check for convergence (centroids do not change significantly)
        if np.linalg.norm(centroids - new_centroids) < tol:
            break
        
        centroids = new_centroids

    return centroids, labels

# ------------
# DEBUG

def main():
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    df = pd.DataFrame(X , columns=feature_names)
    X = preprocess_iris(df)
    k = 3
    final_centroids, final_labels = k_means(X, k)


    plt.figure(figsize=(12, 6))
    
    plt.subplot(1 , 2 , 1)
    plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='plasma', alpha=0.7, edgecolors='k')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")  
    plt.title("Visualization of K-Means Clustering (Sepal Features)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 2], X[:, 3], c=final_labels, cmap='plasma', alpha=0.7, edgecolors='k')
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Visualization of K-Means Clustering (Petal Features)")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets
    main()


