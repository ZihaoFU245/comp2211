import numpy as np
import timeit

def old_compute_centroids(abundance_matrix, labels):
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

    X_3D = abundance_matrix[: , None , :]   # (n_cells , 1 , n_proteins)

    max_ = np.max(labels)


    One_Hot_Encoded = np.eye(max_ + 1)[labels]  # (n_cells , n_classes)
    Encoded_3D = One_Hot_Encoded[: , : , None]  # (n_cells , n_classes , 1)

    preprocessed = X_3D * Encoded_3D
    preprocessed = np.where(preprocessed == 0, np.nan, preprocessed)
    
    # FIND MEAN
    new_centroids = np.nanmean(preprocessed , axis=0)

    return new_centroids

def compute_centroids(abundance_matrix , labels):
    max_label = np.max(labels)
    sums = np.zeros((max_label + 1, abundance_matrix.shape[1]), dtype=abundance_matrix.dtype)
    counts = np.bincount(labels)
    np.add.at(sums, labels, abundance_matrix)
    new_centroids = sums / counts[:, None]
    return new_centroids

def for_loop_compute_centroids(abundance_matrix , labels):
    max_label = np.max(labels)
    sums = np.zeros((max_label + 1, abundance_matrix.shape[1]), dtype=abundance_matrix.dtype)
    counts = np.zeros(max_label + 1, dtype=int)
    
    for i in range(abundance_matrix.shape[0]):
        sums[labels[i]] += abundance_matrix[i]
        counts[labels[i]] += 1

    new_centroids = sums / counts[:, None]
    return new_centroids

def pure_python_compute_centroids(abundance_matrix, labels):
    n_cells = len(abundance_matrix)
    n_proteins = len(abundance_matrix[0])
    max_label = max(labels)
    sums = [[0.0 for _ in range(n_proteins)] for _ in range(max_label + 1)]
    counts = [0]*(max_label + 1)

    for i in range(n_cells):
        lbl = labels[i]
        row = abundance_matrix[i]
        for j in range(n_proteins):
            sums[lbl][j] += row[j]
        counts[lbl] += 1

    new_centroids = []
    for lbl in range(max_label + 1):
        if counts[lbl] == 0:
            new_centroids.append([0.0]*n_proteins)
        else:
            new_centroids.append([val / counts[lbl] for val in sums[lbl]])
    return new_centroids

def test():
    labels = np.random.randint(0, 10, size=(150,))
    abundance_matrix = np.random.rand(150, 500)  # Example abundance matrix with 150 cells and 500 proteins

    # Time the old function
    old_timer = timeit.Timer(lambda: old_compute_centroids(abundance_matrix, labels))
    old_time = old_timer.timeit(number=1000)
    
    # Time the new function
    new_timer = timeit.Timer(lambda: compute_centroids(abundance_matrix, labels))
    new_time = new_timer.timeit(number=1000)
    
    print(f"Old function time: {old_time:.6f} seconds")
    print(f"New function time: {new_time:.6f} seconds")
    
    # Calculate centroids once for comparison
    old_centroids = old_compute_centroids(abundance_matrix, labels)
    new_centroids = compute_centroids(abundance_matrix, labels)
    isEqual = np.allclose(old_centroids, new_centroids)
    print(f"Are the centroids equal? {isEqual}")

    for_loop_timer = timeit.Timer(lambda: for_loop_compute_centroids(abundance_matrix, labels))
    for_loop_time = for_loop_timer.timeit(number=1000)
    print(f"For Loop function time: {for_loop_time:.6f} seconds")

    for_loop_centroids = for_loop_compute_centroids(abundance_matrix, labels)
    isEqual_for_loop = np.allclose(old_centroids, for_loop_centroids)
    print(f"Are the old and for-loop centroids equal? {isEqual_for_loop}")

    pure_python_timer = timeit.Timer(lambda: pure_python_compute_centroids(abundance_matrix, labels))
    pure_python_time = pure_python_timer.timeit(number=1000)
    print(f"Pure Python function time: {pure_python_time:.6f} seconds")

    pure_python_centroids = pure_python_compute_centroids(abundance_matrix, labels)
    isEqual_pure_python = np.allclose(old_centroids, np.array(pure_python_centroids))
    print(f"Are the old and purely python centroids equal? {isEqual_pure_python}")

if __name__ == "__main__":
    test()