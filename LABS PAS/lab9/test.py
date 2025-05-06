import numpy as np

def slide(arr, window_shape):
    """
    Slides a window of shape window_shape over arr starting from the left-bottom,
    moving left to right, then up one row, until all windows are covered.
    Returns a list of tuples: (i, j, window)
    """
    windows = []
    max_row = arr.shape[0] - window_shape[0]
    max_col = arr.shape[1] - window_shape[1]
    for i in range(max_row, -1, -1):  # start from bottom row upwards
        for j in range(0, max_col + 1):  # left to right
            window = arr[i:i+window_shape[0], j:j+window_shape[1]]
            windows.append((i, j, window))
    return windows

board = np.array([
    [0, 1, 0, 2, 1],
    [2, 1, 1, 0, 2],
    [0, 2, 1, 1, 0],
    [1, 0, 2, 2, 1]
])

window_shape = (1, 4)  # (height, width)

windows = slide(board, window_shape)
for i, j, window in windows:
    print(f"Window at ({i},{j}):\n{window}\n")