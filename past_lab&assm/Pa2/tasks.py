import numpy as np
import cv2 as cv
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
import os

# not used for any graded tasks.  Only for you to debug and visualize.
import matplotlib.pyplot as plt # plot images

data_dir = "./pa2_dataset/img.png" # testing image
img = cv.imread(data_dir)
img = cv.cvtColor(img , cv.COLOR_BGR2RGB)

def get_bright_pixels(img : np.ndarray) -> np.ndarray:
    """
    Finds the location of bright pixels in an image.
    Args:
        img: a NumPy array of shape (height, width, 3) storing an image
    Returns:
        A boolean NumPy array of shape (height, width) where True indicates the location of bright pixels.
    """
    pixel_mask = (img > 175)
    bright_pixels = np.all(pixel_mask , axis=2)

    return bright_pixels

def get_lane_beginnings(mask : np.ndarray) -> tuple[int , int]:
    """
    Estimates the x-coordinates of the lane beginnings.
    Args:
        mask: a boolean NumPy array of shape (height, width) where True indicates the location of bright pixels.
    Returns:
        A tuple of two integers, containing the estimated x-coordinate of the beginning of the left and right lane respectively.
    """
    bottom = mask[int(0.95 * mask.shape[0]): , :]
    bottom = bottom.sum(axis=0)
    threshold = bottom.max() / 3


    selected_X_cor = np.argwhere((bottom > threshold))
    spliiting_point = np.mean(selected_X_cor)
    # right is 1 , left is 0
    label = np.where(selected_X_cor < spliiting_point, 0, 1)
    left_median = int(np.median(selected_X_cor[label == 0]))
    right_median = int(np.median(selected_X_cor[label == 1]))

    return left_median , right_median

def get_whole_lanes(mask : np.ndarray, left_lane : int, right_lane : int) -> tuple[np.ndarray , np.ndarray]:
    """
    Uses the beginning of the lanes to refine the mask so that those unrelated bright pixels are filtered out.
    Args:
        mask: a NumPy array of shape (height, width) where True indicates the bright pixels
        left_lane: the x-coordinate of the beginning of the left lane
        right_lane: the x-coordinate of the beginning of the right lane
    Returns:
        A tuple of 2 boolean NumPy arrays, each array with shape (height, width), corresponding to the left lane and right lane.
    """
    depth = mask.shape[0]
    n_bounding_boxes = 9
    # TODO: left lane
    left_img = np.zeros_like(mask)
    y = depth
    x = left_lane
    for _ in range(n_bounding_boxes):
        row_slice = np.copy(mask[y - 72 : y])
        left_bound = max(0 , x - 110)
        right_bound = min(mask.shape[1] , x + 110)

        row_slice[: , : left_bound] = 0
        row_slice[: , right_bound :] = 0

        if not row_slice.any():
            break

        left_img[y - 72 : y, :] |= row_slice
        y -= 72
        x = int(np.mean(np.argwhere(row_slice) , axis=0)[1])
    
    # TODO: right lane
    right_img = np.zeros_like(mask)
    y = depth
    x = right_lane
    for _ in range(n_bounding_boxes):
        row_slice = np.copy(mask[y-72 : y])
        left_bound = max(0, x - 110)
        right_bound = min(mask.shape[1], x + 110)

        row_slice[:, : left_bound] = 0
        row_slice[:, right_bound :] = 0
        
        if not row_slice.any():
            break
        
        right_img[y - 72 : y , :] |= row_slice
        y -= 72
        x = int(np.mean(np.argwhere(row_slice), axis=0)[1])

    return left_img , right_img

def get_lane_edges(left_lane, right_lane):
    """
    Extracts lane edges.
    Args:
        left_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the left lane
        right_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the right lane
    Returns:
        A tuple of 2 boolean NumPy arrays each with shape (height, width), containing the edges of the left lane and the right lane respectively
    """
    # define the kernels
    edge_left_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])      # use this for left edge detection
    edge_right_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])     # use this for right edge detection
    structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8) # use this for erosion and closing

    # TODO: left lane edge
    eroded_l = np.copy(left_lane).astype(np.uint8)
    eroded_l = cv.erode(eroded_l, structuring_element, iterations=3)
    eroded_l = cv.morphologyEx(eroded_l, cv.MORPH_CLOSE, structuring_element, iterations=10)
    l_edge = cv.filter2D(eroded_l, -1, edge_left_kernel)

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,2)
    plt.imshow(l_edge)
    plt.subplot(1,2,1)
    plt.imshow(left_lane)
    plt.show()
    

    return None


