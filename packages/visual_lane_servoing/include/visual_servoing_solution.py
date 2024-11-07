from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    # The image is divided in 2:
    steer_matrix_left = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j < (shape[1] // 2):
                steer_matrix_left[i, j] = -0.7
            else:
                steer_matrix_left[i, j] = 0

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    ## The image is divided in 2:
    steer_matrix_right = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j < (shape[1] // 2):
                steer_matrix_right[i, j] = 0
            else:
                steer_matrix_right[i, j] = 0.35

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    mask_left_edge = np.random.rand(h, w)
    mask_right_edge = np.random.rand(h, w)

    # The image-to-ground homography associated with this image
    H = np.array([-4.137917960301845e-05, -0.00011445854191468058, -0.1595567007347241, 
              0.0008382870319844166, -4.141689222457687e-05, -0.2518201638170328, 
              -0.00023561657746150284, -0.005370140574116084, 0.9999999999999999])

    H = np.reshape(H,(3, 3))
    Hinv = np.linalg.inv(H)

    # Grayscale image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ground mask
    mask_ground = cv2.warpPerspective(np.ones(img.shape, dtype=np.uint8), Hinv, (img.shape[1], img.shape[0]))

    # Smooth the image using a Gaussian kernel
    sigma = 7
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # Gradient magnitude mask
    threshold = np.array([20]) # CHANGE ME
    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    mask_mag = (Gmag > threshold)

    # Color masks
    white_lower_hsv = np.array([0, 0, 180])     
    white_upper_hsv = np.array([180, 50, 255])
    yellow_lower_hsv = np.array([20, 50, 50])
    yellow_upper_hsv = np.array([30, 255, 255])

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # Left and right masks
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0

    # Sobel masks
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
