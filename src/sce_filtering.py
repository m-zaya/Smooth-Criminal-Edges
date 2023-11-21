"""
CS 4391 Project - Smooth Criminal Edges

module docstring goes here
"""
 
import cv2
import numpy as np


#TODO: add any special parameters we want to be tweaked
def sce_filtering(
    img: np.uint8,
    # spatial_variance: float,
    # intensity_variance: float,
    # kernel_size: int,
) -> np.uint8:
    """
    function docstring goes here
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image

    ### our code goes here

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered


if __name__ == "__main__":
    img = cv2.imread("data/boat.png", 0) # read gray image

    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')

    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)

    # SCE filtering
    img_sce = sce_filtering(img)
    cv2.imwrite('results/im_sce.png', img_sce)

