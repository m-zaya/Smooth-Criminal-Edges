"""
CS 4391 Project - Smooth Criminal Edges

module docstring goes here
"""
 
import cv2
import numpy as np


def sp_noise(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    source: https://stackoverflow.com/a/27342545
    '''
    output = np.zeros(image.shape,np.uint8)
    noise_mask = np.zeros(image.shape,np.bool)
    thres = 1 - prob
    noise_count = 0
    norm_count = 0
    total_count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total_count += 1
            rdn = random.random()
            if rdn < prob:
                output[i,j] = 0
                noise_mask[i,j] = True
                noise_count += 1
            elif rdn > thres:
                output[i,j] = 255
                noise_mask[i,j] = True
                noise_count += 1
            else:
                output[i,j] = image[i,j]
                noise_mask[i,j] = False
                norm_count += 1
    
    print("# of set noise pixels: ", noise_count)
    print("# of original pixels: ", norm_count)
    print("# of total pixels: ", total_count)
        
    _, binary_matrix = cv2.threshold(noise_map, 128, 255, cv2.THRESH_BINARY)

    noise_map = (binary_matrix > 0).astype(int)
    return output, noise_mask


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
    noise = np.random.normal(0,0.5,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')

    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)

    # SCE filtering
    img_sce = sce_filtering(img)
    cv2.imwrite('results/im_sce.png', img_sce)

