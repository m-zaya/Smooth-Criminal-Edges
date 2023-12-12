"""
CS 4391 Project - Smooth Criminal Edges

module docstring goes here
"""
 
import cv2
import numpy as np
import random

from filters import nonadaptive_median_filter_detection

def sp_noise(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    source: https://stackoverflow.com/a/27342545
    '''
    output = np.zeros(image.shape,np.uint8)
    noise_mask = np.zeros(image.shape,bool)
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

    return output, noise_mask


def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    """
    Homework 2 Part 2
    Create a kernel_sizexkernel_size gaussian kernel of given the variance. 
    """
    # Todo: given variance: spatial_variance and kernel size, you need to create a kernel_sizexkernel_size gaussian kernel
    # Please check out the formula in slide 15 of lecture 6 to learn how to compute the gaussian kernel weight: g[k, l] at each position [k, l].
    kernel_weights = np.zeros((kernel_size, kernel_size))

    for k in range(kernel_size):
        for l in range(kernel_size):
            kernel_weights[k,l] = np.exp(-(k**2+l**2)/(2*spatial_variance))

    return kernel_weights


#TODO: add any special parameters we want to be tweaked
def sce_filtering(
    img: np.uint8,
    mask: np.bool_,
    spatial_variance: float = 30,
    intensity_variance: float = 0.5,
    kernel_size: int = 7,
) -> np.uint8:
    """
    function docstring goes here
    """
    kernel_radius = kernel_size//2

    img = np.pad(img, kernel_radius, constant_values=0)
    mask = np.pad(mask, kernel_radius, constant_values=True)

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image

    spatial_kernel = gauss_kernel_generator(kernel_size, spatial_variance)

    for i in range(kernel_radius, img.shape[0]-kernel_radius):
        for j in range(kernel_radius, img.shape[1]-kernel_radius):
            if mask[i,j]:
                
                mask_slice = mask[i-kernel_radius:i+kernel_radius+1,j-kernel_radius:j+kernel_radius+1]
                img_slice = img[i-kernel_radius:i+kernel_radius+1,j-kernel_radius:j+kernel_radius+1]

                # median filter
                samples = []
                for ii in range(img_slice.shape[0]):
                    for jj in range(img_slice.shape[1]):
                        if not mask_slice[ii,jj]:
                            samples.append(img_slice[ii,jj])
                
                if len(samples) >= 1:
                    med = np.median(samples)
                else:
                    med = np.median(img_slice)

                # x = med - img_slice
                # r = np.exp(-x**2/(2*intensity_variance))
                # img_filtered[i,j] = np.sum(spatial_kernel*r*img_slice*(1-mask_slice))/np.sum(spatial_kernel*r)
                img_filtered[i,j] = med


            else:
                    img_filtered[i,j] = img[i,j]

    img_filtered = img_filtered[kernel_radius:img_filtered.shape[0]-kernel_radius,kernel_radius:img_filtered.shape[1]-kernel_radius]


    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

def maskgen(img):
    return (img == 0) | (img == 255)

if __name__ == "__main__":
    img = cv2.imread("data/bridge.png", 0) # read gray image

    # # Generate Gaussian noise
    # noise = np.random.normal(0,0.5,img.size)
    # noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')

    # # Add the generated Gaussian noise to the image
    # img_noise = cv2.add(img, noise)
    # cv2.imwrite('results/im_noisy.png', img_noise)

    # add noise to the image, and get the ground truth mask for where the noise is
    img_noisy, mask_gt = sp_noise(img, 0.2)
    mask = nonadaptive_median_filter_detection(img_noisy, 11)

    # SCE filtering
    img_sce = sce_filtering(img_noisy, 1-mask)
    cv2.imwrite('results/im_noisy.png', img_noisy)
    cv2.imwrite('results/im_sce.png', img_sce)

    mask2 = maskgen(img_noisy)

    img_sce2 = sce_filtering(img_noisy, mask2)
    cv2.imwrite('results/im_sce2.png', img_sce2)

    print(np.all(img_sce == img_sce2))

