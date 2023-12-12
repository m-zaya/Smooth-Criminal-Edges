import random
import numpy as np
import cv2

def sp_noise(image, prob=0.05):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    source: https://stackoverflow.com/a/27342545
    '''
    noise_map = np.zeros(image.shape,np.uint8)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    white_noise_count = 0
    black_noise_count = 0
    norm_count = 0
    total_count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total_count += 1
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_map[i][j] = 0
                black_noise_count += 1
            elif rdn > thres:
                output[i][j] = 255
                noise_map[i][j] = 0
                white_noise_count += 1
            else:
                output[i][j] = image[i][j]
                noise_map[i][j] = 1
                norm_count += 1

    return output, noise_map, white_noise_count, black_noise_count, white_noise_count+black_noise_count, norm_count, total_count

def uniform_noise(image, prob=0.05):
    shape = image.shape
    noise_map = np.zeros(shape)
    cv2.randu(noise_map, 0, 255)
    noise_map = (noise_map * prob)
    output = cv2.add(image,noise_map)
    return output, noise_map

def sce_restoration(img: np.uint8, mask: np.bool_, implement_allnoise_fix: bool = True) -> np.uint8:

    img = np.pad(img, 2, constant_values = 0)
    mask = np.pad(mask, 2, constant_values = True)

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image

    for i in range(2, img.shape[0]-2):
        for j in range(2, img.shape[1]-2):
            if mask[i,j]:
                mask_slice = mask[i-2:i+2,j-2:j+2]
                img_slice = img[i-2:i+2,j-2:j+2]

                samples = []
                for ii in range(img_slice.shape[0]):
                    for jj in range(img_slice.shape[1]):
                        if not mask_slice[ii,jj]:
                            samples.append(img_slice[ii,jj])

                if len(samples) >= 1:
                    img_filtered[i,j] = np.median(samples)
                else:
                    img_filtered[i,j] = np.median(img_slice) if implement_allnoise_fix else img[i,j]
                
            else:
                    img_filtered[i,j] = img[i,j]

    img_filtered = img_filtered[2:img_filtered.shape[0]-2,2:img_filtered.shape[1]-2]


    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered
