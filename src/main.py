import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def get_images(directory):
    imgs = []
    for filename in os.listdir(directory):
        if filename[-3:] == "png":
            f = os.path.join(directory, filename)
            imgs.append(cv2.imread(f,0))
    return imgs

def sp_noise(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    source: https://stackoverflow.com/a/27342545
    '''
    noise_map = np.zeros(image.shape,np.uint8)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    noise_count = 0
    norm_count = 0
    total_count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total_count += 1
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_map[i][j] = 0
                noise_count += 1
            elif rdn > thres:
                output[i][j] = 255
                noise_map[i][j] = 0
                noise_count += 1
            else:
                output[i][j] = image[i][j]
                noise_map[i][j] = 255
                norm_count += 1
    
    print("# of set noise pixels: ", noise_count)
    print("# of original pixels: ", norm_count)
    print("# of total pixels: ", total_count)
        
    _, binary_matrix = cv2.threshold(noise_map, 128, 255, cv2.THRESH_BINARY)

    noise_map = (binary_matrix > 0).astype(int)
    return output, noise_map

def get_noise_map(image, window_size):
    noise_map = np.zeros_like(image)

    for i in range(image.shape[0] - window_size + 1):
        for j in range(image.shape[1] - window_size + 1):
            window = image[i:i+window_size, j:j+window_size]
            std_dev = np.std(window)
            noise_map[i+1, j+1] = std_dev

    return noise_map

def get_sharpened_map(image):
    kernel50 = np.array(  [[0, -1, 0],
                        [-1, 50, -1],
                        [0, -1, 0]])
    kernel3 = np.array(  [[0, -1, 0],
                        [-1, 1, -1],
                        [0, -1, 0]])
    img50 = cv2.filter2D(image, -1, kernel50)
    img3 = cv2.filter2D(image, -1, kernel3)
    _, binary_matrix = cv2.threshold(img3, 128, 255, cv2.THRESH_BINARY)
    img3 = (binary_matrix > 0).astype(int)
    img3 = 1 - img3
    _, binary_matrix = cv2.threshold(img50, 128, 255, cv2.THRESH_BINARY)
    img50 = (binary_matrix > 0).astype(int)
    img_comb = (img50+img3)
    img_comb = np.where(img_comb == 2, 1, 0)

    print("Number of 1s (detected noise) in matrix: ", np.count_nonzero(img_comb == 0))
    print("Number of 0s (not detected noise) in matrix: ", np.count_nonzero(img_comb == 1))
    print("Number of elements total: ", img_comb.size)


    return img_comb


def binary_confusion(given, predicted):
    tp = np.sum((given == 1) & (predicted == 1))
    tn = np.sum((given == 0) & (predicted == 0))
    fp = np.sum((given == 0) & (predicted == 1))
    fn = np.sum((given == 1) & (predicted == 0))
    return np.array([[tn, fp], [fn, tp]])


def detect_outliers(image, max_window_size):
    rows, cols = image.shape
    outliers = np.zeros_like(image, dtype=bool)
    pad_size = max_window_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    for i in range(rows):
        for j in range(cols):
            window_size = 3
            while window_size <= max_window_size:
                window = padded_image[i:i + window_size, j:j + window_size]
                if image[i, j] < np.min(window) or image[i, j] > np.max(window):
                    outliers[i, j] = True
                    window_size += 2
                else:
                    break
    return outliers

if __name__ == "__main__":
    images = get_images("data/")
    count = 0
    for image in images:
        count += 1
        print("Image #" + str(count))
        noisy_image, noisy = sp_noise(image)
        sharped_image = get_sharpened_map(noisy_image)
        conf_matrix = binary_confusion(noisy, sharped_image)

        plt.figure(figsize=(24, 12))

        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Noisy Image')
        plt.imshow(noisy_image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('Noisy')
        plt.imshow(noisy, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Sharpened Map Image')
        plt.imshow(sharped_image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Confusion Matrix')
        cell_text = [['TN: ' + str(conf_matrix[0, 0]), 'FP: ' + str(conf_matrix[0, 1])],
                    ['FN: ' + str(conf_matrix[1, 0]), 'TP: ' + str(conf_matrix[1, 1]) ]]
        print(cell_text)
        table = plt.table(cellText=cell_text, loc='center', cellLoc='center', edges='open')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.axis('off')
        
        print("\n\n")

        plt.show()        
