import os
import random
import cv2
import numpy as np

from tools import binary_confusion, display_plots
from filters import get_noise_map_basic, get_sharpened_map, adaptive_median_filter_detection, nonadaptive_median_filter_detection

def sp_noise(image, prob=0.01):
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
    
    print("# of set noise white pixels: ", white_noise_count)
    print("# of set noise black pixels: ", black_noise_count)
    print("# of total noise pixels: ", white_noise_count+black_noise_count)
    print("# of original pixels: ", norm_count)
    print("# of total pixels: ", total_count)

    return output, noise_map, white_noise_count, black_noise_count, white_noise_count+black_noise_count, norm_count, total_count



if __name__ == "__main__":
    directory = "data/data/"
    datafile = open("results/ata.csv", "w")
    datafile.write("filename, photo_h, photo_w, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count, sharped_not_noise_count, sharped_noise_count,sharped_tn, sharped_tp, sharped_fn, sharped_fp,amfd_not_noise_count, amfd_noise_count, amfd_tn, amfd_tp, amfd_fn, amfd_fp,namfd_not_noise_count, namfd_noise_count, namfd_tn,namfd_tp, namfd_fn, namfd_fp\n") # add percentages
    datafile.close()
    for filename in os.listdir(directory):
        if (filename[-3:] != "png"):
            if (filename[-3:] != "jpg"):
                continue
        f = os.path.join(directory, filename)
        image = cv2.imread(f,0)
        print("Image: " + filename)
        photo_h, photo_w = image.shape
        noisy_image, noisy, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count = sp_noise(image)

        sharped_image = get_sharpened_map(noisy_image)
        sharped_not_noise_count = np.count_nonzero(sharped_image == 1)
        sharped_noise_count = np.count_nonzero(sharped_image == 0)
        conf_matrix_sharp = binary_confusion(noisy, sharped_image)
        cell_text_sharp = [['TN: ' + str(conf_matrix_sharp[0, 0]), 'FP: ' + str(conf_matrix_sharp[0, 1])],
                    ['FN: ' + str(conf_matrix_sharp[1, 0]), 'TP: ' + str(conf_matrix_sharp[1, 1]) ]]
        print('Confusion Matrix for Sharpened Map')
        print(cell_text_sharp)

        amfd_image = adaptive_median_filter_detection(noisy_image, 11)
        amfd_not_noise_count = np.count_nonzero(amfd_image == 1)
        amfd_noise_count = np.count_nonzero(amfd_image == 0)
        conf_matrix_amfd = binary_confusion(noisy, amfd_image)
        cell_text_amfd = [['TN: ' + str(conf_matrix_amfd[0, 0]), 'FP: ' + str(conf_matrix_amfd[0, 1])],
                    ['FN: ' + str(conf_matrix_amfd[1, 0]), 'TP: ' + str(conf_matrix_amfd[1, 1]) ]]
        print('Confusion Matrix for AMFD Map')
        print(cell_text_amfd)

        namfd_image = nonadaptive_median_filter_detection(noisy_image, 11)
        namfd_not_noise_count = np.count_nonzero(namfd_image == 1)
        namfd_noise_count = np.count_nonzero(namfd_image == 0)
        conf_matrix_namfd = binary_confusion(noisy, namfd_image)
        cell_text_namfd = [['TN: ' + str(conf_matrix_namfd[0, 0]), 'FP: ' + str(conf_matrix_namfd[0, 1])],
                    ['FN: ' + str(conf_matrix_namfd[1, 0]), 'TP: ' + str(conf_matrix_namfd[1, 1]) ]]
        print('Confusion Matrix for NAMFD Map')
        print(cell_text_namfd)

        # display_plots(image, noisy_image, noisy, sharped_image, cell_text_sharp, amfd_image, cell_text_amfd, namfd_image, cell_text_namfd)

        datafile = open("results/data.csv", "a")
        data = [filename, photo_h, photo_w, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count, sharped_not_noise_count, sharped_noise_count, 'TN: ' + str(conf_matrix_sharp[0, 0]), 'TP: ' + str(conf_matrix_sharp[1, 1]), 'FN: ' + str(conf_matrix_sharp[1, 0]), 'FP: ' + str(conf_matrix_sharp[0, 1]), amfd_not_noise_count, amfd_noise_count, 'TN: ' + str(conf_matrix_amfd[0, 0]), 'TP: ' + str(conf_matrix_amfd[1, 1]), 'FN: ' + str(conf_matrix_amfd[1, 0]), 'FP: ' + str(conf_matrix_amfd[0, 1]), namfd_not_noise_count, namfd_noise_count, 'TN: ' + str(conf_matrix_namfd[0, 0]), 'TP: ' + str(conf_matrix_namfd[1, 1]), 'FN: ' + str(conf_matrix_namfd[1, 0]), 'FP: ' + str(conf_matrix_namfd[0, 1])]
        datafile.write(', '.join(str(var) for var in data) + "\n")
        datafile.close()

        print("\n\n")

