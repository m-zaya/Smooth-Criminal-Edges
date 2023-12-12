import numpy as np
import matplotlib.pyplot as plt

def binary_confusion(given, predicted):
    # 1 = white ; 0 = black
    tn = np.sum((given == 1) & (predicted == 1))
    tp = np.sum((given == 0) & (predicted == 0))
    fn = np.sum((given == 0) & (predicted == 1))
    fp = np.sum((given == 1) & (predicted == 0))
    return np.array([[tn, fp], [fn, tp]])

def collect_noise_data(noise_map, detection_map):
    not_noise_count = np.count_nonzero(detection_map == 1)
    noise_count = np.count_nonzero(detection_map == 0)
    conf_matrix = binary_confusion(noise_map, detection_map)
    cell_text = [['TN: ' + str(conf_matrix[0, 0]), 'FP: ' + str(conf_matrix[0, 1])],
                ['FN: ' + str(conf_matrix[1, 0]), 'TP: ' + str(conf_matrix[1, 1]) ]]
    return not_noise_count, noise_count, conf_matrix, cell_text
    


def display_plots(filename, image, noisy_image, noisy, stddev_image, sharped_image, amfd_image, namfd_image, img_sce_sharp, img_sce_namfd):
    plt.figure(figsize=(24, 12))

    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.title('Noisy')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.title('Standard Deviation Per Pixel')
    plt.imshow(stddev_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title('Sharpened Map Image')
    plt.imshow(sharped_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.title('Adaptive Median Filter Detection')
    plt.imshow(amfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.title('Non Adaptive Median Filter Detection')
    plt.imshow(namfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.title('Sharpened Filter - Result')
    plt.imshow(img_sce_sharp, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.title('Non Adaptive Median Filter - Result')
    plt.imshow(img_sce_namfd, cmap='gray')
    plt.axis('off')

    # plt.show()
    
    plt.savefig("results/"+filename[:-4]+'_plot.png', dpi=300, pad_inches=0.01)
