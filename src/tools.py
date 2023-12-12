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

def display_plots(filename, image, noisy_image, noisy, stddev_image, sharped_image, amfd_image, namfd_image, img_sce_sharp, img_sce_amfd, img_sce_namfd, mse_value_sharp, mse_value_amfd, mse_value_namfd, img_fast_nl_means, img_gaussian, img_box_blur, img_median, img_bilateral, threshold_image, img_threshold):
    plt.figure(figsize=(48, 36))

    plt.subplot(5, 5, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 3)
    plt.title('Noisy')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 4)
    plt.title('Sharpened Map Image')
    plt.imshow(sharped_image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 5)
    plt.title('Adaptive Median Filter Detection')
    plt.imshow(amfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 6)
    plt.title('Non Adaptive Median Filter Detection')
    plt.imshow(namfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 7)
    plt.title('Sharpened Filter - Result')
    plt.imshow(img_sce_sharp, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 8)
    plt.title('Adaptive Median Filter - Result')
    plt.imshow(img_sce_namfd, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 9)
    plt.title('Non Adaptive Median Filter - Result')
    plt.imshow(img_sce_namfd, cmap='gray')
    plt.axis('off')

    differences = image != img_sce_sharp
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 10)
    plt.title('Sharpened Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    differences = image != img_sce_amfd
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 11)
    plt.title('Adaptive Median Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    differences = image != img_sce_namfd
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 12)
    plt.title('Non Adaptive Median Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 13)
    plt.title('NonLocalMeans - Result')
    plt.imshow(img_fast_nl_means, cmap='gray')
    plt.axis('off')

    differences = image != img_fast_nl_means
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 14)
    plt.title('NonLocal Means Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 15)
    plt.title('Gaussian - Result')
    plt.imshow(img_gaussian, cmap='gray')
    plt.axis('off')

    differences = image != img_gaussian
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 16)
    plt.title('Gaussian Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 17)
    plt.title('Median - Result')
    plt.imshow(img_median, cmap='gray')
    plt.axis('off')

    differences = image != img_median
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 18)
    plt.title('Median Filter - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 19)
    plt.title('Box Blur - Result')
    plt.imshow(img_box_blur, cmap='gray')
    plt.axis('off')

    differences = image != img_box_blur
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 20)
    plt.title('Box Blur - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 21)
    plt.title('Bilateral Blur - Result')
    plt.imshow(img_bilateral, cmap='gray')
    plt.axis('off')

    differences = image != img_bilateral
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 22)
    plt.title('Bilateral Blur - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 23)
    plt.title('Threshhold Map Image')
    plt.imshow(threshold_image, cmap='gray')
    plt.axis('off')

    plt.subplot(5, 5, 24)
    plt.title('Threshhold - Result')
    plt.imshow(img_threshold, cmap='gray')
    plt.axis('off')

    differences = image != img_threshold
    coords = np.column_stack(np.where(differences))
    colored_img = np.stack((image, image, image), axis=-1)
    colored_img[coords[:, 0], coords[:, 1], 0] = 255
    plt.subplot(5, 5, 25)
    plt.title('Threshhold - Results Difference')
    plt.imshow(colored_img, cmap='gray')
    plt.axis('off')



    # plt.show()
    
    plt.savefig("results/"+filename[:-4]+'_plot.png', dpi=300, pad_inches=0.01)
