<<<<<<< Updated upstream
import numpy as np
import matplotlib.pyplot as plt

def binary_confusion(given, predicted):
    # 1 = white ; 0 = black
    tn = np.sum((given == 1) & (predicted == 1))
    tp = np.sum((given == 0) & (predicted == 0))
    fn = np.sum((given == 0) & (predicted == 1))
    fp = np.sum((given == 1) & (predicted == 0))
    return np.array([[tn, fp], [fn, tp]])


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
    plt.savefig("results/"+filename[:-4]+'_plot.png')
=======
import numpy as np
import matplotlib.pyplot as plt

def binary_confusion(given, predicted):
    # 1 = white ; 0 = black
    tn = np.sum((given == 1) & (predicted == 1))
    tp = np.sum((given == 0) & (predicted == 0))
    fn = np.sum((given == 0) & (predicted == 1))
    fp = np.sum((given == 1) & (predicted == 0))
    return np.array([[tn, fp], [fn, tp]])


def display_plots(image, noisy_image, noisy, sharped_image, cell_text_sharp, amfd_image, cell_text_amfd, namfd_image, cell_text_namfd):
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
    plt.title('Sharpened Map Image')
    plt.imshow(sharped_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.title('Confusion Matrix for Sharpened Map')
    table = plt.table(cellText=cell_text_sharp, loc='center', cellLoc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title('Adaptive Median Filter Detection')
    plt.imshow(amfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.title('Confusion Matrix for AMFD Map')
    table = plt.table(cellText=cell_text_amfd, loc='center', cellLoc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.title('Non Adaptive Median Filter Detection')
    plt.imshow(namfd_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.title('Confusion Matrix for NAMFD Map')
    table = plt.table(cellText=cell_text_namfd, loc='center', cellLoc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.axis('off')


    plt.show()
>>>>>>> Stashed changes
