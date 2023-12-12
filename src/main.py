import os
import cv2
from datetime import datetime

from tools import display_plots, collect_noise_data
from detectors import get_noise_map_basic, get_sharpened_map, adaptive_median_filter_detection, nonadaptive_median_filter_detection
from noising import sp_noise, sce_restoration


if __name__ == "__main__":

    startTime = datetime.now()
    print("Started...")

    directory = "data/"

    datafile = open("results/data.csv", "w")
    datafile.write("filename, photo_h, photo_w, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count, sharped_not_noise_count, sharped_noise_count,sharped_tn, sharped_tp, sharped_fn, sharped_fp,amfd_not_noise_count, amfd_noise_count, amfd_tn, amfd_tp, amfd_fn, amfd_fp, namfd_not_noise_count, namfd_noise_count, namfd_tn,namfd_tp, namfd_fn, namfd_fp\n")

    datafile.close()

    print("Opening data...")
    print(datetime.now() - startTime)

    for filename in os.listdir(directory):
        if (filename[-3:] != "png"):
            if (filename[-3:] != "jpg"):
                continue
        
        print("Currently working on: " + filename)
        f = os.path.join(directory, filename)
        image = cv2.imread(f,0)
        print("Image: " + filename)
        photo_h, photo_w = image.shape
        noisy_image, noisy, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count = sp_noise(image)

        print("Running AMFD...")
        print(datetime.now() - startTime)

        amfd_image = noisy_image
        amfd_image = adaptive_median_filter_detection(amfd_image, 11)
        amfd_not_noise_count, amfd_noise_count, conf_matrix_amfd, cell_text_amfd = collect_noise_data(noisy, amfd_image)

        print("Done. Now Running NAMFD...")
        print(datetime.now() - startTime)

        namfd_image = noisy_image
        namfd_image = nonadaptive_median_filter_detection(namfd_image, 11)
        namfd_not_noise_count, namfd_noise_count, conf_matrix_namfd, cell_text_namfd = collect_noise_data(noisy, namfd_image)

        print("Done. Now Running Sharped...")
        print(datetime.now() - startTime)

        sharped_image = noisy_image
        sharped_image = get_sharpened_map(sharped_image)
        sharped_not_noise_count, sharped_noise_count, conf_matrix_sharp, cell_text_sharp = collect_noise_data(noisy, sharped_image)

        print("Done. Now Running STDDEV...")
        print(datetime.now() - startTime)

        stddev_image = noisy_image
        stddev_image = get_noise_map_basic(stddev_image, 3)
        stddev_not_noise_count, stddev_noise_count, conf_matrix_stddev, cell_text_stddev = collect_noise_data(noisy, stddev_image)

        print("Done. Now Restoring with SCE...")
        print(datetime.now() - startTime)

        img_sce_sharp = sce_restoration(noisy_image, 1-sharped_image)
        img_sce_namfd = sce_restoration(noisy_image, 1-namfd_image)


        print("Done. Now plotting, saving, and collecting data...")
        print(datetime.now() - startTime)

        display_plots(filename, image, noisy_image, noisy, stddev_image, sharped_image, amfd_image, namfd_image, img_sce_sharp, img_sce_namfd)

        datafile = open("results/data.csv", "a")
        data = [filename, photo_h, photo_w, white_noise_count, black_noise_count, total_noise_count, norm_count, total_count, sharped_not_noise_count, sharped_noise_count, str(conf_matrix_sharp[0, 0]), str(conf_matrix_sharp[1, 1]), str(conf_matrix_sharp[1, 0]), str(conf_matrix_sharp[0, 1]), amfd_not_noise_count, amfd_noise_count, str(conf_matrix_amfd[0, 0]), str(conf_matrix_amfd[1, 1]), str(conf_matrix_amfd[1, 0]), str(conf_matrix_amfd[0, 1]), namfd_not_noise_count, namfd_noise_count, str(conf_matrix_namfd[0, 0]), str(conf_matrix_namfd[1, 1]), str(conf_matrix_namfd[1, 0]), str(conf_matrix_namfd[0, 1])]
        datafile.write(', '.join(str(var) for var in data) + "\n")
        datafile.close()

        print("\n\n")

