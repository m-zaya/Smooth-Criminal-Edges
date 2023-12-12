import cv2
import numpy as np

def get_noise_map_basic(image, window_size):
    result = np.zeros_like(image)

    for i in range(image.shape[0] - window_size + 1):
        for j in range(image.shape[1] - window_size + 1):
            window = image[i:i+window_size, j:j+window_size]
            std_dev = np.std(window)
            result[i, j] = std_dev

    return 1-result

def get_sharpened_map(image):
    kernel = np.array([[-1, -1, -1],
                        [-1, 100, -1],
                        [-1, -1, -1]])
    result = cv2.filter2D(image, -1, kernel)
    _, binary_matrix = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
    result = (binary_matrix > 0).astype(int)

    return result

def threshold_detection(image, low=0, high=255):
    return ~((image <= low) | (image >= high))

def adaptive_median_filter_detection(image, wmax):
    height, width = image.shape
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            w = 3
            while w <= wmax:
                i_start, i_end = max(0, i - w // 2), min(height, i + w // 2 + 1)
                j_start, j_end = max(0, j - w // 2), min(width, j + w // 2 + 1)
                window = image[i_start:i_end, j_start:j_end]
                min_val = np.min(window)
                med_val = np.median(window)
                max_val = np.max(window)
                if min_val < med_val < max_val:
                    break
                else:
                    w += 2
            if min_val < image[i, j] < max_val:
                result[i, j] = 1
            else:
                result[i, j] = 0

    return result

def nonadaptive_median_filter_detection(image, w):
    result = np.zeros_like(image)
    padded_image = np.pad(image, w // 2, mode='constant', constant_values=0)
    window_view = np.lib.stride_tricks.sliding_window_view(padded_image, (w, w))
    min_values = np.min(window_view, axis=(2, 3))
    max_values = np.max(window_view, axis=(2, 3))
    result = np.where((min_values < image) & (image < max_values), 1, 0)

    return result