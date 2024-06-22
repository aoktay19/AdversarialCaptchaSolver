import os
import cv2
import numpy as np
import random
import shutil
import math

def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif 1 - rdn < prob:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def perturb_and_save(original_folder, output_folder, num_images=100, noise_probability=0.01):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    for i in range(num_images):
        original_image_path = os.path.join(original_folder, image_files[i])

        if os.path.exists(original_image_path):
            original_image = cv2.imread(original_image_path, 0)  # Read as grayscale
            perturbed_image = sp_noise(original_image, noise_probability)

            # Save the perturbed image to the output folder
            output_path = os.path.join(output_folder, 'perturbed_' + image_files[i])
            cv2.imwrite(output_path, perturbed_image)
        else:
            print(f"Original image not found: {original_image_path}")

# Example usage
original_folder = 'samples/'
output_folder = 'perturbed/'

perturb_and_save(original_folder, output_folder, num_images=100, noise_probability=0.075)
