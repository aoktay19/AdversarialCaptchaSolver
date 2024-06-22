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
                output[i][j] = 0 #math.floor(random.random() * 256)
            else:
                output[i][j] = image[i][j]
    return output

def generate_noisy_data(original_X, noise_level=0.025):
    noisy_X = original_X + noise_level * np.random.normal(0, 1, original_X.shape)
    noisy_X = np.clip(noisy_X, 0, 1)  # Ensure values are within the valid range [0, 1]
    return noisy_X

def perturb_and_save(original_folder, output_folder, num_images=100, noise_level=0.01):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    for i in range(num_images):
        original_image_path = os.path.join(original_folder, image_files[i])

        if os.path.exists(original_image_path):
            original_image = cv2.imread(original_image_path, 0)  # Read as grayscale
            original_image = original_image / 255.0
            perturbed_image = generate_noisy_data(original_image, noise_level)

            # Scale the perturbed image back to 0-255 range
            perturbed_image *= 255
            perturbed_image = perturbed_image.astype(np.uint8)

            # Save the perturbed image to the output folder
            output_path = os.path.join(output_folder, 'noisy_' + image_files[i])
            cv2.imwrite(output_path, perturbed_image)
        else:
            print(f"Original image not found: {original_image_path}")

# Example usage
original_folder = 'samples/'
output_folder = 'noisy/'

perturb_and_save(original_folder, output_folder, num_images=100, noise_level=0.05)
