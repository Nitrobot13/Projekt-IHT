import cv2, os
import numpy as np

def detect_and_mark_corners(image_path, output_path):
    image =  cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blockSize = 15
    apertureSize = 11
    k = 0.08

    gray = np.float32(gray)

    dest = cv2.cornerHarris(gray, blockSize, apertureSize, k)

    dest = cv2.dilate(dest, None)

    image[dest > 0.01 * dest.max()] = [0,0,255]

    cv2.imshow('image', image)
    cv2.waitKey()

output_folder = 'corner'
input_folder = 'images'

# Create a directory for output images if it doesn't exist


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        detect_and_mark_corners(image_path, output_path)



