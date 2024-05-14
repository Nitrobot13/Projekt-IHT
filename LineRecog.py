import cv2
import os
import numpy as np

# Function to detect lines in an image and draw them
def detect_and_draw_lines(image_path, output_path):
    # Read the image and convert to GreyScale. Also add Blur to mask the noise in the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(gray, (5,5), 0)

    blurred_edges = cv2.Canny(blur, 60, 80)
    
    lines = cv2.HoughLines(blurred_edges, 1, np.pi/360, 80)


    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)

# Create a directory for output images if it doesn't exist
output_folder = 'lined_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the 'output' folder
input_folder = 'output'
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        detect_and_draw_lines(image_path, output_path)

print("Line detection and drawing complete.")
