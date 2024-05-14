import cv2
import os
import numpy as np

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Function to perform K-Means clustering
def kmeans(data, k, max_iterations=100):
    # Initialize cluster centers randomly
    indices = np.random.choice(len(data), k, replace=False)
    centers = data[indices]

    for _ in range(max_iterations):
        # Assign each data point to the nearest cluster
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, center) for center in centers]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Update cluster centers
        new_centers = [np.mean(cluster, axis=0) for cluster in clusters]

        # Check for convergence
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return centers

# Function to detect lines in an image and draw them
def detect_and_draw_lines(image_path, output_path):
    # Read the image and convert to GreyScale. Also add Blur to mask the noise in the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    blurred_edges = cv2.Canny(blur, 80, 100)
    
    lines = cv2.HoughLines(blurred_edges, 1, np.pi/360, 80)

    # Store the (rho, theta) parameters of each line
    line_parameters = []

    # Extract line parameters
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            line_parameters.append((rho, theta))

    # Perform K-Means clustering to group similar lines
    if line_parameters:
        line_parameters = np.array(line_parameters)
        cluster_centers = kmeans(line_parameters, k=4)  # You can adjust the number of clusters as needed

        # Draw the lines based on the cluster centers
        for center in cluster_centers:
            rho, theta = center
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
output_folder = 'clustered_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the 'output' folder
input_folder = 'output'
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        detect_and_draw_lines(image_path, output_path)

print("Line detection and clustering complete.")
