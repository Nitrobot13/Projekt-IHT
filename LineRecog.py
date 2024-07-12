import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_lines(lines, eps=10, min_samples=1):
    if lines is None:
        return []

    # Prepare data for clustering (using mid-point and angle of each line)
    data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2((y2 - y1), (x2 - x1))
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
        data.append([midpoint[0], midpoint[1], angle])

    data = StandardScaler().fit_transform(data)
    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    # Group lines by clusters
    grouped_lines = {}
    for label, line in zip(labels, lines):
        if label in grouped_lines:
            grouped_lines[label].append(line[0])
        else:
            grouped_lines[label] = [line[0]]

    # Average lines in each cluster
    averaged_lines = []
    for lines in grouped_lines.values():
        x1s, y1s, x2s, y2s = zip(*lines)
        avg_line = [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]
        averaged_lines.append([avg_line])
    return averaged_lines
# Function to detect lines in an image and draw them on the image
def detect_and_draw_lines(image_path, output_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    blurred_edges = cv2.Canny(blur, 60, 80)
    
    lines = cv2.HoughLines(blurred_edges, 1, np.pi/360, 80)

    # Draw the lines on the image for HoughLines
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



def detect_and_draw_lines_HoughLinesP(image_path, output_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #convert image to GrayScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #Canny Edge Detection:
    #cthreshold values 150-350 find lines only in modified image GefaltetesBild

    cthreshold1 = 40
    cthreshold2 = 100
    filterSize = 5

    blurred_edges = cv2.Canny(blur, cthreshold1, cthreshold2, filterSize)

    rho = 1
    theta = np.pi/180
    threshold = 50
    minLineLength = 50
    maxLineGap = 80
    
    lines = cv2.HoughLinesP(blurred_edges, rho, theta, threshold, 
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    #clustering lines
    eps = 0.3
    min_samples = 1

    averaged_lines = cluster_lines(lines, eps, min_samples)



    if lines is not None:
        print(f"Found {len(lines)} lines")
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2), (0,255,0),2)

    if averaged_lines is not None:
        print(f"Found {len(averaged_lines)} clusters")
        for line in averaged_lines:
            x1,y1,x2,y2 = line[0]
            print(f"Line at: ({x1},{y1}), ({x2},{y2})")
            cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)


def cluster_lines(lines, eps=10, min_samples=1):
    if lines is None:
        return []

    data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        cos_theta = np.cos(angle) 
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
        data.append([midpoint[0], midpoint[1], cos_theta])

    data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    grouped_lines = {}
    for label, line in zip(labels, lines):
        if label in grouped_lines:
            grouped_lines[label].append(line[0])
        else:
            grouped_lines[label] = [line[0]]

    averaged_lines = []
    for lines in grouped_lines.values():
        x1s, y1s, x2s, y2s = zip(*lines)
        avg_line = [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]
        averaged_lines.append([avg_line])
    return averaged_lines


output_folder = 'cut_lined_output'
input_folder = 'output'
HoughLinesP = True

# Create a directory for output images if it doesn't exist



if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if HoughLinesP:
            detect_and_draw_lines_HoughLinesP(image_path, output_path)
        else: 
            detect_and_draw_lines(image_path,output_path)

print("Line detection and drawing complete. Using HoughLinesP: ", HoughLinesP)
