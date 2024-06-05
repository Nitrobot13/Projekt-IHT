import cv2
import os
import numpy as np

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

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2), (0,255,0),2)

    # Save the output image
    cv2.imwrite(output_path, image)


output_folder = 'new_lined_output'
input_folder = 'new_images'
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

#TODO: Find a way to line the dimmer structures of the image
#TODO: Cluster lines together to form one line using some Algorithm
#TODO: Extract that lines coordinates
