import cv2
import os

def find_contours(image, min_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area -> sort out smaller contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]   
    return contours

def draw_rectangles(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def save_contour_images(image, contours, output_folder, j):
    os.makedirs(output_folder, exist_ok=True)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, f'output_{j}_{i}.jpg'), roi)

def resize_image(image, width):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Compute the scaling factor to maintain the aspect ratio
    scaling_factor = 1920 / original_width

    # Compute the new dimensions
    new_width = 1920
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def process_images_in_folder(input_folder, output_folder):

    min_area = 20000
    width = 2000
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    for filename in os.listdir(input_folder):
        i+=1
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)
            image = resize_image(image, width)
            if image is not None:
                contours = find_contours(image.copy(), min_area)
                image_with_rectangles = draw_rectangles(image.copy(), contours)
                save_contour_images(image.copy(), contours, output_folder, i)
                cv2.imshow('Image', image_with_rectangles)
                cv2.waitKey()
    cv2.destroyAllWindows()

input_folder = 'images'
output_folder = 'output'
process_images_in_folder(input_folder, output_folder)

#TODO: Adapt code for new images if possible
#TODO: Recognize the whole Alignment mark as ONE unique entity if possible

