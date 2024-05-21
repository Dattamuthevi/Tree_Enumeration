!pip install opencv-python==4.5.5.64

!pip install torch

!pip install numpy
import numpy as np

!pip install google-colab

import cv2
import numpy as np
from google.colab import files

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    return blurred

import matplotlib.pyplot as plt

def area(img):
    # Convert the image to a NumPy array
    img = plt.imread(img)

    # fetch the height and width
    height, width, _ = img.shape

    # Calculate the area of the image
    area = height * width

    return area

def detect_trees(image):
    # Perform edge detection
    edges = cv2.Canny(image, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_tree_coverage(contours, image_area):
    tree_area = 0
    for contour in contours:
        # Calculate area for each contour bounding box
        x, y, w, h = cv2.boundingRect(contour)
        tree_area += w * h
    # Calculate percentage of tree coverage
    percentage_coverage = (tree_area / image_area) * 100
    return percentage_coverage

def color_code_percentage(percentage_coverage):
    # Define color codes based on percentage coverage
    if percentage_coverage >= 70:
        color = (0, 255, 0)  # Green for high coverage
    elif percentage_coverage >= 30:
        color = (0, 255, 255)  # Yellow for moderate coverage
    else:
        color = (0, 0, 255)  # Red for low coverage
    return color

def visualize_tree_coverage(image, contours, color):
    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, color, 2)
    return image

import os

from google.colab.patches import cv2_imshow

uploaded = files.upload()

# Read the uploaded image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)

import matplotlib.pyplot as plt

preprocessed_image = preprocess_image(image_path)

# Calculate the total area of the image
image_area = preprocessed_image.shape[0] * preprocessed_image.shape[1]

# Detect trees in the image
contours = detect_trees(preprocessed_image)

# Calculate tree coverage percentage
percentage_coverage = calculate_tree_coverage(contours, image_area)

# Color code the percentage coverage
color = color_code_percentage(percentage_coverage)

# Visualize tree coverage on the original image
result_image = visualize_tree_coverage(image.copy(), contours, color)

# Display the result
print("The Resultant Image:")
print("")
cv2_imshow(result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("")
print("Objects Detected in %:", percentage_coverage, "%")
print("----------------------------------------------------------------")
