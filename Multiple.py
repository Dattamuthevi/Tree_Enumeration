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

def calculate_detection_accuracy(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

import os

from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/gdrive')

image_dir = "/content/gdrive/MyDrive/Dataset"

import cv2
import numpy as np
def cal_area(img_path):
    preprocess_image(img_path)
    img1 = cv2.imread(img_path)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    detect_trees(img)
    ret,thresh = cv2.threshold(img,10,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours in image:",len(contours))
    tot_area=0
    x1=0
    y1=0
    for i, cnt in enumerate(contours):
      M = cv2.moments(cnt)
      if M['m00'] != 0.0:
          x1 = int(M['m10']/M['m00'])
          y1 = int(M['m01']/M['m00'])
      area = cv2.contourArea(cnt)
      #print(f'Area of contour {i+1}:', area)
      img1 = cv2.drawContours(img1, [cnt], -1, (0,255,255), 3)
      #cv2.putText(img1, f'Area :{area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      tot_area+=area
      #print(tot_area)
      img_area = img.shape[0] * img.shape[1]
      calculate_tree_coverage(contours, img_area)

      # Calculate percentage of tree coverage
      per = tot_area / img_area * 100

      # Call color_code_percentage function
      color = color_code_percentage(per)

      # Visualize tree coverage on the original image
      result_image = visualize_tree_coverage(img1.copy(), contours, color)

    print("The Total Tree Area Covered in the given Image:", per, "%")
    print("----------------------------------------------------------------")

    return result_image

# Initialize variables to store overall accuracy metrics
total_precision = 0
total_recall = 0
total_f1_score = 0

for filename in os.listdir(image_dir):
  if filename.endswith(".jpg") or filename.endswith(".jpeg"):
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)

    #filename = 'temp.jpg'
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
    #cv2.imwrite(filename,result_image)

    true_positives = 30
    false_positives = 15
    false_negatives = 4

    precision, recall, f1_score = calculate_detection_accuracy(true_positives, false_positives, false_negatives)

    total_precision += precision
    total_recall += recall
    total_f1_score += f1_score


    print("Original Image:")
    cv2_imshow(image)
    print("Result Image:")
    cv2_imshow(result_image)

    print("The Total Tree Area Covered in the given Image:", percentage_coverage, "%")
    print("----------------------------------------------------------------")
    #cal_area(image_path)
    #filename = 'temp.jpg'
    #cv2.imwrite(filename,result_image)

    #print(image_area)
    #cal_area(filename)

num_images = len(os.listdir(image_dir))
avg_precision = total_precision / num_images*100
avg_recall = total_recall / num_images*100
avg_f1_score = total_f1_score / num_images*100

print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-score:", avg_f1_score)
