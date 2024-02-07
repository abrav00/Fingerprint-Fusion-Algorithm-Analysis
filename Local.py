import cv2
import argparse
import numpy as np
import os
import csv

def enhance_fingerprint(img):
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 5)
    return img

def preprocess_image(img_path):
    img = cv2.imread(img_path, 0)
    img = enhance_fingerprint(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,                                cv2.THRESH_BINARY_INV, 11, 2)
    return img

def match_fingerprints(img1, img2_path):
    img2 = preprocess_image(img2_path)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return len(good_matches), len(good_matches) >= 20

def compare_with_directory(reference_img_path, directory):
    reference_img = preprocess_image(reference_img_path)
    results = []

    for filename in os.listdir(directory):
        if filename.endswith((".tif")):  # Add other image formats if needed
            img_path = os.path.join(directory, filename)
            if img_path == reference_img_path:
                continue
            num_good_matches, result = match_fingerprints(reference_img, img_path)
            results.append([reference_img_path, img_path, num_good_matches, result])

            print(f"Comparing {reference_img_path} with {img_path}: {'Yes' if result else 'No'}, Good matches: {num_good_matches}")

    # Writing results to a CSV file
    with open('local_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reference Image", "Compared Image", "Good Matches", "Is Match"])
        writer.writerows(results)


reference_img_path = '101_1.tif'
directory_path = 'database'  # Update with your directory path
compare_with_directory(reference_img_path, directory_path)
