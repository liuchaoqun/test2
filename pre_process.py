import cv2
import numpy as np
from collections import defaultdict
import os
import pickle

def segment_characters(image_path):
    # 1. Preprocessing
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return thresh

def find_char_bounding_boxes(thresh):
    # 2. Finding Contours
    # Use RETR_EXTERNAL to get only the outer contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)

    # 3. Filtering and Sorting
    char_bounding_boxes = []
    MIN_CONTOUR_AREA = 5 # This value needs tuning based on your image

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            char_bounding_boxes.append((x, y, w, h))

    # Sort bounding boxes by their x-coordinate
    char_bounding_boxes = sorted(char_bounding_boxes, key=lambda box: box[0])
    return char_bounding_boxes

def extract_character_images(thresh, char_bounding_boxes):
    # 4. Extracting Characters
    character_images = []
    for box in char_bounding_boxes:
        x, y, w, h = box
        # Crop from the original grayscale image
        char_image = thresh[y:y+h, x:x+w]
        character_images.append(char_image)
    # print(character_images)
        
    return character_images

def image_to_arrays(image_path, char_bounding_boxes):
    thresh = segment_characters(image_path)
    character_images = extract_character_images(thresh, char_bounding_boxes)
    return character_images

def extract_labels(output_path):
    # read the output.txt file
    with open(output_path, 'r') as file:
        text = file.read().strip()

    return text

if __name__ == "__main__":
    dic_boxes = {'x1':[], 'x2':[], 'x3':[], 'x4':[], 'x5':[], 'y': [], 'w': [], 'h': []}

    for i in range(25):
        img_path = f"sampleCaptchas/input/input{i:02d}.jpg"
        thresh = segment_characters(img_path)
        char_bounding_boxes = find_char_bounding_boxes(thresh)
        for box in char_bounding_boxes:
            dic_boxes['y'].append(box[1])
            dic_boxes['w'].append(box[2])
            dic_boxes['h'].append(box[3])
        dic_boxes['x1'].append(char_bounding_boxes[0][0])
        dic_boxes['x2'].append(char_bounding_boxes[1][0])
        dic_boxes['x3'].append(char_bounding_boxes[2][0])
        dic_boxes['x4'].append(char_bounding_boxes[3][0])
        dic_boxes['x5'].append(char_bounding_boxes[4][0])

    dic_boxes['x1'] = min(dic_boxes['x1'])
    dic_boxes['x2'] = min(dic_boxes['x2'])
    dic_boxes['x3'] = min(dic_boxes['x3'])
    dic_boxes['x4'] = min(dic_boxes['x4'])
    dic_boxes['x5'] = min(dic_boxes['x5'])
    dic_boxes['y'] = min(dic_boxes['y'])
    dic_boxes['w'] = max(dic_boxes['w'])
    dic_boxes['h'] = max(dic_boxes['h'])

    char_bounding_boxes = [
        (dic_boxes['x1'], dic_boxes['y'], dic_boxes['w'], dic_boxes['h']),
        (dic_boxes['x2'], dic_boxes['y'], dic_boxes['w'], dic_boxes['h']),
        (dic_boxes['x3'], dic_boxes['y'], dic_boxes['w'], dic_boxes['h']),
        (dic_boxes['x4'], dic_boxes['y'], dic_boxes['w'], dic_boxes['h']),
        (dic_boxes['x5'], dic_boxes['y'], dic_boxes['w'], dic_boxes['h'])
    ]

    # save the char_bounding_boxes to a pickle file
    os.makedirs('results', exist_ok=True)
    with open('results/char_bounding_boxes.pkl', 'wb') as f:
        pickle.dump(char_bounding_boxes, f)

    print("The char_bounding_boxes are saved to results/char_bounding_boxes.pkl")

    # extract the pattern of each character
    dic_mapping = defaultdict(list)

    for i in range(25):
        # check if the file exists
        if not os.path.exists(f"sampleCaptchas/input/input{i:02d}.jpg"):
            continue
        if not os.path.exists(f"sampleCaptchas/output/output{i:02d}.txt"):
            continue

        img_path = f"sampleCaptchas/input/input{i:02d}.jpg"
        output_path = f"sampleCaptchas/output/output{i:02d}.txt"

        img_arr = image_to_arrays(img_path, char_bounding_boxes)
        labels = extract_labels(output_path)

        for j in range(5):
            dic_mapping[labels[j]].append(img_arr[j])

    dic_mapping = {k: np.mean(v, axis=0) for k, v in dic_mapping.items()}
    dic_mapping = {k: np.uint8(v) for k, v in dic_mapping.items()}

    # sort the dic_mapping by the key
    dic_mapping = dict(sorted(dic_mapping.items()))

    # save the dic_mapping to a pickle file
    with open('results/dic_mapping.pkl', 'wb') as f:
        pickle.dump(dic_mapping, f)

    print("The dic_mapping is saved to results/dic_mapping.pkl")