import pickle
import numpy as np
from pre_process import image_to_arrays
import argparse

# load the dic_mapping
with open('results/dic_mapping.pkl', 'rb') as f:
    dic_mapping = pickle.load(f)

# load the char_bounding_boxes
with open('results/char_bounding_boxes.pkl', 'rb') as f:
    char_bounding_boxes = pickle.load(f)

class Captcha(object):
    def __init__(self):
        self.dic_mapping = dic_mapping
        self.char_bounding_boxes = char_bounding_boxes

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        img_arrs = image_to_arrays(im_path, self.char_bounding_boxes)

        predictions = []
        for img_arr in img_arrs:
            # calculate the distance between the img_arr and the dic_mapping
            distances = {}
            for key, value in self.dic_mapping.items():
                distances[key] = np.linalg.norm(img_arr - value)

            # get the key with the minimum distance
            min_key = min(distances, key=distances.get)
            predictions.append(min_key)

        outcome = ''.join(predictions)
        with open(save_path, 'w') as f:
            f.write(outcome)

        print(f"The predicted outcome is: {outcome}")
        print(f"The predicted outcome is saved to {save_path}")
        return outcome

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="sampleCaptchas/input/input100.jpg")
    parser.add_argument('--save_path', type=str, default="results/output100.txt")
    args = parser.parse_args()

    captcha = Captcha()
    captcha(args.img_path, args.save_path)
