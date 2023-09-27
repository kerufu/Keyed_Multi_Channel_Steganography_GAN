import tensorflow as tf
import glob
import cv2
import numpy as np

import setting

class dataset_worker():
    def __init__(self) -> None:
        try:
            self.dataset = tf.data.Dataset.load(setting.processed_dataset_path)
        except:
            self.dataset = self.load_dataset(setting.dataset_path, setting.processed_dataset_path)

    def load_dataset(self, dataset_path, processed_dataset_path):
        print("load data from raw")
        data = []
        for path in glob.iglob(dataset_path+"*.jpg"):

            img = self.preprocess_image(path)
            data.append(img)

        data = np.array(data)
        data = data / 127.5 - 1
        
        dataset = tf.data.Dataset.from_tensor_slices(data)

        dataset.save(processed_dataset_path)

        return dataset
    
    def preprocess_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img.shape[0] < img.shape[1]:
            padding_size = (img.shape[1] - img.shape[0]) // 2
            img = cv2.copyMakeBorder(img, padding_size, padding_size, 0, 0, cv2.BORDER_REFLECT)
        elif img.shape[0] > img.shape[1]:
            padding_size = (img.shape[0] - img.shape[1]) // 2
            img = cv2.copyMakeBorder(img, 0, 0, padding_size, padding_size, cv2.BORDER_REFLECT)

        img = cv2.resize(img, (setting.image_size, setting.image_size))
        return img