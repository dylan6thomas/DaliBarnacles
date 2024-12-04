import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

def get_frame(image_path):
    # https://stackoverflow.com/questions/55169645/square-detection-in-image

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    # Read the image
    image = cv2.imread(image_path)
    
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green color range (Hue values for green)
    lower_green = np.array([35, 50, 10])  # Adjust values as needed
    upper_green = np.array([100, 255, 255])

    # Create a mask for green
    mask = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))

    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect_kernel)
    
    contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0

    wh_ratio_max = 1.1
    wh_ratio_min = 0.9

    for c in contours[:-2]:
        x,y,w,h = cv2.boundingRect(c)
        a = w*h
        wh_ratio = w/h
        if a > max_area and a < 6000000 and wh_ratio < wh_ratio_max and wh_ratio > wh_ratio_min:
            largest_contour = c
            max_area = a

    x,y,w,h = cv2.boundingRect(largest_contour)

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(close,cmap="gray")

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(image[y:y+h,x:x+w])

    return image, x, y, w, h

def divide_into_segments(image, grid_size):
    h, w, _ = image.shape
    step_x, step_y = w // grid_size[1], h // grid_size[0]

    segments = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_start, y_start = j * step_x, i * step_y
            x_end, y_end = x_start + step_x, y_start + step_y
            segments.append(image[y_start:y_end, x_start:x_end])
    return segments

def count_barnacles(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    plt.subplot(5,1,3)
    plt.imshow(morph,cmap="gray")

    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    min_area = 100

    for c in contours[:-2]:
        a = cv2.contourArea(c)
        if a > min_area:
            filtered_contours.append(c)

    return len(filtered_contours), filtered_contours

class BarnacleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Custom dataset for barnacle images and masks.

        Args:
            image_dir (str): Path to the images directory.
            mask_dir (str): Path to the masks directory.
            transform (callable, optional): Optional transforms to apply.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

        print(len(self.image_names), len(self.mask_names))

        self.image_names = sorted(os.listdir(self.image_dir))
        self.mask_names = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = cv2.imread(img_path)  # BGR format

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load corresponding mask
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        plt.subplot(6,1,1)
        plt.imshow(mask,cmap="gray")

        plt.subplot(6,1,2)
        plt.imshow(image)

        # Apply transformations (if any)
        if self.transform:
            # Concatenate image and mask for joint transformations
            image = self.transform(image)
            mask = self.transform(mask)

        plt.subplot(6,1,3)
        plt.imshow(mask.squeeze(),cmap="gray")

        plt.subplot(6,1,4)
        plt.imshow(image.reshape(256,256,3))

        return image, mask
    
    
