import random

import numpy as np
from scipy.ndimage import rotate
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class CustomDataset(Dataset):
    def __init__(self, csv_path, augmentation=False):
        self.first_display = False
        # Transforms
        self.augmentations = augmentation
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        first_length = 0
        with open(csv_path, 'r') as file_object:
            lines = file_object.readlines()
            data_array = [line.strip().split(' ') for line in lines]
            self.data_info = np.array(data_array)
        # First column contains the image paths
            for i in range(len(data_array)):
                if first_length == 0:
                    first_length = len(data_array[i])
                else:
                    if first_length != len(data_array[i]):
                        print("Different motherfucker")
                        print(len(data_array[i]))

        self.image_arr = np.asarray(self.data_info[:, 0])
        self.label_arr = np.asarray(self.data_info[:, 1])

        temp_image_arr = []
        temp_label_arr = []
        for i, img_path in enumerate(self.image_arr):
            volume = np.load(img_path, allow_pickle=True)
            if volume.shape[0] == 14 and volume.shape[1] == 40 and volume.shape[2] == 40:
                temp_image_arr.append(img_path)
                temp_label_arr.append(self.label_arr[i])
        self.image_arr = temp_image_arr
        self.label_arr = temp_label_arr
        self.original_img_paths = {path: 0 for path in self.image_arr}
        # Second column is the labels
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        # Open image
        img_as_stack = np.load(single_image_name, allow_pickle=True)
        img_as_stack_orginal = img_as_stack[:, :, :]
        single_image_label = self.label_arr[index]

        img_as_stack_orginal[img_as_stack_orginal > 1000] = 1000
        img_as_stack_orginal = (img_as_stack_orginal[:,:,:] - (-1057))/(1000 - (-1057))
        # seed = np.random.randint(2147483647)
        if self.augmentations:
            random_angle = random.randint(-20, 20)
            img_as_stack_orginal = self.rotate_volume(img_as_stack_orginal, random_angle)

        return np.asarray(img_as_stack_orginal).astype(np.float32), int(single_image_label)

    def __len__(self):
        return self.data_len-1

    def rotate_volume(self, stack, angle):
        rotated_stack = []
        for j in range(len(stack)):
            rotated_patch = rotate(stack[j], angle, reshape=False)
            rotated_stack.append(rotated_patch)
        return rotated_stack

