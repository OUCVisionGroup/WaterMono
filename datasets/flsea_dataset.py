from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import skimage
import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class FLseaDataset(data.Dataset):
    """Flsea-VI Dataset
    Args:
        data_path
        filenames
        height
        width
        num_scales
        is_train
        img_ext
        load_depth
        load_enhanced_img
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False,
                 load_enh_img=False):
        super(FLseaDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.load_depth = load_depth
        self.load_enh_img = load_enh_img
        self.frame_idxs = [0, -1, 1] if self.is_train else [0]

        # Define the intrinsic matrix K
        self.K = np.array([[1.214, 0, 0.48, 0],
                           [0, 1.931, 0.44, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (968, 608)

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                #'color' refer to the original image
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_enh" in k:
                n, im, _ = k
                inputs[(n, im, 0)] = self.resize[0](inputs[(n, im, -1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k :
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_enh" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

    def del_useless(self, inputs, load_enh):
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            if load_enh:
                del inputs[("color_enh", i, -1)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("color_enh_aug", <frame_id>, <scale>)  for augmented enhanced colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = line[1]

        img_folder = 'imgs'
        inputs[("color", 0, -1)] = self.get_image(folder, frame_index, img_folder)
        if self.is_train:
            if len(line) == 3:
                if line[2] == 'start' or line[2] == 'end':
                    frame_index_before = int(self.filenames[index + 1 if line[2] == 'start' else index - 1].split()[1])
                    frame_index_after = int(self.filenames[index + 1 if line[2] == 'start' else index - 1].split()[1])
            else:
                frame_index_before = int(self.filenames[index - 1].split()[1])
                frame_index_after = int(self.filenames[index + 1].split()[1])

            inputs[("color", 1, -1)] = self.get_image(folder, frame_index_after, img_folder)
            inputs[("color", -1, -1)] = self.get_image(folder, frame_index_before, img_folder)

        if self.load_enh_img:
            img_folder = 'IEB'
            inputs[("color_enh", 0, -1)] = self.get_image(folder, frame_index, img_folder)
            inputs[("color_enh", 1, -1)] = self.get_image(folder, frame_index_after, img_folder)
            inputs[("color_enh", -1, -1)] = self.get_image(folder, frame_index_before, img_folder)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        self.del_useless(inputs, self.load_enh_img)


        # load gt_depth
        if self.load_depth:
            if self.check_depth():
                depth_gt = self.get_depth(folder, frame_index)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs


    def get_image(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, image_folder, f_str)
        color = self.loader(image_path)

        return color

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "depth/{}_SeaErra_abs_depth.tif".format(int(frame_index)))

        return os.path.isfile(velo_filename)


    def get_depth(self, folder, frame_index):
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "depth/{}_SeaErra_abs_depth.tif".format(int(frame_index)))

        depth_gt = Image.open(velo_filename)
        depth_gt = np.array(depth_gt)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        return depth_gt

class FLseaStereo(FLseaDataset):
    def __init__(self, *args, **kwargs):
        super(FLseaStereo, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        inputs = {}

        line = self.filenames[index].split()
        folder = line[0]
        frame_index = line[1]

        img_folder = 'imgs/LFT'
        inputs[("color", 0, -1)] = self.get_image(folder, frame_index, img_folder)
        self.preprocess(inputs)
        del inputs[("color", 0, -1)]

        return inputs

    def get_image(self, folder, frame_index, image_folder):
        f_str = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, image_folder, f_str)
        color = self.loader(image_path)

        return color

    def preprocess(self, inputs):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        for k in list(inputs):
            f = inputs[k]
            if "color" in k :
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)


    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "depth/LFT/{}_abs_depth.tif".format(frame_index))

        return os.path.isfile(velo_filename)

    def get_depth(self, folder, frame_index):
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "depth/LFT/{}_abs_depth.tif".format(frame_index))

        depth_gt = Image.open(velo_filename)
        depth_gt = np.array(depth_gt)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        return depth_gt

