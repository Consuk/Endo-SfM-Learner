from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)
        #SCARED Dataset
        self.K = np.array([[262.4, 0, 160],
                           [0, 261.12, 128],
                           [0, 0, 1]], dtype=np.float32)

                       
        
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()

        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 1

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        #SCATER
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "data", f_str)
        
            
        return image_path

class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)
        if self.folder_type == "sequence":
            self.train_entries = self.collect_train_entries()


    def get_image_path(self, folder, frame_index, side):
        #SCATER
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "data", f_str)
        
        #COLON10k
        #f_str=str(frame_index) + self.img_ext
        #image_path = os.path.join(self.data_path, folder, f_str)
            
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

        def collect_train_entries(self):
            train_entries = []

            print(f"[INFO] Using folder_type='sequence', collecting samples from: {self.filenames[0].split()[0]} ...")

            for entry in self.filenames:
                folder = entry.split()[0]
                data_path = os.path.join(self.data_path, folder, "data")

                if not os.path.exists(data_path):
                    print(f"[WARNING] No data folder found at {data_path}")
                    continue

                # Obtener y ordenar las imágenes disponibles
                images = sorted([f for f in os.listdir(data_path) if f.endswith(self.img_ext)])
                num_imgs = len(images)

                for i in range(1, num_imgs - 1):  # frame_idx = i; puede acceder a i-1 y i+1
                    train_entries.append((folder, i, 'l'))

                print(f"[INFO] {folder}: {num_imgs} images -> {len(images)-2} sequence samples")

            print(f"[INFO] Total sequence samples collected: {len(train_entries)}")
            return train_entries


