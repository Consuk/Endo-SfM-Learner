# Solución óptima: modificar `SCAREDDataset.__init__` para expandir `self.filenames`

import os
import numpy as np
from .mono_dataset import MonoDataset

class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[262.4, 0, 160],
                           [0, 261.12, 128],
                           [0, 0, 1]], dtype=np.float32)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        # Expandimos cada entrada de carpeta a múltiples entradas por frame
        expanded_filenames = []
        for entry in self.filenames:
            folder = entry.strip()  # ya no incluye 'data'
            folder_path = os.path.join(self.data_path, folder, "data")
            if not os.path.isdir(folder_path):
                continue

            frame_files = sorted(os.listdir(folder_path))
            frame_indices = [
                int(os.path.splitext(f)[0])
                for f in frame_files if f.endswith(self.img_ext)
            ]
            frame_indices.sort()

            # Crear una entrada por frame (evitamos el último si usamos [0,1])
            for idx in frame_indices:
                if (idx - 1) in frame_indices and (idx + 1) in frame_indices:
                    expanded_filenames.append(f"{folder} {idx} l")


        self.filenames = expanded_filenames

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        from PIL import Image as pil
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def index_to_folder_and_frame_idx(self, index):
        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]
        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        f_str = f"{frame_index}{self.img_ext}"
        return os.path.join(self.data_path, folder, "data", f_str)


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = f"{frame_index}{self.img_ext}"
        return os.path.join(self.data_path, folder, "data", f_str)

    def get_depth(self, folder, frame_index, side, do_flip):
        import cv2
        f_str = f"scene_points{frame_index - 1:06d}.tiff"
        depth_path = os.path.join(
            self.data_path,
            folder,
            f"image_0{self.side_map[side]}/data/groundtruth",
            f_str)

        depth_gt = cv2.imread(depth_path, 3)[:, :, 0][:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt
