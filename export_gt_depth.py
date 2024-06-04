from __future__ import absolute_import, division, print_function
import os
import numpy as np
import PIL.Image as pil
from utils import readlines


def export_gt_depths_kitti():

    splits = 'OUC'
    dataset_path = '/home/dingyl/datasets/FLsea'

    split_folder = os.path.join(os.path.dirname(__file__), "splits", splits)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(splits))

    gt_depths = []

    for line in lines:
        try:
            folder, frame_id = line.split()
        except:
            folder, frame_id, _ = line.split()
        frame_id = frame_id

        velo_filename = os.path.join(
            dataset_path,
            folder,
            "depth/{}_SeaErra_abs_depth.tif".format(frame_id))
            # "depth/{}.tif".format(frame_id))

        gt_depth = pil.open(velo_filename)
        gt_depth_np = np.array(gt_depth)
        gt_depth.close()
        gt_depths.append(gt_depth_np.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depth.npz")

    print("Saving to {}".format(splits))
    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
