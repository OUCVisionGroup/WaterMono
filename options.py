from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class LiteMonoOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Lite-Mono options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default= "/home/dingyl/datasets/FLsea")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="./tmp")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="WaterMono")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="OUC")
        self.parser.add_argument("--model",
                                 type=str,
                                 help="which model to load",
                                 choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                 default="lite-mono")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay in AdamW",
                                 default=1e-2)
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.2)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="FLSea")
        self.parser.add_argument("--tiff",
                                 help="if set, trains from raw FLSea tiff files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=448)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=40.0)
        self.parser.add_argument("--profile",
                                 type=bool,
                                 help="profile once at the beginning of the training",
                                 default=True)

        # Our options
        self.parser.add_argument("--load_teacher_network",
                                 type=bool,
                                 help="if set, load teacher network to generate TGAM and pesudo labels",
                                 default=False)
        self.parser.add_argument("--load_teacher_weights_folder",
                                 type=str,
                                 help="name of the teacher model to load",
                                 default='./tmp/original_model_50e/models/weights_49')
        self.parser.add_argument("--anomaly_mask",
                                 type=bool,
                                 help="if set, use Teacher-Guided Anomaly Mask to filter Lpe",
                                 default=False)
        self.parser.add_argument("--TGAM_quantile",
                                 type=float,
                                 help="quantile of TGAM",
                                 default=0.95)
        self.parser.add_argument("--supervised_loss",
                                 type=bool,
                                 help="whether to use pseudo labels from the teacher network",
                                 default=False)
        self.parser.add_argument("--depth_consistency_check",
                                 type=bool,
                                 help="whether to use 3D consistency check to filter pseudo labels",
                                 default=False)
        self.parser.add_argument("--depth_consistency_threshold",
                                 type=float,
                                 help="the threshold of 3D consistency check",
                                 default=0.03)
        self.parser.add_argument("--supervised_loss_weight",
                                 type=float,
                                 help="weight of supervised loss from pseudo labels",
                                 default=0.25)
        self.parser.add_argument("--load_enhanced_img",
                                 type=bool,
                                 help="whether to use enhanced images to compute Lpe",
                                 default=True)

        # Settings related to rotated distillation
        self.parser.add_argument("--rotation_augment",
                                 type=float,
                                 help="whether to perform rotation augmentation",
                                 default=False)
        self.parser.add_argument("--rotation_aug_proportion",
                                 type=float,
                                 help="proportion of rotation augmentation, which should be between 0 and 1",
                                 default=0.1)
        self.parser.add_argument("--rotation_angle_range",
                                 type=float,
                                 help="range of rotation angles for rotation augmentation",
                                 default=[-15,15])
        self.parser.add_argument("--rotation_loss_weight",
                                 type=float,
                                 help="weight of the rotated distillation loss",
                                 default=0.025)

        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=28)
        self.parser.add_argument("--lr",
                                 nargs="+",
                                 type=float,
                                 help="learning rates of DepthNet and PoseNet. "
                                      "Initial learning rate, "
                                      "minimum learning rate, "
                                      "First cycle step size.",
                                 default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=50)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--mypretrain",
                                 type=str,
                                 help="if set, use my pretrained encoder")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=10)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="OUC",
                                 help="which split to run eval on")
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluate on FLSea-stereo",
                                 action="store_true")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
