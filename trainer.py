from __future__ import absolute_import, division, print_function

import random
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from layers import *

import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from ewma import EWMA

torch.backends.cudnn.benchmark = True

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.threshold = EWMA(momentum=0.98)
        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        assert 0<=self.opt.rotation_aug_proportion<=1, "proportion must be between 0 and 1"
        if self.opt.anomaly_mask == True or self.opt.supervised_loss == True or self.opt.rotation_augment == True:
            assert self.opt.load_teacher_network == True, "to use TGAM or RD, load teacher model first"

        self.num_rotated_imgs = int(self.opt.rotation_aug_proportion * self.opt.batch_size)
        self.num_rotated_imgs = max(1,self.num_rotated_imgs)

        self.num_scales = len(self.opt.scales)
        self.opt.frame_ids = [0,-1,1]
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        self.use_pose_net = True
        self.gaussian_blur_transform = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0))

        self.models["encoder"] = networks.LiteMono(model=self.opt.stu_model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())


        if self.opt.pose_model_type == "separate_resnet":
            self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models_pose["pose_encoder"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

            self.models_pose["pose"] = networks.PoseDecoder(
                self.models_pose["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif self.opt.pose_model_type == "shared":
            self.models_pose["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc, self.num_pose_frames)

        elif self.opt.pose_model_type == "posecnn":
            self.models_pose["pose"] = networks.PoseCNN(
                self.num_input_frames if self.opt.pose_model_input == "all" else 2)

        self.models_pose["pose"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.opt.lr[2]),
                            T_mul=1,
                            eta_min=self.opt.lr[1],
                            last_epoch=-1,
                            max_lr=self.opt.lr[0],
                            warmup_steps=0,
                            gamma=0.9
                            )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.load_teacher_network:
            self.teacher_models = {}
            self.teacher_models["encoder"] = networks.LiteMono(model=self.opt.model, drop_path_rate=self.opt.drop_path,
                                                               width=self.opt.width,height=self.opt.height)
            self.teacher_models["depth"] = networks.DepthDecoder(self.teacher_models["encoder"].num_ch_enc, self.opt.scales)
            self.teacher_models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
            self.teacher_models["pose"] = networks.PoseDecoder(self.teacher_models["pose_encoder"].num_ch_enc,
                                                               num_input_features=1,  num_frames_to_predict_for=2)
            self.load_teacher_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"FLSea": datasets.FLseaDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.tiff' if self.opt.tiff else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, 4,
            is_train=True, img_ext=img_ext, load_enh_img=self.opt.load_enhanced_img)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # val_dataset = self.dataset(
        #     self.opt.data_path, val_filenames, self.opt.height, self.opt.width, 4, is_train=False, img_ext=img_ext)
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_filenames)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        self.model_pose_lr_scheduler.step()

        loss_of_epoch = {'reprojection_loss': 0,
                         'supervised_loss':0,
                         'rot_distillation_loss':0}

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            for key in loss_of_epoch.keys():
                loss_of_epoch[key] += losses[key]

            self.model_optimizer.zero_grad()
            self.model_pose_optimizer.zero_grad()

            losses["total_loss"].backward()
            self.model_optimizer.step()
            self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            to_log = batch_idx % self.opt.log_frequency == 0

            if to_log:
                self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)
                # if "depth_gt" in inputs:
                #     self.val()

            self.step += 1

        self.log("train", inputs, outputs, loss_of_epoch)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

         # we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)
        rotation_outputs = {}

        if self.opt.rotation_augment:
            assert -180 <= self.opt.rotation_angle_range[0] and self.opt.rotation_angle_range[1] <= 180, \
                "rotation angle is not within the range [-180, 180]."
            self.rotation_angle = random.randint(self.opt.rotation_angle_range[0], self.opt.rotation_angle_range[1])
            rotation_outputs = self.process_rotated_inputs(inputs,self.rotation_angle)

        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs, rotation_outputs)

        return outputs, losses

    def process_rotated_inputs(self, inputs, angle):
        rotation_inputs = inputs["color_aug", 0, 0][:self.num_rotated_imgs, :, :, :]
        rotation_inputs = Edgeless_Rotate(rotation_inputs, angle)
        features = self.models["encoder"](rotation_inputs)
        rotation_outputs = self.models["depth"](features)
        return  rotation_outputs

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if self.opt.load_enhanced_img:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color_enh", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_supervised_loss(self, pred, target, mask):
        """Compute the error between predicted depth and pseudo labels
        """
        loss = torch.log(torch.abs(target - pred) + 1)
        loss = loss * mask
        supervised_loss = loss.sum() / (mask.sum() + 1e-7)

        return supervised_loss

    def compute_silog_loss(self, depth_est, depth_gt):
        """Compute scale invariant loss
        """
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - 1.0 * (d.mean() ** 2))

    def compute_correlation_coefficient(self, pred, target):
        x = pred.reshape(pred.shape[0], -1)
        y = target.reshape(target.shape[0], -1)

        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)

        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)
        std_y = torch.std(y, dim=1, keepdim=True, unbiased=False)

        r = torch.mean((x - mean_x) * (y - mean_y), dim=1, keepdim=True) / (std_x * std_y)
        return r.mean()

    def compute_consistency_map(self, inputs, outputs):
        """compute inter-frame 3D consistency
        """
        consistency_maps = []
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            T_inverse = torch.inverse(T)
            cam_points_t2c = self.backproject_depth[0](outputs[("depth", 0, 0)],inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](cam_points_t2c, inputs[("K", 0)], T)


            depth_source = F.grid_sample(outputs[("depth", frame_id, 0)], pix_coords,
                                         padding_mode="border", align_corners=True)  # 反向warping
            cam_points_ssc = self.backproject_depth[0](depth_source, inputs[("inv_K", 0)])
            cam_points_s2c = torch.matmul(T_inverse, cam_points_ssc)

            distances = torch.abs(cam_points_t2c - cam_points_s2c)[:, :3, :]
            distance = torch.sum(distances, dim=1)
            consistency_maps.append(distance.reshape(self.opt.batch_size, 1 ,self.opt.height, self.opt.width))
        consistency_maps = torch.cat(consistency_maps, dim=1)
        min_consistency, idxs = torch.min(consistency_maps, dim=1, keepdim=True)  # 寻找最小值
        consistency_map = min_consistency
        return consistency_map

    def consistency_check(self, inputs, teacher_outputs):
        with torch.no_grad():
            output_former = self.teacher_models["depth"](self.teacher_models["encoder"](inputs["color", -1, 0]))
            output_later = self.teacher_models["depth"](self.teacher_models["encoder"](inputs["color", 1, 0]))

            teacher_outputs[('d', 0)] = teacher_outputs[("disp", 0)]
            teacher_outputs[('d', -1)] = output_former[("disp", 0)]
            teacher_outputs[('d', 1)] = output_later[("disp", 0)]

            for i in self.opt.frame_ids:
                disp = teacher_outputs[("d", i)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                teacher_outputs[("depth", i, 0)] = depth
            consistency_map = self.compute_consistency_map(inputs, teacher_outputs)
            consistency_mask = consistency_map < self.opt.depth_consistency_threshold

        return consistency_mask

    def compute_losses(self, inputs, outputs, rotation_outputs):
        """Compute the reprojection and other losses for a minibatch
        """
        losses = {"reprojection_loss":0,
                  "supervised_loss":0,
                  "smooth_loss":0,
                  "rot_distillation_loss":0,
                  "total_loss":0}

        # generate teacher_mask
        if self.opt.load_teacher_network:
            outputs["mask"], teacher_outputs = self.generate_teacher_mask(inputs)
            pseudo_label = teacher_outputs[('disp', 0)].detach()
            # pseudo_depth = teacher_outputs[("depth", 0, 0)].detach()

        # compute consistency map
        if self.opt.load_teacher_network and self.opt.supervised_loss and self.opt.depth_consistency_check:
            outputs["consistency_mask"] = self.consistency_check(inputs, outputs)

        # generate rotated pseudo_label
        if self.opt.load_teacher_network and self.opt.rotation_augment:
            rotation_label = Edgeless_Rotate(pseudo_label[:self.num_rotated_imgs,:,:,:],self.rotation_angle)

        for scale in self.opt.scales:
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            if self.opt.load_enhanced_img:
                target = inputs[("color_enh", 0, source_scale)]
            else:
                target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if self.opt.load_enhanced_img:
                        pred = inputs[("color_enh", frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.opt.load_teacher_network and self.opt.anomaly_mask:
                mask_ratio = outputs["mask"].cpu().numpy().mean()
                to_optimise = outputs["mask"] * to_optimise / mask_ratio

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss_reprojection = to_optimise.mean()
            losses["reprojection_loss"] += loss_reprojection
            losses["total_loss"] += loss_reprojection

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            losses["smooth_loss"] += smooth_loss
            losses["total_loss"] += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            if self.opt.supervised_loss:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                if self.opt.depth_consistency_check:
                    supervised_loss = self.compute_supervised_loss(disp, pseudo_label, outputs["consistency_mask"])
                else:
                    supervised_loss = self.compute_supervised_loss(disp, pseudo_label, torch.ones_like(disp))
                losses["supervised_loss"] += supervised_loss
                declining_weight = (-self.epoch/self.opt.num_epochs) + 1
                losses["total_loss"] += self.opt.supervised_loss_weight * declining_weight * supervised_loss

            # compute rotated distillation loss
            if self.opt.rotation_augment:
                rot_disp = rotation_outputs[("disp", scale)]
                rot_disp = F.interpolate(rot_disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                # _, rot_depth = disp_to_depth(rot_disp, self.opt.min_depth, self.opt.max_depth)
                # rot_distillation_loss = self.compute_silog_loss(rot_depth, rotation_label)
                rot_distillation_loss = 1-self.compute_correlation_coefficient(rot_disp, rotation_label)
                losses["rot_distillation_loss"] += rot_distillation_loss
                losses["total_loss"] += self.opt.rotation_loss_weight * rot_distillation_loss

        losses["total_loss"] /= self.num_scales

        return losses

    def generate_teacher_mask(self, inputs):
        """generate Teacher-Guided Anomaly Mask
        """
        with torch.no_grad():
            features = self.teacher_models["encoder"](inputs["color", 0, 0])
            teacher_outputs = self.teacher_models["depth"](features)

            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.teacher_models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.teacher_models["pose"](pose_inputs)

                teacher_outputs[("axisangle", 0, f_i)] = axisangle
                teacher_outputs[("translation", 0, f_i)] = translation
                teacher_outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # warping
            disp = teacher_outputs[("disp", 0)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            teacher_outputs[("depth", 0, 0)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = teacher_outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)
                teacher_outputs[("sample", frame_id, 0)] = pix_coords

                teacher_outputs[("color", frame_id, 0)] = F.grid_sample(
                    inputs[("color", frame_id, 0)],
                    teacher_outputs[("sample", frame_id, 0)],
                    padding_mode="border", align_corners=True)

            reprojection_loss = []
            target = inputs["color", 0, 0]

            for frame_id in self.opt.frame_ids[1:]:
                pred = teacher_outputs[("color", frame_id, 0)]
                pred = self.gaussian_blur_transform(pred)
                target = self.gaussian_blur_transform(target)
                reprojection_loss.append(self.compute_reprojection_loss(pred, target))

            reprojection_loss = torch.cat(reprojection_loss, 1)
            loss, _ = torch.min(reprojection_loss, dim=1)
            # mask = loss < threshold
            quantile = torch.quantile(loss, self.opt.TGAM_quantile)
            self.threshold.update(quantile)
            mask = loss < self.threshold.running_val


        return mask,teacher_outputs

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]      # writer=SummaryWriter(train/val)
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.epoch)

        for j in range(min(4, self.opt.batch_size)):
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, 0, j),
                    inputs[("color", frame_id, 0)][j].data, self.epoch)
                if frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, 0, j),
                        outputs[("color", frame_id, 0)][j].data, self.epoch)

            writer.add_image(
                "disp_{}/{}".format(0, j),
                normalize_image(outputs[("disp", 0)][j]), self.epoch)

            if not self.opt.disable_automasking:
                writer.add_image(
                    "automask_{}/{}".format(0, j),
                    outputs["identity_selection/{}".format(0)][j][None, ...], self.epoch)

            if self.opt.anomaly_mask:
                writer.add_image(
                    "mask/{}".format(j),
                    outputs["mask"][j][None, ...], self.epoch)

            if self.opt.supervised_loss and self.opt.depth_consistency_check:
                writer.add_image(
                    "consistency_mask/{}".format(j),
                    outputs["consistency_mask"][j], self.epoch)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_teacher_model(self):
        """Load model(s) from disk
        """
        self.opt.load_teacher_weights_folder = os.path.expanduser(self.opt.load_teacher_weights_folder)

        assert os.path.isdir(self.opt.load_teacher_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_teacher_weights_folder)
        print("loading teacher model from folder {}".format(self.opt.load_teacher_weights_folder))

        for n in self.opt.models_to_load:
            path = os.path.join(self.opt.load_teacher_weights_folder, "{}.pth".format(n))

            model_dict = self.teacher_models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.teacher_models[n].load_state_dict(model_dict)
            self.teacher_models[n].cuda()
            self.teacher_models[n].eval()