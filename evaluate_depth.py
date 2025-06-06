from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
from thop import clever_format
from thop import profile

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 40

    test_set = 'stereo_test' if opt.eval_stereo else 'test_files'

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, f"{test_set}.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        if not opt.eval_stereo:
            dataset = datasets.FLseaDataset(opt.data_path, filenames,
                                            encoder_dict['height'], encoder_dict['width'],
                                            4, img_ext='.jpg')

        else:
            dataset = datasets.FLseaStereo(opt.data_path, filenames,encoder_dict['height'],
                                           encoder_dict['width'],4, img_ext='.tif')

        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=6,
                                pin_memory=False, drop_last=False)

        encoder = networks.LiteMono(model=opt.model,
                                    height=encoder_dict['height'],
                                    width=encoder_dict['width'])
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                t1 = time_sync()
                output = depth_decoder(encoder(input_color))
                t2 = time_sync()

                pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    if not opt.eval_stereo:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_stereo.npz")
    # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_challenge.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0.1

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    print("\n-> Done!")


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())
