#!/usr/bin/python3
import argparse
import os
import os.path
from os.path import join as pjn
import q
import skimage
import skimage.io
import numpy as np
import torch
import matplotlib
from omegaconf import OmegaConf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from labels_counts_utils import make_label2count_list
import dataset as dtst
from SDCNetCPU import SDCNet
from matplotlib import cm as CM

def get_predictions(cfg, model):
    """
    Visualize model predictions by running inference on the samples and
    create (and save to disk) compound images that contain the original
    image, the predicted count and the approximate predicted density map
    represented by the tensor `DIV2` (local count values for 16x16 patches).
    """
    model.eval()
    rgb_mean_train = dtst.calc_rgb_mean_train(cfg)

    vis_dir_name = 'visualized_predictions'

    imgs_dir = cfg.test.imgs_for_inference_dir
    flist = sorted(os.listdir(imgs_dir))
    fpaths = [pjn(imgs_dir, fname) for fname in flist]
    fpaths = [fpath for fpath in fpaths if os.path.isfile(fpath)]
    if fpaths:
        print('  <image_filename>: <total_count>')
        if not os.path.isdir(vis_dir_name):
            os.mkdir(vis_dir_name)

    for fpath in fpaths:
        fname = os.path.split(fpath)[1]
        try:
            img_np = np.array(skimage.io.imread(fpath))
        except:
            if os.path.isfile(fpath):
                print(f"  {fname}: cannot read this file as an image")
            continue

        h, w = img_np.shape[:2]
        assert img_np.ndim == 2 or img_np.shape[2] == 3
        if img_np.ndim == 2:
            # monochrome image, only one channel; expand to 3 channels
            img_np = np.repeat(np.expand_dims(img_np, axis=2), 3, axis=2)

        normd_image = (img_np - rgb_mean_train) / 255
        sample = {
            'image_bname': os.path.splitext(fname)[0],
            'image': normd_image.astype(np.float32).transpose((2, 0, 1)),
            'dmap': np.zeros((h, w)),
            'num_annot_headpoints': 0,
        }
        sample_pd = dtst.PadToMultipleOf64()(sample)
        sample_pd['image'] = np.expand_dims(sample_pd['image'], axis=0)
        image_normd = torch.from_numpy(sample_pd['image']).float().to('cpu')
        *cls_logits_list, DIV2, U1, U2, W1, W2 = model(image_normd)
        pred_count_16x16_blocks = DIV2.cpu().detach().numpy()
        pred = pred_count_16x16_blocks.squeeze(0).squeeze(0)
        pred_count = pred.sum()
        print(f"  {fname}: {pred_count:.1f}")
        #
        if cfg.test.visualize:
            bleach_coef = 0.4
            image_bleached = (255 - (255 - img_np) * bleach_coef).astype(int)
            bname = sample['image_bname']
            w, h = 1024, 768
            dpi = 100
            fs_h = int(h / dpi)
            fs_w = int(w / dpi)
            fs = (fs_w, fs_h)
            fig, axs = plt.subplots(figsize=fs, dpi=dpi, ncols=2)
            axs[0].imshow(img_np, vmin=0, vmax=255)
            axs[0].set_title('Original image')
            extent = (0, w, 0, h)
            axs[1].imshow(image_bleached, vmin=0, vmax=255, extent=extent)
            pred_im = axs[1].imshow(
                pred,
                cmap=CM.jet, alpha=0.5,
                extent=extent, interpolation='nearest')
            axs[1].set_title(f"Predicted total count = {pred_count:.1f}")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pred_im, cax=cax)
            fig.tight_layout()
            plt.savefig(pjn(vis_dir_name, bname + '.png'), bbox_inches='tight')
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate density maps ')
    parser.add_argument('--imgs_for_inference_dir',
                        help='directory where images are located.', required=True)
    parser.add_argument('--num_intervals', type=int,
                        help='max size CMax for counter', required=True)
    parser.add_argument('--trained_ckpt_for_inference',
                        help='location of model for evaluation', required=True)
    parser.add_argument('--visualize', type=bool, default=False,
                        help='Determine if visualized predictions are saved')
    args = parser.parse_args()
    return args


def main():
    cfg = OmegaConf.load('conf\config_train_val_test.yaml')
    cfg_args = parse_args()
    cfg.test.imgs_for_inference_dir = cfg_args.imgs_for_inference_dir
    cfg.num_intervals = cfg_args.num_intervals
    cfg.test.trained_ckpt_for_inference = cfg_args.trained_ckpt_for_inference
    cfg.test.visualize = cfg_args.visualize
    ckpt_path = cfg.test.trained_ckpt_for_inference
    print(f"  Running inference using checkpoint "
          f"'{cfg.test.trained_ckpt_for_inference}'")
    loaded_struct = torch.load(ckpt_path)

    interval_bounds, label2count_list = make_label2count_list(cfg)
    model = SDCNet(
        label2count_list,
        cfg.model.supervised,
        load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)

    #if not cfg.resources.disable_cuda and torch.cuda.is_available():
    device = torch.device("cpu")
    model = model.to(device)

    if cfg.test.visualize:
        vis_dir_print = pjn(
            os.path.relpath(os.getcwd()),
            'visualized_predictions')
        print(f"  Visualized predictions are being saved to '{vis_dir_print}'")

    get_predictions(cfg, model)

    # if args_dict['export']:
    #     batch_size = 1
    #     x = torch.randn(batch_size, 3, 64*1, 64*1, requires_grad=False).cuda()
    #     #
    #     p1, ext = os.path.splitext(args_dict['checkpoint'])
    #     torch.onnx.export(model, x, p1 + ".onnx", opset_version=11)
    #     #
    #     traced_script_module = torch.jit.trace(model, x)
    #     traced_script_module.save(p1 + "_jit_trace.pt")
    #     #
    #     script_module = torch.jit.script(model)
    #     script_module.save(p1 + "_jit_script.pt")


if __name__ == "__main__":
    main()
