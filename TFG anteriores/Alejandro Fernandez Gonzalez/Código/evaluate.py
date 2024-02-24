import argparse
import os
import os.path
from os.path import join as pjn
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from labels_counts_utils import make_label2count_list
import dataset as dtst
from SDCNet import SDCNet
import train


def visualize_predictions(cfg, model, data_loader, vis_dir):
    """
    Visualize model predictions by running inference on the samples and
    create (and save to disk) compound images that contain the original
    image, the ground truth count, the predicted count and the approximate
    predicted density map represented by the tensor `DIV2` (local count values
    for 16x16 patches).

    Args:
        cfg: the global configuration (hydra).
        model: the model to be run on the provided samples.
        data_loader: DataLoader object that provides samples.
        vis_dir: path to the directory where the compound images will be 
            placed.

    Returns: 
        None.

    Side effects:
        Creates compound images (for visualization purposes) in `vis_dir`.
    """
    if not os.path.isdir(vis_dir):
        os.mkdir(vis_dir)

    model.eval()
    rgb_mean_train = dtst.calc_rgb_mean_train(cfg)
    rgb_mean_train = rgb_mean_train[:, np.newaxis, np.newaxis]

    for sample in tqdm(data_loader):
        gt_count = float(sample['num_annot_headpoints'].numpy())
        image_normd = sample['image'].float().to('cuda')
        *cls_logits_list, DIV2, U1, U2, W1, W2 = model(image_normd)
        pred_count_16x16_blocks = DIV2.cpu().detach().numpy()
        pred = pred_count_16x16_blocks.squeeze(0).squeeze(0)
        pred_count = pred.sum()
        if gt_count >= 1:
            rel_err_percent = (pred_count - gt_count) / gt_count * 100
        else:
            rel_err_percent = 0.0
        image = image_normd.cpu().numpy() * 255 + rgb_mean_train
        image = image.astype(int).squeeze(0)
        # ^ shape = (3, w, h)
        image = np.transpose(image, (1, 2, 0))
        bleach_coef = 0.4
        image_bleached = (255 - (255 - image) * bleach_coef).astype(int)
        bname = sample['image_bname'][0]
        # ^ bnames appear in numerically sorted order ('IMG_1', 'IMG_2', ...)

        #h, w = image.shape[:2]
        w, h = 1024, 768
        # search for "Relationship between dpi and figure size"
        # on stackoverflow
        dpi = 100
        # ^ dpi = 100 by default
        fs_h = int(h / dpi)
        fs_w = int(w / dpi)
        fs = (fs_w, fs_h)
        fig, axs = plt.subplots(figsize=fs, dpi=dpi, ncols=2)
        axs[0].imshow(image, vmin=0, vmax=255)
        axs[0].set_title(
            f"{bname + '.jpg'} (Dataset)\n"
            f"Ground truth total count = {gt_count:.1f}")
        extent = (0, w, 0, h)
        axs[1].imshow(image_bleached, vmin=0, vmax=255, extent=extent)
        pred_im = axs[1].imshow(
            pred,
            cmap='bwr', alpha=0.5,
            extent=extent, interpolation='nearest')
        axs[1].set_title(
            f"Predicted total count = {pred_count:.1f}\n"
            f"(error = {pred_count - gt_count:+.1f}, "
            f"relative error = {rel_err_percent:+.1f}%)")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pred_im, cax=cax)
        fig.tight_layout()
        plt.savefig(pjn(vis_dir, bname + '.png'), bbox_inches='tight')
        plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser(description='Generate density maps ')
    parser.add_argument('--dataset_rootdir',
                        help='directory where dataset is located.', required=True)
    parser.add_argument('--densmaps_gt_npz',
                        help='density maps files location as ./density_maps_*.npz', required=True)
    parser.add_argument('--num_intervals', type=int,
                        help='max size CMax for counter', required=True)
    parser.add_argument('--trained_ckpt_for_inference',
                        help='location of model for evaluation', required=True)
    parser.add_argument('--visualize', type=bool, default=False,
                        help='Determine if visualized predictions are saved')
    args = parser.parse_args()
    return args

def main():
    cfg_args = parse_args()
    cfg = OmegaConf.load('conf/config_train_val_test.yaml')
    cfg.dataset.dataset_rootdir = cfg_args.dataset_rootdir
    cfg.dataset.densmaps_gt_npz = cfg_args.densmaps_gt_npz
    cfg.dataset.num_intervals = cfg_args.num_intervals
    cfg.test.trained_ckpt_for_inference = cfg_args.trained_ckpt_for_inference
    cfg.test.visualize = cfg_args.visualize
    print(f"  Evaluating the checkpoint "
          f"'{cfg.test.trained_ckpt_for_inference}'")
    loaded_struct = torch.load(
        cfg.test.trained_ckpt_for_inference)
    
    cfg.train.train_val_split = 0.0
    # ^ associate all of the train data with the val_loader below
    #   (do not split the train data into train + validation)
    _, val_loader, test_loader = train.get_dataloaders(cfg, (0, 1, 1))

    interval_bounds, label2count_list = make_label2count_list(cfg)
    model = SDCNet(
        label2count_list,
        cfg.model.supervised,
        load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)

    additional_cfg = {'device': None}
    if not cfg.resources.disable_cuda and torch.cuda.is_available():
        additional_cfg['device'] = torch.device('cuda')
        model = model.cuda()
    else:
        additional_cfg['device'] = torch.device('cpu')

    optimizer = None

    trainer = train.TrainManager(
        model,
        optimizer,
        cfg,
        additional_cfg,
        train_loader=None,
        val_loader=None,
    )

    print()
    datadir = cfg.dataset.dataset_rootdir
    print(f"  Evaluating on the (whole) train data and on the test data "
          f"(in '{datadir}')")

    mae_train, mse_train, mape_train = trainer.validate(val_loader)
    print(f"  Metrics on the (whole) train data: "
          f"MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, MAPE: {mape_train:.2f}")

    mae_test, mse_test, mape_test = trainer.validate(test_loader)
    print(f"  Metrics on the test data:          "
          f"MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, MAPE: {mape_test:.2f}")

    if cfg.test.visualize:
        vis_dir_name = f"visualized_test_predictions"
        vis_dir_print = vis_dir_name
        print(f"  Visualized predictions are being saved to '{vis_dir_print}':")
        visualize_predictions(cfg, model, test_loader, vis_dir_name)


def evaluate(dataset_rootdir, num_intervals, densmaps_gt_npz, trained_ckpt_for_inference):
    cfg = OmegaConf.load('conf/config_train_val_test.yaml')
    cfg.dataset.dataset_rootdir = dataset_rootdir
    cfg.dataset.num_intervals = num_intervals
    cfg.dataset.densmaps_gt_npz = densmaps_gt_npz
    cfg.test.trained_ckpt_for_inference = trained_ckpt_for_inference
    loaded_struct = torch.load(
        cfg.test.trained_ckpt_for_inference)

    cfg.train.train_val_split = 0.0
    # ^ associate all of the train data with the val_loader below
    #   (do not split the train data into train + validation)
    _, val_loader, test_loader = train.get_dataloaders(cfg, (0, 1, 1))

    interval_bounds, label2count_list = make_label2count_list(cfg)
    model = SDCNet(
        label2count_list,
        cfg.model.supervised,
        load_pretr_weights_vgg=False)
    model.load_state_dict(loaded_struct['model_state_dict'], strict=True)

    additional_cfg = {'device': None}
    if not cfg.resources.disable_cuda and torch.cuda.is_available():
        additional_cfg['device'] = torch.device('cuda')
        model = model.cuda()
    else:
        additional_cfg['device'] = torch.device('cpu')

    optimizer = None

    trainer = train.TrainManager(
        model,
        optimizer,
        cfg,
        additional_cfg,
        train_loader=None,
        val_loader=None,
    )


    datadir = cfg.dataset.dataset_rootdir

    mae_train, mse_train, mape_train = trainer.validate(val_loader)

    mae_test, mse_test, mape_test  = trainer.validate(test_loader)

    result=f"  Metrics for Cmax = {cfg.dataset.num_intervals} and Sigma = {cfg.dataset.densmaps_gt_npz[32:34]}: " \
           f"MAE (train/test): {mae_train:.2f}/{mae_test:.2f}, MSE (train/test): {mse_train:.2f}/{mse_test:.2f} "\
           f"MAPE (train/test): {mape_train:.2f}%/{mape_test:.2f}%\n"
    return result


if __name__ == "__main__":
    main()
