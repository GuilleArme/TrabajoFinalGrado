import argparse
import sys
import os
import os.path
import glob
from sys import exit as e
from os.path import join as pjn
import q

import math
import random
from collections import OrderedDict
import hydra
import tempfile
import time
from tqdm import tqdm
from multiprocessing import Process, Manager
from omegaconf import OmegaConf

import skimage
import skimage.io
from PIL import Image
import numpy as np
import scipy.io
from sklearn.neighbors import NearestNeighbors

# skimage throws a lot of warnings like
# /usr/local/lib/python3.5/dist-packages/skimage/io/_io.py:141:
#    UserWarning: fname.png is a low contrast image.
# Let's suppress them.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def check_consist_imgs_annots(imgs_dir, annot_dir):
    """
    Check the correspondence between the images and annotations.
    `imgs_dir` must contain '*.jpg' files, `annot_dir` must contain 
    the same number of '*.mat' files, the basenames of the files must
    differ only by the leading 'GT_' substring
    (in the '*.mat' file basenames).
    """
    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError(f"images directory '{imgs_dir}' is not found")
    
    jpg_files = sorted(glob.glob(pjn(imgs_dir, "*.jpg")))
    if not jpg_files:
        raise FileNotFoundError(
            f"directory '{imgs_dir}' contains no '*.jpg' files")

    jpg_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in jpg_files]

    if not os.path.isdir(annot_dir):
        raise FileNotFoundError(
            f"annotations directory '{annot_dir}' is not found")
    
    mat_files = sorted(glob.glob(pjn(annot_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"directory '{annot_dir}' contains no '*.mat' files")

    mat_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in mat_files]

    assert len(jpg_basenames) == len(mat_basenames), \
        "different number of image files and annotation files"

    corresp_basenames = [
        (bn_mat == "GT_" + bn_jpg)
        for bn_jpg, bn_mat in zip(jpg_basenames, mat_basenames)
    ]
    assert all(corresp_basenames), \
        "image and ground truth file basenames are not consistent"


def get_headpoints_dict(annot_dir):
    """
    Load the '*.mat' files from the annotation directory
    and convert their contents (coordinates of the head points)
    to the {basename: numpy ndarray} dictionary (OrderedDict()).
    """
    mat_files = sorted(glob.glob(pjn(annot_dir, "*.mat")))
    mat_basenames = [
        os.path.splitext(os.path.split(f)[1])[0] for f in mat_files]
    
    basename2headpoints_dict = OrderedDict()
    
    for f, bn in zip(mat_files, mat_basenames):
        mat = scipy.io.loadmat(f)
        numpy_void_obj = mat['image_info'][0][0][0][0]
        headpoints = numpy_void_obj[0]
        num_headpoints = numpy_void_obj[1][0][0]
        assert headpoints.shape[0] == num_headpoints, \
            "number of headpoints entries != specified " \
            "total number of headpoints"
        assert headpoints.shape[1] == 2, \
            "<2 or >2 coordinate values for one headpoint entry"
        basename2headpoints_dict[bn] = headpoints

    return basename2headpoints_dict


def get_one_head_gaussian(side_len, r, sigma):
    """
    Pre-calculate the values of the Gaussian function in the 
    specified spatial square region (in the points with integer
    coordinates).

    Args:
        side_len: side of the square inside which the Gaussian values
            should be calculated.
        r: the Gaussian is cenetered in the point (r, r).
        sigma: the Gaussian RMS width.

    Returns:
        Two-dimensional array containing the Gaussian function values.
    """
    one_head_gaussian = np.zeros((side_len + 2, side_len + 2))
    for i in range(side_len + 1):
        for j in range(side_len + 1):
            t = -(i - r + 1)**2 - (j - r + 1)**2
            t /= 2 * sigma**2
            one_head_gaussian[i, j] = math.exp(t) / (sigma**2 * 2*math.pi)

    return one_head_gaussian


def generate_density_maps(
        basename2headpoints_dict_part,
        basename2dmap_dict,
        imgs_dir,
        cfg):
    """
    Generate the density maps. They are the sums of normalized Gaussian
    functions centered at the people's head points.

    Implementation details: for each headpoint, a Gaussian 2d array is 
    constructed. It is clipped to the image boundaries if needed.
    The remaining part of the array is normalized such that the values 
    corresponding to one head sum to 1. Density map is the sum of all
    (normalized) Gaussians for all heads. Total sum of the density map
    values is equal to the number of annotated heads.

    The Gaussian RMS width is adaptive. Consider one head point. 
    `cfg.knn` (3 by default) nearest neighbors for that
    point are found and average distance to them is calculated. That average 
    distance is capped by the constant pre-defined value 
    `cfg.max_knn_avg_dist` (50.0 by default) for ShanghaiTech
    part_B dataset (the average distance is not capped for ShanghaiTech
    part_A). The `sigma` (Gaussian RMS width) is the product of the average
    distance and a pre-defined constant `cfg.sigma_coef`
    (0.3 by default).
    The sum of the density map values across the whole image area must be
    equal to the number of annotated heads.

    Args:
        basename2headpoints_dict_part: Part of the dictionary containing 
            the mapping between basenames and 2d headpoints numpy ndarrays
            (returned by get_headpoints_dict()).
        basename2dmap_dict: Dictionary that will be filled with the mapping
            between the basenames and density maps (each density map has
            the same height and width as the corresponding image).
        imgs_dir: Directory containing images (only their width and hight
            values are needed).
        cfg: the global configuration (hydra).

    Returns:
    """
    side_len = cfg.sqr_side
    r = 1 + side_len // 2

    for bn, points in basename2headpoints_dict_part.items():
        img_fpath = pjn(imgs_dir, bn[3:] + '.jpg')
        # bn[3:] means skipping the initial 'GT_' from the basename
        w, h = Image.open(img_fpath).size

        ## points.shape == (num_heads, 2)
        # `points` contains pairs (coord_along_w, coord_along_h) as floats

        neigh = NearestNeighbors(
            n_neighbors=(1 + cfg.knn),
            # each point ^ is the closest one to itself
            metric='euclidean',
            n_jobs=-1)
        neigh.fit(points)
        knn_dists, knn_inds = neigh.kneighbors(points)

        dmap = np.zeros((h, w))

        for j, w_h_pair in enumerate(points):
            knn_dist_avg = knn_dists[j, 1:].mean()
            # excluding the point itself^ (zero distance)
            max_d = cfg.max_knn_avg_dist

            #Modificado para no condicionar la limitación de max distance a un dataset específico
            if (max_d > 0) and (knn_dist_avg > max_d) :
                knn_dist_avg = max_d

            #modificado para permitir valores fijos de sigma
            if cfg.sigma_fixed:
                sigma = cfg.sigma
            else:
                sigma = cfg.sigma_coef * knn_dist_avg
            one_head_gaussian = get_one_head_gaussian(side_len, r, sigma)
            one_head_sum = np.sum(one_head_gaussian)

            w_center = int(w_h_pair[0])
            h_center = int(w_h_pair[1])
            ##
            left = max(0, w_center - r)
            right = min(w, w_center + r)
            up = max(0, h_center - r)
            down = min(h, h_center + r)
            # ^ clip to the image boundaries
            ##
            left_g = left - w_center + r
            right_g = right - w_center + r
            up_g = up - h_center + r
            down_g = down - h_center + r
            # ^ one_head_gaussian must also be clipped to the image boundaries
            # after placing the gaussian center to the required location
            ##
            one_head_gaus_subset = one_head_gaussian[up_g:down_g,
                                                     left_g:right_g]
            dmap[up:down, left:right] += \
                one_head_gaus_subset / np.sum(one_head_gaussian)
            # seems that xhp uses division by np.sum(one_head_gaussian) ^ here
            # instead of np.sum(one_head_gaus_subset)!

        basename2dmap_dict[f"{bn}/density_map"] = dmap
        basename2dmap_dict[f"{bn}/num_annot_headpoints"] = points.shape[0]
        #print(np.sum(dmap), points.shape[0])
        integral_eq_annot_num = (int(round(np.sum(dmap))) == points.shape[0])
        #assert integral_eq_annot_num
        # ^ Integral (sum) over the density map must be equal
        # to the annotated number of people
        # if dmap is normalized by np.sum(one_head_gaus_subset).
        # It will not hold if dmap is normalized by np.sum(one_head_gaussian).


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    

def generate_density_maps_paral(basename2headpoints_dict, imgs_dir, cfg):
    basenames = list(basename2headpoints_dict.keys())
    random.shuffle(basenames)
    
    chunk_size = int(math.ceil(len(basenames) / cfg.num_proc))
    basenames_chunks = chunks(basenames, chunk_size)
    
    manager = Manager()
    basename2dmap_dict = manager.dict()
    procs = []
    
    for basenames_chunk in basenames_chunks:
        bn2hp_dict_part = {
            bn: basename2headpoints_dict[bn] for bn in basenames_chunk}
        p = Process(
            target=generate_density_maps, 
            args=(bn2hp_dict_part, basename2dmap_dict, imgs_dir, cfg)
        )
        p.start()
        procs.append(p)
    
    for p in procs:
        p.join()
    
    return basename2dmap_dict
    
def parse_args():
    parser = argparse.ArgumentParser(description='Generate density maps ')
    parser.add_argument('--dataset_rootdir',
                        help='directory where dataset is located.')
    parser.add_argument('--knn', type=int, default=3,
                        help='number of nearest neigbors to calculate distance to')
    parser.add_argument('--max_knn_avg_dist', type=float, default=0.0,
                        help='average knn distance is set to this value if exceeds this value. Set to 0 if no max limit wanted')
    parser.add_argument('--sigma_fixed', type=bool, default=False,
                        help='Determine if sigma value is fixed or relative to Knn_avg_dist')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='value for sigma if sigma is fixed')
    parser.add_argument('--sigma_coef', type=float, default=0.3,
                        help='Gaussian\'s sigma = sigma_coef * knn_avg_dist')
    parser.add_argument('--sqr_side', type=float, default=40,
                        help='Gaussian values are set to 0.0 outside of [-sqr_side/2, +sqr_side/2]')
    parser.add_argument('--num_proc', type=int, default=8,
                        help='number of processes to use')
    args = parser.parse_args()
    return args


def main():
    #cfg = OmegaConf.load('conf\config_density_maps')
    cfg = parse_args()
    for t in ['train_data', 'test_data']:
        the_dir = pjn(
            cfg.dataset_rootdir,
            t)
        imgs_dir = pjn(the_dir, "images")
        annot_dir = pjn(the_dir, "ground-truth")
        check_consist_imgs_annots(imgs_dir, annot_dir)

        bn2points_dict = get_headpoints_dict(annot_dir)

        print(f"  Calling generate_density_maps_paral()  "
              f"{t[:-5]}... ",
              end='',
              flush=True)

        if cfg.num_proc == 1:
            # single-process mode for debugging
            dmaps_dict = {}
            generate_density_maps(bn2points_dict, dmaps_dict, imgs_dir, cfg)
        else:
            dmaps_dict = generate_density_maps_paral(
                bn2points_dict,
                imgs_dir,
                cfg)

        print(f"Done")
        
        npz_name = f"density_maps_S{str(int(cfg.sigma))}_{t[:-5]}.npz"
        print(f"  Saving the file {npz_name}")
        np.savez(pjn(cfg.dataset_rootdir, npz_name), **dmaps_dict)


if __name__ == "__main__":
    main()
