#
# Copyright (C) 2023, Inria
# GRAPHDECO research group
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d


# =========================
# TSDF → POINT CLOUD
# =========================
def tsdf_to_pointcloud(volume):
    pcd = volume.extract_point_cloud()
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        raise ValueError("Empty TSDF point cloud")

    return pts


# =========================
# DSM / DOM
# =========================
def compute_dsm_dom(points, res=0.05):

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = np.arange(x_min, x_max + res, res)
    y_bins = np.arange(y_min, y_max + res, res)

    dsm = np.full((len(y_bins), len(x_bins)), -np.inf, dtype=np.float32)
    dom_sum = np.zeros_like(dsm, dtype=np.float32)
    dom_cnt = np.zeros_like(dsm, dtype=np.float32)

    xi = np.searchsorted(x_bins, x) - 1
    yi = np.searchsorted(y_bins, y) - 1

    for i in range(len(points)):
        ix = xi[i]
        iy = yi[i]

        if 0 <= ix < dsm.shape[1] and 0 <= iy < dsm.shape[0]:
            zi = z[i]

            # DSM (max)
            if zi > dsm[iy, ix]:
                dsm[iy, ix] = zi

            # DOM (mean)
            dom_sum[iy, ix] += zi
            dom_cnt[iy, ix] += 1

    dom = dom_sum / (dom_cnt + 1e-8)
    dsm[dsm == -np.inf] = np.nan

    return dsm, dom, x_min, y_min, res


# =========================
# GT ALIGNMENT
# =========================
def align_with_gt(pred, gt):

    mask = np.isfinite(gt) & np.isfinite(pred)

    if np.sum(mask) < 10:
        return pred

    p = pred[mask]
    g = gt[mask]

    scale = np.std(g) / (np.std(p) + 1e-8)
    shift = np.mean(g) - scale * np.mean(p)

    return pred * scale + shift


# =========================
# RENDER + TSDF FUSION
# =========================
def render_set(model_path, name, iteration, views, scene, gaussians,
               pipeline, background, volume=None,
               max_depth=5.0):

    for idx, view in enumerate(tqdm(views, desc="Rendering")):

        out = render(view, gaussians, pipeline, background)

        depth = out["plane_depth"].squeeze().detach().cpu().numpy()

        if volume is None:
            continue

        H, W = depth.shape

        pose = np.identity(4)
        pose[:3, :3] = view.R.transpose(-1, -2).cpu().numpy()
        pose[:3, 3] = view.T.cpu().numpy()

        color = o3d.geometry.Image(
            (out["render"].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        )

        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, view.Fx, view.Fy, view.Cx, view.Cy
        )

        volume.integrate(rgbd, intrinsic, pose)


# =========================
# MAIN PIPELINE
# =========================
def render_sets(dataset, iteration, pipeline,
                 skip_train, skip_test,
                 max_depth, voxel_size,
                 use_gt_alignment=True):

    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg = [1,1,1] if dataset.white_background else [0,0,0]
        background = torch.tensor(bg, device="cuda")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # =========================
        # TRAIN (build TSDF)
        # =========================
        if not skip_train:

            render_set(dataset.model_path, "train",
                       scene.loaded_iter,
                       scene.getTrainCameras(),
                       scene, gaussians,
                       pipeline, background,
                       volume=volume,
                       max_depth=max_depth)

            print("Extract TSDF point cloud...")

            points = tsdf_to_pointcloud(volume)

            print("Computing DSM / DOM...")

            dsm, dom, x_min, y_min, res = compute_dsm_dom(points)

            # =========================
            # LOAD GT (optional)
            # =========================
            gt_dsm_path = os.path.join(dataset.model_path, "gt_dsm.npy")
            gt_dom_path = os.path.join(dataset.model_path, "gt_dom.npy")

            if use_gt_alignment and os.path.exists(gt_dsm_path):

                gt_dsm = np.load(gt_dsm_path)
                gt_dom = np.load(gt_dom_path)

                print("Aligning with GT...")

                dsm = align_with_gt(dsm, gt_dsm)
                dom = align_with_gt(dom, gt_dom)

            # =========================
            # SAVE OUTPUT
            # =========================
            out_dir = os.path.join(dataset.model_path, "dsm_dom")
            os.makedirs(out_dir, exist_ok=True)

            np.save(os.path.join(out_dir, "dsm.npy"), dsm)
            np.save(os.path.join(out_dir, "dom.npy"), dom)

            np.save(os.path.join(out_dir, "grid.npy"),
                    np.array([x_min, y_min, res]))

            print("Saved DSM/DOM to:", out_dir)

        # =========================
        # TEST (optional render only)
        # =========================
        if not skip_test:
            render_set(dataset.model_path, "test",
                       scene.loaded_iter,
                       scene.getTestCameras(),
                       scene, gaussians,
                       pipeline, background)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":

    torch.set_num_threads(8)

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--use_gt_alignment", action="store_true")
    parser.add_argument("--gt_dsm_path", default=None, type=str,
                    help="Path to GT DSM GeoTIFF")

    parser.add_argument("--gt_dom_path", default=None, type=str,
                    help="Path to GT DOM GeoTIFF")
    args = get_combined_args(parser)

    print("Rendering:", args.model_path)

    safe_state(False)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.max_depth,
        args.voxel_size,
        args.use_gt_alignment
    )