#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from lpips import LPIPS
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, edged_normal_loss
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from utils.loss_utils import VGGPerceptualLoss
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_single_view_dep_for_log = 0.0
    ema_single_view_embedding_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    depth_loss, normal_loss, geo_loss, ncc_loss, edged_loss, embedding_loss, hr_loss = None, None, None, None, None, None, None
    perc_loss = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration == 1:
            print(">> 阶段1：冻结颜色，只训练几何")

        # #在训练过程中动态控制 requires_grad
        if iteration < 3000:
            # 冻结颜色参数
            gaussians._features_dc.requires_grad_(False)
            gaussians._features_rest.requires_grad_(False)
        else:
            # 解冻颜色参数（只做一次）
            if not gaussians._features_dc.requires_grad:
                print(f">> 阶段2：解冻颜色参数（iteration={iteration}）")
                gaussians._features_dc.requires_grad_(True)
                gaussians._features_rest.requires_grad_(True)
        
        # Pick a random Camera
        if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.depth_weight_from_iter, return_depth_normal=iteration>opt.depth_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # hr_every_iters = getattr(opt, "hr_supervision_interval", 20)   # 每 N iter 做一次
        # hr_crop_size = getattr(opt, "hr_crop_size", 512)               # HR patch 大小
        # hr_lambda = getattr(opt, "hr_lambda", 0.5)                     # HR loss 权重（按需调）
        # use_hard_mining = getattr(opt, "hr_hard_mining", True)
        # if iteration < 7000:
        #     hr_lambda = 0.2
        #     use_hard_mining = False
        # else:
        #     hr_lambda = 1.2
        #     use_hard_mining = True
        # Loss
        # if iteration > 0:  # 或你想开始加 perceptual loss 的阶段
        #     perc_loss = vgg_loss_fn(image.unsqueeze(0), gt_image.unsqueeze(0))
        #     loss += 0.1 * perc_loss  
        # if (iteration % hr_every_iters == 0) and (hr_lambda > 0):
        #     try:
        #         # 1) 尝试直接读取原始 high-res 图像（cv2读取 BGR）
        #         #    组合路径： viewpoint_cam.image_path + viewpoint_cam.image_name （与你现有代码风格一致）
        #         img_path = os.path.join(viewpoint_cam.image_path, viewpoint_cam.image_name) if hasattr(viewpoint_cam, "image_path") else None
        #         if img_path and os.path.exists(img_path):
        #             full_img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #             full_img = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB)
        #             full_img = full_img.astype(np.float32) / 255.0
        #             # convert to torch CHW
        #             full_img_t = torch.from_numpy(full_img).permute(2, 0, 1).unsqueeze(0).to("cuda")
        #         else:
        #             # 备选：如果找不到文件，用已有的 gt_image（可能已被 downsample）
        #             # 将 gt_image 放大到更高分辨率作为 proxy（效果不如原始）
        #             full_img_t = F.interpolate(gt_image.unsqueeze(0), scale_factor=opt.hr_proxy_upscale if hasattr(opt, "hr_proxy_upscale") else 4, mode='bilinear', align_corners=False)
                
        #         _, _, H_full, W_full = full_img_t.shape

        #         # 2) 选择 HR patch 的左上角位置
        #         if use_hard_mining:
        #             # 用当前 render（低分辨率）上采样到 full 图尺寸，计算像素误差热图，取误差最大区域
        #             # render: shape (C,H_r,W_r)  -> upsample to (C,H_full,W_full)
        #             render_upsampled = F.interpolate(image.unsqueeze(0), size=(H_full, W_full), mode='bilinear', align_corners=False).squeeze(0)
        #             # 计算 per-pixel L1 error (灰度)
        #             render_gray = 0.2989 * render_upsampled[0] + 0.5870 * render_upsampled[1] + 0.1140 * render_upsampled[2]
        #             gt_gray = 0.2989 * full_img_t[0,0] + 0.5870 * full_img_t[0,1] + 0.1140 * full_img_t[0,2]
        #             err_map = (render_gray - gt_gray).abs()
        #             # 将 err_map 用窗口求和得到 patch-score map（快速卷积）
        #             # 使用平均池化模拟窗口求和
        #             kernel = hr_crop_size
        #             # 注意：当 full 图小于 crop size 时 fallback
        #             if H_full < hr_crop_size or W_full < hr_crop_size:
        #                 top, left = 0, 0
        #                 cur_crop_h, cur_crop_w = H_full, W_full
        #             else:
        #                 err_map_unsq = err_map.unsqueeze(0).unsqueeze(0)  # 1x1xH x W
        #                 score_map = F.avg_pool2d(err_map_unsq, kernel_size=hr_crop_size, stride=1, padding=0).squeeze()
        #                 # 找到 score_map 最大位置
        #                 max_idx = torch.argmax(score_map)
        #                 # convert to 2D coords in score_map (H_full-hr+1, W_full-hr+1)
        #                 score_H = H_full - hr_crop_size + 1
        #                 score_W = W_full - hr_crop_size + 1
        #                 top = (max_idx // score_W).item()
        #                 left = (max_idx % score_W).item()
        #                 cur_crop_h, cur_crop_w = hr_crop_size, hr_crop_size
        #         else:
        #             # 随机采样一个 patch
        #             if H_full < hr_crop_size or W_full < hr_crop_size:
        #                 top, left = 0, 0
        #                 cur_crop_h, cur_crop_w = H_full, W_full
        #             else:
        #                 top = np.random.randint(0, H_full - hr_crop_size + 1)
        #                 left = np.random.randint(0, W_full - hr_crop_size + 1)
        #                 cur_crop_h, cur_crop_w = hr_crop_size, hr_crop_size

        #         # 3) 切出 GT HR patch
        #         hr_gt_patch = full_img_t[:, :, top:top+cur_crop_h, left:left+cur_crop_w]  # 1x3xh x w

        #         # 4) 从当前 render（low res）上采样得到对应的 rendered patch（效率高）
        #         render_upsampled_full = F.interpolate(image.unsqueeze(0), size=(H_full, W_full), mode='bilinear', align_corners=False)
        #         hr_render_patch = render_upsampled_full[:, :, top:top+cur_crop_h, left:left+cur_crop_w]

        #         # 5) 计算 HR losses：L1 + gradient loss（Sobel-like），可加 SSIM/LPIPS（如有）
        #         # L1
        #         hr_l1 = F.l1_loss(hr_render_patch, hr_gt_patch)
        #         # gradient loss: 强化高频（Sobel kernels 实现）
        #         def image_gradient_loss(x, y):
        #             # x,y : 1x3xhxw
        #             sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
        #             sobel_y = sobel_x.transpose(-2,-1)
        #             gx_x = F.conv2d(x, sobel_x.repeat(3,1,1,1), groups=3, padding=1)
        #             gy_x = F.conv2d(x, sobel_y.repeat(3,1,1,1), groups=3, padding=1)
        #             gx_y = F.conv2d(y, sobel_x.repeat(3,1,1,1), groups=3, padding=1)
        #             gy_y = F.conv2d(y, sobel_y.repeat(3,1,1,1), groups=3, padding=1)
        #             return F.l1_loss(torch.sqrt(gx_x**2 + gy_x**2 + 1e-6), torch.sqrt(gx_y**2 + gy_y**2 + 1e-6))
        #         hr_grad = image_gradient_loss(hr_render_patch, hr_gt_patch)

        #         # combine
        #         hr_loss = hr_lambda * (hr_l1 + 0.5 * hr_grad)
        #         loss = loss + hr_loss

        #         # optional: log to tensorboard
        #         if tb_writer and (iteration % (hr_every_iters*5) == 0):
        #             tb_writer.add_scalar('train_hr/hr_l1', hr_l1.item(), iteration)
        #             tb_writer.add_scalar('train_hr/hr_grad', hr_grad.item(), iteration)
        #     except Exception as e:
        #         # 如果任何地方出错，打印但不影响主训练（保持鲁棒）
        #         print(f"[WARN] HR supervision skipped at iter {iteration} due to {e}")

        if iteration > 3000:
            ssim_loss = (1.0 - ssim(image, gt_image))
            if 'app_image' in render_pkg and ssim_loss < 0.5:
                app_image = render_pkg['app_image']
                Ll1 = l1_loss(app_image, gt_image)
            else:
                Ll1 = l1_loss(image, gt_image)
            image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
            loss = image_loss.clone()
        else :
            ssim_loss = (1.0 - ssim(image, gt_image))
            Ll1 = l1_loss(image, gt_image)
            image_loss = 0 * ssim_loss
            loss = image_loss.clone()
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()

        # single-view loss
        def get_depth_weight(opt, iteration):
            if not hasattr(opt, 'depth_weight'):
                return 0.0
                
            if iteration < opt.depth_weight_from_iter:
                return 0.0
            elif iteration > opt.depth_weight_until_iter:
                decay_iterations = 5000
                end_iter = opt.depth_weight_until_iter + decay_iterations
                if iteration > end_iter:
                    return 0.0
                else:
                    decay_ratio = (end_iter - iteration) / decay_iterations
                    return opt.depth_weight * decay_ratio
            else:
                return opt.depth_weight
        if iteration > opt.depth_weight_from_iter:
            depth_weight = get_depth_weight(opt, iteration)
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            depth_normal = render_pkg["depth_normal"]
            plane_depth = render_pkg["plane_depth"]
            murre_depth = render_pkg["murre_depth"]
            depth_mask = (murre_depth > 0).float()
            depth_diff = (plane_depth.squeeze() - murre_depth).abs()
            if not opt.wo_image_weight:
                depth_loss = depth_weight * (image_weight * depth_mask * depth_diff).mean()
            else:
                depth_loss = depth_weight * (depth_mask * depth_diff).mean()

            loss += (depth_loss)
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            edged_weight = opt.edged_weight
            depth_normal = render_pkg["depth_normal"]
            normal = render_pkg["rendered_normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            # edged_loss = edged_normal_loss(normal, depth_normal, gt_image, 0.15)
            # edged_loss = edged_weight * edged_loss
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()    
            loss += (normal_loss)
        
        if iteration > opt.start_embed_iter and iteration < opt.end_embed_iter:
            depth_normal = render_pkg["depth_normal"]
            normal = render_pkg["rendered_normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            norm_pred_flat = normal.reshape(-1,3)
            norm_dep_flat = depth_normal.reshape(-1,3)
            
            norm_pred_flat = F.normalize(norm_pred_flat, dim=-1)
            norm_dep_flat = F.normalize(norm_dep_flat, dim=-1)
            target = torch.ones(norm_pred_flat.shape[0], device=norm_pred_flat.device)

            embedding_loss = image_weight * F.cosine_embedding_loss(norm_pred_flat.detach(), norm_dep_flat.detach(), target)
            embedding_loss = F.cosine_embedding_loss(norm_pred_flat.detach(), norm_dep_flat.detach(), target)
            embedding_loss = opt.embed_weight * embedding_loss
            loss += (embedding_loss)
        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss  is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_single_view_embedding_for_log = 0.4 * embedding_loss.item() if embedding_loss  is not None else 0.0 + 0.6 * ema_single_view_embedding_for_log
            ema_single_view_dep_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + 0.6 * ema_single_view_dep_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single-normal": f"{ema_single_view_for_log:.{5}f}",
                    "Single-depth": f"{ema_single_view_dep_for_log:.{5}f}",
                    "embed": f"{ema_single_view_embedding_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        lpips_fn = LPIPS(net='vgg').to("cuda") 
        lpips_fn.eval()
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
