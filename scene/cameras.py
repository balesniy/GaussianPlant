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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

def soft_mask_from_binary(mask, inner_band=6, outer_band=2, edge_low=0.35):
    if inner_band <= 0 and outer_band <= 0:
        return mask.astype(np.float32)

    binary = (mask > 0.5).astype(np.uint8)
    inv = 1 - binary

    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    soft = np.zeros_like(dist_in, dtype=np.float32)
    inside = binary > 0
    outside = ~inside

    if inner_band > 0:
        deep_inside = inside & (dist_in >= inner_band)
        soft[deep_inside] = 1.0
        band_inside = inside & (dist_in < inner_band)
        soft[band_inside] = edge_low + (1.0 - edge_low) * (dist_in[band_inside] / float(inner_band))
    else:
        soft[inside] = 1.0

    if outer_band > 0:
        band_outside = outside & (dist_out < outer_band)
        soft[band_outside] = edge_low * (1.0 - dist_out[band_outside] / float(outer_band))

    return np.clip(soft, 0.0, 1.0)

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap, 
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 , mask_path=None, soft_mask_inner_band=6, soft_mask_outer_band=2, soft_mask_edge_low=0.35
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # self.mask = mask

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        self.has_alpha_mask = False

        if mask_path:
            try:
                mask_image = Image.open(mask_path).convert("L")
                resized_mask = PILtoTorch(mask_image, resolution)[:1, ...]
                mask_np = resized_mask.squeeze(0).cpu().numpy().astype(np.float32)
                mask_np = soft_mask_from_binary(
                    mask_np,
                    inner_band=soft_mask_inner_band,
                    outer_band=soft_mask_outer_band,
                    edge_low=soft_mask_edge_low,
                )
                self.alpha_mask = torch.from_numpy(mask_np[None]).to(self.data_device)
                self.has_alpha_mask = True
            except Exception as exc:
                print(f"[Warning] Failed to load alpha mask at {mask_path}: {exc}")
                self.alpha_mask = None

        if self.alpha_mask is None and resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            self.has_alpha_mask = True
        elif self.alpha_mask is None:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        self.alpha_mask = self.alpha_mask.clamp(0.0, 1.0)

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
