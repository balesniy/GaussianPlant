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
from typing import Literal
import torch
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
try:
    from simple_knn._C import distCUDA2
except ImportError:
    def distCUDA2(points, chunk_size=2048):
        """PyTorch fallback for environments where simple-knn is not built."""
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}")
        if points.shape[0] <= 1:
            return torch.ones((points.shape[0],), dtype=points.dtype, device=points.device)

        points = points.contiguous()
        num_points = points.shape[0]
        k = min(3, num_points - 1)
        mean_dists = torch.empty((num_points,), dtype=points.dtype, device=points.device)
        chunk_indices = torch.arange(chunk_size, device=points.device)

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            dist = torch.cdist(points[start:end], points, p=2).square()
            row_indices = chunk_indices[: end - start]
            dist[row_indices, torch.arange(start, end, device=points.device)] = float("inf")
            mean_dists[start:end] = dist.topk(k, largest=False, dim=1).values.mean(dim=1)

        return mean_dists
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
try:
    import faiss
except ImportError:
    faiss = None
import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert, quaternion_apply, matrix_to_quaternion
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
import networkx as nx
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import trimesh
from utils.gs_utils import fit_cylinder_ransac, estimate_gs_para_from_cluster, branch_to_cylinder, leaf_to_disk, stpr_to_cylinder, gs_to_disk_distance, gs_to_cylinder_distance, stpr_to_disk, build_edge, build_mst_from_endpoints,save_mst_ply, is_leaf
import time
from utils.loss_utils import mst_loss

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


def _minimum_spanning_forest(num_nodes, edges, costs, max_edge_length=0.0, edge_lengths=None, max_edge_cost=0.0):
    if num_nodes <= 1 or len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)

    order = np.argsort(np.asarray(costs, dtype=np.float64))
    parent = np.arange(num_nodes, dtype=np.int64)
    rank = np.zeros(num_nodes, dtype=np.int8)
    selected = []

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    for edge_idx in order:
        if max_edge_cost > 0 and costs[edge_idx] > max_edge_cost:
            continue
        if max_edge_length > 0 and edge_lengths is not None and edge_lengths[edge_idx] > max_edge_length:
            continue
        a, b = edges[edge_idx]
        if union(a, b):
            selected.append((int(a), int(b)))
            if max_edge_length <= 0 and len(selected) == num_nodes - 1:
                break
    return np.asarray(selected, dtype=np.int64)


def _orient_tree_edges(num_nodes, undirected_edges, root):
    if num_nodes <= 1 or len(undirected_edges) == 0:
        return np.empty((0, 2), dtype=np.int64), np.zeros((num_nodes,), dtype=np.int64)
    adjacency = [[] for _ in range(num_nodes)]
    for a, b in undirected_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    parent = np.full((num_nodes,), -1, dtype=np.int64)
    oriented = []
    roots = [int(root)] + [idx for idx in range(num_nodes) if idx != int(root)]
    for component_root in roots:
        if parent[component_root] != -1:
            continue
        parent[component_root] = component_root
        queue = [component_root]
        for node in queue:
            for nbr in adjacency[node]:
                if parent[nbr] != -1:
                    continue
                parent[nbr] = node
                oriented.append((node, nbr))
                queue.append(nbr)
    return np.asarray(oriented, dtype=np.int64), parent

def run_kmeans(features, k, niter=25, nredo=3):
    if faiss is not None:
        try:
            kmeans = faiss.Kmeans(d=features.shape[1], k=k, niter=niter, nredo=nredo, gpu=True)
            kmeans.train(features)
            return kmeans.index.search(features, 1)[1].flatten()
        except Exception as exc:
            print(f"[Warning] faiss GPU KMeans failed, falling back to sklearn KMeans: {exc}")

    kmeans = KMeans(n_clusters=k, n_init=nredo, max_iter=niter, random_state=0)
    return kmeans.fit_predict(features)


class StPrGraphEdgeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.edge_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def forward(self, node_feats, edge_idx, edge_feats):
        h = self.node_proj(node_feats)
        e = self.edge_proj(edge_feats)
        src = edge_idx[:, 0]
        dst = edge_idx[:, 1]
        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            msg_fwd = message_mlp(torch.cat([h[src], h[dst], e], dim=-1))
            msg_rev = message_mlp(torch.cat([h[dst], h[src], e], dim=-1))
            agg = torch.zeros_like(h)
            deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            agg.index_add_(0, dst, msg_fwd)
            agg.index_add_(0, src, msg_rev)
            deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
            deg.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=h.dtype, device=h.device))
            agg = agg / deg.clamp(min=1.0)
            h = h + update_mlp(torch.cat([h, agg], dim=-1))
        return self.edge_head(torch.cat([h[src], h[dst], e], dim=-1)).view(-1)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default", device=None):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.device = device
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_to_track = 4
        self.setup_functions()
        self.knn_dists = None
        self.knn_idx = None
        self.n_points = None
        self.nn_stpr_appgs = None
        self.structure_gs = None
        self.appgs = None
        self.leaf_disks = None
        self.branch_cylinders = None
        self.stpr_label = None 
        self.app_label = None
        self._pst_logit = None
        self._stpr_type_logit = None
        self._semantic_logit = None
        self._semantic_feature = torch.empty(0)
        self._stpr_graph_gnn = None
        self._stpr_graph_gnn_config = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.n_points,
            {
                "app_label": self.app_label,
                "stpr_label": self.stpr_label,
                "exposure": self._exposure,
                "exposure_optimizer": self.exposure_optimizer.state_dict() if hasattr(self, "exposure_optimizer") else None,
                "pst_logit": self._pst_logit,
                "stpr_type_logit": self._stpr_type_logit,
                "semantic_logit": self._semantic_logit,
                "semantic_feature": self._semantic_feature,
                "stpr_graph_gnn_config": self._stpr_graph_gnn_config,
                "stpr_graph_gnn_state": self._stpr_graph_gnn.state_dict() if self._stpr_graph_gnn is not None else None,
            },
        )
    
    def restore(self, model_args, training_args):
        metadata = {}
        if len(model_args) > 14:
            metadata = model_args[14]
            model_args = model_args[:14]
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self._mask,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        self.n_points) = model_args
        if metadata.get("exposure") is not None:
            self._exposure = metadata["exposure"]
        self._pst_logit = metadata.get("pst_logit")
        self._stpr_type_logit = metadata.get("stpr_type_logit")
        self._semantic_logit = metadata.get(
            "semantic_logit",
            nn.Parameter(torch.zeros((self._xyz.shape[0], 1), dtype=torch.float, device=self.device).requires_grad_(True))
        )
        self._semantic_feature = metadata.get("semantic_feature", torch.empty(0, device=self.device))
        gnn_config = metadata.get("stpr_graph_gnn_config")
        if gnn_config is not None:
            self._stpr_graph_gnn_config = gnn_config
            self._stpr_graph_gnn = StPrGraphEdgeGNN(**gnn_config).to(self.device)
            if metadata.get("stpr_graph_gnn_state") is not None:
                self._stpr_graph_gnn.load_state_dict(metadata["stpr_graph_gnn_state"])
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as exc:
            print(f"[Warning] Optimizer state did not match current Gaussian parameters; continuing with a fresh optimizer. Details: {exc}")
        if metadata.get("exposure_optimizer") is not None:
            self.exposure_optimizer.load_state_dict(metadata["exposure_optimizer"])
        self.app_label = metadata.get("app_label")
        self.stpr_label = metadata.get("stpr_label")
        

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_n_points(self):
        return len(self._xyz)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_semantic(self):
        if self._semantic_logit is None:
            return torch.ones((self.get_xyz.shape[0], 1), dtype=torch.float, device=self.device)
        return torch.sigmoid(self._semantic_logit)
    
    @property
    def get_exposure(self):
        return self._exposure

    @property
    def get_mask(self):
        self._ensure_mask_shape()
        return self._mask

    def _ensure_mask_shape(self, expected_count=None):
        if expected_count is None:
            expected_count = self._xyz.shape[0]
        device = self.device if self.device is not None else self._xyz.device
        if self._mask.numel() == expected_count:
            return
        if self._mask.numel() == 0:
            self._mask = torch.ones((expected_count,), dtype=torch.float, device=device)
            return
        mask = self._mask.detach().to(device)
        if mask.numel() > expected_count:
            self._mask = mask[:expected_count].clone()
        else:
            padding = torch.ones((expected_count - mask.numel(),), dtype=mask.dtype, device=device)
            self._mask = torch.cat((mask, padding), dim=0)

    def opacity_regularizer(self):
        return torch.mean(self.get_opacity * (1 - self.get_opacity))
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1, return_full = False):
        if not return_full:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        else:
            cov = self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
            cov_full = torch.zeros((cov.shape[0], 3, 3), device=self.device)
            cov_full[:, 0, 0] = cov[:, 0]
            cov_full[:, 1, 1] = cov[:, 3]
            cov_full[:, 2, 2] = cov[:, 5]
            cov_full[:, 0, 1] = cov[:, 1]
            cov_full[:, 0, 2] = cov[:, 2]
            cov_full[:, 1, 2] = cov[:, 4]
            cov_full[:, 1, 0] = cov[:, 1]
            cov_full[:, 2, 0] = cov[:, 2]
            cov_full[:, 2, 1] = cov[:, 4]
            return cov_full

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def ensure_stpr_graph_gnn(self, node_dim, edge_dim, hidden_dim=64, num_layers=2, lr=0.0025):
        config = {
            "node_dim": int(node_dim),
            "edge_dim": int(edge_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
        }
        if self._stpr_graph_gnn is None or self._stpr_graph_gnn_config != config:
            self._stpr_graph_gnn = StPrGraphEdgeGNN(**config).to(self.get_xyz.device)
            self._stpr_graph_gnn_config = config
            if self.optimizer is not None:
                self.optimizer.add_param_group({
                    "params": self._stpr_graph_gnn.parameters(),
                    "lr": lr,
                    "name": "stpr_graph_gnn",
                })
        return self._stpr_graph_gnn

    def _ransac_trunk_axis(self, xyz, radius, root_axis=2, iterations=128, radius_factor=2.5):
        if xyz.shape[0] < 2:
            return None
        best_score = None
        best_axis = None
        best_point = None
        n = xyz.shape[0]
        threshold = (torch.median(radius).clamp(min=1e-5) * float(radius_factor)).detach()
        generator = torch.Generator(device=xyz.device)
        generator.manual_seed(17)
        for _ in range(int(iterations)):
            pair = torch.randperm(n, generator=generator, device=xyz.device)[:2]
            p0 = xyz[pair[0]]
            p1 = xyz[pair[1]]
            axis = F.normalize(p1 - p0, dim=0, eps=1e-8)
            if not torch.isfinite(axis).all() or torch.linalg.norm(p1 - p0) < 1e-6:
                continue
            rel = xyz - p0
            projected = (rel * axis).sum(dim=-1, keepdim=True) * axis
            radial_dist = torch.linalg.norm(rel - projected, dim=-1)
            inliers = radial_dist < threshold
            vertical = torch.abs(axis[int(root_axis)]) if 0 <= int(root_axis) < 3 else torch.abs(axis[2])
            score = inliers.float().sum() + 0.25 * vertical * n
            if best_score is None or score > best_score:
                best_score = score
                best_axis = axis
                best_point = p0
        if best_axis is None:
            return None
        return best_point, best_axis, threshold

    def initialize_stpr_type_logits(self, labels, confidence=2.0, root_axis=2):
        labels = labels or []
        n = len(labels)
        if n == 0:
            return
        logits = torch.full((n, 3), -float(confidence), dtype=torch.float, device=self.device)
        branch_indices = [idx for idx, label in enumerate(labels) if label == "branch"]
        leaf_indices = [idx for idx, label in enumerate(labels) if label == "leaf"]
        if branch_indices:
            branch_idx = torch.tensor(branch_indices, dtype=torch.long, device=self.device)
            xyz = self.get_xyz[branch_idx].detach()
            scales = self.get_scaling[branch_idx].detach()
            radius = scales[:, 1:].mean(dim=-1)
            root_axis = int(root_axis) if 0 <= int(root_axis) < 3 else 2
            height = xyz[:, root_axis]
            radius_score = (radius - radius.min()) / (radius.max() - radius.min()).clamp(min=1e-6)
            base_score = (height.max() - height) / (height.max() - height.min()).clamp(min=1e-6)
            ransac_score = torch.zeros_like(radius_score)
            ransac = self._ransac_trunk_axis(xyz, radius, root_axis=root_axis)
            if ransac is not None:
                axis_point, axis_dir, axis_radius = ransac
                rel = xyz - axis_point
                projected = (rel * axis_dir).sum(dim=-1, keepdim=True) * axis_dir
                radial_dist = torch.linalg.norm(rel - projected, dim=-1)
                ransac_score = torch.exp(-radial_dist / axis_radius.clamp(min=1e-6))
            trunk_score = 0.45 * radius_score + 0.35 * base_score + 0.20 * ransac_score
            trunk_count = max(1, min(int(np.ceil(0.15 * len(branch_indices))), len(branch_indices)))
            trunk_local = torch.topk(trunk_score, k=trunk_count, largest=True).indices
            trunk_idx = branch_idx[trunk_local]
            logits[branch_idx, 1] = float(confidence)
            logits[trunk_idx, 0] = float(confidence) + trunk_score[trunk_local]
            logits[trunk_idx, 1] = 0.0
        if leaf_indices:
            leaf_idx = torch.tensor(leaf_indices, dtype=torch.long, device=self.device)
            logits[leaf_idx, 2] = float(confidence)
        self._stpr_type_logit = nn.Parameter(logits.requires_grad_(True))

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic_logit = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device=self.device)
        self.pretrained_exposures = None
        self.n_points = len(self._xyz)
        exposure = torch.eye(3, 4, device=self.device)[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._semantic_logit], 'lr': training_args.opacity_lr, "name": "semantic"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self._semantic_feature.numel():
            l.append({'params': [self._semantic_feature], 'lr': training_args.feature_lr, "name": "semantic_feature"})
        if self._stpr_graph_gnn is not None:
            l.append({'params': self._stpr_graph_gnn.parameters(), 'lr': training_args.feature_lr, "name": "stpr_graph_gnn"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        if self._pst_logit is not None and self.optimizer is not None:
            self.optimizer.add_param_group({'params': [self._pst_logit], 'lr': training_args.opacity_lr, "name": "pst"})
        if self._stpr_type_logit is not None and self.optimizer is not None:
            self.optimizer.add_param_group({'params': [self._stpr_type_logit], 'lr': training_args.opacity_lr, "name": "stpr_type"})

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('semantic_logit')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        semantics = self._semantic_logit.detach().cpu().numpy() if self._semantic_logit is not None else np.zeros((xyz.shape[0], 1), dtype=np.float32)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, semantics, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_label_ply(self, path, keep_labels=None):
        if keep_labels is None or self.stpr_label is None:
            self.save_ply(path)
            return
        keep = torch.tensor([lbl in keep_labels for lbl in self.stpr_label], dtype=torch.bool, device=self.device)
        self.clone_subset(keep, copy_structure_metadata=True).save_ply(path)

    def clone_subset(self, keep_mask, copy_structure_metadata=False):
        keep_mask = keep_mask.to(self.device).bool()
        self._ensure_mask_shape(keep_mask.shape[0])
        clone = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        clone.active_sh_degree = self.active_sh_degree
        clone.spatial_lr_scale = self.spatial_lr_scale
        clone._xyz = nn.Parameter(self._xyz.detach()[keep_mask].clone().requires_grad_(True))
        clone._features_dc = nn.Parameter(self._features_dc.detach()[keep_mask].clone().requires_grad_(True))
        clone._features_rest = nn.Parameter(self._features_rest.detach()[keep_mask].clone().requires_grad_(True))
        clone._opacity = nn.Parameter(self._opacity.detach()[keep_mask].clone().requires_grad_(True))
        clone._semantic_logit = nn.Parameter(self._semantic_logit.detach()[keep_mask].clone().requires_grad_(True)) if self._semantic_logit is not None else nn.Parameter(torch.zeros((int(keep_mask.sum().item()), 1), dtype=torch.float, device=self.device).requires_grad_(True))
        clone._scaling = nn.Parameter(self._scaling.detach()[keep_mask].clone().requires_grad_(True))
        clone._rotation = nn.Parameter(self._rotation.detach()[keep_mask].clone().requires_grad_(True))
        clone._mask = self._mask.detach()[keep_mask].clone()
        if self._semantic_feature.numel():
            clone._semantic_feature = nn.Parameter(self._semantic_feature.detach()[keep_mask].clone().requires_grad_(True))
        clone.max_radii2D = torch.zeros((clone.get_xyz.shape[0]), device=self.device)
        clone.xyz_gradient_accum = torch.zeros((clone.get_xyz.shape[0], 1), device=self.device)
        clone.denom = torch.zeros((clone.get_xyz.shape[0], 1), device=self.device)
        clone.n_points = clone.get_xyz.shape[0]
        clone.exposure_mapping = self.exposure_mapping
        clone.pretrained_exposures = self.pretrained_exposures
        clone._exposure = nn.Parameter(self._exposure.detach().clone().requires_grad_(True))
        if copy_structure_metadata and self.stpr_label is not None:
            clone.stpr_label = [lbl for lbl, keep in zip(self.stpr_label, keep_mask.detach().cpu().tolist()) if keep]
        if copy_structure_metadata and self.app_label is not None:
            clone.app_label = [lbl for lbl, keep in zip(self.app_label, keep_mask.detach().cpu().tolist()) if keep]
        if copy_structure_metadata and self._pst_logit is not None:
            clone._pst_logit = nn.Parameter(self._pst_logit.detach()[keep_mask].clone().requires_grad_(True))
        if copy_structure_metadata and self._stpr_type_logit is not None:
            clone._stpr_type_logit = nn.Parameter(self._stpr_type_logit.detach()[keep_mask].clone().requires_grad_(True))
        return clone

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def reset_opacity_stpr(self):
        # set opacity to 1
        opacities_new = self.inverse_opacity_activation(torch.ones_like(self.get_opacity)*0.5)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        property_names = {p.name for p in plydata.elements[0].properties}
        if "semantic_logit" in property_names:
            semantic_logit = np.asarray(plydata.elements[0]["semantic_logit"])[..., np.newaxis]
        else:
            semantic_logit = np.zeros_like(opacities)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._semantic_logit = nn.Parameter(torch.tensor(semantic_logit, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device=self.device)
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.n_points = self._xyz.shape[0]

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_param = group["params"][0]
                stored_state = self.optimizer.state.get(old_param, None)

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[old_param]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1 or group["name"] == "stpr_graph_gnn":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask,flag=Literal['app','stpr']):
        valid_points_mask = ~mask
        self._ensure_mask_shape(valid_points_mask.shape[0])
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        if "semantic" in optimizable_tensors:
            self._semantic_logit = optimizable_tensors["semantic"]
        if "semantic_feature" in optimizable_tensors:
            self._semantic_feature = optimizable_tensors["semantic_feature"]
        if "stpr_type" in optimizable_tensors:
            self._stpr_type_logit = optimizable_tensors["stpr_type"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.knn_dists = None
        self.knn_idx = None
        if flag=='app':
            self.app_label = [lbl for lbl, m in zip(self.app_label, valid_points_mask) if m]
            assert len(self.app_label) == self.get_xyz.shape[0]
        elif flag=='stpr':
            self.stpr_label = [lbl for lbl, m in zip(self.stpr_label, valid_points_mask) if m]
            assert len(self.stpr_label) == self.get_xyz.shape[0]
            if self._pst_logit is not None:
                self._pst_logit = optimizable_tensors["pst"]
        self._mask = self._mask[valid_points_mask].clone()
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_label=None, flag=None, new_pst_logit=None, new_semantic=None, new_semantic_feature=None, new_stpr_type_logit=None):
        self._ensure_mask_shape()
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "semantic": new_semantic if new_semantic is not None else torch.zeros_like(new_opacities),
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self._pst_logit is not None and new_pst_logit is not None:
            d["pst"] = new_pst_logit
        if self._stpr_type_logit is not None and new_stpr_type_logit is not None:
            d["stpr_type"] = new_stpr_type_logit
        if self._semantic_feature.numel():
            if new_semantic_feature is None:
                new_semantic_feature = torch.zeros((new_xyz.shape[0], self._semantic_feature.shape[1]), dtype=self._semantic_feature.dtype, device=self.device)
            d["semantic_feature"] = new_semantic_feature

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._semantic_logit = optimizable_tensors["semantic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "pst" in optimizable_tensors:
            self._pst_logit = optimizable_tensors["pst"]
        if "stpr_type" in optimizable_tensors:
            self._stpr_type_logit = optimizable_tensors["stpr_type"]
        if "semantic_feature" in optimizable_tensors:
            self._semantic_feature = optimizable_tensors["semantic_feature"]

        new_mask = torch.ones((new_xyz.shape[0],), dtype=self._mask.dtype, device=self._mask.device)
        self._mask = torch.cat((self._mask, new_mask), dim=0)
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        if flag == 'app':
            self.app_label.extend(new_label)
            assert len(self.app_label) == self.get_xyz.shape[0]
        elif flag == 'stpr':
            self.stpr_label.extend(new_label)
            assert len(self.stpr_label) == self.get_xyz.shape[0]
        elif flag is not None:
            raise ValueError(f"Unknown densification flag: {flag}")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.knn_dists = None
        self.knn_idx = None
        

    def densify_and_split(self, grads, grad_threshold, scene_extent, flag=Literal['app','stpr'],N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic = self._semantic_logit[selected_pts_mask].repeat(N,1) if self._semantic_logit is not None else torch.zeros_like(new_opacity)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1) if self._semantic_feature.numel() else None
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_label = None
        new_pst_logit = None
        new_stpr_type_logit = None
        if flag == 'app':
            new_label = [lbl for lbl, m in zip(self.app_label, selected_pts_mask) if m]
            new_label = new_label * N
        elif flag == 'stpr':
            new_label = [lbl for lbl, m in zip(self.stpr_label, selected_pts_mask) if m]
            new_label = new_label * N
            if self._pst_logit is not None:
                new_pst_logit = self._pst_logit[selected_pts_mask].repeat(N, 1)
            if self._stpr_type_logit is not None:
                new_stpr_type_logit = self._stpr_type_logit[selected_pts_mask].repeat(N, 1)
        elif flag is not None:
            raise ValueError(f"Unknown densification flag: {flag}")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii,new_label,flag,new_pst_logit,new_semantic,new_semantic_feature,new_stpr_type_logit)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,flag=Literal['app', 'stpr']):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_semantic = self._semantic_logit[selected_pts_mask] if self._semantic_logit is not None else torch.zeros_like(new_opacities)
        new_semantic_feature = self._semantic_feature[selected_pts_mask] if self._semantic_feature.numel() else None
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_label = None
        new_pst_logit = None
        new_stpr_type_logit = None
        if flag == 'app':
            new_label = [lbl for lbl, m in zip(self.app_label, selected_pts_mask) if m]
        elif flag == 'stpr':
            new_label = [lbl for lbl, m in zip(self.stpr_label, selected_pts_mask) if m]
            if self._pst_logit is not None:
                new_pst_logit = self._pst_logit[selected_pts_mask]
            if self._stpr_type_logit is not None:
                new_stpr_type_logit = self._stpr_type_logit[selected_pts_mask]
        elif flag is not None:
            raise ValueError(f"Unknown densification flag: {flag}")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,new_label,flag,new_pst_logit,new_semantic,new_semantic_feature,new_stpr_type_logit)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, flag='stpr', only_prune=False, size_threshold_small=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        if not only_prune:
            self.densify_and_clone(grads, max_grad, extent,flag)
            self.densify_and_split(grads, max_grad, extent,flag)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_low_opacity_count = prune_mask.sum()
        prune_large_count = 0
        prune_small_count = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            prune_large_count = torch.logical_or(big_points_vs, big_points_ws).sum()
        if size_threshold_small:
            small_points_ws = self.get_scaling.max(dim=1).values < 0.05 * extent
            prune_mask = torch.logical_or(prune_mask, small_points_ws)
            prune_small_count = small_points_ws.sum()
        self.prune_points(prune_mask,flag=flag)
        print(f"Pruned {prune_low_opacity_count} points with low opacity, {prune_large_count} points with large screen size, {prune_small_count} points with small screen size.")


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def build_stprs_from_gs(self, num_clusters=100,method: Literal['coarse_kmeans', 'kmeans', 'random', '3dgs', 'feature_kmeans'] = 'coarse_kmeans', min_cluster_points=100,
                            point_features=None, stpr_feature_weight=1.0, stpr_xyz_weight=0.25, stpr_appgs_per_stpr=50,
                            stpr_semantic_dim=0, stpr_appgs_max_scale_ratio=0.03,
                            scene_extent=None, stpr_min_scale_ratio=1e-5, stpr_max_scale_ratio=0.5,
                            debug_dir=None, plant_prior="mixed", no_leaf_mode=False,
                            stpr_dbscan_eps=0.005, stpr_dbscan_min_samples=5,
                            geometry_refine_labels=False, geometry_knn=12,
                            geometry_cost_threshold=0.55, geometry_max_dist_factor=6.0,
                            geometry_axis_threshold=0.35, geometry_tangent_threshold=0.55,
                            geometry_radius_threshold=0.8, geometry_radius_graph_r=None):
        """
        structural primitives (StPrs) from optimized Gaussians by clustering them.
        param num_clusters: Number of clusters for grouping Gaussians into structural primitives.
        """
        print(f"Building {num_clusters} structural primitives using {method} method.")
        # Extract Gaussian properties
        xyz = self._xyz.detach().cpu().numpy()  # Gaussian positions
        scaling = self.get_scaling.detach().cpu().numpy()  # Gaussian scales
        rotation = self.get_rotation.detach().cpu().numpy()  # Gaussian rotations
        covariance = self.get_covariance().detach().cpu().numpy()  # Gaussian covariance matrices
        feature = self.get_features.detach().cpu().numpy()  # Gaussian features
        feature_dc = self.get_features_dc.detach().cpu().numpy()  # Gaussian features
        feature_rest = self.get_features_rest.detach().cpu().numpy()  # Gaussian features
        # Initialize lists for StPr parameters
        stpr_positions = []
        stpr_scales = []
        stpr_rotations = []
        stpr_features_dc = []
        stpr_features_rest = []
        branch_positions = []
        branch_scales = []
        branch_rotations = []
        branch_points_all = []
        leaf_positions = []
        leaf_scales = []
        leaf_rotations = []
        branch_quat = []
        leaf_quat = []
        branch_feature_dc = []
        branch_feature_rest = []
        leaf_feature_dc = []
        leaf_feature_rest = []
        stpr_label = []
        branch_index = []
        leaf_index = []
        surf_rotations = []
        if method in ('coarse_kmeans', 'kmeans'):
            k = max(1, min(num_clusters, xyz.shape[0] // max(min_cluster_points, 1)))
            xyz_center = xyz.mean(axis=0, keepdims=True)
            xyz_scale = scene_extent if scene_extent is not None and scene_extent > 0 else np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
            feature_vectors = ((xyz - xyz_center) / max(float(xyz_scale), 1e-6)).astype(np.float32)
            print(f"[DEBUG][coarse-kmeans] points={xyz.shape[0]} k={k} target_points_per_stpr~{max(xyz.shape[0] // max(k, 1), 1)}")
            labels = run_kmeans(feature_vectors, k, niter=25, nredo=3)
            point_colors = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)
            used_points = 0
            skipped_tiny = 0
            index = 0

            for label in np.unique(labels):
                cluster_mask = labels == label
                cluster_points = xyz[cluster_mask]
                if len(cluster_points) < 3:
                    skipped_tiny += len(cluster_points)
                    continue
                used_points += len(cluster_points)
                mean, stpr_rot, stpr_scale, rot_cylinder, rot_disk = estimate_gs_para_from_cluster(cluster_points, test_flag=False)
                feature_dc_stpr = feature_dc[cluster_mask].mean(axis=0)
                feature_rest_stpr = feature_rest[cluster_mask].mean(axis=0)
                is_branch = no_leaf_mode or not is_leaf(cluster_points)

                stpr_positions.append(mean)
                stpr_rotations.append(stpr_rot)
                stpr_features_dc.append(feature_dc_stpr)
                stpr_features_rest.append(feature_rest_stpr)
                if is_branch:
                    stpr_scales.append(stpr_scale)
                    branch_positions.append(mean)
                    branch_scales.append(stpr_scale)
                    branch_rotations.append(rot_cylinder)
                    surf_rotations.append(rot_cylinder)
                    branch_quat.append(stpr_rot)
                    branch_points_all.append(cluster_points)
                    branch_feature_dc.append(feature_dc_stpr)
                    branch_feature_rest.append(feature_rest_stpr)
                    stpr_label.append('branch')
                    branch_index.append(index)
                    point_colors[cluster_mask] = [1.0, 0.0, 0.0]
                else:
                    stpr_scale[2] = 1e-6
                    stpr_scales.append(stpr_scale)
                    leaf_positions.append(mean)
                    leaf_scales.append(stpr_scale)
                    leaf_quat.append(stpr_rot)
                    leaf_rotations.append(rot_disk)
                    leaf_feature_dc.append(feature_dc_stpr)
                    leaf_feature_rest.append(feature_rest_stpr)
                    surf_rotations.append(rot_disk)
                    stpr_label.append('leaf')
                    leaf_index.append(index)
                    point_colors[cluster_mask] = [0.0, 1.0, 0.0]
                index += 1

            if debug_dir is not None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(point_colors)
                o3d.io.write_point_cloud(os.path.join(debug_dir, "coarse_kmeans_leaf_branch.ply"), pcd)
            print(
                f"[DEBUG][coarse-kmeans] initialized_stprs={len(stpr_positions)} "
                f"used_points={used_points}/{xyz.shape[0]} ({used_points / max(xyz.shape[0], 1):.1%}) "
                f"skipped_tiny={skipped_tiny} leaf={len(leaf_index)} branch={len(branch_index)}"
            )
        elif method == 'random':
            pass
        elif method in ('3dgs', 'feature_kmeans'):
            """
            Build structural primitives (StPrs) from optimized Gaussians using 3D Gaussian clustering.
            Using segmented leaf&branch points information for initialize leaf stpr(disk) and branch stpr(cylinder).
            """
            if method == 'feature_kmeans':
                if point_features is None:
                    raise ValueError("feature_kmeans requires point_features")
                point_features = np.asarray(point_features, dtype=np.float32)
                if point_features.shape[0] != xyz.shape[0]:
                    raise ValueError(f"point_features has {point_features.shape[0]} rows, expected {xyz.shape[0]}")

                xyz_center = xyz.mean(axis=0, keepdims=True)
                xyz_scale = scene_extent if scene_extent is not None and scene_extent > 0 else np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
                xyz_norm = (xyz - xyz_center) / max(float(xyz_scale), 1e-6)
                feat_mean = point_features.mean(axis=0, keepdims=True)
                feat_std = point_features.std(axis=0, keepdims=True) + 1e-6
                feat_norm = (point_features - feat_mean) / feat_std
                feat_norm /= np.linalg.norm(feat_norm, axis=1, keepdims=True) + 1e-6
                cluster_features = np.hstack((stpr_xyz_weight * xyz_norm, stpr_feature_weight * feat_norm)).astype(np.float32)
                k = max(1, min(num_clusters, xyz.shape[0] // max(min_cluster_points, 1)))
                print(
                    f"[DEBUG][feature-kmeans] points={xyz.shape[0]} feature_dim={point_features.shape[1]} "
                    f"k={k} xyz_weight={stpr_xyz_weight} feature_weight={stpr_feature_weight}"
                )
                labels = run_kmeans(cluster_features, k, niter=25, nredo=3)

                unique_labels = set(labels)
                label_leaf = []
                label_branch = []
                point_colors = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)
                rng = np.random.default_rng(0)
                colors = rng.random((len(unique_labels), 3), dtype=np.float32)
                used_points = 0
                skipped_tiny = 0
                for color_idx, label in enumerate(sorted(unique_labels)):
                    cluster_points = xyz[labels == label]
                    if len(cluster_points) < 3:
                        skipped_tiny += len(cluster_points)
                        continue
                    used_points += len(cluster_points)
                    if no_leaf_mode:
                        point_colors[labels == label] = [1.0, 0.0, 0.0]
                        label_branch.append(label)
                    elif is_leaf(cluster_points):
                        point_colors[labels == label] = [0.0, 1.0, 0.0]
                        label_leaf.append(label)
                    else:
                        point_colors[labels == label] = [1.0, 0.0, 0.0]
                        label_branch.append(label)
                    colors[color_idx] = point_colors[labels == label][0]
                if debug_dir is not None:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    pcd.colors = o3d.utility.Vector3dVector(point_colors)
                    o3d.io.write_point_cloud(os.path.join(debug_dir, "feature_kmeans_leaf_branch.ply"), pcd)
                print(
                    f"[DEBUG][feature-kmeans] clusters={len(unique_labels)} valid_clusters={len(label_leaf) + len(label_branch)} "
                    f"skipped_tiny={skipped_tiny} "
                    f"used_points={used_points}/{xyz.shape[0]} ({used_points / max(xyz.shape[0], 1):.1%}) "
                    f"leaf={len(label_leaf)} branch={len(label_branch)}"
                )
            else:
                label_leaf, label_branch, labels = fit_cylinder_ransac(
                    xyz,
                    eps=stpr_dbscan_eps,
                    min_samples=stpr_dbscan_min_samples,
                    save_ply=debug_dir is not None,
                    min_cluster_points=min_cluster_points,
                    save_prefix=os.path.join(debug_dir, "fit_cylinder_ransac") if debug_dir else None,
                    force_branch=no_leaf_mode,
                    geometry_refine=geometry_refine_labels,
                    geometry_knn=geometry_knn,
                    geometry_cost_threshold=geometry_cost_threshold,
                    geometry_max_dist_factor=geometry_max_dist_factor,
                    geometry_axis_threshold=geometry_axis_threshold,
                    geometry_tangent_threshold=geometry_tangent_threshold,
                    geometry_radius_threshold=geometry_radius_threshold,
                    geometry_radius_graph_r=geometry_radius_graph_r,
                    scene_extent=scene_extent,
                )
                print(f"[DEBUG][ransac] num ransac labels={len(np.unique(labels))} num leaf labels={len(label_leaf)} num branch labels={len(label_branch)}")
            index = 0
            for i,label in enumerate(np.unique(labels)):
                if label in label_leaf:
                    # build disk gs
                    leaf_points = xyz[labels == label]
                    mean,stpr_rot, stpr_scale,rot_cylinder,rot_disk = estimate_gs_para_from_cluster(leaf_points,test_flag=False)
                    stpr_scale[2] = 1e-6
                    stpr_positions.append(mean)
                    stpr_scales.append(stpr_scale)
                    stpr_rotations.append(stpr_rot)
                    feature_dc_stpr = feature_dc[labels == label].mean(axis=0)
                    feature_rest_stpr = feature_rest[labels == label].mean(axis=0)
                    stpr_features_dc.append(feature_dc_stpr)
                    stpr_features_rest.append(feature_rest_stpr)
                    leaf_positions.append(mean)
                    leaf_scales.append(stpr_scale)
                    leaf_quat.append(stpr_rot)
                    leaf_rotations.append(rot_disk)
                    leaf_feature_dc.append(feature_dc_stpr)
                    leaf_feature_rest.append(feature_rest_stpr)
                    surf_rotations.append(rot_disk)
                    stpr_label.append('leaf')
                    leaf_index.append(index)
                    index += 1
                    
                elif label in label_branch:
                    # build 3dgs
                    branch_points = xyz[labels == label]
                    mean,stpr_rot, stpr_scale,rot_cylinder,rot_disk = estimate_gs_para_from_cluster(branch_points,test_flag=False)
                    stpr_scales.append(stpr_scale)
                    stpr_rotations.append(stpr_rot)
                    stpr_positions.append(mean)
                    feature_dc_stpr = feature_dc[labels == label].mean(axis=0)
                    feature_rest_stpr = feature_rest[labels == label].mean(axis=0)
                    stpr_features_dc.append(feature_dc_stpr)
                    stpr_features_rest.append(feature_rest_stpr)
                    # record branch stprs for later use
                    branch_positions.append(mean)
                    branch_scales.append(stpr_scale)
                    branch_rotations.append(rot_cylinder)
                    surf_rotations.append(rot_cylinder)
                    branch_quat.append(stpr_rot)
                    branch_points_all.append(branch_points)
                    branch_feature_dc.append(feature_dc_stpr)
                    branch_feature_rest.append(feature_rest_stpr)
                    stpr_label.append('branch')
                    branch_index.append(index)
                    index += 1
                    # build cylinder gs

            if not stpr_positions:
                fallback_label = "branch" if no_leaf_mode else "leaf"
                print(f"Warning: no DBSCAN leaf/branch clusters detected; falling back to one {fallback_label}-like primitive from all Gaussians.")
                mean, stpr_rot, stpr_scale, rot_cylinder, rot_disk = estimate_gs_para_from_cluster(xyz, test_flag=False)
                stpr_positions.append(mean)
                stpr_scales.append(stpr_scale)
                stpr_rotations.append(stpr_rot)
                feature_dc_stpr = feature_dc.mean(axis=0)
                feature_rest_stpr = feature_rest.mean(axis=0)
                stpr_features_dc.append(feature_dc_stpr)
                stpr_features_rest.append(feature_rest_stpr)
                if no_leaf_mode:
                    branch_positions.append(mean)
                    branch_scales.append(stpr_scale)
                    branch_rotations.append(rot_cylinder)
                    branch_quat.append(stpr_rot)
                    branch_points_all.append(xyz)
                    branch_feature_dc.append(feature_dc_stpr)
                    branch_feature_rest.append(feature_rest_stpr)
                    surf_rotations.append(rot_cylinder)
                    stpr_label.append('branch')
                    branch_index.append(index)
                else:
                    stpr_scale[2] = 1e-6
                    leaf_positions.append(mean)
                    leaf_scales.append(stpr_scale)
                    leaf_quat.append(stpr_rot)
                    leaf_rotations.append(rot_disk)
                    leaf_feature_dc.append(feature_dc_stpr)
                    leaf_feature_rest.append(feature_rest_stpr)
                    surf_rotations.append(rot_disk)
                    stpr_label.append('leaf')
                    leaf_index.append(index)
                index += 1

        else:
            raise ValueError("Unknown clustering method. Use 'coarse_kmeans', 'feature_kmeans', or '3dgs'.")

        if not stpr_positions:
            fallback_label = "branch" if no_leaf_mode else "leaf"
            print(f"Warning: no StPr clusters detected; falling back to one {fallback_label}-like primitive from all Gaussians.")
            mean, stpr_rot, stpr_scale, rot_cylinder, rot_disk = estimate_gs_para_from_cluster(xyz, test_flag=False)
            feature_dc_stpr = feature_dc.mean(axis=0)
            feature_rest_stpr = feature_rest.mean(axis=0)
            stpr_positions.append(mean)
            stpr_rotations.append(stpr_rot)
            stpr_features_dc.append(feature_dc_stpr)
            stpr_features_rest.append(feature_rest_stpr)
            if no_leaf_mode:
                stpr_scales.append(stpr_scale)
                branch_positions.append(mean)
                branch_scales.append(stpr_scale)
                branch_rotations.append(rot_cylinder)
                surf_rotations.append(rot_cylinder)
                branch_quat.append(stpr_rot)
                branch_points_all.append(xyz)
                branch_feature_dc.append(feature_dc_stpr)
                branch_feature_rest.append(feature_rest_stpr)
                stpr_label.append('branch')
                branch_index.append(0)
            else:
                stpr_scale[2] = 1e-6
                stpr_scales.append(stpr_scale)
                leaf_positions.append(mean)
                leaf_scales.append(stpr_scale)
                leaf_quat.append(stpr_rot)
                leaf_rotations.append(rot_disk)
                leaf_feature_dc.append(feature_dc_stpr)
                leaf_feature_rest.append(feature_rest_stpr)
                surf_rotations.append(rot_disk)
                stpr_label.append('leaf')
                leaf_index.append(0)

        if branch_points_all:
            mesh_cylinder = branch_to_cylinder(branch_points=np.vstack(branch_points_all), branch_positions=branch_positions,
                               branch_scales=branch_scales, branch_rotations=branch_rotations) # list of open3d mesh
        else:
            print("Warning: no branch primitives detected; continuing with leaf-only StPr initialization.")
            mesh_cylinder = None

        if leaf_positions and not no_leaf_mode:
            leaf_disk = leaf_to_disk(leaf_positions=leaf_positions, leaf_scales=leaf_scales, leaf_rotations=leaf_rotations,save_flag=False) # list of open3d mesh
        else:
            print("Warning: no leaf primitives detected; continuing with branch-only StPr initialization.")
            leaf_disk = None

        self.appgs = self.build_appgs_from_stprs(mesh_cylinder,branch_scales,branch_quat, branch_feature_dc,branch_feature_rest,
                                                 leaf_disk,leaf_scales, leaf_rotations, leaf_feature_dc, leaf_feature_rest,
                                                 branch_label=branch_index, leaf_label=leaf_index,
                                                 samples_per_branch=stpr_appgs_per_stpr, samples_per_leaf=stpr_appgs_per_stpr,
                                                 semantic_dim=stpr_semantic_dim,
                                                 branch_positions=branch_positions, leaf_positions=leaf_positions,
                                                 scene_extent=scene_extent, max_scale_ratio=stpr_appgs_max_scale_ratio,
                                                 )
        self.leaf_disks = leaf_disk
        self.branch_cylinders = mesh_cylinder
        self.branch_label = branch_index
        self.leaf_label = leaf_index

        # Convert lists to tensors
        # stpr_rotations = np.roll(np.array(stpr_rotations), 1, axis=1)
        stpr_positions = torch.tensor(np.array(stpr_positions), dtype=torch.float, device=self.device)
        stpr_scales = torch.tensor(np.array(stpr_scales), dtype=torch.float, device=self.device)
        stpr_rotations = torch.tensor(np.array(stpr_rotations), dtype=torch.float, device=self.device)
        stpr_features_dc = torch.tensor(np.array(stpr_features_dc), dtype=torch.float, device=self.device)
        stpr_features_rest = torch.tensor(np.array(stpr_features_rest), dtype=torch.float, device=self.device)
        # stpr_opacities = torch.tensor(stpr_opacities, dtype=torch.float, device=self.device)
        # check nan in stpr_scales, rTypeError: can't convert cuda:7 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.eplace nan with 0.1
        stpr_scales = torch.nan_to_num(stpr_scales, nan=0.01, posinf=0.01, neginf=0.01).clamp(min=1e-5)
        stpr_sur_rots = torch.tensor(np.array(surf_rotations), dtype=torch.float, device=self.device)
        print(f"[DEBUG][stpr] num StPr before scale_filter={stpr_scales.shape[0]}")
        if scene_extent is None:
            extent_tensor = stpr_positions.max(dim=0).values - stpr_positions.min(dim=0).values
            scene_extent = float(torch.linalg.norm(extent_tensor).detach().cpu())
        print(f"[DEBUG][stpr] scene extent={scene_extent:.6g}")
        max_scale = stpr_scales.max(dim=1).values
        print(
            f"[DEBUG][stpr] scale summary min/mean/max="
            f"{max_scale.min().item():.6g}/{max_scale.mean().item():.6g}/{max_scale.max().item():.6g}"
        )
        scale_filter = (max_scale < stpr_max_scale_ratio * scene_extent) & (max_scale > stpr_min_scale_ratio * scene_extent)
        if not scale_filter.any():
            print("Warning: scale filter would remove all StPrs; keeping unfiltered primitives.")
            scale_filter = torch.ones_like(scale_filter, dtype=torch.bool)
        print(f"[DEBUG][stpr] num StPr after scale_filter={int(scale_filter.sum().item())}")
        stpr_positions = stpr_positions[scale_filter]
        stpr_scales = stpr_scales[scale_filter]
        stpr_scales = torch.log(stpr_scales)
        stpr_rotations = stpr_rotations[scale_filter]
        stpr_features_dc = stpr_features_dc[scale_filter]
        stpr_features_rest = stpr_features_rest[scale_filter]
        stpr_sur_rots = stpr_sur_rots[scale_filter]
        stpr_label = [lbl for lbl, keep in zip(stpr_label, scale_filter.detach().cpu().tolist()) if keep]
        if self.appgs is not None and self.nn_stpr_appgs is not None and scale_filter.numel() > int(scale_filter.sum().item()):
            if isinstance(self.nn_stpr_appgs, list):
                parent_old = torch.stack(self.nn_stpr_appgs, dim=0).to(self.device).view(-1)
            else:
                parent_old = self.nn_stpr_appgs.to(self.device).view(-1)
            app_keep = scale_filter[parent_old]
            remap = torch.full((scale_filter.shape[0],), -1, dtype=torch.long, device=self.device)
            remap[scale_filter] = torch.arange(int(scale_filter.sum().item()), dtype=torch.long, device=self.device)
            self.appgs = self.appgs.clone_subset(app_keep, copy_structure_metadata=True)
            self.nn_stpr_appgs = remap[parent_old[app_keep]].unsqueeze(1)
            print(f"[DEBUG][appgs] filtered AppGS after StPr scale_filter: {int(app_keep.sum().item())}/{app_keep.shape[0]}")
        stpr_opacities = self.inverse_opacity_activation( 0.5* torch.ones((stpr_positions.shape[0], 1),device=self.device))
        # Initialize a new GaussianModel for StPrs and return it
        self.structure_gs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.structure_gs.spatial_lr_scale = self.spatial_lr_scale
        self.structure_gs._xyz = nn.Parameter(stpr_positions.requires_grad_(True))
        self.structure_gs._scaling = nn.Parameter(stpr_scales.requires_grad_(True))
        self.structure_gs._rotation = nn.Parameter(stpr_rotations.requires_grad_(True))
        self.structure_gs._features_dc = nn.Parameter(stpr_features_dc.requires_grad_(True))
        self.structure_gs._features_rest = nn.Parameter(stpr_features_rest.requires_grad_(True))
        self.structure_gs._opacity = nn.Parameter(stpr_opacities.requires_grad_(True))
        self.structure_gs._semantic_logit = nn.Parameter(torch.ones((stpr_positions.shape[0], 1), dtype=torch.float, device=self.device).requires_grad_(True))
        if stpr_semantic_dim > 0:
            self.structure_gs._semantic_feature = nn.Parameter(torch.zeros((stpr_positions.shape[0], stpr_semantic_dim), dtype=torch.float, device=self.device).requires_grad_(True))
        self.structure_gs.max_radii2D = torch.zeros((stpr_positions.shape[0]), device=self.device)
        self.structure_gs._mask = torch.ones((stpr_positions.shape[0],), dtype=torch.float, device=self.device)
        self.structure_gs.exposure_mapping = self.exposure_mapping
        self.structure_gs.pretrained_exposures = None
        self.structure_gs.surf_rotations = stpr_sur_rots
        self.structure_gs.stpr_label = stpr_label
        p0 = []
        for label in stpr_label:
            if plant_prior == "branch_only" or no_leaf_mode:
                p0.append(0.9)
            elif label == "branch":
                p0.append(0.6)
            elif label == "leaf":
                p0.append(0.4)
            else:
                p0.append(0.5)
        p0 = torch.tensor(p0, dtype=torch.float, device=self.device).clamp(1e-4, 1.0 - 1e-4).unsqueeze(1)
        self.structure_gs._pst_logit = nn.Parameter(torch.log(p0 / (1.0 - p0)).requires_grad_(True))
        self.structure_gs.initialize_stpr_type_logits(stpr_label)
        exposure = self._exposure.detach()
        self.structure_gs._exposure = nn.Parameter(exposure.requires_grad_(True))
        print(f"Initialized {len(stpr_positions)} Structural Primitives (StPrs) from Gaussian clustering.")
        return self.structure_gs,self.appgs

    
    def build_appgs_from_stprs(self, mesh_cylinder,branch_scales,branch_rotations, branch_feature_dc, branch_feature_rest,
                               leaf_disk,leaf_scales, leaf_rotations,leaf_feature_dc, leaf_feature_rest,
                               branch_label, leaf_label,
                               samples_per_branch=10, samples_per_leaf=10, semantic_dim=0,
                               branch_positions=None, leaf_positions=None, scene_extent=None, max_scale_ratio=0.03):
        """
        Build Appearance Gaussians (AppGs) from the structural primitives (StPrs).
        """
        positions = []
        scales= [] 
        rotations = []
        new_features_dc = []
        new_features_rest = []
        app_label = []
        app_stpr_nn = []
        max_scale = None
        if scene_extent is not None and max_scale_ratio > 0:
            max_scale = float(scene_extent) * float(max_scale_ratio)

        if branch_positions is not None and len(branch_positions) > 0:
            for i, center in enumerate(branch_positions):
                scale_np = np.asarray(branch_scales[i], dtype=np.float32).copy()
                if max_scale is not None:
                    scale_np = np.clip(scale_np, 1e-6, max_scale)
                theta = torch.rand((samples_per_branch,), dtype=torch.float32) * (2.0 * torch.pi)
                axial = (torch.rand((samples_per_branch,), dtype=torch.float32) - 0.5) * (3.0 * float(scale_np[0]))
                radius = float(scale_np[1])
                local = torch.stack(
                    [axial, radius * torch.cos(theta), radius * torch.sin(theta)],
                    dim=1,
                )
                quat_np = np.asarray(branch_rotations[i], dtype=np.float32)
                rot_matrix = torch.tensor(R.from_quat(np.roll(quat_np, -1)).as_matrix(), dtype=torch.float32)
                pos = local @ rot_matrix.T + torch.tensor(center, dtype=torch.float32).view(1, 3)
                scale = torch.tensor((scale_np / np.sqrt(max(samples_per_branch, 1))), dtype=torch.float32).unsqueeze(0).repeat(pos.shape[0], 1)
                rot = torch.tensor(quat_np).unsqueeze(0).repeat(pos.shape[0], 1)
                feature_dc = torch.tensor(branch_feature_dc[i]).repeat(pos.shape[0], 1)
                feature_rest = torch.tensor(branch_feature_rest[i]).repeat(pos.shape[0], 1)
                positions.append(pos)
                scales.append(scale)
                rotations.append(rot)
                new_features_dc.append(feature_dc)
                new_features_rest.append(feature_rest)
                labels = ['branch'] * pos.shape[0]
                app_label.extend(labels)
                app_index = torch.tensor(branch_label[i]).unsqueeze(0).repeat(pos.shape[0], 1)
                app_stpr_nn.extend(app_index)


        if leaf_positions is not None and len(leaf_positions) > 0:
            for i, center in enumerate(leaf_positions):
                scale_np = np.asarray(leaf_scales[i], dtype=np.float32).copy()
                if max_scale is not None:
                    scale_np = np.clip(scale_np, 1e-6, max_scale)
                theta = torch.rand((samples_per_leaf,), dtype=torch.float32) * (2.0 * torch.pi)
                radius = torch.sqrt(torch.rand((samples_per_leaf,), dtype=torch.float32))
                local = torch.stack(
                    [2.0 * float(scale_np[0]) * radius * torch.cos(theta),
                     float(scale_np[1]) * radius * torch.sin(theta),
                     torch.zeros_like(theta)],
                    dim=1,
                )
                quat_np = np.asarray(leaf_rotations[i], dtype=np.float32)
                if quat_np.shape == (3, 3):
                    rot_matrix_np = quat_np
                    quat_np = np.roll(R.from_matrix(rot_matrix_np).as_quat(), 1).astype(np.float32)
                else:
                    rot_matrix_np = R.from_quat(np.roll(quat_np, -1)).as_matrix()
                rot_matrix = torch.tensor(rot_matrix_np, dtype=torch.float32)
                pos = local @ rot_matrix.T + torch.tensor(center, dtype=torch.float32).view(1, 3)
                scale = torch.tensor(scale_np / np.sqrt(max(samples_per_leaf, 1)), dtype=torch.float32).unsqueeze(0).repeat(pos.shape[0], 1)
                scale[:,2] = 1e-6
                rot = torch.tensor(quat_np).unsqueeze(0).repeat(pos.shape[0], 1)
                feature_dc = torch.tensor(leaf_feature_dc[i]).repeat(pos.shape[0], 1)
                feature_rest = torch.tensor(leaf_feature_rest[i]).repeat(pos.shape[0], 1)
                positions.append(pos)
                scales.append(scale)
                rotations.append(rot)
                new_features_dc.append(feature_dc)
                new_features_rest.append(feature_rest)
                labels = ['leaf']*pos.shape[0]
                app_label.extend(labels)
                app_index = torch.tensor(leaf_label[i]).unsqueeze(0).repeat(pos.shape[0], 1)
                app_stpr_nn.extend(app_index)


        num_total_samples = torch.vstack(positions).shape[0]
        new_opacities = self.inverse_opacity_activation(0.5* torch.ones((num_total_samples, 1),device=self.device))
        appgs_features_dc = torch.tensor(np.array(new_features_dc), dtype=torch.float, device=self.device)
        appgs_features_rest = torch.tensor(np.array(new_features_rest), dtype=torch.float, device=self.device)  
        appgs_features_dc = appgs_features_dc.reshape(-1, 1, 3)
        appgs_features_rest = appgs_features_rest.reshape(-1,15,3)
        self.appgs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.appgs.spatial_lr_scale = self.spatial_lr_scale
        self.appgs._xyz = nn.Parameter(torch.vstack(positions).to(self.device).requires_grad_(True))
        self.appgs._scaling = nn.Parameter(torch.log(torch.vstack((scales))).float().to(self.device).requires_grad_(True))
        self.appgs._rotation = nn.Parameter(torch.vstack(rotations).float().to(self.device).requires_grad_(True))
        self.appgs._features_dc = nn.Parameter(appgs_features_dc.requires_grad_(True))
        self.appgs._features_rest = nn.Parameter(appgs_features_rest.requires_grad_(True))
        self.appgs._opacity = nn.Parameter(new_opacities.requires_grad_(True))
        self.appgs._semantic_logit = nn.Parameter(torch.ones((num_total_samples, 1), dtype=torch.float, device=self.device).requires_grad_(True))
        if semantic_dim > 0:
            self.appgs._semantic_feature = nn.Parameter(torch.zeros((num_total_samples, semantic_dim), dtype=torch.float, device=self.device).requires_grad_(True))
        self.appgs.max_radii2D = torch.zeros((num_total_samples), device=self.device)
        self.appgs._mask = torch.ones((num_total_samples), dtype=torch.float, device=self.device)
        self.appgs.exposure_mapping = self.exposure_mapping
        self.appgs.pretrained_exposures = None
        self.appgs.app_label = app_label
        self.nn_stpr_appgs = app_stpr_nn
        exposure = self._exposure.detach()
        self.appgs._exposure = nn.Parameter(exposure.requires_grad_(True))
        print(f"Initialized {num_total_samples} Appearance Gaussians (AppGs) from StPrs.")
        return self.appgs

    def build_appgs_from_cylinder(self, cylinder_params, num_samples=50):
        """
        Build Appearance Gaussians (AppGs) by uniformly sampling points on the surface of the cylinder.
        
        Args:
            cylinder_params (dict): Dictionary containing cylinder parameters (center, axis, radius, height).
            num_samples (int): Number of samples to generate.
        
        Returns:
            GaussianModel: Initialized AppGs.
        """
        # Extract cylinder parameters
        C = cylinder_params["center"]  # (N, 3)
        A = cylinder_params["axis"]    # (N, 3)
        R = cylinder_params["radius"]  # (N,)
        H = cylinder_params["height"]  # (N,)

        # Normalize the axis vector
        A = A / torch.norm(A, dim=-1, keepdim=True)

        # Sample points on the top and bottom circles
        num_circle_samples = num_samples // 3
        theta = torch.linspace(0, 2 * torch.pi, num_circle_samples, device=self.device)  # (num_circle_samples,)
        circle_x = torch.cos(theta) * R[:, None]  # (N, num_circle_samples)
        circle_y = torch.sin(theta) * R[:, None]  # (N, num_circle_samples)

        # Compute two orthogonal vectors to the axis
        v1 = torch.tensor([1, 0, 0], device=self.device).repeat(A.shape[0], 1)
        v1 = v1 - (torch.sum(v1 * A, dim=-1, keepdim=True) * A)
        v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)

        v2 = torch.cross(A, v1, dim=-1)  # Get the second orthogonal vector

        # Compute the points on the circles
        top_center = C + (H[:, None] / 2) * A
        bottom_center = C - (H[:, None] / 2) * A

        circle_points_top = top_center[:, None, :] + circle_x[:, :, None] * v1[:, None, :] + circle_y[:, :, None] * v2[:, None, :]
        circle_points_bottom = bottom_center[:, None, :] + circle_x[:, :, None] * v1[:, None, :] + circle_y[:, :, None] * v2[:, None, :]

        # Sample points on the side surface
        num_side_samples = num_samples - 2 * num_circle_samples
        side_theta = torch.linspace(0, 2 * torch.pi, num_side_samples, device=self.device)
        z_offsets = torch.linspace(-0.5, 0.5, num_side_samples, device=self.device) * H[:, None]

        side_x = torch.cos(side_theta) * R[:, None]
        side_y = torch.sin(side_theta) * R[:, None]
        side_z = z_offsets

        side_points = C[:, None, :] + side_x[:, :, None] * v1[:, None, :] + side_y[:, :, None] * v2[:, None, :] + side_z[:, :, None] * A[:, None, :]

        # Combine all sampled points
        sampled_points = torch.cat([circle_points_top, circle_points_bottom, side_points], dim=1).reshape(-1, 3)

        # Initialize Appearance Gaussians (AppGs)
        num_total_samples = sampled_points.shape[0]
        scaling_st = self.structure_gs._scaling
        scales = scaling_st.unsqueeze(1).repeat(1, num_samples, 1) / num_samples

        scales= scales.reshape(-1,3)
        # set all scale to -1 
        scales = torch.log(scales)
        scales[torch.isnan(scales)] = -3
        quaternion_st = self.structure_gs._rotation
        rots = quaternion_st.unsqueeze(1).repeat(1, num_samples, 1)
        rots = rots.reshape(-1,4)
        # init features
        features_dc = self.structure_gs._features_dc.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1,1,3)
        # assign features to each gaussian from the same structure
        features_rest = self.structure_gs._features_rest.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(-1,15,3)
        # init opacity
        # new_opacities = self.structure_gs._opacity.unsqueeze(1).repeat(1, samples_per_stgs, 1).reshape(-1,1)    
        new_opacities = self.inverse_opacity_activation(0.1 * torch.ones((sampled_points.shape[0], 1), dtype=torch.float, device=self.device))
        # new_opacities = torch.ones((sampled_points.shape[0], 1), dtype=torch.float, device=self.device)
        self.appgs = GaussianModel(sh_degree=self.max_sh_degree, optimizer_type=self.optimizer_type, device=self.device)
        self.appgs.spatial_lr_scale = self.spatial_lr_scale
        self.appgs._xyz = nn.Parameter(sampled_points.requires_grad_(True))
        self.appgs._scaling = nn.Parameter(scales.requires_grad_(True))
        self.appgs._rotation = nn.Parameter(rots.requires_grad_(True))
        self.appgs._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self.appgs._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self.appgs._opacity = nn.Parameter(new_opacities.requires_grad_(True))
        self.appgs._semantic_logit = nn.Parameter(torch.ones((num_total_samples, 1), dtype=torch.float, device=self.device).requires_grad_(True))
        self.appgs.max_radii2D = torch.zeros((sampled_points.shape[0]), device=self.device)
        self.appgs._mask = torch.ones((num_total_samples,), dtype=torch.float, device=self.device)
        self.appgs.exposure_mapping = self.exposure_mapping
        self.appgs.pretrained_exposures = None
        exposure = self._exposure.detach()
        self.appgs._exposure = nn.Parameter(exposure.requires_grad_(True))

        # print(f"Initialized {num_total_samples} Appearance Gaussians (AppGs) from Cylinder Surface.")
        return self.appgs

    def compute_gaussian_overlap_with_neighbors(
        self, 
        neighbor_idx,
        use_gaussian_center_only=True,
        n_samples_to_compute_overlap=32,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:, 0]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self._scaling.detach()
            quaternions = self._rotation.detach()
        else:
            scaling = self._scaling
            quaternions = self._rotation
        
        # Samples points in the corresponding gaussians
        if use_gaussian_center_only:
            n_samples_to_compute_overlap = 1
            gaussian_samples = self._xyz[point_idx].unsqueeze(1) + 0.  # (n_points, n_samples_to_compute_overlap, 3)
        else:
            gaussian_samples = self._xyz[point_idx].unsqueeze(1) + quaternion_apply(
                quaternions[point_idx].unsqueeze(1), 
                scaling[point_idx].unsqueeze(1) * torch.randn(
                    n_points, n_samples_to_compute_overlap, 3, 
                    device=self.device)
                )  # (n_points, n_samples_to_compute_overlap, 3)
        
        # >>> We will now compute the gaussian weight of all samples, for each neighbor gaussian.
        # We start by computing the shift between the samples and the neighbor gaussian centers.
        neighbor_center_to_samples = gaussian_samples.unsqueeze(1) - self._xyz[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # We compute the inverse of the scaling of the neighbor gaussians. 
        # For 2D gaussians, we implictly project the samples on the plane of each gaussian; 
        # We do so by setting the inverse of the scaling of the gaussian to 0 in the direction of the gaussian normal (i.e. 0-axis).
        inverse_scales = 1. / scaling[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, 1, 3)
        
        # We compute the "gaussian distance" of all samples to the neighbor gaussians, i.e. the norm of the unrotated shift,
        # weighted by the inverse of the scaling of the neighbor gaussians.
        gaussian_distances = inverse_scales * quaternion_apply(
            quaternion_invert(quaternions[neighbor_idx[:, neighbor_start_idx:]]).unsqueeze(2), 
            neighbor_center_to_samples
            )  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # Now we can compute the gaussian weights of all samples, for each neighbor gaussian.
        # We then sum them to get the gaussian overlap of each neighbor gaussian.
        gaussian_weights = torch.exp(-1./2. * (gaussian_distances ** 2).sum(dim=-1))  # (n_points, n_neighbors-1, n_samples_to_compute_overlap)
        gaussian_overlaps = gaussian_weights.mean(dim=-1)  # (n_points, n_neighbors-1)
        
        # If needed, we weight the gaussian overlaps by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_overlaps = gaussian_overlaps * weights
            
        return gaussian_overlaps
    
    def compute_gaussian_alignment_with_neighbors(
        self,
        neighbor_idx,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        std_factor = 1.,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:,]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self._scaling.detach()
            quaternions = self._rotation.detach()
        else:
            scaling = self._scaling
            quaternions = self._rotation
        
        # We compute scaling, inverse quaternions and centers for all gaussians and their neighbors
        all_scaling = scaling[neighbor_idx]
        all_invert_quaternions = quaternion_invert(quaternions)[neighbor_idx]
        all_centers = self._xyz[neighbor_idx]
        
        # We compute direction vectors between the gaussians and their neighbors
        neighbor_shifts = all_centers[:, neighbor_start_idx:] - all_centers[:, :neighbor_start_idx]
        neighbor_distances = neighbor_shifts.norm(dim=-1).clamp(min=1e-8)
        neighbor_directions = neighbor_shifts / neighbor_distances.unsqueeze(-1)
        
        # We compute the standard deviations of the gaussians in the direction of their neighbors,
        # and reciprocally in the direction of the gaussians.
        standard_deviations_gaussians = (
            all_scaling[:, 0:neighbor_start_idx]
            * quaternion_apply(all_invert_quaternions[:, 0:neighbor_start_idx], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        standard_deviations_neighbors = (
            all_scaling[:, neighbor_start_idx:]
            * quaternion_apply(all_invert_quaternions[:, neighbor_start_idx:], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        # The distance between the gaussians and their neighbors should be the sum of their standard deviations (up to a factor)
        stabilized_distance = (standard_deviations_gaussians + standard_deviations_neighbors) * std_factor
        gaussian_alignment = (neighbor_distances / stabilized_distance.clamp(min=1e-8) - 1.).abs()
        
        # If needed, we weight the gaussian alignments by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_alignment = gaussian_alignment * weights
            
        return gaussian_alignment

    def get_normals(self, estimate_from_points=False, neighborhood_size:int=32):
        """Returns the normals of the Gaussians.

        Args:
            estimate_from_points (bool, optional): _description_. Defaults to False.
            neighborhood_size (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        if estimate_from_points:
            normals = estimate_pointcloud_normals(
                self.points[None], #.detach(), 
                neighborhood_size=neighborhood_size,
                disambiguate_directions=True
                )[0]
        else:
            if self.binded_to_surface_mesh:
                normals = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1).view(-1, 1, 3)
                normals = normals.expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 3)
            else:
                normals = self.get_smallest_axis()
        return normals
    
    def get_neighbors_of_random_points(self, num_samples):
        if num_samples >= 0:
            sampleidx = torch.randperm(len(self._xyz), device=self.device)[:num_samples]        
            return self.knn_idx[sampleidx]
        else:
            return self.knn_idx
    
    def get_local_variance(self, values:torch.Tensor, neighbor_idx:torch.Tensor):
        """_summary_

        Args:
            values (_type_): Shape is (n_points, n_values)
            neighbor_idx (_type_): Shape is (n_points, n_neighbors)
        """
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        return (neighbor_values - neighbor_values.mean(dim=1, keepdim=True)).pow(2).sum(dim=-1).mean(dim=1)
    
    def get_local_distance2(
        self, 
        values:torch.Tensor, 
        neighbor_idx:torch.Tensor, 
        weights:torch.Tensor=None,
        ):
        """_summary_

        Args:
            values (torch.Tensor): Shape is (n_points, n_values)
            neighbor_idx (torch.Tensor): Shape is (n_points, n_neighbors)
            weights (torch.Tensor, optional): Shape is (n_points, n_neighbors). Defaults to None.

        Returns:
            _type_: _description_
        """
        
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        distance2 = neighbor_values[:, 1:] - neighbor_values[:, :1]  # Shape is (n_points, n_neighbors-1, n_values)
        distance2 = distance2.pow(2).sum(dim=-1)  # Shape is (n_points, n_neighbors-1)
        
        if weights is not None:
            distance2 = distance2 * weights

        return distance2.mean(dim=1)  # Shape is (n_points,)
    
    def reset_neighbors(self):
        # Compute KNN               
        with torch.no_grad():
            knns = knn_points(self._xyz[None], self._xyz[None], K=self.knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

    def update_nn_between_appgs_and_stprs(self):
        app_points = self.appgs._xyz
        stpr_points = self.structure_gs._xyz
        knns = knn_points(app_points[None], stpr_points[None], K=1)
        self.nn_stpr_appgs = knns.idx[0]
    
    def get_edge_neighbors(self, k_neighbors, 
                           edges=None, triangle_vertices=None,):
        if edges is None:
            edges = self.triangle_border_edges
        if triangle_vertices is None:
            triangle_vertices = self.triangle_vertices
        
        # We select the closest edges based on the position of the edge center
        edge_centers = triangle_vertices[edges].mean(dim=-2)
        
        # TODO: Compute only for vertices with high opacity? Remove points with low opacity?
        edge_knn = knn_points(edge_centers[None], edge_centers[None], K=8)
        edge_knn_idx = edge_knn.idx[0]
        
        return edge_knn_idx

    def get_smallest_axis(self, return_idx=False):  
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = quaternion_to_matrix(self._rotation)
        smallest_axis_idx = self._scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True,):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mask is None:
            scaling = self._scaling
        else:
            scaling = self._scaling[mask]
        
        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])
        
        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        # cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)
        cum_probs = areas / areas.sum(dim=-1, keepdim=True)
        
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)
        if mask is not None:
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]
        
        random_points = self._xyz[random_indices] + quaternion_apply(
            self._rotation[random_indices], 
            sampling_scale_factor * self._scaling[random_indices] * torch.randn_like(self._xyz[random_indices]))
        
        return random_points, random_indices
    
    def drop_low_opacity_points(self, opacity_threshold=0.5):
        mask = self.get_opacity[...,0] < opacity_threshold  # 1e-3, 0.5
        self.prune_points(mask)
        print(f"Dropped {mask.sum()} points with opacity below {opacity_threshold}.")
        print(f"""Remaining points: {len(self._xyz)}""")  
    
    def convert_gs_to_cylinders(self, sigma=3.0):
        cov = self.get_covariance(return_full=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        lambda1, lambda2, lambda3 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
        radius = (sigma * torch.sqrt(torch.maximum(lambda1, lambda2)))
        axis = eigenvectors[:, :, 2]  # (N, 3)
        height = (sigma * torch.sqrt(lambda3))
        center = self._xyz
        cylinder_params = {
            "center": center,
            "axis": axis,
            "radius": radius,
            "height": height,
        }
        return cylinder_params

    def gs_to_graph(self, k=3, filename="gaussian_graph.ply"):
        """
        Save Structural Gaussians (StPrs) as a PLY file with edges based on nearest neighbors.

        Parameters:
        - stprs_xyz (torch.Tensor): Tensor of shape (N, 3), Gaussian center positions.
        - k (int): Number of nearest neighbors to connect.
        - filename (str): Output file name for the PLY file.
        """
        # Ensure tensor is detached and converted to numpy
        xyz = self._xyz.detach().cpu().numpy()
        num_points = xyz.shape[0]

        # Compute k-nearest neighbors
        tree = cKDTree(xyz)
        distances, indices = tree.query(xyz, k=k+1)  # k+1 to include self, will remove later

        # Prepare vertex data
        vertices = [(xyz[i][0], xyz[i][1], xyz[i][2]) for i in range(num_points)]

        # Prepare edge data (store as list of tuples)
        edges = []
        for i in range(num_points):
            for j in indices[i][1:]:  # Skip self-connection (first index)
                edges.append((i, j))

        # Convert to NumPy structured arrays
        vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        edge_dtype = np.dtype([("vertex1", "i4"), ("vertex2", "i4")])

        vertex_array = np.array(vertices, dtype=vertex_dtype)
        edge_array = np.array(edges, dtype=edge_dtype)

        # Create PLY elements
        vertex_element = PlyElement.describe(vertex_array, "vertex")
        edge_element = PlyElement.describe(edge_array, "edge")

        # Save to PLY file
        PlyData([vertex_element, edge_element], text=True).write(filename)
        print(f"PLY file saved as {filename}")
        
    def gs_cylinder_distance(self, nn_index, cylinder_params):
        # Extract Gaussian parameters
        C = cylinder_params["center"][nn_index].squeeze(1)  # (M, 3) Nearest cylinder centers
        A = cylinder_params["axis"][nn_index].squeeze(1)    # (M, 3) Nearest cylinder axes
        R = cylinder_params["radius"][nn_index].squeeze(1)  # (M,) Nearest cylinder radii
        H = cylinder_params["height"][nn_index].squeeze(1)  # (M,) Nearest cylinder heights

        # Extract AppG positions
        P = self._xyz  # (M, 3) AppG center positions

        # Step 1: Project AppG centers onto the cylinder axis
        AP = P - C  # (M, 3) Vector from cylinder center to AppG
        proj_scalar = torch.sum(AP * A, dim=-1)  # (M,) Projection length along axis
        proj_point = C + proj_scalar.unsqueeze(-1) * A  # (M, 3) Projected points on the cylinder axis

        # Step 2: Compute radial distance from projected point to AppG
        radial_vector = P - proj_point  # (M, 3)
        radial_distance = torch.norm(radial_vector, dim=-1)  # (M,)
        distance_to_surface = radial_distance - R  # (M,)
        return distance_to_surface.mean()
    
    def low_freq_loss(self):
        cov = self.get_covariance(return_full=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        freq = 1.0 / torch.sqrt(eigenvalues+ 1e-6)
        return freq.mean() 

    def merge_gaussians(self, gaussian_overlaps, nn_index,threshold=0.5):
        """
        Merge Gaussians in a GaussianModel based on overlap.

        Args:
            gaussian_model: GaussianModel instance
            gaussian_overlaps (torch.Tensor): (N, N) overlap matrix.
            threshold (float): Overlap threshold for merging.
        """
        clusters = self.find_merge_groups(gaussian_overlaps, threshold)

        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_opacity = []
        new_features_dc = []
        new_features_rest = []
        new_max_radii2D = []
        new_tmp_radii = []
        new_denom = []
        new_xyz_gradient_accum = []
        new_exposure = []

        for cluster in clusters:
            if len(cluster) == 1:
                # If only one Gaussian, keep it unchanged
                idx = cluster[0]
                new_xyz.append(self._xyz[idx])
                new_scaling.append(self._scaling[idx])
                new_rotation.append(self._rotation[idx])
                new_opacity.append(self._opacity[idx])
                new_features_dc.append(self._features_dc[idx])
                new_features_rest.append(self._features_rest[idx])
                new_max_radii2D.append(self.max_radii2D[idx])
                new_tmp_radii.append(self.tmp_radii[idx])
                new_denom.append(self.denom[idx])
                new_xyz_gradient_accum.append(self.xyz_gradient_accum[idx])
                new_exposure.append(self._exposure[idx])
            else:
                # Merge Gaussians in the cluster
                indices = torch.tensor(cluster, device=self.device)
                merged_xyz = torch.mean(self._xyz[indices], dim=0)
                merged_scaling = torch.mean(self._scaling[indices], dim=0)
                merged_rotation = torch.mean(self._rotation[indices], dim=0)  # Simple average
                merged_opacity = torch.mean(self._opacity[indices])  # Simple average

                new_xyz.append(merged_xyz)
                new_scaling.append(merged_scaling)
                new_rotation.append(merged_rotation)
                new_opacity.append(merged_opacity)
                new_features_dc.append(self._features_dc[indices[0]])  # Use the first Gaussian's features
                new_features_rest.append(self._features_rest[indices[0]])  # Use the first Gaussian's features
                new_max_radii2D.append(self.max_radii2D[indices].max())
                new_tmp_radii.append(self.tmp_radii[indices].max())
                new_denom.append(self.denom[indices].sum())
                new_xyz_gradient_accum.append(self.xyz_gradient_accum[indices].sum())
                new_exposure.append(self._exposure[indices[0]])  # Use the first Gaussian's exposure
                
        new_opacity_fixed = [op.unsqueeze(0) if op.dim() == 0 else op for op in new_opacity]
        new_xyz = torch.stack(new_xyz)
        new_scaling = torch.stack(new_scaling)
        new_rotation = torch.stack(new_rotation)
        new_opacity_fixed = torch.stack(new_opacity_fixed)
        new_features_dc = torch.stack(new_features_dc)
        new_features_rest = torch.stack(new_features_rest)
        new_max_radii2D = torch.tensor(new_max_radii2D, device=self.device)
        new_tmp_radii = torch.tensor(new_tmp_radii, device=self.device)
        new_denom = torch.tensor(new_denom, device=self.device)
        new_xyz_gradient_accum = torch.tensor(new_xyz_gradient_accum, device=self.device)
        new_exposure = torch.stack(new_exposure)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity_fixed, new_scaling, new_rotation, new_tmp_radii)
        
        print(f"Merged Gaussians. New count: {len(new_xyz)}")

    def find_merge_groups(self,gaussian_overlaps, threshold=0.5):
        """
        Find connected components in the Gaussian overlap graph.
        
        Args:
            gaussian_overlaps (torch.Tensor): (N, N) overlap matrix.
            threshold (float): Overlap threshold for merging.

        Returns:
            List[List[int]]: List of clusters, where each cluster is a list of indices.
        """
        N = gaussian_overlaps.shape[0]
        adjacency_matrix = (gaussian_overlaps > threshold).float()
        
        # Union-Find to find connected components
        parent = list(range(N))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x  # Merge groups

        # Build the merge groups
        for i in range(N):
            for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
                if adjacency_matrix[i, j] > 0:
                    union(i, j)

        # Group Gaussians by connected components
        clusters = {}
        for i in range(N):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        return list(clusters.values())

    def build_surface(self, plant_prior="mixed"):
        """Soft cylinder/disk mixture binding loss for every AppGS/StPr pair."""
        stpr_pos = self.structure_gs.get_xyz
        stpr_scale = self.structure_gs.get_scaling
        stpr_rot = self.structure_gs.get_rotation
        cyl_param, cylinder_mesh = stpr_to_cylinder(stpr_pos, stpr_scale, stpr_rot, save_flag=False)
        disk_param = stpr_to_disk(stpr_pos, stpr_scale, stpr_rot, save_flag=False)
        self.cylinder_mesh = cylinder_mesh

        if isinstance(self.nn_stpr_appgs, list):
            parent = torch.stack(self.nn_stpr_appgs, dim=0).to(self.device).view(-1)
        else:
            parent = self.nn_stpr_appgs.to(self.device).view(-1)

        xyz = self.appgs._xyz
        d_cy = gs_to_cylinder_distance(xyz, parent, cyl_param).view(-1)
        d_di = gs_to_disk_distance(xyz, parent, disk_param).view(-1)
        if plant_prior == "branch_only":
            return (0.95 * d_cy + 0.05 * d_di).mean()
        p_st = torch.sigmoid(self.structure_gs._pst_logit).view(-1)[parent]
        return (p_st * d_cy + (1.0 - p_st) * d_di).mean()
        
        

        
        
    def compute_gaussian_binding_loss(self, n_samples=1, reduction=Literal['sum','mean'],method=Literal['surface', 'mahalanobis'], plant_prior="mixed"):
        """
        Compute binding loss from appgs to structure_gs.
        This reflects how well appgs are spatially explained by stprs.

        Args:
            nn_idx: LongTensor (N, K) - stpr neighbors for each appg
            n_samples: int - how many samples to simulate per gaussian
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            loss: scalar or (N,) if reduction='none'
        """
        nn_idx = self.nn_stpr_appgs
        N = len(nn_idx)

        if method == "mahalanobis":
            # 1. Sample appg points
            if n_samples == 1:
                samples = self.appgs._xyz.unsqueeze(1)  # (N, 1, 3)
            else:
                samples = self.appgs._xyz.unsqueeze(1) + quaternion_apply(
                    self.appgs._rotation.unsqueeze(1),
                    self.appgs._scaling.unsqueeze(1) * torch.randn(N, n_samples, 3, device=self.device)
                )  # (N, n_samples, 3)

            # 2. Neighbor gaussians
            nbr_xyz   = self.structure_gs._xyz[nn_idx]         # (N, K, 3)
            nbr_rot   = self.structure_gs._rotation[nn_idx]    # (N, K, 4)
            nbr_scale = self.structure_gs._scaling[nn_idx]     # (N, K, 3)

            # 3. Compute offset from appgs to neighbor centers
            offset = samples.unsqueeze(1) - nbr_xyz.unsqueeze(2)  # (N, K, n_samples, 3)

            # 4. Transform offset to local stpr space
            local_offset = quaternion_apply(
                quaternion_invert(nbr_rot).unsqueeze(2),  # (N, K, 1, 4)
                offset  # (N, K, n_samples, 3)
            )

            # 5. Normalize by inverse scale
            normed = local_offset / nbr_scale.unsqueeze(2)  # (N, K, n_samples, 3)

            # 6. Distance as binding loss
            dist = torch.norm(normed, dim=-1)  # (N, K, n_samples)

            dist = dist.mean(dim=-1)  # (N, K) → mean over samples

            # 7. Aggregate per-appg
            loss_per_appg = dist.mean(dim=-1)  # (N,)

            if reduction == 'mean':
                return loss_per_appg.mean()
            elif reduction == 'sum':
                return loss_per_appg.sum()
            else:
                return loss_per_appg  # (N,)
        elif method == "surface":
            """ approximate the distance between appg and stpr by the distance between appg and the surface of stpr """
            loss_bind = self.build_surface(plant_prior=plant_prior)
            return loss_bind

    def stpr_to_graph(self,opacity_threshold=0, anisotrpopy_threshold=1,save_mst=False, min_branch_candidates=16):  # 0.2 ,30
        # MST for grpah extraction
        # step1: Noise filtering
        if self.structure_gs._pst_logit is not None:
            p_branch = torch.sigmoid(self.structure_gs._pst_logit).view(-1)
            branch_mask = p_branch > 0.5
        else:
            p_branch = None
            branch_mask = torch.tensor([lbl == 'branch' for lbl in self.structure_gs.stpr_label], device=self.device)
        stpr_opcaity = self.structure_gs._opacity
        # keep = stpr_opcaity > opacity_threshold # low opacity filter
        # keep = keep.flatten()
        # leaf like filter
        anisotropy = self.structure_gs.get_scaling[:,0] / self.structure_gs.get_scaling[:,1]
        anisotropy = torch.max(anisotropy, 1.0/anisotropy)
        keep = (anisotropy > anisotrpopy_threshold).flatten()
        
        # step2: stpr to edge
        keep = keep & branch_mask
        if int(keep.sum().item()) < min_branch_candidates and self.structure_gs.get_xyz.shape[0] >= 2:
            fallback_pool = (anisotropy > anisotrpopy_threshold).flatten()
            if not fallback_pool.any():
                fallback_pool = torch.ones_like(keep, dtype=torch.bool)
            pool_idx = torch.nonzero(fallback_pool, as_tuple=False).view(-1)
            if p_branch is not None:
                aniso_score = torch.log(anisotropy.flatten().clamp(min=1.0))
                aniso_score = aniso_score / aniso_score.max().clamp(min=1e-6)
                score = p_branch + 0.25 * aniso_score
            else:
                score = anisotropy.flatten()
            k = min(max(min_branch_candidates, int(branch_mask.sum().item())), pool_idx.shape[0])
            selected = pool_idx[torch.topk(score[pool_idx], k=k, largest=True).indices]
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[selected] = True
            print(
                f"[DEBUG][graph] branch candidates below {min_branch_candidates}; "
                f"using top-{k} soft/elongated StPr candidates for graph debug."
            )
        print(f"[DEBUG][graph] num branch StPr after pruning={int(keep.sum().item())}")
        if not keep.any():
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_edges = np.empty((0, 2), dtype=np.int32)
            return empty_edges, empty_points, torch.tensor(0.0, device=self.device)
        center = self.structure_gs.get_xyz[keep] # (N,3)
        scales = self.structure_gs.get_scaling[keep]
        rot = self.structure_gs.get_rotation[keep]
        rot_matrix = quaternion_to_matrix(rot) #
        u = rot_matrix[:,:,0]
        h = self.structure_gs.get_scaling[keep][:,0] * 1.5  
        top = center + h.unsqueeze(1) * u
        bottom = center - h.unsqueeze(1) * u
        mst_edges, points = build_mst_from_endpoints(top,bottom)
        if save_mst:
            save_mst_ply(points, mst_edges)
        # graph loss 
        loss_graph = mst_loss(top,bottom,rot_matrix,mst_edges) 
        return mst_edges, points,loss_graph

    def tree_constrained_stpr_loss(
            self,
            min_branch_candidates=16,
            knn=12,
            constraint_mode="sfs_invariants",
            projection="forest",
            root_axis=2,
            forest_max_edge_length=0.0,
            forest_max_edge_cost=0.0,
            distance_weight=1.0,
            center_distance_weight=0.25,
            angle_weight=0.25,
            radius_cost_weight=0.25,
            branch_weight=0.5,
            root_direction_weight=0.0,
            use_graph_gnn=False,
            graph_gnn_hidden_dim=64,
            graph_gnn_layers=2,
            graph_gnn_weight=1.0,
            graph_gnn_lr=0.0025,
            sfs_weight=1.0,
            radius_weight=0.25,
            angle_loss_weight=0.25,
            degree_weight=0.02,
            leaf_attachment_weight=0.0,
            vascular_flow_weight=0.0,
            trunk_root_weight=0.0,
            trunk_flow_weight=0.0,
            trunk_radius_weight=0.0,
            max_degree=4,
            radius_margin=0.0,
            branch_label_weight=0.05,
            branch_label_target=0.8,
            branch_label_min_prob=0.25,
            neg_per_pos=3,
            anisotropy_threshold=1.0):
        if self.structure_gs is None or self.structure_gs.get_xyz.shape[0] < 2:
            device = self.device if self.device is not None else self.get_xyz.device
            return torch.tensor(0.0, device=device)

        stprs = self.structure_gs
        device = stprs.get_xyz.device
        type_probs = F.softmax(stprs._stpr_type_logit, dim=-1) if stprs._stpr_type_logit is not None else None
        if type_probs is not None:
            p_branch = (type_probs[:, 0] + type_probs[:, 1]).clamp(0.0, 1.0)
            branch_mask = p_branch > 0.5
        elif stprs._pst_logit is not None:
            p_branch = torch.sigmoid(stprs._pst_logit).view(-1)
            branch_mask = p_branch > 0.5
        elif stprs.stpr_label is None:
            p_branch = None
            branch_mask = torch.ones((stprs.get_xyz.shape[0],), dtype=torch.bool, device=device)
        else:
            p_branch = None
            branch_mask = torch.tensor([lbl == "branch" for lbl in stprs.stpr_label], dtype=torch.bool, device=device)

        scales_all = stprs.get_scaling
        anisotropy = scales_all[:, 0] / scales_all[:, 1].clamp(min=1e-8)
        anisotropy = torch.maximum(anisotropy, 1.0 / anisotropy.clamp(min=1e-8)).flatten()
        keep = (anisotropy > anisotropy_threshold) & branch_mask
        if int(keep.sum().item()) < min_branch_candidates:
            pool = torch.nonzero(anisotropy > anisotropy_threshold, as_tuple=False).view(-1)
            if pool.numel() == 0:
                pool = torch.arange(stprs.get_xyz.shape[0], device=device)
            if p_branch is not None:
                aniso_score = torch.log(anisotropy.clamp(min=1.0))
                aniso_score = aniso_score / aniso_score.max().clamp(min=1e-6)
                score = p_branch + 0.25 * aniso_score
            else:
                score = anisotropy
            k = min(max(min_branch_candidates, int(branch_mask.sum().item())), pool.numel())
            selected = pool[torch.topk(score[pool], k=k, largest=True).indices]
            keep = torch.zeros_like(branch_mask, dtype=torch.bool)
            keep[selected] = True

        branch_indices = torch.nonzero(keep, as_tuple=False).view(-1)
        n = int(branch_indices.numel())
        if n < 2:
            return stprs.get_xyz.sum() * 0.0

        xyz = stprs.get_xyz[branch_indices]
        scales = stprs.get_scaling[branch_indices]
        rot = stprs.get_rotation[branch_indices]
        rot_matrix = quaternion_to_matrix(rot)
        axis = F.normalize(rot_matrix[:, :, 0], dim=-1, eps=1e-8)
        radius = scales[:, 1:].mean(dim=-1)
        half_length = scales[:, 0] * 1.5
        top = xyz + half_length[:, None] * axis
        bottom = xyz - half_length[:, None] * axis
        p_branch_kept = torch.sigmoid(stprs._pst_logit[branch_indices]).view(-1) if stprs._pst_logit is not None else torch.ones((n,), device=device)
        if type_probs is not None:
            type_probs_kept = type_probs[branch_indices]
            p_trunk_kept = type_probs_kept[:, 0]
            p_branch_only_kept = type_probs_kept[:, 1]
            p_leaf_kept = type_probs_kept[:, 2]
            p_branch_kept = (p_trunk_kept + p_branch_only_kept).clamp(0.0, 1.0)
        else:
            p_trunk_kept = torch.zeros((n,), dtype=xyz.dtype, device=device)
            p_branch_only_kept = p_branch_kept
            p_leaf_kept = 1.0 - p_branch_kept

        query_k = min(max(int(knn), 1) + 1, n)
        with torch.no_grad():
            dist_detached = torch.cdist(xyz.detach(), xyz.detach(), p=2)
            _, nbr = torch.topk(dist_detached, k=query_k, largest=False, dim=1)
            edge_set = set()
            for i in range(n):
                for j in nbr[i, 1:].detach().cpu().tolist():
                    a, b = sorted((int(i), int(j)))
                    if a != b:
                        edge_set.add((a, b))
            endpoints = torch.stack([top.detach(), bottom.detach()], dim=1).reshape(-1, 3)
            endpoint_dist = torch.cdist(endpoints, endpoints, p=2)
            endpoint_dist[torch.arange(2 * n, device=device), torch.arange(2 * n, device=device)] = float("inf")
            endpoint_k = min(max(int(knn), 1) + 1, 2 * n)
            _, endpoint_nbr = torch.topk(endpoint_dist, k=endpoint_k, largest=False, dim=1)
            for endpoint_i in range(2 * n):
                node_i = endpoint_i // 2
                for endpoint_j in endpoint_nbr[endpoint_i].detach().cpu().tolist():
                    if endpoint_j >= 2 * n:
                        continue
                    node_j = int(endpoint_j) // 2
                    a, b = sorted((int(node_i), node_j))
                    if a != b:
                        edge_set.add((a, b))
            if not edge_set:
                return xyz.sum() * 0.0
            edge_np = np.asarray(sorted(edge_set), dtype=np.int64)

        edge_idx = torch.as_tensor(edge_np, dtype=torch.long, device=device)
        a = edge_idx[:, 0]
        b = edge_idx[:, 1]
        delta = xyz[b] - xyz[a]
        center_dist = torch.linalg.norm(delta, dim=-1).clamp(min=1e-8)
        endpoint_pairs_a = torch.stack([top[a], top[a], bottom[a], bottom[a]], dim=1)
        endpoint_pairs_b = torch.stack([top[b], bottom[b], top[b], bottom[b]], dim=1)
        endpoint_delta = endpoint_pairs_b - endpoint_pairs_a
        endpoint_pair_dist = torch.linalg.norm(endpoint_delta, dim=-1).clamp(min=1e-8)
        endpoint_dist_min, endpoint_choice = endpoint_pair_dist.min(dim=1)
        endpoint_delta_min = endpoint_delta[torch.arange(endpoint_delta.shape[0], device=device), endpoint_choice]
        edge_dir = endpoint_delta_min / endpoint_dist_min[:, None]
        axis_align = torch.maximum(torch.abs((axis[a] * edge_dir).sum(dim=-1)), torch.abs((axis[b] * edge_dir).sum(dim=-1)))
        radius_ratio = torch.minimum(radius[a], radius[b]) / torch.maximum(radius[a], radius[b]).clamp(min=1e-8)
        radius_penalty_cost = 1.0 - radius_ratio
        branch_conf = torch.minimum(p_branch_kept[a], p_branch_kept[b])
        trunk_continuity = torch.minimum(p_trunk_kept[a], p_trunk_kept[b])
        leaf_pair_penalty = torch.maximum(p_leaf_kept[a], p_leaf_kept[b])
        root_axis_idx = int(root_axis) if 0 <= int(root_axis) < 3 else 2
        root_direction_cost = 1.0 - torch.abs(delta[:, root_axis_idx]) / center_dist
        cost = (
            distance_weight * endpoint_dist_min
            + center_distance_weight * center_dist
            + angle_weight * (1.0 - axis_align)
            + radius_cost_weight * radius_penalty_cost
            - branch_weight * branch_conf
            - 0.25 * branch_weight * trunk_continuity
            + 0.5 * branch_weight * leaf_pair_penalty
            + root_direction_weight * root_direction_cost
        )
        learned_edge_logit = None
        if use_graph_gnn:
            graph_scale = torch.quantile(center_dist.detach(), 0.75).clamp(min=1e-4)
            xyz_centered = (xyz - xyz.detach().mean(dim=0, keepdim=True)) / graph_scale
            node_height = xyz_centered[:, root_axis_idx:root_axis_idx + 1] if 0 <= root_axis_idx < 3 else xyz_centered[:, 2:3]
            node_feats = torch.cat([
                xyz_centered,
                axis,
                (radius / graph_scale).unsqueeze(-1),
                (half_length / graph_scale).unsqueeze(-1),
                p_trunk_kept.unsqueeze(-1),
                p_branch_only_kept.unsqueeze(-1),
                p_leaf_kept.unsqueeze(-1),
                p_branch_kept.unsqueeze(-1),
                node_height,
            ], dim=-1)
            edge_feats = torch.stack([
                endpoint_dist_min / graph_scale,
                center_dist / graph_scale,
                axis_align,
                radius_ratio,
                branch_conf,
                torch.minimum(p_trunk_kept[a], p_trunk_kept[b]),
                torch.maximum(p_leaf_kept[a], p_leaf_kept[b]),
                torch.abs(delta[:, root_axis_idx]) / center_dist,
                p_branch_kept[a],
                p_branch_kept[b],
                radius_penalty_cost,
            ], dim=-1)
            edge_gnn = stprs.ensure_stpr_graph_gnn(
                node_dim=node_feats.shape[-1],
                edge_dim=edge_feats.shape[-1],
                hidden_dim=graph_gnn_hidden_dim,
                num_layers=graph_gnn_layers,
                lr=graph_gnn_lr,
            )
            learned_edge_logit = edge_gnn(node_feats, edge_idx, edge_feats)
            cost = cost - float(graph_gnn_weight) * learned_edge_logit
        edge_logits = -cost

        with torch.no_grad():
            projection = str(projection)
            use_forest_thresholds = projection == "forest"
            selected_np = _minimum_spanning_forest(
                n,
                edge_np,
                cost.detach().cpu().numpy(),
                max_edge_length=float(forest_max_edge_length) if use_forest_thresholds else 0.0,
                edge_lengths=endpoint_dist_min.detach().cpu().numpy(),
                max_edge_cost=float(forest_max_edge_cost) if use_forest_thresholds else 0.0,
            )
            if selected_np.shape[0] == 0:
                return edge_logits.sum() * 0.0
            selected_pairs = {tuple(sorted((int(a0), int(b0)))) for a0, b0 in selected_np.tolist()}
            target_np = np.asarray([1.0 if tuple(edge) in selected_pairs else 0.0 for edge in edge_np], dtype=np.float32)
            if 0 <= int(root_axis) < 3:
                height_detached = xyz[:, int(root_axis)].detach()
            else:
                height_detached = xyz[:, 2].detach()
            height_score = (height_detached - height_detached.min()) / (height_detached.max() - height_detached.min()).clamp(min=1e-6)
            radius_score = (radius.detach() - radius.detach().min()) / (radius.detach().max() - radius.detach().min()).clamp(min=1e-6)
            root_score = height_score - 0.35 * radius_score - 0.5 * p_trunk_kept.detach()
            root = int(torch.argmin(root_score).item())
            oriented_np, _ = _orient_tree_edges(n, selected_np, root)

        targets = torch.as_tensor(target_np, dtype=edge_logits.dtype, device=device)
        selected_mask = targets > 0.5
        rejected_mask = ~selected_mask
        pos_logits = edge_logits[selected_mask]
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) if pos_logits.numel() else edge_logits.sum() * 0.0
        if rejected_mask.any() and pos_logits.numel() and neg_per_pos > 0:
            neg_logits_all = edge_logits[rejected_mask]
            num_neg = min(int(neg_per_pos) * int(pos_logits.numel()), int(neg_logits_all.numel()))
            neg_logits = torch.topk(neg_logits_all, k=num_neg, largest=True).values
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        else:
            neg_loss = edge_logits.sum() * 0.0

        with torch.no_grad():
            selected_edge_indices = np.flatnonzero(target_np > 0.5)
            edge_lengths_np = endpoint_dist_min.detach().cpu().numpy()
            component_parent = np.arange(n, dtype=np.int64)

            def component_find(x):
                x = int(x)
                while component_parent[x] != x:
                    component_parent[x] = component_parent[component_parent[x]]
                    x = int(component_parent[x])
                return x

            for a0, b0 in selected_np:
                ra = component_find(a0)
                rb = component_find(b0)
                if ra != rb:
                    component_parent[rb] = ra
            component_count = len({component_find(idx) for idx in range(n)})
            selected_nodes_np = np.unique(oriented_np.reshape(-1)) if oriented_np.size else np.empty((0,), dtype=np.int64)
            selected_nodes_t = torch.as_tensor(selected_nodes_np, dtype=torch.long, device=device)
            self.tree_constraint_stats = {
                "num_branch_candidates": n,
                "num_candidate_edges": int(edge_np.shape[0]),
                "num_projected_edges": int(selected_np.shape[0]),
                "component_count": int(component_count),
                "mean_selected_edge_length": float(edge_lengths_np[selected_edge_indices].mean()) if selected_edge_indices.size else 0.0,
                "mean_hard_negative_logit": float(neg_logits.detach().mean().item()) if "neg_logits" in locals() and neg_logits.numel() else 0.0,
                "gnn_edge_logit_mean": float(learned_edge_logit.detach().mean().item()) if learned_edge_logit is not None else 0.0,
                "root_index": int(root),
                "root_trunk_prob": float(p_trunk_kept[root].detach().item()) if p_trunk_kept.numel() else 0.0,
                "radius_violation_rate": 0.0,
                "pst_selected_mean": float(p_branch_kept[selected_nodes_t].detach().mean().item()) if selected_nodes_t.numel() else 0.0,
                "max_degree": 0.0,
            }

        constraint_mode = str(constraint_mode)
        if constraint_mode == "mst_only":
            return sfs_weight * pos_loss

        sfs_loss = pos_loss + neg_loss
        if constraint_mode == "sfs":
            return sfs_weight * sfs_loss

        oriented = torch.as_tensor(oriented_np, dtype=torch.long, device=device)
        parent = oriented[:, 0]
        child = oriented[:, 1]
        parent_xyz = xyz[parent]
        child_xyz = xyz[child]
        tree_delta = child_xyz - parent_xyz
        tree_dist = torch.linalg.norm(tree_delta, dim=-1).clamp(min=1e-8)
        tree_dir = tree_delta / tree_dist[:, None]

        radius_loss = F.relu(radius[child] - radius[parent] + radius_margin).mean()
        parent_angle = 1.0 - torch.abs((axis[parent] * tree_dir).sum(dim=-1))
        child_angle = 1.0 - torch.abs((axis[child] * tree_dir).sum(dim=-1))
        angle_loss = 0.5 * (parent_angle + child_angle).mean()
        degree = torch.zeros((n,), dtype=edge_logits.dtype, device=device)
        degree.index_add_(0, parent, torch.ones_like(parent, dtype=edge_logits.dtype))
        degree.index_add_(0, child, torch.ones_like(child, dtype=edge_logits.dtype))
        degree_loss = F.relu(degree - float(max_degree)).square().mean()
        selected_nodes = torch.unique(oriented)
        branch_gate = p_branch_kept[selected_nodes].detach() >= float(branch_label_min_prob)
        if stprs._pst_logit is not None and branch_gate.any() and branch_label_weight > 0:
            selected_logits = stprs._pst_logit[branch_indices[selected_nodes[branch_gate]]].view(-1)
            branch_targets = torch.full_like(selected_logits, float(branch_label_target))
            branch_label_loss = F.binary_cross_entropy_with_logits(selected_logits, branch_targets)
        else:
            branch_label_loss = edge_logits.sum() * 0.0

        leaf_attachment_loss = edge_logits.sum() * 0.0
        flow_loss = edge_logits.sum() * 0.0
        trunk_root_loss = edge_logits.sum() * 0.0
        trunk_flow_loss = edge_logits.sum() * 0.0
        trunk_radius_loss = edge_logits.sum() * 0.0
        leaf_attachment_count = 0
        trunk_mass = p_trunk_kept.sum().clamp(min=1e-6)
        height_norm = (xyz[:, root_axis_idx] - xyz[:, root_axis_idx].min()) / (xyz[:, root_axis_idx].max() - xyz[:, root_axis_idx].min()).clamp(min=1e-6)
        trunk_root_loss = (p_trunk_kept * height_norm).sum() / trunk_mass
        branch_radius_mean = (p_branch_only_kept.detach() * radius).sum() / p_branch_only_kept.detach().sum().clamp(min=1e-6)
        trunk_radius_loss = (p_trunk_kept * F.relu(branch_radius_mean - radius + radius_margin)).sum() / trunk_mass
        if (leaf_attachment_weight > 0 or vascular_flow_weight > 0) and stprs.get_xyz.shape[0] > n:
            if type_probs is not None:
                leaf_mask = type_probs[:, 2] > 0.5
            elif stprs._pst_logit is not None:
                leaf_mask = torch.sigmoid(stprs._pst_logit).view(-1) < 0.5
            elif stprs.stpr_label is not None:
                leaf_mask = torch.tensor([lbl != "branch" for lbl in stprs.stpr_label], dtype=torch.bool, device=device)
            else:
                leaf_mask = torch.zeros((stprs.get_xyz.shape[0],), dtype=torch.bool, device=device)
            leaf_mask[branch_indices] = False
            leaf_indices = torch.nonzero(leaf_mask, as_tuple=False).view(-1)
            leaf_attachment_count = int(leaf_indices.numel())
            if leaf_indices.numel() > 0:
                leaf_xyz = stprs.get_xyz[leaf_indices]
                leaf_scales = stprs.get_scaling[leaf_indices]
                leaf_area = leaf_scales[:, 0:2].prod(dim=-1).detach().clamp(min=1e-6)
                leaf_to_center = torch.cdist(leaf_xyz, xyz, p=2)
                leaf_parent = torch.argmin(leaf_to_center.detach(), dim=1)
                assigned_delta = leaf_xyz - xyz[leaf_parent]
                assigned_axis = axis[leaf_parent]
                axial = (assigned_delta * assigned_axis).sum(dim=-1).clamp(min=-half_length[leaf_parent], max=half_length[leaf_parent])
                closest = xyz[leaf_parent] + axial[:, None] * assigned_axis
                attach_dist = torch.linalg.norm(leaf_xyz - closest, dim=-1)
                leaf_branch_prob = p_branch_kept[leaf_parent]
                leaf_self_prob = torch.sigmoid(stprs._pst_logit[leaf_indices]).view(-1) if stprs._pst_logit is not None else torch.zeros_like(attach_dist)
                leaf_attachment_loss = (attach_dist / radius[leaf_parent].clamp(min=1e-4)).mean()
                leaf_attachment_loss = leaf_attachment_loss + 0.1 * leaf_branch_prob.neg().add(1.0).mean()
                leaf_attachment_loss = leaf_attachment_loss + 0.1 * leaf_self_prob.mean()

                if vascular_flow_weight > 0 and child.numel() > 0:
                    with torch.no_grad():
                        node_demand = torch.zeros((n,), dtype=radius.dtype, device=device)
                        node_demand.index_add_(0, leaf_parent, leaf_area.to(device=device, dtype=radius.dtype))
                        downstream = node_demand.clone()
                        for edge_i in range(oriented.shape[0] - 1, -1, -1):
                            downstream[parent[edge_i]] += downstream[child[edge_i]]
                        edge_flow = downstream[child].clamp(min=1e-6)
                    if edge_flow.numel() > 1 and torch.var(edge_flow) > 1e-10:
                        log_flow = torch.log(edge_flow)
                        log_flow = (log_flow - log_flow.mean()) / log_flow.std().clamp(min=1e-6)
                        log_radius = torch.log(radius[parent].clamp(min=1e-6))
                        log_radius = (log_radius - log_radius.mean()) / log_radius.std().clamp(min=1e-6)
                        flow_loss = F.smooth_l1_loss(log_radius, log_flow)
                        max_flow = edge_flow.detach().max().clamp(min=1e-6)
                        parent_flow = torch.zeros((n,), dtype=radius.dtype, device=device)
                        parent_flow.index_add_(0, parent, edge_flow.detach())
                        flow_norm = (parent_flow / max_flow).clamp(0.0, 1.0)
                        trunk_flow_loss = F.binary_cross_entropy(p_trunk_kept.clamp(1e-6, 1.0 - 1e-6), flow_norm)

        with torch.no_grad():
            self.tree_constraint_stats.update({
                "radius_violation_rate": float((radius[child] > radius[parent] + radius_margin).float().mean().item()) if child.numel() else 0.0,
                "pst_selected_mean": float(p_branch_kept[selected_nodes].detach().mean().item()) if selected_nodes.numel() else self.tree_constraint_stats["pst_selected_mean"],
                "max_degree": float(degree.detach().max().item()) if degree.numel() else 0.0,
                "leaf_attachment_count": int(leaf_attachment_count),
                "leaf_attachment_loss": float(leaf_attachment_loss.detach().item()) if torch.is_tensor(leaf_attachment_loss) else 0.0,
                "vascular_flow_loss": float(flow_loss.detach().item()) if torch.is_tensor(flow_loss) else 0.0,
                "trunk_prob_mean": float(p_trunk_kept.detach().mean().item()) if p_trunk_kept.numel() else 0.0,
                "trunk_root_loss": float(trunk_root_loss.detach().item()) if torch.is_tensor(trunk_root_loss) else 0.0,
            })

        return (
            sfs_weight * sfs_loss
            + radius_weight * radius_loss
            + angle_loss_weight * angle_loss
            + degree_weight * degree_loss
            + branch_label_weight * branch_label_loss
            + leaf_attachment_weight * leaf_attachment_loss
            + vascular_flow_weight * flow_loss
            + trunk_root_weight * trunk_root_loss
            + trunk_flow_weight * trunk_flow_loss
            + trunk_radius_weight * trunk_radius_loss
        )
