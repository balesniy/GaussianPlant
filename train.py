import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim,  align_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_tensor_as_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch3d.loss import chamfer_distance
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def refresh_neighbors_if_needed(gaussians, iteration, interval):
    if gaussians.knn_idx is None or interval <= 1 or iteration % interval == 0:
        gaussians.reset_neighbors()

def masked_l1_loss(network_output, gt, mask):
    mask_sum = mask.sum().clamp(min=1.0)
    return (torch.abs(network_output - gt) * mask).sum() / (mask_sum * network_output.shape[0])

def render_alpha_approx(viewpoint_cam, gaussians, pipe, device):
    alpha_bg = torch.zeros((3), dtype=torch.float32, device=device)
    alpha_color = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device=device)
    alpha_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        alpha_bg,
        override_color=alpha_color,
        separate_sh=False,
    )
    return alpha_pkg["render"].mean(dim=0, keepdim=True).clamp(0.0, 1.0)

def find_checkpoint_pair(checkpoint):
    root, ext = os.path.splitext(checkpoint)
    if ext != ".pth":
        root = checkpoint

    if root.endswith("_appgs"):
        app_checkpoint = root + ".pth"
        stpr_checkpoint = root[:-6] + "_stprs.pth"
    elif root.endswith("_stprs"):
        stpr_checkpoint = root + ".pth"
        app_checkpoint = root[:-6] + "_appgs.pth"
    else:
        app_checkpoint = root + "_appgs.pth"
        stpr_checkpoint = root + "_stprs.pth"

    if os.path.exists(app_checkpoint) and os.path.exists(stpr_checkpoint):
        return app_checkpoint, stpr_checkpoint
    return None

def restore_child_gaussians(checkpoint, dataset, opt, reference_gaussians):
    model_params, iteration = torch.load(checkpoint, weights_only=False)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, reference_gaussians.device)
    gaussians.exposure_mapping = reference_gaussians.exposure_mapping
    gaussians.pretrained_exposures = reference_gaussians.pretrained_exposures
    gaussians._exposure = torch.nn.Parameter(reference_gaussians._exposure.detach().clone().requires_grad_(True))
    gaussians.restore(model_params, opt)
    return gaussians, iteration

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_init = GaussianModel(dataset.sh_degree, opt.optimizer_type, args.device)
    appgs = None
    stprs = None
    process_state = "init"
    scene = Scene(dataset, gaussians_init)
    gaussians_init.training_setup(opt)
    gt_xyz = gaussians_init.get_xyz.detach()
    if checkpoint:
        checkpoint_pair = find_checkpoint_pair(checkpoint)
        if checkpoint_pair is not None:
            app_checkpoint, stpr_checkpoint = checkpoint_pair
            appgs, app_iter = restore_child_gaussians(app_checkpoint, dataset, opt, gaussians_init)
            stprs, stpr_iter = restore_child_gaussians(stpr_checkpoint, dataset, opt, gaussians_init)
            if app_iter != stpr_iter:
                raise ValueError(f"Checkpoint iteration mismatch: {app_checkpoint} is {app_iter}, {stpr_checkpoint} is {stpr_iter}")
            first_iter = app_iter
            gaussians_init.appgs = appgs
            gaussians_init.structure_gs = stprs
            gaussians_init.update_nn_between_appgs_and_stprs()
            process_state = 'appgs'
        else:
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
            gaussians_init.restore(model_params, opt)
            process_state = 'init'

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # regularizer           
    loss_align = 0
    loss_overlap = 0    
    loss_freq = 0
    num_appgs = 0
    num_stprs = 0
    loss_bind = 0
    loss_opacity_app = 0
    loss_opacity_stprs = 0
    loss_mst = 0
    image_stprs = None
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians_init, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        if stprs is None:
            # build our representation
            if os.path.exists(os.path.join(args.source_path, "points_3dgs.ply")):
                opt.iter_build_stprs= 1
                opt.iter_build_appgs = 1
            else:
                opt.iter_build_stprs = 7000
                opt.iter_build_appgs = 7000
            if iteration == opt.iter_build_stprs:   # build stprs from inital gaussians
                points_3dgs_path = os.path.join(args.source_path, "points_3dgs.ply")
                if os.path.exists(points_3dgs_path):
                    gaussians_init.load_ply(points_3dgs_path)
                stprs,appgs = gaussians_init.build_stprs_from_gs(num_clusters=100, method='3dgs', min_cluster_points=args.min_cluster_points)
                stprs.training_setup(opt)
                appgs.training_setup(opt)
                stprs.save_ply(os.path.join(scene.model_path, "stprs_init.ply"))
                appgs.save_ply(os.path.join(scene.model_path, "appgs_init.ply"))
                process_state = "appgs"
            # gaussians_init.update_nn_between_appgs_and_stprs()
 
         # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        # Load GT
        gt_image = viewpoint_cam.original_image.to(args.device)
        # gt_mask = viewpoint_cam.mask.to(args.device).unsqueeze(0)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background       
        # defaults for all training stages
        Ll1 = torch.tensor(0.0, device=args.device)
        Ll1depth_pure = 0.0

        image = None
        image_stprs = None

        invDepth_appgs = None
        invDepth_stprs = None
        mono_invdepth = None

        loss_align = 0.0
        loss_overlap = 0.0
        loss_freq = 0.0

        loss_opacity_app = 0.0
        loss_opacity_stprs = 0.0
        loss_bind = 0.0
        loss_mst = 0.0

        num_appgs = 0
        num_stprs = 0
        

        # stage 1: init
        if process_state == "init":
            gaussians_init.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians_init.oneupSHdegree()

            # render

            render_pkg = render(viewpoint_cam, gaussians_init, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image,  depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            mask = viewpoint_cam.alpha_mask.to(args.device) if getattr(viewpoint_cam, "has_alpha_mask", False) else None
            
            # Basic Loss
            masked_image = image * mask if mask is not None else image
            masked_gt_image = gt_image * mask if mask is not None else gt_image
            Ll1 = masked_l1_loss(image, gt_image, mask) if mask is not None else l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(masked_image.unsqueeze(0), masked_gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(masked_image, masked_gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            if mask is not None and opt.lambda_mask > 0:
                outside_mask = (1.0 - mask).clamp(0.0, 1.0)
                outside_sum = outside_mask.sum().clamp(min=1.0)
                rendered_alpha = render_alpha_approx(viewpoint_cam, gaussians_init, pipe, args.device)
                bg_target = bg.view(-1, 1, 1) if bg.ndim == 1 else bg
                outside_alpha_loss = (rendered_alpha * outside_mask).sum() / outside_sum
                outside_rgb_loss = (torch.abs(image - bg_target) * outside_mask).sum() / (outside_sum * image.shape[0])
                loss += opt.lambda_mask * (outside_alpha_loss + outside_rgb_loss)
            
            # Depth regularization
            Ll1depth_pure = 0.0
            invDepth = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            # regularizer
            
            loss_align = 0
            loss_overlap = 0
            if args.reg_align or args.reg_overlap:
                refresh_neighbors_if_needed(gaussians_init, iteration, args.neighbor_update_interval)
                neighbor_idx  = gaussians_init.get_neighbors_of_random_points(gaussians_init.get_xyz.shape[0] // 10)
            if args.reg_align:
                # gaussian_align = gaussians_init.compute_gaussian_alignment_with_neighbors(neighbor_idx)
                # loss_align = (1-gaussian_align).mean()
                loss_align = align_loss(stprs, neighbor_idx)
                loss+= loss_align * opt.lambda_align
            if args.reg_overlap:
                gaussian_overlap = gaussians_init.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                loss_overlap = gaussian_overlap.mean()
                loss += loss_overlap * opt.lambda_overlap
            loss.backward()
            
        elif process_state == "stprs" :
            stprs.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                stprs.oneupSHdegree()

            # render
            render_pkg = render(viewpoint_cam, stprs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            # render_pkg = render_with_depth(viewpoint_cam, stprs, pipe, bg)
            image_stprs, depth_stpr, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = image_stprs
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.to(args.device)
                image_stprs *= alpha_mask
            
            # Basic Loss
            # Ll1 = l1_loss(image_stprs, gt_image)
            # if FUSED_SSIM_AVAILABLE:
            #     ssim_value = fused_ssim(image_stprs.unsqueeze(0), gt_image.unsqueeze(0))
            # else:
            #     ssim_value = ssim(image_stprs, gt_image)
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss = 0
            # Depth regularization
            Ll1depth_pure = 0.0
            invDepth = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0
            
            # mask loss
            # loss_mask = l1_loss(mask, gt_mask)
            # loss += loss_mask * opt.lambda_mask


            if args.reg_align or args.reg_overlap:
                refresh_neighbors_if_needed(stprs, iteration, args.neighbor_update_interval)
                neighbor_idx  = stprs.get_neighbors_of_random_points(stprs.get_xyz.shape[0] )
            if args.reg_align:
                # gaussian_align = stprs.compute_gaussian_alignment_with_neighbors(neighbor_idx)
                # loss_align = (1-gaussian_align).mean()
                loss_align = align_loss(stprs, neighbor_idx)
                loss+= loss_align * opt.lambda_align
            if args.reg_overlap:
                gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                loss_overlap = (1-gaussian_overlap.mean())
                loss += loss_overlap * opt.lambda_overlap
            if args.reg_freq:
                loss_freq = stprs.low_freq_loss()
                loss += loss_freq * opt.lambda_freq
            # binding loss
            
            loss.backward()
            
        elif process_state == "appgs":
            # appgs.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                appgs.oneupSHdegree()
                
            render_pkg_stprs = render(viewpoint_cam, stprs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            render_pkg_appgs = render(viewpoint_cam, appgs, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image_stprs, depth_stprs, viewspace_point_tensor_stprs, visibility_filter_stprs, radii_stprs = render_pkg_stprs["render"], render_pkg_stprs["depth"], render_pkg_stprs["viewspace_points"], render_pkg_stprs["visibility_filter"], render_pkg_stprs["radii"]
            image, depth_appgs, viewspace_point_tensor, visibility_filter, radii = render_pkg_appgs["render"], render_pkg_appgs["depth"], render_pkg_appgs["viewspace_points"], render_pkg_appgs["visibility_filter"], render_pkg_appgs["radii"]
            mask = viewpoint_cam.alpha_mask.to(args.device) if getattr(viewpoint_cam, "has_alpha_mask", False) else None
            # Basic Loss
            masked_image = image * mask if mask is not None else image
            masked_gt_image = gt_image * mask if mask is not None else gt_image
            Ll1 = masked_l1_loss(image, gt_image, mask) if mask is not None else l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(masked_image.unsqueeze(0), masked_gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(masked_image, masked_gt_image)
            Ll1_stpr = masked_l1_loss(image_stprs, gt_image, mask) if mask is not None else l1_loss(image_stprs, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + 0.1 * Ll1_stpr
            # Depth regularization for stpr and appgs
            Ll1depth_pure = 0.0
            invDepth_appgs = None
            invDepth_stprs = None
            mono_invdepth = None
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth_stprs = render_pkg_stprs["depth"]
                invDepth_appgs = render_pkg_appgs["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(args.device)
                depth_mask = viewpoint_cam.depth_mask.to(args.device)

                Ll1depth_pure_appgs = torch.abs((invDepth_appgs  - mono_invdepth) * depth_mask).mean()
                Ll1depth_pure_stprs = torch.abs((invDepth_stprs  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * (Ll1depth_pure_appgs + Ll1depth_pure_stprs) / 2 
                loss += Ll1depth * 0.001
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            # regularizer           

            if iteration >100:
                # appgs.reset_neighbors()
                refresh_neighbors_if_needed(stprs, iteration, args.neighbor_update_interval)
                neighbor_idx  = stprs.get_neighbors_of_random_points(-1)
                refresh_neighbors_if_needed(appgs, iteration, args.neighbor_update_interval)
                neighbor_idx_app  = appgs.get_neighbors_of_random_points(-1)
                # neighbor_idx_appg  = appgs.get_neighbors_of_random_points(stprs.get_xyz.shape[0]//10)
                if args.reg_align:
                    loss_align = align_loss(appgs, neighbor_idx_app)
                    loss+= loss_align * opt.lambda_align
                if args.reg_overlap:
                    gaussian_overlap = stprs.compute_gaussian_overlap_with_neighbors(neighbor_idx)
                    loss_overlap = 1-gaussian_overlap.mean()
                    loss += loss_overlap * opt.lambda_overlap
                if args.reg_opacity:
                    loss_opacity_app = appgs.opacity_regularizer()
                    loss_opacity_stprs = stprs.opacity_regularizer()
                    loss += loss_opacity_app * opt.lambda_opacity + loss_opacity_stprs * opt.lambda_opacity
                if args.reg_mst:
                    _,_,loss_mst = gaussians_init.stpr_to_graph()
                    loss += loss_mst * opt.lambda_mst
                
            loss_bind = gaussians_init.compute_gaussian_binding_loss(method='surface')
            loss += loss_bind * opt.lambda_bind
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if appgs is not None:
                    progress_bar.set_postfix({"L1": f"{Ll1.item():.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",  "Num of appgs":f"{appgs.get_xyz.shape[0]}", "Num of stprs":f"{stprs.get_xyz.shape[0]}","Align Loss": f"{loss_align:.{7}f}", 
                                              "Overlap Loss": f"{loss_overlap:.{7}f}",})
                else:
                    progress_bar.set_postfix({"L1": f"{Ll1.item():.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Align Loss": f"{loss_align:.{7}f}", 
                                          "Overlap Loss": f"{loss_overlap:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if appgs is not None:
                num_appgs = appgs.get_xyz.shape[0]
            if stprs is not None:
                num_stprs = stprs.get_xyz.shape[0]
            training_report(tb_writer, iteration, Ll1, loss, Ll1depth_pure,loss_align, loss_overlap,l1_loss, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, image,  invDepth_appgs,invDepth_stprs,mono_invdepth, image_stprs,loss_freq, num_appgs, num_stprs, loss_opacity_app, loss_opacity_stprs, loss_bind,loss_mst, args.tb_image_interval)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,save_app=True,save_stpr=True,save_branch=True)

            # Densification
            if iteration < opt.densify_until_iter:
                if process_state == "init":
                # Keep track of max radii in image-space for pruning
                    gaussians_init.max_radii2D[visibility_filter] = torch.max(gaussians_init.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians_init.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians_init.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, flag=None)
                       
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians_init.reset_opacity()
                
                elif process_state == "stprs":
                    stprs.max_radii2D[visibility_filter] = torch.max(stprs.max_radii2D[visibility_filter], radii[visibility_filter])
                    stprs.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval:
                        if stprs._xyz.shape[0]<=args.max_stpr_num:
                            size_threshold = 30 if iteration > opt.opacity_reset_interval else None # size_threshold:20
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, flag='stpr', size_threshold_small=None)
                        else:
                            size_threshold = 30 if iteration > opt.opacity_reset_interval else None
                            stprs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*10, size_threshold, radii, flag='stpr', only_prune=True)
                    if iteration % opt.opacity_reset_interval == 0:
                        stprs.reset_opacity_stpr()


                elif process_state == "appgs":
                    appgs.max_radii2D[visibility_filter] = torch.max(appgs.max_radii2D[visibility_filter], radii[visibility_filter])
                    appgs.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    stprs.max_radii2D[visibility_filter_stprs] = torch.max(stprs.max_radii2D[visibility_filter_stprs], radii_stprs[visibility_filter_stprs])
                    stprs.add_densification_stats(stprs._xyz, visibility_filter_stprs)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        appgs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,radii,flag='app')
                   
                        """ Adaptive densification control of stprs """
                        grad_threshold_stpr = 0.001
                        if stprs.get_xyz.shape[0] < args.max_stpr_num:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            stprs.densify_and_prune(grad_threshold_stpr, 0.005, scene.cameras_extent, size_threshold, radii_stprs,flag='stpr')

                        else:
                            stprs.densify_and_prune(grad_threshold_stpr, 0.005, scene.cameras_extent, size_threshold, radii_stprs, only_prune=True,flag='stpr')
                        gaussians_init.update_nn_between_appgs_and_stprs()
               
            # Optimizer step
            if iteration < opt.iterations:
                if process_state == "init":
                    gaussians_init.optimizer.step()
                    gaussians_init.optimizer.zero_grad(set_to_none = True)
                if process_state == "stprs":
                    stprs.optimizer.step()
                    stprs.optimizer.zero_grad(set_to_none = True)
                if process_state == "appgs":
                    appgs.optimizer.step()
                    appgs.optimizer.zero_grad(set_to_none = True)
                    stprs.optimizer.step()
                    stprs.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if appgs is not None:
                    torch.save((appgs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_appgs.pth")
                if stprs is not None:
                    torch.save((stprs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_stprs.pth")
                if appgs is None and stprs is None:
                    torch.save((gaussians_init.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, loss_depth,loss_align,loss_overlap,l1_loss, testing_iterations, 
                    scene : Scene, renderFunc, renderArgs, train_test_exp, image, depth_appgs, depth_stprs
,mono_invdepth,image_stprs,loss_freq, num_appgs, num_stprs, loss_opacity_appgs, loss_opacity_stprs, loss_bind,loss_mst, tb_image_interval):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth, iteration)
        tb_writer.add_scalar('train_loss_patches/align_loss', loss_align, iteration)
        tb_writer.add_scalar('train_loss_patches/overlap_loss', loss_overlap, iteration)
        tb_writer.add_scalar('train_loss_patches/freq_loss', loss_freq, iteration)
        tb_writer.add_scalar('train_loss_patches/num_appgs', num_appgs, iteration)
        tb_writer.add_scalar('train_loss_patches/num_stprs', num_stprs, iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_loss_appgs', loss_opacity_appgs, iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_loss_stprs', loss_opacity_stprs, iteration)
        tb_writer.add_scalar('train_loss_patches/binding_loss', loss_bind, iteration)
        tb_writer.add_scalar('train_loss_patches/mst_loss', loss_mst, iteration)
        
        if tb_image_interval > 0 and iteration % tb_image_interval == 0:
            tb_writer.add_images('train_image_patches/render', image[None], global_step=iteration)
            if depth_stprs is not None:
                tb_writer.add_images('train_image_patches/depth_stprs', depth_stprs[None], global_step=iteration)
            if depth_appgs is not None:
                tb_writer.add_images('train_image_patches/depth_appgs', depth_appgs[None], global_step=iteration)
            if mono_invdepth is not None:
                tb_writer.add_images('train_image_patches/mono_invdepth', mono_invdepth[None], global_step=iteration)
            if image_stprs is not None:
                tb_writer.add_images('train_image_patches/stprs_render', image_stprs[None], global_step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[300,1000,2000,3000,5000,7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000,7000,15000,30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--tb_image_interval", type=int, default=1000)
    parser.add_argument("--neighbor_update_interval", type=int, default=50)
    parser.add_argument("--min_cluster_points", type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument("--reg_mask", action="store_true", default=False)
    parser.add_argument("--reg_align", action="store_true", default=False)
    parser.add_argument("--reg_overlap", action="store_true", default=False)
    parser.add_argument("--reg_freq", action="store_true", default=False)
    parser.add_argument("--reg_opacity", action="store_true", default=False)
    parser.add_argument("--reg_mst", action="store_true", default=False)
    parser.add_argument("--max_stpr_num", type=int, default=2000 )

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device {device}")
    args.device = device
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # All done
    print("\nTraining complete.")
