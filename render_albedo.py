import os
import json
from argparse import ArgumentParser
from typing import Dict, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
from lpips import LPIPS

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from gs_ir import recon_occlusion, IrradianceVolumes
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap, psnr as get_psnr
from utils.loss_utils import ssim as get_ssim

@torch.no_grad()
def render_albedo(model_path, checkpoint_path, dataset, pipeline):

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_params = checkpoint["gaussians"]

    gaussians.restore(model_params)


    

    iteration = checkpoint_path.split("chkpnt")[-1].split(".")[0]
    albedo_path = os.path.join(model_path, "albedo", f"ours_{iteration}")
    test_path = os.path.join(albedo_path, "test")
    train_path = os.path.join(albedo_path, "train")
    test_light_path = os.path.join(albedo_path, "test_white")
    train_light_path = os.path.join(albedo_path, "train_white")
    os.makedirs(albedo_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_light_path, exist_ok=True)
    os.makedirs(train_light_path, exist_ok=True)

    duplicate_views = ["_bridge","_city","_courtyard","_forest","_fireplace","_night","_snow","_office","_sunset","_sunset_000","_sunset_120","_sunset_240"]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    views = scene.getTestCameras()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if any(duplicate in view.image_name for duplicate in duplicate_views):
            continue
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        render_img = rendering_result["render"]
        torchvision.utils.save_image(render_img, os.path.join(test_path, f"{view.image_name}.png"))

    views = scene.getTrainCameras()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if any(duplicate in view.image_name for duplicate in duplicate_views):
            continue
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        render_img = rendering_result["render"]
        torchvision.utils.save_image(render_img, os.path.join(train_path, f"{view.image_name}.png"))

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    views = scene.getTestCameras()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if any(duplicate in view.image_name for duplicate in duplicate_views):
            continue
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        render_img = rendering_result["render"]
        torchvision.utils.save_image(render_img, os.path.join(test_light_path, f"{view.image_name}.png"))

    views = scene.getTrainCameras()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if any(duplicate in view.image_name for duplicate in duplicate_views):
            continue
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=True,
            derive_normal=True,
        )

        render_img = rendering_result["render"]
        torchvision.utils.save_image(render_img, os.path.join(train_light_path, f"{view.image_name}.png"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    checkpoint_path=args.checkpoint
    dataset=model.extract(args)

    render_albedo(model_path=model_path, checkpoint_path=checkpoint_path, dataset=dataset, pipeline=pipeline)
    
