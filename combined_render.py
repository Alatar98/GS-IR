import math
import os
import json
from argparse import ArgumentParser
from typing import Dict, NamedTuple, Optional

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
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap, psnr as get_psnr
from utils.loss_utils import ssim as get_ssim


from scene.cameras import Camera
import matplotlib.pyplot as plt


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True)

def quat_mul(q1, q2):
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def quat_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def quat_rotate(q, v):
    """
    q: (..., 4) quaternion [w, x, y, z]
    v: (..., 3) vector
    returns: (..., 3) rotated vector
    """
    q = quat_normalize(q)
    # treat v as pure quaternion (0, vx, vy, vz)
    zeros = torch.zeros_like(v[..., :1])
    v_q = torch.cat((zeros, v), dim=-1)
    qv = quat_mul(q, v_q)
    q_conj = quat_conj(q)
    res = quat_mul(qv, q_conj)
    return res[..., 1:]  # drop scalar part

"""
_xyz: torch.Size([5956075, 3]) 
_features_dc: torch.Size([5956075, 1, 3]) 
_features_rest: torch.Size([5956075, 15, 3]) 
_scaling: torch.Size([5956075, 3]) 
_rotation: torch.Size([5956075, 4]) 
_opacity: torch.Size([5956075, 1]) 
_normal: torch.Size([5956075, 3]) 
_albedo: torch.Size([5956075, 3]) 
_roughness: torch.Size([5956075, 1]) 
_metallic: torch.Size([5956075, 1]) 
max_radii2D: torch.Size([5956075]) 
"""

to_combine = [
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_normal",
    "_albedo",
    "_roughness",
    "_metallic",
    "max_radii2D",
]

to_translate = [
    "_xyz",
]

to_rotate = [
    "_xyz",
#    "_scaling",
    "_normal",
#    "_albedo",
]

to_scale = [
    "_xyz",
#    "_scaling",
#    "_normal",
#    "_albedo",
]


SAVE_COMBINED = False

translate_test = torch.tensor([0, 1, 0], device='cuda', dtype=torch.float32)
#0.7071, 0.7071, 0, 0
rotate_test = torch.tensor([[0.7071, 0.7071, 0, 0]], device='cuda', dtype=torch.float32)
scale_test = torch.tensor([1, 1, 1], device='cuda', dtype=torch.float32)

def combine_gaussians(gaussians1: GaussianModel, gaussians2: GaussianModel, translation: torch.Tensor=torch.tensor([0,0,0],device='cuda', dtype=torch.float32), rotation: torch.Tensor=torch.tensor([[1, 0, 0, 0]], device='cuda', dtype=torch.float32), scale: torch.Tensor=torch.tensor([1.0, 1.0, 1.0], device='cuda', dtype=torch.float32)) -> GaussianModel:
    """
    Combines two GaussianModel instances into a new GaussianModel instance. The second model is transformed by the specified translation, rotation, and scale before combining.
    Translation is a 3D vector, rotation is a quaternion, and scale is a 3D vector.
    If the scale is less than 1.0, it will downsample the second model's attributes to reduce memory usage.
    """

    new_gaussians = GaussianModel(gaussians1.active_sh_degree)
    for key,val in gaussians2.__dict__.items():
        if key in to_combine:

            ## SCALING 
            if key in to_scale:                
                print(f"Scaling {key} by {scale}")
                val = val * scale

            if key == "_scaling":
                print(f"Scaling {key} by {scale}")
                val = val + torch.log(scale)


            ## ROTATION
            if key in to_rotate:
                print(f"Rotating {key} by {rotation}")
                val = quat_rotate(rotation, val)

            if key == "_rotation":
                print(f"Rotating {key} by {rotation}")
                val = quat_mul(rotation, val)

            #if key == "_scaling":
            #    print(f"Rotating {key} by {rotation}")
            #    val_nolog = torch.exp(val)
            #    #val_nolog = quat_rotate(rotation, val_nolog)
            #    val = torch.log(val_nolog)


            ## TRANSLATION
            if key in to_translate:
                print(f"Translating {key} by {translation}")
                val = val + translation.view(1, 3)
            #remove every 2 values from val and gaussians1.__dict__[key] to reduce memory usage
            min_scale = min(scale)
            if min_scale < 1.0:
                val = val[::int(1/min_scale*1/min_scale)]

            new_gaussians.__dict__[key] = torch.cat((val, gaussians1.__dict__[key]), dim=0)
        else:
            new_gaussians.__dict__[key] = val
    return new_gaussians



def get_latest_checkpoint(model_path: str) -> Optional[str]:
    checkpoint_files = [f for f in os.listdir(model_path) if f.startswith("chkpnt") and f.endswith(".pth")]
    if not checkpoint_files:
        return None

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x[6:-4]))
    return os.path.join(model_path, latest_checkpoint)

def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))

@torch.no_grad()
def render_combined(model1_path, model2_path, source1_path, source2_path):
    print(f"Rendering combined model from {model1_path} and {model2_path}")

    sh_degree = 3
    gaussians1 = GaussianModel(sh_degree)
    gaussians2 = GaussianModel(sh_degree)

    checkpoint_path1 = get_latest_checkpoint(model1_path)
    checkpoint_path2 = get_latest_checkpoint(model2_path)

    print(f"Latest checkpoint for model 1: {checkpoint_path1}")
    print(f"Latest checkpoint for model 2: {checkpoint_path2}")

    if not checkpoint_path1 or not checkpoint_path2:
        print("Error: Could not find latest checkpoints.")
        return

    checkpoint1 = torch.load(checkpoint_path1, weights_only=False)
    checkpoint2 = torch.load(checkpoint_path2, weights_only=False)

    model_params1 = checkpoint1["gaussians"]
    first_iter1 = checkpoint1["iteration"]
    # cubemap_params = checkpoint["cubemap"]
    # light_optimizer_params = checkpoint["light_optimizer"]
    # irradiance_volumes_params = checkpoint["irradiance_volumes"]
    gaussians1.restore(model_params1, None)

    model_params2 = checkpoint2["gaussians"]
    first_iter2 = checkpoint2["iteration"]
    # cubemap_params = checkpoint["cubemap"]
    # light_optimizer_params = checkpoint["light_optimizer"]
    # irradiance_volumes_params = checkpoint["irradiance_volumes"]
    gaussians2.restore(model_params2, None)

    gaussians_combined = combine_gaussians(gaussians1, gaussians2, translation=translate_test, rotation=rotate_test, scale=scale_test)
    if SAVE_COMBINED:
        gaussians_combined.save_ply(os.path.join(model1_path, "point_cloud_combined.ply"))

    cameras = {}

    cameras_extrinsic_file = os.path.join(source1_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(source1_path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    for idx, key in enumerate(cam_extrinsics):    
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width
    
            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
    
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert (
                    False
                ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
    
   
    
            cam_info = Camera(
                colmap_id=intr.id,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=torch.zeros((3, height, width), device="cuda"),
                image_name=extr.name,
                uid=uid,
            )
            cameras[uid] = cam_info

    pipeline = PipelineParams(ArgumentParser())

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    rendering_result = render(
                viewpoint_camera=cameras[1],
                pc=gaussians_combined,
                pipe=pipeline,
                bg_color=background,
                inference=True,
                pad_normal=True,
                derive_normal=True,
            )

    imshow = rendering_result["render"]
    plt.imshow(imshow.permute(1, 2, 0).cpu().numpy())
    plt.show()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Combiner script parameters")
    #model = ModelParams(parser, sentinel=True)
    #pipeline = PipelineParams(parser)
    parser.add_argument("-m1", "--model1", type=str, required=True, help="Path to the first model directory")
    parser.add_argument("-m2", "--model2", type=str, required=True, help="Path to the second model directory")
    parser.add_argument("-s1", "--source1", type=str, required=True, help="Path to the source data for the first model")
    parser.add_argument("-s2", "--source2", type=str, required=True, help="Path to the source data for the second model")
    args = parser.parse_args()


    # Initialize system state (RNG)
    #safe_state(args.quiet)

    #dataset=model.extract(args)

    render_combined(model1_path=args.model1, model2_path=args.model2, source1_path=args.source1, source2_path=args.source2)