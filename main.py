import argparse
import numpy as np
import torch
import rembg
from PIL import Image
import cv2
from torchvision.transforms import v2

from omegaconf import OmegaConf
from einops import rearrange
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    get_zero123plus_input_cameras,
    spherical_camera_pose
)
from pytorch_lightning import seed_everything
from src.utils.infer_util import remove_background, resize_foreground, save_video

from pathlib import Path

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from utils import *
from inference import inference_single_image

import time
import json


def InstantMesh(config_path, input_object_path, azimuth, elevation, device='cuda'):
    
    config_file_name = Path(config_path)
    config = OmegaConf.load(config_file_name)
    config_name = config_file_name.stem
    model_config = config.model_config
    
    load_start = time.time()
    
    device = torch.device('cuda')
    
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device)

    model = instantiate_from_config(model_config)
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    
    rembg_session = rembg.new_session()
    
    load_end = time.time()
    
    print(f'Instant mesh model load time {load_end-load_start:3f} sec')
    
    inference_start = time.time()
    
    
    input_image = Image.open(input_object_path)
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    
    zero123plus_generate_image = pipeline(input_image, num_inference_steps=75).images[0]
    
    multiview_images = np.asarray(zero123plus_generate_image, dtype=np.float32) / 255.0
    multiview_images = torch.from_numpy(multiview_images).permute(2, 0, 1).contiguous().float()
    multiview_images = rearrange(multiview_images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    
    images = multiview_images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
        
        c2ws = spherical_camera_pose(azimuth, elevation, 4.0)
        cameras = torch.linalg.inv(c2ws).unsqueeze(0).unsqueeze(0).to(device)
        
        frame = model.forward_geometry(planes, cameras)['img'].squeeze()
        
        novel_view_image = Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    inference_end = time.time()
    print(f'Instant mesh inference time {inference_end-inference_start:3f} sec')
    return zero123plus_generate_image, novel_view_image

def AnyDoor(config_path, venue_path, object_img, object_size, object_position, device='cuda'):
    venue_path = Path(venue_path)
    
    config_file_name = Path(config_path)
    config = OmegaConf.load(config_file_name)
    model_ckpt =  config.pretrained_model
    model_config = config.config_file

    load_start = time.time()
    
    model = create_model(model_config ).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
    model = model.to(device)
    load_end = time.time()
    
    print(f'Anydoor model load time {load_end-load_start:3f} sec')
    
    inference_start = time.time()
    
    venue_image = cv2.imread(venue_path.as_posix(), cv2.IMREAD_COLOR)
    venue_image = cv2.cvtColor(venue_image, cv2.COLOR_BGR2RGB)
    

    object_img, object_mask_image = generate_object_img_mask(object_img)
    object_mask_image = (object_mask_image > 128).astype(np.uint8)
    
    location_mask, _ = make_mask_from_point(venue_image, object_mask_image, object_size, object_position)
    
    gen_image, hint = inference_single_image(model, object_img, object_mask_image,
                                             venue_image.copy(), location_mask,
                                             guidance_scale=3.0)
    inference_end = time.time()
    print(f'Anydoor inference time {inference_end-inference_start:3f} sec')
    gen_image = Image.fromarray(gen_image.astype(np.uint8))
    location_mask = Image.fromarray(location_mask.astype(np.uint8))
    return gen_image, location_mask



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instantmesh_config', type=str,
                        default='configs/instant-mesh-large.yaml')
    parser.add_argument('--anydoor_config', type=str,
                        default='configs/anydoor_inference.yaml')
    parser.add_argument('--bg_img', type=str, required=True)
    parser.add_argument('--object_img', type=str, required=True)
    parser.add_argument('--azimuth', type=float, default=60)
    parser.add_argument('--elevation', type=float, default=0)
    parser.add_argument('--object_size', type=float, default=0.5)
    parser.add_argument('--object_location', type=json.loads, default="[0.6, 0.5]")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=0)
    
    opt = parser.parse_args()
    
    seed_everything(opt.seed)
    instatmesh_config = opt.instantmesh_config
    anydoor_config = opt.anydoor_config
    bg_img = opt.bg_img
    object_img = opt.object_img
    azimuth = opt.azimuth
    elevation = opt.elevation
    object_size = [opt.object_size, opt.object_size,]
    object_location = opt.object_location
    output_dir = opt.output_dir
    
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    z, n = InstantMesh(instatmesh_config, object_img, azimuth, elevation)
    anydoorgen, location_mask = AnyDoor(anydoor_config, bg_img, n, object_size, object_location)
   
    zero123_result = output/'zero123plus.png'
    novel_view = output/f'azimuth={azimuth},elevation={elevation}_view.png'
    anydoor_result = output/'anydoor.png'
    location_mask_results = output/'location_mask.png'
    z.save(zero123_result)
    n.save(novel_view)
    anydoorgen.save(anydoor_result)
    location_mask.save(location_mask_results)
    
    
    