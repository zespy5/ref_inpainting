import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

from pathlib import Path


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


if __name__ == '__main__':
    
    save_root = Path('instantmesh_results')
    save_root.mkdir(exist_ok=True)
    
    data_root = Path('../resized_data/objects_data')
    
    
    config_file_name = Path('configs/instant-mesh-large.yaml')
    config = OmegaConf.load(config_file_name)
    config_name = config_file_name.stem
    model_config = config.model_config
    infer_config = config.infer_config
    
    device = torch.device('cuda')

    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device)
    
    
    # load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    
    rembg_session = rembg.new_session()
    
    outputs = []
    for cls in data_root.glob('*'):
        class_name = cls.stem
        for obj in cls.glob('*'):
            object_name = obj.stem
            save_object_dir = save_root/object_name
            save_object_dir.mkdir(exist_ok=True)
            
            input_image = Image.open(obj).convert('RGB')
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)
            
            output_image = pipeline(
                input_image, 
                num_inference_steps=75, 
            ).images[0]
            
            save_zero123plus_gen_file_name = save_object_dir/f'zero123plus_gen_image.jpg'
            output_image.save(save_zero123plus_gen_file_name)
            images = np.asarray(output_image, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
            images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

            outputs.append({'save_dir': save_object_dir, 'images': images})
            
    del pipeline
    
        
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    chunk_size = 20

    for idx, sample in enumerate(outputs):
        save_dir = sample['save_dir']

        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
        

        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)

            # get mesh
            mesh_path_idx = save_dir/ f'mesh.obj'

            mesh_out = model.extract_mesh(
                planes,
                **infer_config,
            )
            
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx.as_posix())


            video_path_idx = save_dir/f'video.gif'
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=6, 
                elevation=0.0,
                is_flexicubes=True,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=True,
            )

            save_video(
                frames,
                video_path_idx.as_posix(),
                fps=30,
            )
            angle = [60,120,180,240,300]
            frames = [Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for frame in frames]
            for i in range(1,6):
                image_path = save_dir/f'gen_image_{i*60}.png'
                frames[i*20].save(image_path)

                
                
                