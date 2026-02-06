"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
A standalone inference script for generating long videos with action control.
"""

import sys
import os
import cv2
import math
import torch
import random
import json
import datetime
import importlib
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist
import torchvision.transforms as transforms
import re

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.utils.data_utils import get_first_clip_from_video, save_silent_video
from infworld.utils.dataset_utils import is_vid, is_img

# ============================================================================
# Action Mapping Dictionaries
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0,
    'go forward': 1,
    'go back': 2,
    'go left': 3,
    'go right': 4,
    'go forward and go left': 5,
    'go forward and go right': 6,
    'go back and go left': 7,
    'go back and go right': 8,
    'uncertain': 9
}

VIEW_ACTION_MAP = {
    'no-op': 0,
    'turn up': 1,
    'turn down': 2,
    'turn left': 3,
    'turn right': 4,
    'turn up and turn left': 5,
    'turn up and turn right': 6,
    'turn down and turn left': 7,
    'turn down and turn right': 8,
    'uncertain': 9
}

# ============================================================================
# Utility Functions
# ============================================================================
def extract_ckpt_step(path):
    """Extract checkpoint step number from path."""
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0

def resize_and_center_crop(image, target_size):
    """Resize image and center crop to target size."""
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)
    
    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]  # [1, C, 1, H, W]

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_gc():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_action_sequence(action_path):
    """Load action sequence from JSON file."""
    with open(action_path, 'r') as f:
        actions = json.load(f)
    
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices

def load_condition_image(image_path, bucket_config):
    """Load and preprocess condition image."""
    if is_vid(image_path):
        frames = get_first_clip_from_video(image_path, clip_len=1)
    elif is_img(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = [image]
    else:
        raise ValueError(f'Unsupported file format: {image_path}')
    
    processed_frames = []
    for frame in frames:
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        
        tensor = resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2  # Normalize to [-1, 1]
        processed_frames.append(tensor)
    
    return torch.cat(processed_frames, dim=2)

# ============================================================================
# Distributed Setup (support single-GPU without torchrun to avoid port conflict)
# ============================================================================
def setup_distributed():
    """Setup distributed or single-GPU mode."""
    if 'RANK' in os.environ:
        # Launched by torchrun or similar
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        return local_rank, global_rank, num_processes, True  # use_cp_init=True
    else:
        # Single process (no torchrun) - avoid port conflict, no dist init
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, num_processes, False  # use_cp_init=False

local_rank, global_rank, num_processes, use_dist = setup_distributed()
print(f"[InfWorld] local_rank: {local_rank} | global_rank: {global_rank} | world_size: {num_processes}")

# Context parallel setup
context_parallel_size = 1
import infworld.context_parallel.context_parallel_util as cp_util
if use_dist:
    from infworld.context_parallel.context_parallel_util import init_context_parallel, get_dp_size, get_dp_rank
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
else:
    # Single process: set globals so get_dp_rank/get_dp_size work without dist
    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    dp_rank = 0
    dp_size = 1
enable_context_parallel = (context_parallel_size > 1)

# ============================================================================
# Configuration
# ============================================================================
# Inference settings
GLOBAL_SEED = 42
setup_seed(GLOBAL_SEED + global_rank)

TEXT_CFG_SCALE = 5.0
NUM_SAMPLING_STEPS = 30
SHIFT = 7  # PX256: 3, PX627: 7, PX960: 11
NUM_CHUNKS = 13  # Number of video chunks to generate
HIGH_QUALITY_SAVE = True

# Paths - checkpoint_path is read from config (configs/infworld_config.yaml)
# Model config - use standalone config
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'infworld_config.yaml')

PROMPTS_YAML = os.path.join(PROJECT_ROOT, 'prompts', 'demo.yaml')
BUCKET_CONFIG_NAME = 'ASPECT_RATIO_627_F64'

# Output directory
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'outputs')

# Negative prompt for generation quality
NEGATIVE_PROMPT = "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."

# ============================================================================
# Main Inference Loop
# ============================================================================
def resolve_path(path, root=PROJECT_ROOT):
    """Resolve path: if relative, join with project root."""
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def load_dit_state_dict(checkpoint_path):
    """Load DiT state dict from .ckpt (torch) or .safetensors."""
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def main():
    torch_gc()
    
    config_path = CONFIG_PATH
    args = OmegaConf.load(config_path)
    checkpoint_path = resolve_path(args.get("checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))
    
    ckpt_step = extract_ckpt_step(checkpoint_path)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, f"infworld-ckpt{ckpt_step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")
    print(f"[InfWorld] Output directory: {output_dir}")
    
    # Resolve relative paths in config for models that load from disk
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = resolve_path(args.text_encoder_cfg.tokenizer_path)
    
    # Initialize models
    print("[InfWorld] Loading VAE...")
    vae = get_obj_from_str(args.vae_target)(**args.vae_cfg).to(local_rank)
    
    print("[InfWorld] Loading Text Encoder...")
    text_encoder = get_obj_from_str(args.text_encoder_target)(device=local_rank, **args.text_encoder_cfg)
    text_encoder.t5.model.to(local_rank)
    
    print("[InfWorld] Loading Scheduler...")
    scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
    scheduler.num_sampling_steps = NUM_SAMPLING_STEPS
    scheduler.shift = SHIFT
    
    print("[InfWorld] Loading DiT Model...")
    dtype = getattr(torch, args.amp_dtype)
    dit = get_obj_from_str(args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=enable_context_parallel,
        **args.model_cfg
    ).to(dtype)
    dit.eval()
    
    # Load DiT checkpoint (from config)
    state_dict = load_dit_state_dict(args.checkpoint_path)
    
    # Remove position embeddings (will be recomputed)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    dit.to(local_rank)
    
    # Load bucket config
    from infworld.configs import bucket_config as bucket_config_module
    bucket_config = getattr(bucket_config_module, BUCKET_CONFIG_NAME)
    
    # Load prompts
    prompts_path = os.path.abspath(PROMPTS_YAML)
    target_prompts = OmegaConf.load(prompts_path).prompts
    print(f"[InfWorld] Loaded {len(target_prompts)} prompts")
    
    # Process each prompt
    for task_idx, (prompt, image_path, action_path) in enumerate(target_prompts):
        if task_idx % dp_size != dp_rank:
            continue
        
        if not os.path.exists(image_path):
            print(f"[InfWorld] Skipping task {task_idx}: Image not found - {image_path}")
            continue
        
        if not os.path.exists(action_path):
            print(f"[InfWorld] Skipping task {task_idx}: Action not found - {action_path}")
            continue
        
        print(f"[InfWorld] Task {task_idx}: {prompt[:50]}...")
        
        # Load condition image
        cond_video = load_condition_image(image_path, bucket_config).to(local_rank)
        
        with torch.no_grad():
            cond_latent = vae.encode(cond_video)
        
        # Load action sequence
        move_indices, view_indices = load_action_sequence(action_path)
        
        # Initialize video buffer
        video_buffer = cond_video.clone().cpu()
        
        # Latent size for generation
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21  # Output frames per chunk
        latent_size = torch.Size(latent_size)
        
        # Generate video chunks
        for chunk_idx in range(NUM_CHUNKS):
            print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{NUM_CHUNKS}")
            
            with torch.no_grad():
                current_cond = video_buffer.to(local_rank)
                current_latent = vae.encode(current_cond)
            
            # Get action slice for current chunk
            curr_start = video_buffer.shape[2] - 1
            curr_end = curr_start + args.validation_data.num_frames
            
            move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            
            # Pad if needed
            num_frames = args.validation_data.num_frames
            if move.shape[0] < num_frames:
                pad_len = num_frames - move.shape[0]
                move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
                view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
            
            additional_args = {
                "image_cond": current_latent,
                "move": move.unsqueeze(0),
                "view": view.unsqueeze(0),
            }
            
            torch_gc()
            
            with torch.no_grad():
                samples = scheduler.sample(
                    model=dit,
                    text_encoder=text_encoder,
                    null_embedder=dit.y_embedder,
                    z_size=latent_size,
                    prompts=[prompt],
                    guidance_scale=TEXT_CFG_SCALE,
                    negative_prompts=[NEGATIVE_PROMPT],
                    device=torch.device(local_rank),
                    additional_args=additional_args,
                )
                
                decoded_chunk = vae.decode(samples).cpu()
                video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
                
                print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")
                torch_gc()
        
        # Save final video
        video_name = f"{task_idx:04d}_{prompt[:30].replace(' ', '_')}"
        save_path = os.path.join(output_dir, video_name)
        
        quality = 10 if HIGH_QUALITY_SAVE else 5
        save_silent_video(video_buffer.to(local_rank), save_path, fps=30, quality=quality)
        print(f"[InfWorld] Saved: {save_path}.mp4")

if __name__ == "__main__":
    main()
