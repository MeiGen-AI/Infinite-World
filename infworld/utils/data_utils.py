import os
import io
import re
import math
import tempfile
import imageio
import random
from tqdm import tqdm
import subprocess

import cv2
import numpy as np
from decord import VideoReader
from PIL import Image
from moviepy.editor import AudioFileClip, VideoClip


import torch
from torchvision.io import write_video
from torchvision.utils import save_image
import torchvision.transforms as transforms

import binascii
import torchvision
import imageio
import os.path as osp


def infinite_iterator(iter):
    while True:
        for sample in iter:
            yield sample

### Moved from opensora dataset utils
def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    Returns:
        x (Tensor): shape [T, H, W, C]
    """
    assert x.ndim == 4

    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1) # [C, H, W]
        save_image([x], save_path, normalize=normalize, value_range=value_range)
        x = x.unsqueeze(0)  # [1, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [1, H, W, C]
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x = x.clamp(min=low, max=high)
            x = x.sub(low).div(max(high - low, 1e-5))

        x = x.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return x


def video_reader_from_data_meta(datameta, use_tempfile, num_threads_decord):
    """ Get VideoReader from data meta; data meta needs to be video.
    """
    if not datameta.is_video:
        raise NotImplementedError('Unknown data type.')

    if 'raw_frames' in datameta:
        raw_data = datameta.raw_frames
        if use_tempfile:
            # write raw frames to a temp file before loading
            # this avoids some codec problems
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(raw_data)
                video_reader = VideoReader(temp.name, num_threads=num_threads_decord)
        else:
            # Use io.BytesIO to read image data from memory
            dataBytesIO = io.BytesIO(raw_data)
            # Convert raw data to numpy array
            # Use decord to read video data from memory
            video_reader = VideoReader(dataBytesIO, num_threads=num_threads_decord)
    elif "tar_dir" in datameta and "tar_filename" in datameta and "tar_key" in datameta:
        raw_data = datameta.load_tar_videodata()
        if use_tempfile:
            # write raw frames to a temp file before loading
            # this avoids some codec problems
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(raw_data)
                video_reader = VideoReader(temp.name, num_threads=num_threads_decord)
        else:
            # Use io.BytesIO to read image data from memory
            dataBytesIO = io.BytesIO(raw_data)
            # Convert raw data to numpy array
            # Use decord to read video data from memory
            video_reader = VideoReader(dataBytesIO, num_threads=num_threads_decord)
    elif os.path.exists(datameta.filename):
        video_reader = VideoReader(datameta.filename, num_threads=num_threads_decord)
    else:
        raise NotImplementedError('Not supported data format. rawframes or filename is needed.')

    return video_reader


def cap_from_data_meta(datameta):
    if not datameta.is_video:
        raise NotImplementedError('Unknown data type.')

    if 'raw_frames' in datameta:
        raw_data = datameta.raw_frames
        # write raw frames to a temp file before loading
        # this avoids some codec problems
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(raw_data)
            cap = cv2.VideoCapture(temp.name)
    elif "tar_dir" in datameta and "tar_filename" in datameta and "tar_key" in datameta:
        raw_data = datameta.load_tar_videodata()
        # write raw frames to a temp file before loading
        # this avoids some codec problems
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(raw_data)
            cap = cv2.VideoCapture(temp.name)        
    elif os.path.exists(datameta.filename):
        cap = cv2.VideoCapture(datameta.filename)
    else:
        raise NotImplementedError('Not supported data format. rawframes or filename is needed.')

    return cap


def none_node_splitter(src, group=None):
    yield from src


def resize_and_covert_to_gray(np_frames, pixel_value=16, interpolation=cv2.INTER_LINEAR, resize_only=False):
    # Get the dimensions of the first frame
    height, width, *_ = np_frames[0].shape
    # Determine the new dimensions based on the aspect ratio of the original frame
    if width < height:
        new_width = pixel_value
        new_height = int((new_width / width) * height)
    else:
        new_height = pixel_value
        new_width = int((new_height / height) * width)

    # Function to preprocess each frame
    def transform(frame):
        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
        # Convert the frame to grayscale
        if not resize_only:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    # Apply the transformation to each frame
    resize_frames = [transform(frame) for frame in np_frames]
    resize_frames = np.stack(resize_frames)

    return resize_frames

def get_top_m_percent(arr, m_percent):
    B, H, W = arr.shape
    N = int(H * W * m_percent / 100)
    result = np.zeros((B, N))
    for i in range(B):
        flattened_frame = arr[i].flatten()
        flattened_frame = flattened_frame[~np.isnan(flattened_frame)]
        top_m_percent_values = np.partition(flattened_frame, -N)[-N:]
        result[i] = top_m_percent_values
    return np.nanmean(result,axis=1)

def compute_optical_flow_score(np_frames, pixel_value=16):
    video_length = np_frames.shape[0]
    # Calculate the optical flow for each pair of frames
    flow_scores = []
    for i in range(1, video_length):
        # Calculate the optical flow between the current and previous frame
        flow = cv2.calcOpticalFlowFarneback(np_frames[i - 1], np_frames[i], None,  0.5, 3, 15, 3, 5, 1.2, 0)
        # Convert the flow vectors to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Append the mean magnitude of the flow vectors to the list of scores
        flow_scores.append(magnitude)

    # Return the flow score
    return np.array(flow_scores)

def get_first_frame_from_video_path(video_path):
    # get cv2 video capture data meta
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # get first frame, ret will be False if the read operation fails.
    ret, frame = cap.read()
    if ret is False:
        return None
    cap.release()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert the numpy frame to Image.
    frame = Image.fromarray(frame)

    return frame

def get_first_clip_from_video(video_path, clip_len=1):
    """
    获取视频前n帧（默认第1帧）
    
    参数：
    video_path: 视频文件路径
    n: 需要获取的帧数（从第1帧开始）

    返回：
    list: 包含前n帧PIL.Image对象的列表，空列表表示读取失败
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    
    if clip_len is None:
        clip_len = 100000000
    # 循环读取前n帧
    for frame_idx in range(clip_len):
        # 设置当前帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break  # 视频长度不足时提前终止
            
        # 格式转换
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def get_last_clip_from_video(video_path, clip_len=1):
    """
    获取视频最后n帧
    
    参数：
    video_path: 视频文件路径
    clip_len: 需要获取的帧数（从末尾开始）

    返回：
    list: 包含最后n帧的RGB帧列表，空列表表示读取失败
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算起始帧位置
    start_frame = max(0, total_frames - clip_len)
    
    # 设置起始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 读取剩余所有帧
    while len(frames) < clip_len:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换颜色空间并存储
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # 如果视频长度不足，返回实际能读取的帧
    return frames[-clip_len:] if len(frames) >= clip_len else frames


def pad_to_square_ndarray(image, pad_value=255):
    H, W, C = image.shape
    max_size = max(H, W)
    
    padded_image = np.full((max_size, max_size, C), pad_value, dtype=image.dtype)
    
    top_left_y = (max_size - H) // 2
    top_left_x = (max_size - W) // 2
    
    padded_image[top_left_y:top_left_y + H, top_left_x:top_left_x + W, :] = image
    
    return padded_image

def pad_to_square_pil(image, pad_value=255):
    width, height = image.size
    
    max_size = max(width, height)
    
    new_image = Image.new("RGB", (max_size, max_size), (pad_value, pad_value, pad_value))
    
    top_left_x = (max_size - width) // 2
    top_left_y = (max_size - height) // 2
    
    new_image.paste(image, (top_left_x, top_left_y))
    
    return new_image

def separate_connected_components(mask):

    labeled_array, num_features = label(mask)

    separate_masks = []
    bboxes = []

    slices = find_objects(labeled_array)

    for i in range(1, num_features + 1):

        component_mask = (labeled_array == i).astype(np.uint8)
        separate_masks.append(component_mask)

        slice_ = slices[i - 1]

        bbox = (slice_[1].start, slice_[0].start, slice_[1].stop, slice_[0].stop)  # (xmin, ymin, xmax, ymax)
        bboxes.append(bbox)

    return separate_masks, bboxes

def bbox_random_crop(bbox):

    xmin, ymin, xmax, ymax = bbox

    width = xmax - xmin
    height = ymax - ymin

    if height > width:
        square_size = width
        max_y_start = ymax - square_size
        y_start = random.randint(ymin, max_y_start)
        return (xmin, y_start, xmin + square_size, y_start + square_size)
    else:
        square_size = height
        max_x_start = xmax - square_size
        x_start = random.randint(xmin, max_x_start)
        return (x_start, ymin, x_start + square_size, ymin + square_size)

def inflate_bbox(bbox, d):

    x_min, y_min, x_max, y_max = bbox
  
    original_width = x_max - x_min
    original_height = y_max - y_min
    
    new_width = d * original_width
    new_height = new_width

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_new_width = new_width / 2
    half_new_height = new_height / 2

    new_x_min = int(center_x - half_new_width)
    new_x_max = int(center_x + half_new_width)
    new_y_min = int(center_y - half_new_height)
    new_y_max = int(center_y + half_new_height)

    return (new_x_min, new_y_min, new_x_max, new_y_max)

def get_frame_by_idx(cap, frame_idxs):
    if isinstance(frame_idxs, np.ndarray) or isinstance(frame_idxs, list):
        frames = []
        for frame_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            assert ret
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        return frames
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idxs)
        ret, frame = cap.read()
        assert ret
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


def recover_mask(array, shape):
    size = np.prod(shape)
    mask = np.unpackbits(array)[:size].reshape(shape).astype(np.uint8)
    return mask


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        inter_area = 0
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def extract_number_from_suffix(s):
    match = re.search(r'_\[([\d.]+)\]$', s)
    if match:
        return float(match.group(1))
    else:
        return 0

def tensor_to_video(tensor, output_video_path, input_audio_path, fps=30, dynamic_fps=True, audio_range=None, video_length=None):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_path (str): The file path where the output video will be saved.
        input_audio_path (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 30 fps.
    """
    if tensor.shape[1] == 1:
        output_video_path += '.png'
    else:
        output_video_path += '.mp4'

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()  # convert to [f, h, w, c]
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # to [0, 255]

    def make_frame(t):
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]

    if not dynamic_fps:
        video_duration = tensor.shape[0] / fps

    audio_clip = AudioFileClip(input_audio_path)
    audio_duration = audio_clip.duration
    
    if not dynamic_fps:
        final_duration = min(video_duration, audio_duration)
        audio_clip = audio_clip.subclip(0, final_duration)
    else:
        select_start, select_end = audio_range[0] / video_length, audio_range[1] / video_length
        audio_clip = audio_clip.subclip(select_start * audio_duration, select_end * audio_duration)
        final_duration = (select_end - select_start) * audio_duration
        fps = tensor.shape[0] / final_duration

    new_video_clip = VideoClip(make_frame, duration=final_duration)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    print(f"video save fps is: {fps}")
    new_video_clip.write_videofile(output_video_path, fps=fps, audio_codec="aac")

def resize_and_centercrop(cond_image, target_size):
        """
        Resize image to the target size without padding.
        """

        # Get the original size
        orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        
        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        
        # Resize
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)

        # tensor and crop
        resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) # 1 C H W
        cropped_tensor = cropped_tensor[:, :, None, :, :] # 1 C H W --> 1 C 1 H W

        return cropped_tensor


def compute_face_to_front_angle(rvec):
    # 参考姿态（正对镜头）
    rvec_ref = np.zeros((3, 1), dtype=np.float32)
    # rvec_ref = np.array([[0], [0], [1]], dtype=np.float32)
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    R_face, _ = cv2.Rodrigues(rvec)
    R_diff = R_face @ R_ref.T
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    return 180 - angle_rad * 180 / np.pi



def rotation_vector_to_euler_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def head_pose_calculation(face_landmarks, image_size=(720, 480)):
    # ========== 可选：模型中的 3D 点定义 ==========
    # 依照通用五点模型（左眼、右眼、鼻尖、左嘴角、右嘴角）
    model_points = np.array([
            [-30.0,  35.0,  0.0],  # 左眼
            [30.0,   35.0,  0.0],  # 右眼
            [0.0,     0.0,  0.0],  # 鼻尖
            [-25.0, -35.0,  0.0],  # 左嘴角
            [25.0,  -35.0,  0.0],  # 右嘴角
        ])

    # ========== 相机内参 ==========
    focal_length = image_size[0]
    center = (image_size[0] / 2, image_size[1] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # 假设无畸变

    success, rvec, tvec = cv2.solvePnP(
        model_points, face_landmarks,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # # # 转换为旋转矩阵
    # # R1, _ = cv2.Rodrigues(rvec)
    # angle_face_to_front = compute_face_to_front_angle(rvec)


    # 转换为欧拉角（单位：度）
    yaw, pitch, roll = rotation_vector_to_euler_angles(rvec)


    return abs(yaw), abs(pitch)




def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name



def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
       
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=10, ffmpeg_params=["-crf", "10"])
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file

def save_silent_video(gen_video_samples, save_path, fps=25, quality=10, high_quality_save=True):
    """
    保存无声音视频（支持追加帧到已有视频）
    
    参数：
    gen_video_samples: 生成的视频张量 [B,C,T,H,W]
    save_path: 保存路径（不带扩展名）
    fps: 视频帧率
    quality: 视频质量 (0-10)
    high_quality_save: 是否启用高质量模式
    """
    gen_video_samples = gen_video_samples[0]  # 取第一个样本
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 统一保存为MP4格式
    final_save_path = f"{save_path}.mp4"
    
    # 张量转视频帧
    video_frames = (gen_video_samples + 1) / 2  # [-1,1] -> [0,1]
    video_frames = video_frames.permute(1, 2, 3, 0).cpu().numpy()  # T H W C
    video_frames = np.clip(video_frames * 255, 0, 255).astype(np.uint8)
    
    # 处理已有视频
    all_frames = []
    existing_fps = fps  # 默认使用新视频的fps
    if os.path.exists(final_save_path):
        # 读取已有视频信息
        with imageio.get_reader(final_save_path) as reader:
            # 先获取元数据再读取帧
            meta_data = reader.get_meta_data()
            existing_fps = meta_data['fps']
            existing_frames = [frame for frame in reader]
            
            # 检查参数一致性
            if existing_fps != fps:
                raise ValueError(f"Existing video fps {existing_fps} conflicts with new fps {fps}")
            if existing_frames[0].shape != video_frames[0].shape:
                raise ValueError("Frame resolution mismatch between existing and new video")
                
            all_frames.extend(existing_frames)
    
    # 添加新帧
    all_frames.extend(video_frames)

    # 设置编码参数
    if high_quality_save:
        ffmpeg_params = [
            '-c:v', 'libx264',
            '-crf', '0',          # 无损模式
            '-preset', 'veryslow' # 最高压缩率
        ]
    else:
        ffmpeg_params = [
            '-c:v', 'libx264',
            '-crf', '23',         # 默认质量 (0-51, 越小质量越高)
            '-preset', 'medium'
        ]
    
    # 使用imageio保存
    with imageio.get_writer(
        final_save_path,
        fps=existing_fps,  # 使用已有视频的fps（当存在时）
        codec='libx264',
        quality=quality,
        ffmpeg_params=ffmpeg_params
    ) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"Silent video saved to: {final_save_path}")

def save_silent_video_overwrite(gen_video_samples, save_path, fps=25, quality=5, high_quality_save=False):
    """
    保存无声音视频（支持追加帧到已有视频）
    
    参数：
    gen_video_samples: 生成的视频张量 [B,C,T,H,W]
    save_path: 保存路径（不带扩展名）
    fps: 视频帧率
    quality: 视频质量 (0-10)
    high_quality_save: 是否启用高质量模式
    """
    gen_video_samples = gen_video_samples[0]  # 取第一个样本
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 统一保存为MP4格式
    final_save_path = f"{save_path}.mp4"
    
    # 张量转视频帧
    video_frames = (gen_video_samples + 1) / 2  # [-1,1] -> [0,1]
    video_frames = video_frames.permute(1, 2, 3, 0).cpu().numpy()  # T H W C
    video_frames = np.clip(video_frames * 255, 0, 255).astype(np.uint8)
    
    # 处理已有视频
    all_frames = []
    
    # 添加新帧
    all_frames.extend(video_frames)

    # 设置编码参数
    if high_quality_save:
        ffmpeg_params = [
            '-c:v', 'libx264',
            '-crf', '0',          # 无损模式
            '-preset', 'veryslow' # 最高压缩率
        ]
    else:
        ffmpeg_params = [
            '-c:v', 'libx264',
            '-crf', '23',         # 默认质量 (0-51, 越小质量越高)
            '-preset', 'medium'
        ]
    
    # 使用imageio保存
    with imageio.get_writer(
        final_save_path,
        fps=fps,  # 使用已有视频的fps（当存在时）
        codec='libx264',
        quality=quality,
        ffmpeg_params=ffmpeg_params
    ) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"Silent video saved to: {final_save_path}")

def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5, high_quality_save=False):
    
    gen_video_samples = gen_video_samples[0]

    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"
    
    os.makedirs(os.path.dirname(save_path_tmp), exist_ok=True)
    

    if high_quality_save:
        # Experiment version
        # NOTE: to be verified effects
        cache_video(
                    tensor=gen_video_samples.unsqueeze(0),
                    save_file=save_path_tmp,
                    fps=fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                    )
    else:
        video_audio = (gen_video_samples+1)/2 # C T H W
        video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
        video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
        save_video(video_audio, save_path_tmp, fps=fps, quality=quality)


    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/gaofeng49/conda/memo/bin/ffmpeg",
        "-i",
        vocal_audio_list[0],
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)

    # generate video with audio
    save_path = save_path + ".mp4"
    if high_quality_save:
        final_command = [
            "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/gaofeng49/conda/memo/bin/ffmpeg",
            "-y",
            "-i", save_path_tmp,
            "-i", save_path_crop_audio,
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryslow", # 可选，压缩率更高但更慢
            "-c:a", "aac",  # mp4下只能用aac或copy
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)
    else:
        final_command = [
            "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/gaofeng49/conda/memo/bin/ffmpeg",
            "-y",
            "-i",
            save_path_tmp,
            "-i",
            save_path_crop_audio,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)

def audio_move_from_hdfs(src_path):
    map_dict = {
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_0808_35w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_0808_35w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese",

        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_0829_52w_merge/tar_record_caption_qwen2vlm_pose_audioemb_part2_lipsync_camera_face_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_0829_52w_merge/tar_record_caption_qwen2vlm_pose_audioemb_part2_lipsync_camera_face_chinese",

        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_0912_28w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_0912_28w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese",

        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_0926_105w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_0926_105w_merge/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_chinese",

        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_1129_65w_part1/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_facecropcaption_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_1129_65w_part1/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_facecropcaption_chinese",

        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/data_digitalhuman/talkingbody/yt_runway_sub/singlehuman_lipsync/yt_runway_1129_65w_part2/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_facecropcaption_chinese":
        "/mnt/hdfs/user/hadoop-vision-data/llm/dataset/videogen_dataset/data/digital_human_video/talkingbody/runway_chinese/singlehuman_lipsync/yt_runway_1129_65w_part2/tar_record_caption_qwen2vlm_pose_audioemb_lipsync_camera_face_facecropcaption_chinese"
    }

    for src_p in map_dict:
        if src_p in src_path:
            src_path = src_path.replace(src_p, map_dict[src_p])

    return src_path