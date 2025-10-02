import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pathlib
from tqdm import tqdm
from train import RecurrentUNet


# ==============================================================================
# 2. 推理主函数
# ==============================================================================

def postprocess_and_write(output_tensor, writer):
    """辅助函数：后处理并写入文件 (无需修改)"""
    output_tensor = output_tensor.squeeze(0).cpu()
    for i in range(output_tensor.shape[0]):
        frame = (output_tensor[i] + 1.0) / 2.0
        frame = (frame.clamp(0, 1) * 255).byte()
        frame_np = frame.permute(1, 2, 0).numpy()
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)


def infer_video_4channel(config):
    """
    使用4通道输入(RGB+Mask)对视频进行分段推理。
    """
    device = torch.device(config["device"])
    target_size = config["input_size"]
    target_size_torch = (target_size[1], target_size[0])

    print("正在加载模型...")
    # *** 关键改动: in_channels=4 ***
    # 模型输出仍然是修复后的RGB图像，所以 out_channels=3
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    print(f"模型已加载到设备: {device}")

    # --- 新增: 加载并预处理掩膜 ---
    print(f"正在加载并处理掩膜: {config['mask_path']}")
    mask_image = torchvision.io.read_image(config['mask_path'])
    mask_resized = TF.resize(mask_image, target_size_torch, antialias=True)
    mask_tensor = mask_resized.float() / 255.0
    mask_tensor[mask_tensor > 0.5] = 1.0
    mask_tensor[mask_tensor <= 0.5] = 0.0
    # 确保是单通道 [1, H, W]，并放在CPU上以便和每帧拼接
    mask_tensor_cpu = mask_tensor[0:1, :, :]

    print(f"正在处理输入视频: {config['input_video_path']}")
    cap = cv2.VideoCapture(config["input_video_path"])
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {config['input_video_path']}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'VP09')
    out_writer = cv2.VideoWriter(config["output_video_path"], fourcc, original_fps, target_size)

    hidden_state = None
    chunk_frames = []

    with torch.no_grad():
        with tqdm(total=total_frames, desc="正在推理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 预处理RGB帧
                frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_tensor_3ch = TF.to_tensor(frame_rgb) * 2.0 - 1.0

                # 2. *** 关键改动: 拼接成4通道输入 ***
                # torch.cat 沿着第0维(通道维)拼接 [3,H,W] 和 [1,H,W] -> [4,H,W]
                four_channel_tensor = torch.cat([frame_tensor_3ch, mask_tensor_cpu], dim=0)
                chunk_frames.append(four_channel_tensor)

                if len(chunk_frames) == config["chunk_size"]:
                    input_chunk = torch.stack(chunk_frames).unsqueeze(0).to(device)
                    restored_chunk, hidden_state = model(input_chunk, hidden_state)
                    postprocess_and_write(restored_chunk, out_writer)
                    chunk_frames = []

                pbar.update(1)

            if chunk_frames:
                pbar.set_description("处理最后一段")
                input_chunk = torch.stack(chunk_frames).unsqueeze(0).to(device)
                restored_chunk, hidden_state = model(input_chunk, hidden_state)
                postprocess_and_write(restored_chunk, out_writer)

    cap.release()
    out_writer.release()
    print(f"/n视频推理完成并保存到: {config['output_video_path']}")


# ==============================================================================
# 3. 配置和执行
# ==============================================================================

if __name__ == '__main__':
    inference_config = {
        "model_path": r"model_gan_2/gen_epoch_9.pth",
        # 这是需要修复的视频，例如视频中某些区域被涂黑或有水印
        "input_video_path": r"D:/Dataset/mask_clips/14938.mp4",
        # 这是对应的单张二值化掩膜图片，白色区域代表需要修复的地方
        "mask_path": r"D:/Dataset/masks/14938.png",
        "output_video_path": r"restored_video.mp4",
        "input_size": (480, 270),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "chunk_size": 10,
    }

    if not pathlib.Path(inference_config["model_path"]).exists():
        print(f"错误: 模型文件未找到 -> {inference_config['model_path']}")
    elif not pathlib.Path(inference_config["input_video_path"]).exists():
        print(f"错误: 输入视频未找到 -> {inference_config['input_video_path']}")
    # 新增对掩膜文件路径的检查
    elif not pathlib.Path(inference_config["mask_path"]).exists():
        print(f"错误: 掩膜文件未找到 -> {inference_config['mask_path']}")
    else:
        infer_video_4channel(inference_config)
