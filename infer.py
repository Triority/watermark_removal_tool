import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pathlib
import os
import re
from tqdm import tqdm
from train import RecurrentUNet


# ==============================================================================
# 1. 配置区域 (在此处修改所有参数)
# ==============================================================================

# --- 输入路径配置 ---
MODELS_DIR = r"model_gan/"  # 存放所有模型文件的文件夹
MODEL_PATTERN = "gen_epoch_*.pth" # 匹配模型文件的模式，例如 'gen_*.pth' 或 '*.pth'
INPUT_VIDEO_PATH = r"/media/B/Triority/Dataset/mask_clips/14938.mp4"
MASK_PATH = r"/media/B/Triority/Dataset/masks/14938.png"

# --- 输出模式配置 ---
# 'separate': 每个模型输出一个独立视频
# 'combined': 将所有模型的输出加水印后合并成一个长视频
OUTPUT_MODE = "combined" # 可选 'separate' 或 'combined'
OUTPUT_DIR = r"restored_videos_output/" # 'separate' 模式下的输出文件夹及 'combined' 模式的临时目录
COMBINED_OUTPUT_PATH = r"restored_video_comparison.mp4" # 'combined' 模式下的最终文件名

# --- 模型和推理参数 ---
INPUT_SIZE = (480, 270) # (宽度, 高度)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 75


# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def postprocess_and_write(output_tensor, writer):
    """辅助函数：后处理并写入文件 (无需修改)"""
    output_tensor = output_tensor.squeeze(0).cpu()
    for i in range(output_tensor.shape[0]):
        frame = (output_tensor[i] + 1.0) / 2.0
        frame = (frame.clamp(0, 1) * 255).byte()
        frame_np = frame.permute(1, 2, 0).numpy()
        # 修复：使用正确的 cv2 常量名 COLOR_RGB2BGR
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

def add_text_to_frame(frame, text):
    """在帧的左上角添加文字水印"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 0.6
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(frame, text, position, font, font_scale, (0,0,0), line_type + 1, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, font_scale, font_color, line_type, cv2.LINE_AA)
    return frame


# ==============================================================================
# 3. 推理主函数
# ==============================================================================

def infer_video_4channel(model_path, input_video_path, mask_path, output_video_path, 
                         input_size, device_name, chunk_size):
    """
    使用4通道输入(RGB+Mask)对单个视频进行分段推理。
    """
    device = torch.device(device_name)
    target_size_torch = (input_size[1], input_size[0])

    print("-" * 60)
    print(f"正在加载模型: {pathlib.Path(model_path).name}")
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已加载到设备: {device}")

    print(f"正在加载并处理掩膜: {mask_path}")
    mask_image = torchvision.io.read_image(mask_path)
    mask_resized = TF.resize(mask_image, target_size_torch, antialias=True)
    mask_tensor = mask_resized.float() / 255.0
    mask_tensor[mask_tensor > 0.5] = 1.0
    mask_tensor[mask_tensor <= 0.5] = 0.0
    mask_tensor_cpu = mask_tensor[0:1, :, :]

    print(f"正在处理输入视频: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {input_video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, original_fps, input_size)

    hidden_state = None
    chunk_frames = []

    with torch.no_grad():
        with tqdm(total=total_frames, desc=f"推理 -> {pathlib.Path(output_video_path).name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_tensor_3ch = TF.to_tensor(frame_rgb) * 2.0 - 1.0
                four_channel_tensor = torch.cat([frame_tensor_3ch, mask_tensor_cpu], dim=0)
                chunk_frames.append(four_channel_tensor)

                if len(chunk_frames) == chunk_size:
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
    print(f"视频推理完成，已保存到: {output_video_path}")
    print("-" * 60)
    return output_video_path


# ==============================================================================
# 4. 合并视频函数
# ==============================================================================

def combine_videos(video_paths, model_names, output_path, target_size, fps):
    """将多个视频片段合并为一个，并为每个片段添加水印"""
    print("\n" + "=" * 60)
    print("所有推理已完成，正在合并视频...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_writer = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    for video_path, model_name in tqdm(zip(video_paths, model_names), total=len(video_paths), desc="正在合并"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告: 无法打开临时视频 {video_path}，跳过此片段。")
            continue
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_with_watermark = add_text_to_frame(frame, model_name)
            final_writer.write(frame_with_watermark)
        
        cap.release()

    final_writer.release()
    print(f"合并视频完成！已保存到: {output_path}")
    print("=" * 60)

# ==============================================================================
# 5. 主执行逻辑
# ==============================================================================

def main():
    """主执行函数，使用顶部的配置变量"""
    # 1. 查找所有匹配的模型文件
    models_dir = pathlib.Path(MODELS_DIR)
    # 按文件名中的第一个数字进行排序（例如 gen_epoch_1.pth, gen_epoch_2.pth, ...）
    def _extract_number(p: pathlib.Path):
        m = re.search(r'(\d+)', p.stem)
        return int(m.group(1)) if m else float('inf')

    model_paths = sorted(list(models_dir.glob(MODEL_PATTERN)), key=_extract_number)

    if not model_paths:
        print(f"错误: 在文件夹 '{models_dir}' 中未找到任何匹配 '{MODEL_PATTERN}' 的模型文件。")
        return

    print(f"成功找到 {len(model_paths)} 个模型文件，准备开始推理。")
    for path in model_paths:
        print(f"  - {path.name}")

    # 2. 准备输出目录
    output_dir_path = pathlib.Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 根据选择的模式执行推理
    input_video_path = pathlib.Path(INPUT_VIDEO_PATH)
    
    temp_video_paths = []
    model_names = []

    for model_path in model_paths:
        model_stem = model_path.stem
        input_stem = input_video_path.stem
        
        if OUTPUT_MODE == 'separate':
            output_filename = f"{input_stem}_{model_stem}.mp4"
            output_video_path = str(output_dir_path / output_filename)
        elif OUTPUT_MODE == 'combined':
            temp_filename = f"temp_{input_stem}_{model_stem}.mp4"
            output_video_path = str(output_dir_path / temp_filename)
        else:
            print(f"错误: 无效的 OUTPUT_MODE '{OUTPUT_MODE}'。请选择 'separate' 或 'combined'。")
            return
            
        # 执行单次推理
        result_path = infer_video_4channel(
            model_path=str(model_path),
            input_video_path=INPUT_VIDEO_PATH,
            mask_path=MASK_PATH,
            output_video_path=output_video_path,
            input_size=INPUT_SIZE,
            device_name=DEVICE,
            chunk_size=CHUNK_SIZE
        )

        if result_path and OUTPUT_MODE == 'combined':
            temp_video_paths.append(result_path)
            model_names.append(model_stem)

    # 4. 如果是 'combined' 模式，则执行合并和清理
    if OUTPUT_MODE == 'combined':
        if not temp_video_paths:
            print("没有成功生成的视频片段，无法合并。")
            return
            
        cap = cv2.VideoCapture(temp_video_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        combine_videos(
            video_paths=temp_video_paths,
            model_names=model_names,
            output_path=COMBINED_OUTPUT_PATH,
            target_size=INPUT_SIZE,
            fps=fps
        )
        
        print("正在清理临时文件...")
        for path in temp_video_paths:
            try:
                os.remove(path)
                print(f"  - 已删除: {path}")
            except OSError as e:
                print(f"错误: 删除文件 {path} 失败: {e}")

    print("\n所有任务已完成！")


if __name__ == '__main__':
    main()
