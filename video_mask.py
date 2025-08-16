import os
import random
import string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
from tqdm import tqdm

# --------------------------------------------------------------------------
#  在这里设置 JPEG 压缩质量 (0-100) 来控制最终文件大小
#  这是一个需要您进行实验和调整的参数！
#  推荐从 75 开始尝试。数值越低，文件越小，但画面质量也越差。
# --------------------------------------------------------------------------
JPEG_QUALITY = 75

# 预定义的常用字体列表
COMMON_FONTS = [
    'arial.ttf', 'calibri.ttf', 'times.ttf', 'cour.ttf', 'verdana.ttf'
]


def create_random_multi_text_mask(width, height):
    """生成带有随机数量、随机属性文字的复杂掩膜图片。"""
    mask_image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    num_texts_to_add = random.randint(2, 10)
    for _ in range(num_texts_to_add):
        random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(4, 12)))
        # 字体大小根据新的目标尺寸进行调整
        font_size = random.randint(int(height / 20), int(height / 6))
        font_name = random.choice(COMMON_FONTS)
        try:
            font = ImageFont.truetype(font_name, font_size)
        except IOError:
            font = ImageFont.load_default()
        try:
            # 使用 textbbox 获取更精确的文本边界
            bbox = draw.textbbox((0, 0), random_text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # 兼容旧版 Pillow
            text_width, text_height = draw.textsize(random_text, font=font)

        max_x, max_y = max(0, width - text_width), max(0, height - text_height)
        random_x, random_y = random.randint(0, max_x), random.randint(0, max_y)
        draw.text((random_x, random_y), random_text, font=font, fill=255)
    return mask_image


def process_one_segment(job_details):
    """
    “工人”函数，负责独立处理一个视频片段的所有工作，包含帧压缩和尺寸调整。
    """
    input_path = job_details['input_path']
    clip_start, clip_end = job_details['clip_start'], job_details['clip_end']
    fps = job_details['fps']
    # 从任务详情中获取目标输出尺寸
    target_width = job_details['target_width']
    target_height = job_details['target_height']
    output_size = (target_width, target_height)

    clip_output_path = job_details['clip_output_path']
    mask_output_path = job_details['mask_output_path']
    final_video_output_path = job_details['final_video_output_path']
    segment_name = job_details['segment_name']

    try:
        cap = cv2.VideoCapture(input_path)
        start_frame, end_frame = int(clip_start * fps), int(clip_end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        # 使用目标尺寸初始化 VideoWriter
        writer_clip = cv2.VideoWriter(clip_output_path, fourcc_mp4, fps, output_size)
        writer_final = cv2.VideoWriter(final_video_output_path, fourcc_mp4, fps, output_size)

        # 使用目标尺寸创建掩膜
        mask_pil_image = create_random_multi_text_mask(target_width, target_height)
        mask_pil_image.save(mask_output_path)
        mask_np = np.array(mask_pil_image) > 0
        white_pixels = np.array([255, 255, 255], dtype=np.uint8)

        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break

            # --- 核心改动 1: 调整帧大小 ---
            resized_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)

            # --- 核心改动 2: 对调整大小后的帧进行 JPEG 压缩 ---
            _, encoded_image = cv2.imencode('.jpg', resized_frame, jpeg_params)
            compressed_frame = cv2.imdecode(encoded_image, 1)

            # 将调整大小后的原始质量帧写入 "1_clips" 文件夹
            writer_clip.write(resized_frame)

            # 将压缩后的帧与文字掩膜结合，并写入 "3_final_videos" 文件夹
            modified_frame = np.where(mask_np[:, :, np.newaxis], white_pixels, compressed_frame)
            writer_final.write(modified_frame)

        cap.release()
        writer_clip.release()
        writer_final.release()

        return f"片段 {segment_name} (尺寸 {target_width}x{target_height}) 处理成功。"
    except Exception as e:
        return f"错误: 处理片段 {segment_name} 失败 - {e}"


def full_video_processing_pipeline_parallel(long_video_folder, clips_output_folder, masks_output_folder,
                                            final_video_output_folder, target_width, target_height, num_workers=None):
    """主函数，负责分解任务并使用并行进程池来执行。"""
    os.makedirs(clips_output_folder, exist_ok=True)
    os.makedirs(masks_output_folder, exist_ok=True)
    os.makedirs(final_video_output_folder, exist_ok=True)

    jobs = []
    global_segment_counter = 11502
    print("正在扫描视频并创建任务列表...")

    for dirpath, _, filenames in os.walk(long_video_folder):
        for filename in filenames:
            if not filename.lower().endswith(('.mkv', '.mp4', '.avi', '.mov')): continue
            input_path = os.path.join(dirpath, filename)
            try:
                cap_check = cv2.VideoCapture(input_path)
                if not cap_check.isOpened(): continue
                fps, total_frames = cap_check.get(cv2.CAP_PROP_FPS), cap_check.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps <= 0: fps = 25.0 # 设置默认帧率
                duration = total_frames / fps
                cap_check.release()

                # 跳过少于51分钟*2的视频 (根据原逻辑)
                if duration <= (10 * 60 * 2): continue

                trim_start_sec, trim_end_sec, seg_dur_sec = (15 * 60), (duration - (15 * 60)), 3
                current_pos_sec = trim_start_sec
                while current_pos_sec < trim_end_sec:
                    clip_start, clip_end = current_pos_sec, min(current_pos_sec + seg_dur_sec, trim_end_sec)
                    if clip_start >= clip_end: break
                    new_base_name = f"{global_segment_counter:04d}"
                    jobs.append({
                        'input_path': input_path, 'clip_start': clip_start, 'clip_end': clip_end,
                        'fps': fps,
                        # --- 核心改动：加入目标尺寸到任务详情 ---
                        'target_width': target_width, 'target_height': target_height,
                        'clip_output_path': os.path.join(clips_output_folder, f"{new_base_name}.mp4"),
                        'mask_output_path': os.path.join(masks_output_folder, f"{new_base_name}.png"),
                        'final_video_output_path': os.path.join(final_video_output_folder, f"{new_base_name}.mp4"),
                        'segment_name': new_base_name
                    })
                    global_segment_counter += 1
                    current_pos_sec += seg_dur_sec
            except Exception as e:
                print(f"扫描视频 '{filename}' 时出错: {e}")

    if not jobs:
        print("未找到需要处理的视频片段。程序结束。")
        return

    if num_workers is None: num_workers = os.cpu_count() or 1
    print(f"\n任务列表创建完毕，共 {len(jobs)} 个片段。")
    print(f"将使用 {num_workers} 个并行进程，输出尺寸 {target_width}x{target_height}，JPEG质量={JPEG_QUALITY} 开始处理...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_one_segment, jobs), total=len(jobs)))

    print(f"\n{'=' * 60}\n所有任务均已处理完毕！\n{'=' * 60}")


if __name__ == "__main__":
    # --- 新增：在这里设置期望的输出视频尺寸 ---
    output_width = 480
    output_height = 270

    # 1. 存放原始长视频的文件夹
    input_long_video_folder = "D:\迅雷下载\BBC行星地球全集"  # <--- 修改这里

    # 2. 用于保存视频片段的文件夹
    output_clips_folder = "D:\Dataset\sec\clips"  # <--- 修改这里

    # 3. 用于保存掩膜图片的文件夹
    output_masks_folder = "D:\Dataset\sec\masks"  # <--- 修改这里

    # 4. 用于保存最终视频的文件夹
    output_final_videos_folder = "D:\Dataset\sec\mask_clips"  # <--- 修改这里

    # 设置并行处理的进程数
    workers = 12

    full_video_processing_pipeline_parallel(
        input_long_video_folder,
        output_clips_folder,
        output_masks_folder,
        output_final_videos_folder,
        target_width=output_width,
        target_height=output_height,
        num_workers=workers
    )
