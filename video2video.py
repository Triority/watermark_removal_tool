import cv2
import numpy as np
import random
import string
import time
import threading
from queue import Queue
import os
import multiprocessing
import math
from tqdm import tqdm
from functools import partial

# --- 全局配置 ---
OUTPUT_RESOLUTION = (1280, 720)
MIN_WATERMARKS = 5
MAX_WATERMARKS = 20
SUPPORTED_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv')


def generate_random_string(length=8):
    """生成指定长度的随机字符串"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))


def process_frame(frame, frame_count, watermark_properties):
    """
    处理单帧，添加水印并生成掩膜。
    每个水印根据自己的独立随机间隔更新内容。
    """
    watermarked_frame = frame.copy()
    mask_frame = np.zeros_like(frame, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- 逻辑更新: 逐个检查并更新水印 ---
    for prop in watermark_properties:
        # 检查是否到达此水印的更新时间
        if frame_count >= prop['next_update_frame']:
            # 更新文本内容
            prop['text'] = generate_random_string()
            # 安排下一次更新时间
            prop['next_update_frame'] += prop['update_interval']

        # 绘制当前状态的水印
        cv2.putText(watermarked_frame, prop['text'], prop['pos'], font, prop['size'], (255, 255, 255),
                    prop['thickness'], cv2.LINE_AA)
        cv2.putText(mask_frame, prop['text'], prop['pos'], font, prop['size'], (255, 255, 255), prop['thickness'],
                    cv2.LINE_AA)

    return watermarked_frame, mask_frame


def video_reader(video_path, frame_queue, max_queue_size=128):
    """视频读取线程（生产者）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    while True:
        if frame_queue.qsize() < max_queue_size:
            ret, frame = cap.read()
            if not ret: break
            frame_queue.put(frame)
        else:
            time.sleep(0.01)
    frame_queue.put(None)
    cap.release()


def video_writer(output_path, frame_queue, fps, frame_size):
    """视频写入线程"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not out.isOpened(): return
    while True:
        frame = frame_queue.get()
        if frame is None: break
        out.write(frame)
    out.release()


def process_single_video(task_args, progress_queue):
    """处理单个视频文件的核心函数"""
    input_video_path, watermarked_prefix, mask_prefix, original_prefix = task_args

    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened(): return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_size = OUTPUT_RESOLUTION
        frames_per_clip = 5 * fps
        reader_frame_queue = Queue(maxsize=256)

        reader_thread = threading.Thread(target=video_reader, args=(input_video_path, reader_frame_queue))
        reader_thread.start()

        processed_frame_count_total = 0
        clip_count = 0
        while processed_frame_count_total < total_frames:
            clip_count += 1
            num_watermarks = random.randint(MIN_WATERMARKS, MAX_WATERMARKS)
            watermark_properties = []

            # --- 逻辑更新: 初始化每个水印的独立状态 ---
            for _ in range(num_watermarks):
                update_interval = random.randint(2, 8)
                watermark_properties.append({
                    'text': generate_random_string(),
                    'pos': (random.randint(50, output_size[0] - 250), random.randint(50, output_size[1] - 50)),
                    'size': random.uniform(0.3, 4.0),
                    'thickness': random.randint(1, 6),
                    'update_interval': update_interval,  # 自己的更新间隔
                    'next_update_frame': update_interval  # 第一次更新的帧
                })

            clip_suffix = f"_clip_{clip_count}.mp4"
            watermarked_clip_path = f"{watermarked_prefix}{clip_suffix}"
            mask_clip_path = f"{mask_prefix}{clip_suffix}"
            original_clip_path = f"{original_prefix}{clip_suffix}"

            watermarked_queue, mask_queue, original_queue = Queue(maxsize=256), Queue(maxsize=256), Queue(maxsize=256)

            threads = [
                threading.Thread(target=video_writer,
                                 args=(watermarked_clip_path, watermarked_queue, fps, output_size)),
                threading.Thread(target=video_writer, args=(mask_clip_path, mask_queue, fps, output_size)),
                threading.Thread(target=video_writer, args=(original_clip_path, original_queue, fps, output_size))
            ]
            for t in threads: t.start()

            frame_count_in_clip = 0
            while frame_count_in_clip < frames_per_clip:
                frame = reader_frame_queue.get()
                if frame is None:
                    processed_frame_count_total = total_frames
                    break

                resized_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
                original_queue.put(resized_frame)

                # 传入片段内的帧计数
                watermarked_frame, mask_frame = process_frame(resized_frame, frame_count_in_clip, watermark_properties)

                watermarked_queue.put(watermarked_frame)
                mask_queue.put(mask_frame)
                frame_count_in_clip += 1
                processed_frame_count_total += 1

            for q in [watermarked_queue, mask_queue, original_queue]: q.put(None)
            for t in threads: t.join()

            progress_queue.put(1)

        reader_thread.join()
    except Exception as e:
        print(f"处理文件 {input_video_path} 时出错: {e}")


def main():
    """主函数，负责扫描、计算、创建任务并启动多进程池和进度条。"""
    # --- 用户配置 ---
    input_folder = r"D:\movie\data"
    watermarked_output_folder = r"D:\Dataset_HD2\watermarked_videos"
    mask_output_folder = r"D:\Dataset_HD2\mask_videos"
    original_output_folder = r"D:\Dataset_HD2\original_clips"
    num_processes = multiprocessing.cpu_count()

    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在。")
        return

    for path in [watermarked_output_folder, mask_output_folder, original_output_folder]:
        os.makedirs(path, exist_ok=True)

    # --- 阶段 1: 扫描文件，计算总片段数，并准备任务列表 ---
    tasks = []
    total_clip_count = 0
    print("正在扫描视频文件并计算总片段数...")

    all_video_paths = []
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                all_video_paths.append(os.path.join(dirpath, filename))

    for video_path in tqdm(all_video_paths, desc="扫描进度"):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30

            clips_in_this_video = math.ceil(total_frames / (5 * fps))
            total_clip_count += clips_in_this_video
            cap.release()

            relative_path = os.path.relpath(video_path, input_folder)
            sanitized_path = relative_path.replace(os.sep, '_')
            filename_without_ext = os.path.splitext(sanitized_path)[0]

            watermarked_prefix = os.path.join(watermarked_output_folder, filename_without_ext)
            mask_prefix = os.path.join(mask_output_folder, filename_without_ext)
            original_prefix = os.path.join(original_output_folder, filename_without_ext)

            tasks.append((video_path, watermarked_prefix, mask_prefix, original_prefix))
        except Exception:
            print(f"警告: 无法读取视频元数据 {video_path}")
            continue

    if not tasks:
        print(f"在 '{input_folder}' 及其子目录中未找到任何可处理的视频文件。")
        return

    print(f"扫描完成！共找到 {len(tasks)} 个视频，将被分割成 {total_clip_count} 个片段。")
    print(f"将使用 {num_processes} 个进程进行处理...")

    # --- 阶段 2: 启动多进程池执行任务，并显示总进度条 ---
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    worker_func = partial(process_single_video, progress_queue=progress_queue)

    with multiprocessing.Pool(processes=num_processes) as pool, tqdm(total=total_clip_count, desc="总处理进度") as pbar:
        pool.map_async(worker_func, tasks)

        for _ in range(total_clip_count):
            progress_queue.get()
            pbar.update(1)

    print("\n所有视频任务均已处理完毕！")


if __name__ == "__main__":
    main()
