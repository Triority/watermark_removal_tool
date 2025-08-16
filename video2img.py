import cv2
import os
from tqdm import tqdm
import multiprocessing


num_processes = multiprocessing.cpu_count()
img_time_interval = 5
img_size = (480, 270)
img_Dir = 'data\img'
video_Dir = 'D:\movie\data'
video_Type = '.mkv'
start_frame_num = 0

def video2img(video_path, img_path, interval, size, progress_queue, shared_total_counter):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open file {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)
    if frame_interval == 0:
        print("Error: Frame interval is calculated as 0.")
        cap.release()
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, size)
            output_image_path = os.path.join(img_path, f"Frame_{shared_total_counter.value}.jpg")
            cv2.imwrite(output_image_path, frame)
            shared_total_counter.value += 1
        frame_count += 1
        progress_queue.put(1)
    cap.release()


def find_video_files(directory, file_type=".mkv"):
    mkv_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                full_path = os.path.join(root, file)
                mkv_files_list.append(full_path)
    return mkv_files_list



if __name__ == '__main__':
    video_path_list = find_video_files(video_Dir, video_Type)
    frame_total = 0
    for i in video_path_list:
        cap = cv2.VideoCapture(i)
        frame_total = frame_total + int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print(f"The total number of frames of the video is {frame_total}")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    frame_completed = 0
    shared_total_counter = manager.Value('i', start_frame_num)
    pool = multiprocessing.Pool(processes=num_processes)
    pbar = tqdm(total=frame_total, desc="Video frame processing")
    for i in video_path_list:
        pool.apply_async(video2img, args=(i, img_Dir, img_time_interval, img_size, progress_queue, shared_total_counter))

    while frame_completed < frame_total:
        _ = progress_queue.get()
        frame_completed += 1
        pbar.update(1)

    pbar.close()
    pool.close()
    pool.join()
    print("\nCompleted!")
