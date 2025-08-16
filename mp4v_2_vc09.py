import cv2
import os
import argparse
from tqdm import tqdm

# 之前一些程序生成的视频编码格式是mp4v（MPEG-4 Part 2 编码），但是web不支持无法播放，用这个程序可以把编码格式改为VP09（VP9）（H.264需要额外安装ffmpeg）
# 使用方法：命令行运行: python mp4v_2_vc09.py 0628.mp4 video_0628.mp4

def reconvert_to_vp9(input_path, output_path):
    """
    使用 OpenCV 将一个视频文件重新编码为 VP9 格式。

    Args:
        input_path (str): 输入视频文件的路径 (例如，使用 'mp4v' 编码的视频)。
        output_path (str): 输出的 VP9 编码视频文件的路径。
    """
    print(f"开始将视频转换为 VP9 格式...")
    print(f"  - 输入: {input_path}")
    print(f"  - 输出: {output_path}")

    # --- 1. 打开输入视频 ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开输入视频文件: {input_path}")
        return

    # --- 2. 获取视频属性 ---
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频尺寸 (寬度, 高度)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    # 获取总帧数用于进度条
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频属性: {width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 帧")

    # --- 3. 配置 VP9 视频写入器 ---
    # 使用你已验证可行的 'VP09' FourCC
    fourcc = cv2.VideoWriter_fourcc(*'VP09')
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    if not writer.isOpened():
        print("=" * 50)
        print("错误: OpenCV 无法初始化 VP9 视频写入器 ('VP09')。")
        print("请确认你的 OpenCV 环境配置没有改变。")
        print("=" * 50)
        cap.release()
        return

    # --- 4. 逐帧读取、写入 ---
    print("正在进行逐帧转换...")
    # 使用 tqdm 创建进度条
    with tqdm(total=total_frames, unit="frame", desc="转换进度") as pbar:
        while True:
            ret, frame = cap.read()
            # 如果视频读取完毕，ret会是False
            if not ret:
                break

            # 直接将读取到的帧写入新的视频文件
            writer.write(frame)
            pbar.update(1)

    # --- 5. 释放资源 ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("-" * 30)
    print(f"🎉 转换成功! Web兼容的视频已保存到: {output_path}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将已有的MP4视频重新编码为Web兼容的VP9格式。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="需要转换的旧视频文件路径。\n例如: restored_video_old.mp4"
    )
    parser.add_argument(
        "output_file",
        help="转换后输出的新视频文件路径。\n例如: restored_video_vp9.mp4"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在 -> {args.input_file}")
        exit(1)

    reconvert_to_vp9(args.input_file, args.output_file)