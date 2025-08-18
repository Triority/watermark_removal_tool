import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pathlib
import torchvision
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 将输入门、遗忘门、输出门和细胞门的卷积操作合并计算
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # 4 for i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 计算4*门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class RecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(RecurrentUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # ConvLSTM瓶颈
        self.bottleneck_dim = features[-1]
        self.conv_lstm = ConvLSTMCell(input_dim=self.bottleneck_dim,
                                      hidden_dim=self.bottleneck_dim,
                                      kernel_size=(3, 3), bias=True)

        # 解码器
        in_channels = features[-1]
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            in_channels = feature

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state=None):
        # 视频片段x的期望形状:[batch_size, sequence_length, Channels, H, W]
        batch_size, seq_len, _, H, W = x.shape
        if hidden_state is None:
            bottleneck_h, bottleneck_w = H // (2 ** (len(self.downs) - 1)), W // (2 ** (len(self.downs) - 1))
            hidden_state = self.conv_lstm.init_hidden(batch_size, (bottleneck_h, bottleneck_w))
        outputs = []

        # 序列帧循环
        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            skip_connections_t = []

            # 编码器
            for i, down in enumerate(self.downs):
                current_frame = down(current_frame)
                skip_connections_t.append(current_frame)
                if i < len(self.downs) - 1:
                    current_frame = self.pool(current_frame)
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            current_frame = h + current_frame
            # 反转跳跃连接列表
            skip_connections_t = skip_connections_t[::-1]

            # 解码器
            for i in range(0, len(self.ups), 2):
                current_frame = self.ups[i](current_frame)
                skip_connection = skip_connections_t[i // 2]
                # 如果池化导致奇数尺寸，上采样后的尺寸与跳跃连接不匹配，则强制修改尺寸
                if current_frame.shape != skip_connection.shape:
                    current_frame = nn.functional.interpolate(current_frame, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, current_frame), dim=1)
                current_frame = self.ups[i + 1](concat_skip)

            # 生成帧
            frame_output = self.tanh(self.final_conv(current_frame))
            outputs.append(frame_output)

        return torch.stack(outputs, dim=1), hidden_state


class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None, size=(480, 270)):
        self.root_dir = pathlib.Path(root_dir)
        self.clips_dir = self.root_dir / 'clips'
        self.mask_clips_dir = self.root_dir / 'mask_clips'
        self.mask_dir = self.root_dir / 'masks'

        self.clips_files = sorted([p for p in self.clips_dir.glob('*.mp4')])
        self.mask_clips_files = sorted([p for p in self.mask_clips_dir.glob('*.mp4')])
        self.mask_files = sorted([p for p in self.mask_dir.glob('*.png')])
        assert len(self.clips_files) == len(self.mask_clips_files) == len(self.mask_files), "The number of dataset files does not match!"

        self.sequence_length = sequence_length
        self.transform = transform
        # 输入格式(width, height)，PyTorch(height, width)
        self.target_size = size
        self.target_size_torch = (size[1], size[0])

    def __len__(self):
        return len(self.clips_files)

    def __getitem__(self, idx):
        clips_path = str(self.clips_files[idx])
        mask_clips_path = str(self.mask_clips_files[idx])
        mask_path = str(self.mask_files[idx])

        def read_and_resize_frames(video_path, num_frames, size):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                cap.release()
                raise ValueError(f"Video {video_path} : total_frames ({total_frames}) < num_frames ({num_frames})。")

            frames = []
            start_frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(TF.to_tensor(frame_rgb))
            cap.release()

            if len(frames) != num_frames:
                raise ValueError(f"Read frame failed: {video_path}")

            return torch.stack(frames)

        clips_seq = read_and_resize_frames(clips_path, self.sequence_length, self.target_size)
        masked_seq = read_and_resize_frames(mask_clips_path, self.sequence_length, self.target_size)
        mask_image = torchvision.io.read_image(str(mask_path))
        mask_image_resized = TF.resize(mask_image, self.target_size_torch, antialias=True)

        # 归一化
        clips_seq = clips_seq * 2.0 - 1.0
        masked_seq = masked_seq * 2.0 - 1.0

        mask_seq = mask_image_resized.float() / 255.0
        mask_seq[mask_seq > 0.5] = 1.0
        mask_seq[mask_seq <= 0.5] = 0.0
        mask_seq = mask_seq.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        mask_seq = mask_seq[:, 0:1, :, :]
        masked_seq = torch.cat((masked_seq, mask_seq), dim=1)

        if self.transform:
            pass

        return masked_seq, clips_seq, mask_seq


if __name__ == '__main__':
    lr = 1e-4
    batch_size = 2
    epochs = 50
    sequence_len = 4
    size = (480, 270)
    dataset_loader_workers = 6

    dataset_path = r"D:\Dataset"
    # 继续训练时加载模型路径和已完成轮次，路径为空字符串则从零开始训练且设置的轮次无效
    load_model_path = r"model\epoch_10.pth"
    load_model_epoch = 10


    writer = SummaryWriter(r'runs\gradient_monitoring')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    if load_model_path == "":
        load_model_epoch = 0
    else:
        model.load_state_dict(torch.load(load_model_path, map_location=device))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_loader_workers,
        pin_memory=False)

    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seq, clips_seq, mask_seq) in enumerate(train_loader):
                masked_seq = masked_seq.to(device)
                clips_seq = clips_seq.to(device)
                mask_seq = mask_seq.to(device)

                optimizer.zero_grad()
                restored_seq, h_last = model(masked_seq)

                loss = criterion(restored_seq, clips_seq)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 使用writer.add_scalar来记录，标签格式 'grads/层名' 可以在 TensorBoard 中分组
                        writer.add_scalar(f'grads/{name}_norm', param.grad.norm(2), epoch)
                # 记录总的梯度范数，以监控梯度爆炸
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                writer.add_scalar('grads/total_norm', total_norm, epoch)

                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=f'{loss.item():.4f}')
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss: {avg_loss:.4f} ---")

        torch.save(model.state_dict(), f"model\epoch_{epoch + 1}.pth")

    writer.close()
    print("Completed!")
