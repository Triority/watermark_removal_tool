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

from model import RecurrentUNet, VideoDataset


class VideoDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        # 3D卷积输入视频片段[Batch, Channels, Time, Height, Width]，卷积核在时间维度上覆盖了3帧，在空间高度上覆盖4个像素
        # stride滑动步长，在时间维度上每次只移动1帧，在空间上每次移动2个像素，起到下采样的作用
        # padding输入视频数据块的三个维度的两侧填充0
        layers.append(nn.Conv3d(in_channels, features[0], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)))
        layers.append(nn.InstanceNorm3d(features[0]))
        # inplace=True会直接在存储输入数据的内存上进行计算并覆盖，节省一些GPU显存
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(features) - 1):
            layers.append(
                nn.Conv3d(features[i], features[i + 1], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),
                          bias=False))
            layers.append(nn.InstanceNorm3d(features[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv3d(features[-1], 1, kernel_size=(3, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    lr_gen = 2e-4
    lr_disc = 1e-4
    L1_weigth = 50

    epochs = 100
    batch_size = 1
    # 数据集加载的长度，应为数据集最短帧长度，输入低于此帧长度的数据将会报错
    sequence_len = 72
    # 单次输入模型的帧长度，最好是sequence_len的因数，否则多余的将被丢弃
    batch_len = 6
    size = (480, 270)
    dataset_loader_workers = 4
    # 记录权重梯度直方图的batch间隔
    Gradient_intervals = 50

    # 数据集和模型保存路径
    dataset_path = r"D:/Dataset"
    model_save_dir = r"model_gan_3"
    # 继续训练时加载模型路径和已完成轮次，输入0则从零开始训练
    load_model_epoch = 0
    load_model_path_gen = r"model_gan_3/gen_epoch_1.pth"
    load_model_path_disc = r"model_gan_3/disc_epoch_1.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    writer = SummaryWriter('runs/GAN_3')

    gen = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    disc = VideoDiscriminator(in_channels=3).to(device)

    if load_model_epoch != 0:
        print(f"Loading Generator from {load_model_path_gen}")
        gen.load_state_dict(torch.load(load_model_path_gen, map_location=device))
        print(f"Loading Discriminator from {load_model_path_disc}")
        disc.load_state_dict(torch.load(load_model_path_disc, map_location=device))

    # Adam优化器，学习率lr，beta1默认值0.9的动量大约是过去10个时间步梯度的平均，降低到0.5降低动量惯性
    # beta2默认值0.999的二阶矩估计大约是过去1000个时间步梯度平方的平均，保持较高的值有助于保持自适应学习率的稳定性，防止因为单次梯度爆炸而导致学习率剧烈变化
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    # 二元交叉熵和L1损失函数
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()

    num_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    num_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator has {num_params_gen:,} trainable parameters.")
    print(f"Discriminator has {num_params_disc:,} trainable parameters.")
    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=dataset_loader_workers, pin_memory=False)


    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        total_loss_g = 0.0
        total_loss_d = 0.0
        total_loss_g_L1 = 0.0
        total_loss_g_adv = 0.0
        gen.train()
        disc.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seqs, clips_seqs, mask_seqs) in enumerate(train_loader):
                for batch_seq in range(int(sequence_len / batch_len)):
                    # masked_seq: [B, T, 4, H, W], clips_seq: [B, T, 3, H, W]
                    masked_seq = masked_seqs[:, batch_seq * batch_len: (batch_seq + 1) * batch_len, :, :, :].to(device)
                    clips_seq = clips_seqs[:, batch_seq * batch_len: (batch_seq + 1) * batch_len, :, :, :].to(device)

                    # gen推理用于disc训练
                    clips_fake, gen_hidden = gen(masked_seq, None if batch_seq == 0 else gen_hidden)
                    opt_disc.zero_grad()
                    # 将视频维度从[B, T, C, H, W]转换到[B, C, T, H, W]以匹配Conv3d
                    real_clip_for_disc = clips_seq.permute(0, 2, 1, 3, 4)
                    fake_clip_for_disc = clips_fake.permute(0, 2, 1, 3, 4)
                    # 判别器分别推理真实视频与全1张量、虚假视频与全0张量，计算二元交叉熵损失
                    disc_real = disc(real_clip_for_disc)
                    loss_disc_real = adversarial_loss_fn(disc_real, torch.ones_like(disc_real))
                    # 用 .detach() 阻止梯度传回生成器
                    disc_fake = disc(fake_clip_for_disc.detach())
                    loss_disc_fake = adversarial_loss_fn(disc_fake, torch.zeros_like(disc_fake))
                    # 判别器总损失
                    loss_disc = (loss_disc_real + loss_disc_fake) / 2
                    loss_disc.backward()

                    # 训练生成器
                    opt_gen.zero_grad()
                    disc_fake_for_gen = disc(fake_clip_for_disc)
                    loss_g_adv = adversarial_loss_fn(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                    loss_g_l1 = l1_loss_fn(clips_fake, clips_seq) * L1_weigth
                    loss_g = loss_g_adv + loss_g_l1
                    loss_g.backward()

                    # 分离隐状态截断反向传播
                    gen_hidden = (gen_hidden[0].detach(), gen_hidden[1].detach())

                # 记录梯度权重
                if batch_idx % Gradient_intervals == 0:
                    for name, param in disc.named_parameters():
                        if param.grad is not None:
                            # 使用 f-string 为每个梯度直方图创建唯一的、有组织的标签
                            # 'Gradients/' 会在 TensorBoard 中创建一个名为 Gradients 的分组
                            writer.add_histogram(
                                tag=f'Grad_disc/{name}', values=param.grad,
                                global_step=epoch * len(train_loader) + batch_idx)
                opt_disc.step()
                if batch_idx % Gradient_intervals == 0:
                    for name, param in gen.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(tag=f'Grad_gan/{name}', values=param.grad,
                                                 global_step=epoch * len(train_loader) + batch_idx)
                opt_gen.step()

                # 统计记录
                total_loss_g += loss_g.item()
                total_loss_d += loss_disc.item()
                total_loss_g_L1 += loss_g_l1.item()
                total_loss_g_adv += loss_g_adv.item()
                writer.add_scalar('Loss/loss_g_l1', loss_g_l1.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_g_adv', loss_g_adv.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_g', loss_g.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_disc', loss_disc.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/D_real', loss_disc_real.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/D_fake', loss_disc_fake.item(), epoch * len(train_loader) + batch_idx)

                pbar.set_postfix(
                    D_real=f'{loss_disc_real.item():.4f}',
                    D_fake=f'{loss_disc_fake.item():.4f}',
                    Loss_D=f'{loss_disc.item():.4f}',
                    Loss_G=f'{loss_g.item():.4f}',
                    G_adv=f'{loss_g_adv.item():.4f}',
                    G_L1=f'{loss_g_l1.item():.4f}')
                pbar.update(1)

            avg_loss_g = total_loss_g / len(train_loader)
            avg_loss_d = total_loss_d / len(train_loader)
            avg_loss_g_L1 = total_loss_g_L1 / len(train_loader)
            avg_loss_d_adv = total_loss_g_adv / len(train_loader)
            print(
                f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss_G: {avg_loss_g:.4f}, avg_loss_D: {avg_loss_d:.4f}, avg_loss_g_L1: {avg_loss_g_L1:.4f}, avg_loss_d_adv: {avg_loss_d_adv:.4f} ---")

        pathlib.Path("model_gan").mkdir(parents=True, exist_ok=True)
        torch.save(gen.state_dict(), f"{model_save_dir}/gen_epoch_{epoch + 1}.pth")
        torch.save(disc.state_dict(), f"{model_save_dir}/disc_epoch_{epoch + 1}.pth")

    writer.close()
    print("Completed!")
