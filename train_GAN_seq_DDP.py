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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from model import RecurrentUNet, VideoDataset
from train_GAN_seq import VideoDiscriminator


def setup(rank, world_size):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'nccl' 是 NVIDIA GPU 推荐的后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    """
    主训练函数，每个 GPU 进程都会执行此函数。
    rank: 当前进程的 ID (也是 GPU 的 ID)
    world_size: 进程总数 (GPU 总数)
    args: 包含所有超参数的字典
    """
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # 每个进程只在自己的 GPU 上工作
    device = rank
    torch.cuda.set_device(device)

    # 只有主进程 (rank 0) 才需要进行日志记录
    writer = SummaryWriter('runs/GAN_DDP') if rank == 0 else None

    # 模型初始化后必须移动到对应的 GPU 上
    gen = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    disc = VideoDiscriminator(in_channels=3).to(device)

    # 使用 DDP 包裹模型
    # find_unused_parameters=True 在某些情况下（如图中有部分输出未用于损失计算）是必要的
    gen = DDP(gen, device_ids=[rank], find_unused_parameters=True)
    disc = DDP(disc, device_ids=[rank], find_unused_parameters=True)

    if args['load_model_epoch'] != 0:
        # 加载模型时，需要将模型权重映射到当前进程对应的 GPU
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        print(f"Rank {rank} loading Generator from {args['load_model_path_gen']}")
        gen.load_state_dict(torch.load(args['load_model_path_gen'], map_location=map_location))
        print(f"Rank {rank} loading Discriminator from {args['load_model_path_disc']}")
        disc.load_state_dict(torch.load(args['load_model_path_disc'], map_location=map_location))

    opt_gen = optim.Adam(gen.parameters(), lr=args['lr_gen'], betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=args['lr_disc'], betas=(0.5, 0.999))
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()

    if rank == 0:
        # .module 访问 DDP 包装下的原始模型
        num_params_gen = sum(p.numel() for p in gen.module.parameters() if p.requires_grad)
        num_params_disc = sum(p.numel() for p in disc.module.parameters() if p.requires_grad)
        print(f"Generator has {num_params_gen:,} trainable parameters.")
        print(f"Discriminator has {num_params_disc:,} trainable parameters.")
        print("Preparing dataset...")

    train_dataset = VideoDataset(root_dir=args['dataset_path'], sequence_length=args['sequence_len'], size=args['size'])
    # 使用 DistributedSampler 替代 DataLoader 中的 shuffle=True
    # 它会为每个 GPU 分配不同的数据子集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'],
                              sampler=train_sampler, num_workers=args['dataset_loader_workers'],
                              pin_memory=True) # pin_memory=True 通常能加速数据到 GPU 的传输

    print(f"Rank {rank} start training...")
    for epoch in range(args['load_model_epoch'], args['epochs']):
        # 每个 epoch 开始时，需要设置 sampler 的 epoch，以确保 shuffle 生效
        train_sampler.set_epoch(epoch)

        total_loss_g = 0.0
        total_loss_d = 0.0
        total_loss_g_L1 = 0.0
        total_loss_g_adv = 0.0

        gen.train()
        disc.train()

        # 只有主进程显示 TQDM 进度条
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args['epochs']}", unit="batch", disable=rank!=0)

        for batch_idx, (masked_seqs, clips_seqs, mask_seqs) in enumerate(train_loader):
            for batch_seq in range(int(args['sequence_len'] / args['batch_len'])):
                # 将数据移动到当前进程的 GPU 上
                masked_seq = masked_seqs[:, batch_seq * args['batch_len']: (batch_seq + 1) * args['batch_len'], :, :, :].to(device)
                clips_seq = clips_seqs[:, batch_seq * args['batch_len']: (batch_seq + 1) * args['batch_len'], :, :, :].to(device)

                # gen推理用于disc训练
                clips_fake, gen_hidden = gen(masked_seq, None if batch_seq == 0 else gen_hidden)
                opt_disc.zero_grad()
                real_clip_for_disc = clips_seq.permute(0, 2, 1, 3, 4)
                fake_clip_for_disc = clips_fake.permute(0, 2, 1, 3, 4)
                disc_real = disc(real_clip_for_disc)
                loss_disc_real = adversarial_loss_fn(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake_clip_for_disc.detach())
                loss_disc_fake = adversarial_loss_fn(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake ) / 2
                loss_disc.backward()

                # 训练生成器
                opt_gen.zero_grad()
                disc_fake_for_gen = disc(fake_clip_for_disc)
                loss_g_adv = adversarial_loss_fn(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                loss_g_l1 = l1_loss_fn(clips_fake, clips_seq) * args['L1_weigth']
                loss_g = loss_g_adv + loss_g_l1
                loss_g.backward()

                gen_hidden = (gen_hidden[0].detach(), gen_hidden[1].detach())
            
            # DDP 会自动同步所有进程的梯度，然后才执行 step
            opt_disc.step()
            opt_gen.step()
            
            # --- 日志记录和统计 (仅在主进程中执行) ---
            if rank == 0:
                # 记录梯度权重
                if batch_idx % args['Gradient_intervals'] == 0:
                    for name, param in disc.module.named_parameters(): # 使用 .module
                        if param.grad is not None:
                            writer.add_histogram(
                                tag=f'Grad_disc/{name}', values=param.grad,
                                global_step=epoch * len(train_loader) + batch_idx)
                    for name, param in gen.module.named_parameters(): # 使用 .module
                        if param.grad is not None:
                            writer.add_histogram(tag=f'Grad_gan/{name}', values=param.grad,
                                                 global_step=epoch * len(train_loader) + batch_idx)

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
        
        # 等待所有进程完成当前 epoch 的所有操作
        dist.barrier()

        # --- 模型保存和打印信息 (仅在主进程中执行) ---
        if rank == 0:
            pbar.close()
            avg_loss_g = total_loss_g / len(train_loader)
            avg_loss_d = total_loss_d / len(train_loader)
            avg_loss_g_L1 = total_loss_g_L1 / len(train_loader)
            avg_loss_d_adv = total_loss_g_adv / len(train_loader)
            print(
                f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss_G: {avg_loss_g:.4f}, avg_loss_D: {avg_loss_d:.4f}, avg_loss_g_L1: {avg_loss_g_L1:.4f}, avg_loss_d_adv: {avg_loss_d_adv:.4f} ---")

            pathlib.Path(args['model_save_dir']).mkdir(parents=True, exist_ok=True)
            # 保存时，需要使用 .module 来获取原始模型
            torch.save(gen.module.state_dict(), f"{args['model_save_dir']}/gen_epoch_{epoch + 1}.pth")
            torch.save(disc.module.state_dict(), f"{args['model_save_dir']}/disc_epoch_{epoch + 1}.pth")

    if rank == 0:
        writer.close()
        print("Completed!")
    cleanup()

if __name__ == '__main__':
    # 将所有超参数放入一个字典中，方便传递
    args = {
        'lr_gen': 5e-4,
        'lr_disc': 5e-4,
        'L1_weigth': 50,
        'epochs': 100,
        'batch_size': 6, # 这是每个 GPU 的 batch_size, 总 batch_size = batch_size * num_gpus
        'sequence_len': 24,
        'batch_len': 6,
        'size': (480, 270),
        'dataset_loader_workers': 4,
        'Gradient_intervals': 50,
        'dataset_path': r"/media/B/Triority/Dataset",
        'model_save_dir': r"model_gan_ddp",
        'load_model_epoch': 0,
        'load_model_path_gen': r"model_gan/gen_epoch_1.pth",
        'load_model_path_disc': r"model_gan/disc_epoch_1.pth"
    }

    # 确定要使用的 GPU 数量
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")
    
    # 使用 mp.spawn 启动 world_size 个 main_worker 进程
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    