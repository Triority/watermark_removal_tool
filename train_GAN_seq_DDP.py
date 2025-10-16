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
from collections import OrderedDict

from train import RecurrentUNet, VideoDataset
from train_GAN_seq import VideoDiscriminator


def main_worker(rank, world_size, args):
    """
    主训练函数，每个 GPU 进程都会执行此函数。
    rank: 当前进程的 ID (也是 GPU 的 ID)
    world_size: 进程总数 (GPU 总数)
    args: 包含所有参数的字典
    """
    print(f"Running DDP on rank {rank}.")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 每个进程只在自己的 GPU 上工作
    device = rank
    torch.cuda.set_device(device)

    # 只有主进程 (rank 0) 才需要进行日志记录
    writer = SummaryWriter(args['summary_writer']) if rank == 0 else None

    gen = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    disc = VideoDiscriminator(in_channels=3).to(device)
    gen = DDP(gen, device_ids=[rank], find_unused_parameters=True)
    disc = DDP(disc, device_ids=[rank], find_unused_parameters=True)

    # 加载模型
    if args['load_model_epoch'] != 0:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        print(f"Rank {rank} loading Generator from {args['load_model_path_gen']}")
        gen_state_dict = torch.load(args['load_model_path_gen'], map_location=map_location)
        # 创建一个新的字典，并为每个 key 添加 'module.' 前缀
        if args['load_model_add_.model']:
            new_gen_state_dict = OrderedDict()
            for k, v in gen_state_dict.items():
                name = 'module.' + k
                new_gen_state_dict[name] = v
        
        gen.load_state_dict(new_gen_state_dict)

        print(f"Rank {rank} loading Critic from {args['load_model_path_disc']}")
        disc_state_dict = torch.load(args['load_model_path_disc'], map_location=map_location)
        if args['load_model_add_.model']:
            new_disc_state_dict = OrderedDict()
            for k, v in disc_state_dict.items():
                name = 'module.' + k
                new_disc_state_dict[name] = v

        disc.load_state_dict(new_disc_state_dict)

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
    # 它会为每个 GPU 分配不同的数据子集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'],
                              sampler=train_sampler, num_workers=args['dataset_loader_workers'])

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
            # 将长序列分割成多个小批次进行处理
            for batch_seq in range(int(args['sequence_len'] / args['batch_len'])):
                masked_seq = masked_seqs[:, batch_seq * args['batch_len']: (batch_seq + 1) * args['batch_len'], :, :, :].to(device)
                clips_seq = clips_seqs[:, batch_seq * args['batch_len']: (batch_seq + 1) * args['batch_len'], :, :, :].to(device)

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
                opt_disc.step()

                opt_gen.zero_grad()
                disc_fake_for_gen = disc(fake_clip_for_disc)
                loss_g_adv = adversarial_loss_fn(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                loss_g_l1 = l1_loss_fn(clips_fake, clips_seq) * args['L1_weigth']
                loss_g = loss_g_adv + loss_g_l1
                loss_g.backward()

                gen_hidden = (gen_hidden[0].detach(), gen_hidden[1].detach())
                opt_gen.step()
            
            # 日志记录和统计 (仅在主进程中执行)
            if rank == 0:
                if batch_idx % args['Gradient_intervals'] == 0:
                    for name, param in disc.module.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(
                                tag=f'Grad_disc/{name}', values=param.grad,
                                global_step=epoch * len(train_loader) + batch_idx)
                    for name, param in gen.module.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(tag=f'Grad_gan/{name}', values=param.grad,
                                                 global_step=epoch * len(train_loader) + batch_idx)

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
                    G_adv=f'{loss_g_adv.item():.4f}',
                    G_L1=f'{loss_g_l1.item():.4f}')
                pbar.update(1)
        
        # 等待所有进程完成当前 epoch 的所有操作
        dist.barrier()

        # 模型保存和打印信息 (仅在主进程中执行)
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
    dist.destroy_process_group()


if __name__ == '__main__':
    args = {
        'lr_gen': 1e-4,
        'lr_disc': 1e-4,
        'L1_weigth': 50,
        'epochs': 110,
        'batch_size': 6,
        'sequence_len': 16,
        'batch_len': 4,
        'size': (480, 270),
        'dataset_loader_workers': 2,
        'Gradient_intervals': 50,
        'dataset_path': r"D:\Dataset",
        'model_save_dir': r"model_gan_try",
        'load_model_epoch': 100,
        'load_model_path_gen': r"model_gan/gen_epoch_100.pth",
        'load_model_path_disc': r"model_gan/disc_epoch_100.pth",
        'load_model_add_.model': True,
        'summary_writer': 'runs/GAN_DDP_try'
    }

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")
    
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    