## 概述
仓库用于实践测试深度学习视觉处理内容，顺便给我西藏旅行的无人机素材去掉osd水印

训练好的模型文件和推理输入输出视频详见`published_model&video`文件夹

## 环境设置
### Git LFS
本项目使用 Git LFS 来管理大型模型文件(>100MB)，请确保你已经安装并配置好了 Git LFS (Large File Storage)

[git-lfs.github.com](https://git-lfs.github.com)：根据你的操作系统进行安装。

打开终端或命令行，运行以下命令安装LFS：
```bash
git lfs install
```

现在你可以像往常一样克隆本仓库，LFS会自动下载所需的模型文件
```bash
git clone https://github.com/Triority/watermark_removal_tool.git
```


### python库
```bash
pip install -r requirements.txt
```
如果网络较慢，可以指定国内镜像源加速安装
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动命令
定时刷新nvidia-smi：
```
watch -n 1 nvidia-smi
```
### 推理
默认分辨率480*270直接启动批量推理：
```
python3 infer.py
```

高分辨率视频切分推理：


### 训练
直接启动：
```
python3 train_GAN_seq.py
```
指定显卡训练：
```
CUDA_VISIBLE_DEVICES=0 python3 train_GAN_seq.py
```
多显卡并行训练：
```
python3 train_GAN_seq_DDP.py
```

## 文件说明
### 训练
+ `train.py`：定义了u-net+LSTM网络，并直接进行训练
+ `train_GAN.py`：定义了一个三维卷积网络作为判别器进行对抗训练
+ `train_GAN_seq.py`：训练过程增加更多输入的帧序列的计算
+ `train_GAN_seq_DDP.py`：加入了多卡并行训练提高训练速度(未验证)
### 推理
+ `infer.py`：直接进行模型推理
+ `infer_split.py`：针对更大分辨率的视频，将视频分块处理再推理(画饼)
### 其他


## 可能的问题
### 训练中断导致显卡未释放
查看python进程pid
```
ps ax
```
关闭进程
```
kill -p <pid>
```
