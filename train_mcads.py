import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import datetime
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from data_loader import Dataset
import augumentations as augu
import plots

# ========= 你项目里的模块（按你的实际路径导入）=========
# from your_modules import DSUB, EUB, RLAB, CASAB, ConvBlock
# =====================================================

def set_global_random_seed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- 自定义 Decoder（4层，对齐 f1..f4） ---------------------

# (c) ConvBlock
class ConvBlock(nn.Module):
    """
    # Conv → BN → ReLU → Pointwise Conv → BN → ReLU
    """

    def __init__(self, in_ch,out_ch,nagative_slop=0.01):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1,groups=in_ch,bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch) # 归一化
        self.act1 = nn.LeakyReLU(nagative_slop,inplace=True)
        self.pointwise = nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(nagative_slop,inplace=True)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x

# ---------------------------
# RB（用于 ResPath）：DW(k) -> BN -> (no act) -> PW(1x1) -> BN -> (no act)
# ResPath：每次迭代后做 Add -> LeakyReLU -> BN
# ---------------------------
class ConvBlockResPath(nn.Module):
    def __init__(self, in_ch,out_ch,k=3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch,in_ch,kernel_size=k,padding=k//2,groups=in_ch,bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch,out_ch,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.pw(x)
        x = self.bn2(x)
        return x


class ResPath(nn.Module):
    """
    ResPath 模块

    功能：
        实现一条由多个 Conv-BN-ReLU 组成的残差路径，用于特征逐步匹配。
        常用于 U-Net 或编码器-解码器之间的桥接连接。

    参数：
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
        length (int): 卷积块个数（ResPath 的深度）

    返回：
        torch.Tensor: 输出特征图，尺寸与输入相同，但通道调整为 out_ch
    """
    # 3*3 conv -> BN -> relu
    # 第一层 in_ch x out_ch *3 *3
    def __init__(self,in_ch,out_ch,length):
        super().__init__()
        self.length = length
        self.rb3 = nn.ModuleList()
        self.rb1 = nn.ModuleList()
        self.post_bn = nn.ModuleList()
        self.post_act = nn.ModuleList()

        for i in range(length):
            cin = in_ch if i == 0 else out_ch
            self.rb3.append(ConvBlockResPath(cin,out_ch,k=3))
            self.rb1.append(ConvBlockResPath(cin,out_ch,k=1))
            self.post_bn.append(nn.BatchNorm2d(out_ch))
            self.post_act.append(nn.LeakyReLU(inplace=True))
    def forward(self,x):
        for i in range(self.length):
            x0 = x
            x = self.rb3[i](x0) + self.rb1[i](x0)
            x = self.post_act[i](x)  # 这里的残差 是先激活后归一化
            x = self.post_bn[i](x)
        return x

# 线性自注意力
class LinearSelfAttention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        # 把特征空间平展成一维序列 N = H x W,每个像素点视为一个token
        self.query_conv = nn.Conv2d(channels,channels // 8,1,bias=False)
        self.key_conv = nn.Conv2d(channels,channels // 8,1,bias=False)
        self.value_conv = nn.Conv2d(channels,channels,1,bias=False)
        self.gamma = nn.Parameter(torch.zeros(1)) # 残差缩放系数

    def forward(self,x):
        B,C,H,W = x.size()
        Cq= C //8 # q,k 通道数
        N= H * W

        # Q, K, V 计算
        q = self.query_conv(x).view(B, Cq, N)
        k = self.key_conv(x).view(B, Cq, N)
        v = self.value_conv(x).view(B, C, N)

        # 注意力矩阵计算
        attn = torch.bmm(q.permute(0,2,1), k)/ math.sqrt(Cq)
        attn = F.softmax(attn,dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.view(B,C,H,W)
        return self.gamma * out + x

# ---------------------------
# (f) CASAB：CAM + SAM，然后用加法融合
# CAM：GAP+GMP -> 1x1(SiLU) -> 1x1(Sigmoid)，乘到 x
# SAM：mean/max/min/sum(按通道) -> concat -> 7x7+SiLU -> 1x1+Sigmoid，乘到 x
# 输出： (x*CAM) + (x*SAM)
# ---------------------------
class ChannelAttentionModule(nn.Module):
    def __init__(self,channels, reduction=16):
        super().__init__()
        hidden = max(1,channels//reduction)
        self.gap = nn.AdaptiveAvgPool2d(1) # 平均池化
        self.gmp = nn.AdaptiveMaxPool2d(1) # 最大值池化
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1,bias=False)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        se = self.gap(x) + self.gmp(x)
        se = self.fc1(se) # 降维
        se = F.silu(se) # 给中间hidden 层增加平滑非线性，提升拟合能力
        se = self.fc2(se) # 升维
        se = self.sig(se) # 压缩到（0,1） 用于权重门控
        return x * se

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv7 = nn.Conv2d(4,1,7,padding=3,bias=False)
        self.conv1 = nn.Conv2d(1,1,1,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x_mean = torch.mean(x,dim=1,keepdim=True)
        x_max,_ = torch.max(x,dim=1,keepdim=True)
        x_min,_ = torch.min(x,dim=1,keepdim=True)
        x_sum = torch.sum(x,dim=1,keepdim=True)
        feats = torch.cat([x_mean,x_max,x_min,x_sum],dim=1)
        feats = F.silu(self.conv7(feats))
        feats = self.sig(self.conv1(feats))
        return x * feats

class CASAB(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__()
        self.cam = ChannelAttentionModule(channels,reduction)
        self.sam = SpatialAttentionModule()

    def forward(self,x):
        return self.cam(x) + self.sam(x)


# ---------------------------
# (g) RLAB：ResPath(若干 RB) -> concat -> ConvBlock -> LinearAttention
#          -> 与 decoder 分支做残差相加 -> ConvBlock
# ---------------------------
class RLAB(nn.Module):
    """
    Args:
        dec_ch:DSUB-EUB out_channels
        skip_ch:skip_connection out_channels
        out_ch:
        res_len:循环的层数
    """
    def __init__(self,dec_ch,skip_ch,out_ch,res_len,):
        super().__init__()
        self.respath = ResPath(skip_ch,skip_ch,res_len) # 残差跳跃层
        self.fuse_conv = ConvBlock(dec_ch+skip_ch,out_ch)
        self.attn = LinearSelfAttention(out_ch)
        self.post_conv = ConvBlock(out_ch,out_ch)

    def forward(self,dec_x,skip_x):

        skip_x = self.respath(skip_x)

        x = torch.cat([dec_x,skip_x],dim=1)

        x = self.fuse_conv(x)
        x = self.attn(x)
        x = x+ dec_x
        x = self.post_conv(x)
        return x

# ---------------------------
# (d) DSUB：3x3 -> PixelShuffle(2) -> 3x3（通道不变），后接 ConvBlock 调整到目标通道
# (e) EUB：上采样2x + ConvBlock
# ---------------------------
class DSUB(nn.Module):
    def __init__(self,channels,scale=2):
        super().__init__()
        self.pre = nn.Conv2d(channels,channels*(scale**2),3,padding=1,bias=False) # 输出是亚像素采样，用通道数来换空间分辨率
        self.ps = nn.PixelShuffle(scale)
        self.post = nn.Conv2d(channels,channels,3,padding=1,bias=False)

    def forward(self,x):
        x = self.pre(x)
        x = F.relu(x,inplace=True)

        x = self.ps(x)

        x = self.post(x)
        x = F.relu(x,inplace=True)
        return x

class EUB(nn.Module):
    def __init__(self,in_ch,out_ch):

        super().__init__()
        self.conv = ConvBlock(in_ch,out_ch)

    def forward(self,x):

        x = F.interpolate(x,scale_factor=2.0,mode="bilinear",align_corners=False) # 通道不变，分辨率扩大一倍
        x = self.conv(x)
        return x

class MCADSDecoder(nn.Module):
    """
    与 ConvMAE encoder 输出对齐的 4 层解码器：
      输入:  s3(1/4, C=c3), s4(1/8, C=c4), s5(1/16, C=c5), b1(1/32, C=cb)
      路径:  b1 -> s5 -> s4 -> s3
      输出:  final(B,1,H,W) + 侧输出(用于深监督，可选)
    """
    def __init__(self, c3, c4, c5, cb, num_class=1,final_act = 'sigmoid'):
        super().__init__()
        # ======= 下面这几个模块使用你的实现 =======
        self.dsub1 = DSUB(cb)
        self.conv_to_s5 = ConvBlock(cb, c5)
        self.rlab5 = RLAB(c5, c5, c5, res_len=5)
        self.casab5 = CASAB(c5)

        self.dsub2 = DSUB(c5)
        self.conv_to_s4 = ConvBlock(c5, c4)
        self.rlab4 = RLAB(c4, c4, c4, res_len=4)
        self.casab4 = CASAB(c4)

        self.eub3 = EUB(c4, c3)
        self.rlab3 = RLAB(c3, c3, c3, res_len=3)
        self.casab3 = CASAB(c3)

        # 最终分割输出（在 1/4 尺度产生 logits，再上采样到 1×）
        self.final_seg_head = nn.Sequential(
            nn.Conv2d(c3, c3 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3 // 2, num_class, 1)
        )
        # activation 可选
        if final_act is None or final_act == 'none':
            self.final_activation = nn.Identity()
        elif final_act.lower() == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported final_act: {final_act}")

    def forward(self, s3, s4, s5, b1):
        # b1(1/32) -> s5(1/16)
        u1 = self.conv_to_s5(self.dsub1(b1))
        d1 = self.casab5(self.rlab5(u1, s5))

        # s5(1/16) -> s4(1/8)
        u2 = self.conv_to_s4(self.dsub2(d1))
        d2 = self.casab4(self.rlab4(u2, s4))

        # s4(1/8) -> s3(1/4)
        u3 = self.eub3(d2)
        d3 = self.casab3(self.rlab3(u3, s3))  # 1/4

        # 在 1/4 产生 logits
        logits_1_4 = self.final_seg_head(d3)  # [B,C,H/4,W/4]

        # 上采样到原图大小（若未提供 input_hw，则默认放大 4 倍）
        H_in = s3.shape[-2] * 4
        W_in = s3.shape[-1] * 4
        logits = F.interpolate(logits_1_4, size=(H_in, W_in), mode='bilinear', align_corners=False)

        final = self.final_activation(logits)
        return final


# --------------------- 整体模型封装：Encoder + Decoder ---------------------
class ConvMAE_SegModel(nn.Module):
    """
    取 SMP 的自定义 encoder（convmae/dconvmae），接入我们自定义的解码器。
    encoder(x) 返回 [f1,f2,f3,f4] 分别对应：
        f1: 1/4,  C=256
        f2: 1/8,  C=384
        f3: 1/16, C=768
        f4: 1/32, C=768
    我们映射到 decoder 的 (s3,s4,s5,b1) 顺序。
    """
    def __init__(self, encoder_name, encoder_weights, in_ch=3, n_class=1):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder
        self.encoder = get_encoder(
            name=encoder_name,  # 'convmae' 或 'dconvmae'（你注册的名字）
            in_channels=in_ch,
            weights=encoder_weights,  # ckpt 路径
        )
        # 读取通道定义，确保与打印一致
        # 期望 out_channels = (3, 256, 384, 768, 768)
        oc = self.encoder.out_channels
        assert len(oc) >= 5, f"Unexpected encoder.out_channels: {oc}"
        c3, c4, c5, cb = oc[1], oc[2], oc[3], oc[4]
        self.decoder = MCADSDecoder(c3, c4, c5, cb, num_class=n_class)

    def forward(self, x):
        feats = self.encoder(x)  # [f1,f2,f3,f4]  -> 1/4,1/8,1/16,1/32
        assert len(feats) >= 4, f"Encoder should return 4 feature maps, got {len(feats)}"
        s3, s4, s5, b1 = feats[0], feats[1], feats[2], feats[3]
        final = self.decoder(s3, s4, s5, b1)
        return final  # smp.utils 的 Train/ValidEpoch 默认用这个作为 y_pred


# --------------------- 训练主程序（复刻你现有风格） ---------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--encoder', default='convmae', type=str, help='encoder name (convmae or dconvmae)')
    parser.add_argument('--encoder_weights', default='/home/jackiechen/learingProjects/learning/DeblurringMIM/DeblurringMIM-main/output_pretrain/1110_pretrain_tn3k/checkpoint-10769.pth', type=str,
                        help='encoder weights (pretrained ckpt path)')
    parser.add_argument('--input_size', default=512, type=int, help='images input size')
    parser.add_argument('--datapath', default='/home/jackiechen/learingProjects/dataset_segmention_ddti', type=str, help='dataset path')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--output_dir', default='./runs/', type=str, help='output directory')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--early_stops', default=100, type=int, help='early stop patience (epochs)')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    set_global_random_seed(args.seed)

    DATA_DIR = args.datapath
    x_train_dir = os.path.join(DATA_DIR, 'images/train/')
    x_valid_dir = os.path.join(DATA_DIR, 'images/val/')
    y_train_dir = os.path.join(DATA_DIR, 'masks/train/')
    y_valid_dir = os.path.join(DATA_DIR, 'masks/val/')

    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 1
    batch_size = args.batch_size
    t_size = args.input_size

    # ---- 模型：自定义 Encoder+Decoder ----
    model = ConvMAE_SegModel(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_ch=3,
        n_class=n_class
    )

    if torch.cuda.is_available():
        print("CUDA is available, using GPU.")
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(DEVICE)

    # ---- 日志目录 ----
    tag = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_custom_{ENCODER}_bs{batch_size}_{t_size}_seed{args.seed}"
    model_dir = os.path.join(args.output_dir, tag)
    os.makedirs(model_dir, exist_ok=True)

    # ---- Datasets & Loaders ----
    train_dataset = Dataset(x_train_dir, y_train_dir, augmentation=augu.get_training_augmentation(), t_size=t_size,need_down=False)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, t_size=t_size,need_down=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    print("Train:", len(train_dataset))
    print("Valid:", len(valid_dataset))

    # ---- Loss & Metrics（与原脚本一致）----
    loss = smp.utils.losses.DiceLoss()  # 你也可以换成 Dice + BCE
    metrics = [
        smp.utils.metrics.Fscore(),  # Dice 系列
        smp.utils.metrics.IoU(),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
    ])

    from segmentation_models_pytorch import utils
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_dice = 0
    best_epoch = 0
    EARLY_STOPS = int(args.early_stops)
    train_dict = {'loss': [], 'dice': [], 'iou': []}
    val_dict = {'loss': [], 'dice': [], 'iou': []}

    for i in range(0, 100000):
        print(f"\nEpoch: {i}")
        print("Best epoch:", best_epoch, "\tDICE:", max_dice)

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_dict['loss'].append(train_logs['dice_loss'])
        train_dict['dice'].append(train_logs['fscore'])
        train_dict['iou'].append(train_logs['iou_score'])
        val_dict['loss'].append(valid_logs['dice_loss'])
        val_dict['dice'].append(valid_logs['fscore'])
        val_dict['iou'].append(valid_logs['iou_score'])

        # 保存 loss & dice 曲线
        plots.save_loss_dice(train_dict, val_dict, model_dir)

        # 保存最优
        if max_dice < valid_logs['fscore']:
            if max_dice != 0:
                old_filepath = os.path.join(model_dir, f"{best_epoch}_dice_{max_dice}.pt")
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            max_dice = float(np.round(valid_logs['fscore'], 5))
            torch.save(model, os.path.join(model_dir, f"{i}_dice_{max_dice}.pt"))
            print('Model saved!')
            best_epoch = i

        # 早停
        if i - best_epoch > EARLY_STOPS:
            print(f"{EARLY_STOPS} epochs didn't improve, early stop.")
            print("Best dice:", max_dice)
            break
