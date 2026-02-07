import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

# ---------------------------
# MCADS Decoder（多尺度侧输出顺序一致：z6,z5,z4,z3,z2,z1）
# 输入：s1..s5（encoder 各 stage）、b1（bridge）
# ---------------------------
class MCADSDecoder(nn.Module):
    def __init__(self,c3,c4,c5,cb,num_class = 1):
        super().__init__()
        # Bridge -> s5
        self.dsub1 = DSUB(cb)
        self.conv_to_s5 = ConvBlock(cb,c5)
        self.rlab5 = RLAB(c5,c5,c5,res_len=5)
        self.casab5 = CASAB(c5)

        # S5->S4
        self.dsub2 = DSUB(c5)
        self.conv_to_s4 = ConvBlock(c5,c4)
        self.rlab4 = RLAB(c4,c4,c4,res_len=4)
        self.casab4 = CASAB(c4)

        #s4->s3
        self.eub3 = EUB(c4,c3)
        self.rlab3 = RLAB(c3,c3,c3,res_len=3)
        self.casab3 = CASAB(c3)

        # # s3->s2
        # self.eub2 = EUB(c3,c2)
        # self.rlab2 = RLAB(c2,c2,c2,res_len=2)
        # self.casab2 = CASAB(c2)
        #
        # #s s2 -> s1
        # self.eub1 = EUB(c2,c1)
        # self.rlab1 = RLAB(c1,c1,c1,res_len=1)
        # self.casab1 = CASAB(c1)

        # 侧输出头
        self.head_d5 = nn.Conv2d(c5,num_class,3,padding=1,bias=False)
        self.head_d4 = nn.Conv2d(c4,num_class,3,padding=1,bias=False)
        self.head_d3 = nn.Conv2d(c3,num_class,3,padding=1,bias=False)
        # self.head_d2 = nn.Conv2d(c2,num_class,3,padding=1,bias=False)
        # self.head_d1 = nn.Conv2d(c1,num_class,3,padding=1,bias=False)
        self.head_b1 = nn.Conv2d(cb,num_class,3,padding=1,bias=False)

        self.final_seg_head = nn.Sequential(
            nn.Conv2d(c3, c3 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3 // 2, num_class, 1),
            nn.Sigmoid()
        )
    def forward(self,s3,s4,s5,b1):
        # bridge ->s5
        u1 = self.conv_to_s5(self.dsub1(b1))

        d1 = self.casab5(self.rlab5(u1,s5))

        # s5 -s4
        u2 = self.conv_to_s4(self.dsub2(d1))

        d2 = self.casab4(self.rlab4(u2,s4))

        # s4 -> s3
        u3 = self.eub3(d2)

        d3 = self.casab3(self.rlab3(u3,s3))

        # # s3 -> s2
        # u4 = self.eub2(d3)
        #
        # d4 = self.casab2(self.rlab2(u4,s2))
        #
        # # s2 -> s1
        # u5 = self.eub1(d4)
        # d5 = self.casab1(self.rlab1(u5,s1))


        # 测输出（全部上采样到s1 分辨率，顺序保持）
        H, W = s3.shape[-2:]
        z3 = self.head_d3(d3)
        # z2 = F.interpolate(self.head_d2(d4), size=(H, W), mode='bilinear', align_corners=False)
        # z3 = F.interpolate(self.head_d3(d3), size=(H, W), mode='bilinear', align_corners=False)
        z4 = F.interpolate(self.head_d4(d2), size=(H, W), mode='bilinear', align_corners=False)
        z5 = F.interpolate(self.head_d5(d1), size=(H, W), mode='bilinear', align_corners=False)
        z6 = F.interpolate(self.head_b1(b1), size=(H, W), mode='bilinear', align_corners=False)

        # 最终输出到原图大小
        final = self.final_seg_head(d3)  # [B,1,H/4,W/4]
        final = F.interpolate(final, scale_factor=4, mode='bilinear', align_corners=False)  # -> [B,1,H,W]

        return final, (z6, z5, z4, z3)


if __name__ == "__main__":
    B, H, W = 1, 512, 512
    c3, c4, c5, cb = 256, 384, 768, 768
    dec = MCADSDecoder(c3, c4, c5, cb, num_class=1)

    s3 = torch.randn(B, c3, H // 4, W // 4)
    s4 = torch.randn(B, c4, H // 8, W // 8)
    s5 = torch.randn(B, c5, H // 16, H // 16)
    b1 = torch.randn(B, cb, H // 32, H // 32)

    final, outs = dec(s3, s4, s5, b1)
    print("final:", final.shape)   # -> [1,1,512,512]
    print([o.shape for o in outs]) # -> [1,1,128,128] 侧输出尺寸一致






