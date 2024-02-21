import time
from collections import OrderedDict

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from pykeops.torch import LazyTensor

__all__ = ['MinkUNetDiff']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkGlobalEnc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x4


class MinkUNetDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs] 
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        # Stage1 temp embed proj and conv
        self.latent_stage1 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),              
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_stage1 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[0]),
        )

        self.stage1_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage2 temp embed proj and conv
        self.latent_stage2 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_stage2 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[1]),
        )

        self.stage2_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        # Stage3 temp embed proj and conv
        self.latent_stage3 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_stage3 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[2]),
        )

        self.stage3_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage4 temp embed proj and conv
        self.latent_stage4 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),              
        )

        self.latemp_stage4 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[3]),
        )

        self.stage4_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Up1 temp embed proj and conv
        self.latent_up1 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_up1 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.up1_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up2 temp embed proj and conv
        self.latent_up2 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_up2 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[5]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[5], cs[5]),
        )

        self.up2_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up3 temp embed proj and conv
        self.latent_up3 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_up3 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[6]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[6], cs[6]),
        )

        self.up3_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up4 temp embed proj and conv
        self.latent_up4 = nn.Sequential(
            nn.Linear(cs[4], cs[4]),              
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[4], cs[4]),
        )

        self.latemp_up4 = nn.Sequential(
            nn.Linear(cs[4]+cs[4], cs[7]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[7], cs[7]),
        )

        self.up4_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, cs[4]),
        )

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.last  = nn.Sequential(
            nn.Linear(cs[8], 20),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(20, 3),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_timestep_embedding(self, timesteps):
        assert len(timesteps.shape) == 1 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def match_part_to_full(self, x_full, x_part):
        full_c = x_full.C.clone().float()
        part_c = x_part.C.clone().float()

        # hash batch coord
        max_coord = full_c.max()
        full_c[:,0] *= max_coord * 2.
        part_c[:,0] *= max_coord * 2.

        f_coord = LazyTensor(full_c[:,None,:])
        p_coord = LazyTensor(part_c[None,:,:])

        dist_fp = ((f_coord - p_coord)**2).sum(-1)
        match_feats = dist_fp.argKmin(1,dim=1)[:,0]

        return x_part.F[match_feats]

    def forward(self, x, x_sparse, part_feats, t):
        temp_emb = self.get_timestep_embedding(t)

        x0 = self.stem(x_sparse)
        match0 = self.match_part_to_full(x0, part_feats)
        p0 = self.latent_stage1(match0) 
        t0 = self.stage1_temp(temp_emb)
        batch_temp = torch.unique(x0.C[:,0], return_counts=True)[1]
        t0 = torch.repeat_interleave(t0, batch_temp, dim=0) 
        w0 = self.latemp_stage1(torch.cat((p0,t0),-1))

        x1 = self.stage1(x0*w0)
        match1 = self.match_part_to_full(x1, part_feats)
        p1 = self.latent_stage2(match1) 
        t1 = self.stage2_temp(temp_emb)
        batch_temp = torch.unique(x1.C[:,0], return_counts=True)[1]
        t1 = torch.repeat_interleave(t1, batch_temp, dim=0)
        w1 = self.latemp_stage2(torch.cat((p1,t1),-1))

        x2 = self.stage2(x1*w1)
        match2 = self.match_part_to_full(x2, part_feats)
        p2 = self.latent_stage3(match2) 
        t2 = self.stage3_temp(temp_emb)
        batch_temp = torch.unique(x2.C[:,0], return_counts=True)[1]
        t2 = torch.repeat_interleave(t2, batch_temp, dim=0)
        w2 = self.latemp_stage3(torch.cat((p2,t2),-1))

        x3 = self.stage3(x2*w2)
        match3 = self.match_part_to_full(x3, part_feats)
        p3 = self.latent_stage4(match3) 
        t3 = self.stage4_temp(temp_emb)
        batch_temp = torch.unique(x3.C[:,0], return_counts=True)[1]
        t3 = torch.repeat_interleave(t3, batch_temp, dim=0)
        w3 = self.latemp_stage4(torch.cat((p3,t3),-1))

        x4 = self.stage4(x3*w3)
        match4 = self.match_part_to_full(x4, part_feats)
        p4 = self.latent_up1(match4) 
        t4 = self.up1_temp(temp_emb)
        batch_temp = torch.unique(x4.C[:,0], return_counts=True)[1]
        t4 = torch.repeat_interleave(t4, batch_temp, dim=0)
        w4 = self.latemp_up1(torch.cat((t4,p4),-1))

        y1 = self.up1[0](x4*w4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)
        match5 = self.match_part_to_full(y1, part_feats)
        p5 = self.latent_up2(match5) 
        t5 = self.up2_temp(temp_emb)
        batch_temp = torch.unique(y1.C[:,0], return_counts=True)[1]
        t5 = torch.repeat_interleave(t5, batch_temp, dim=0)
        w5 = self.latemp_up2(torch.cat((p5,t5),-1))

        y2 = self.up2[0](y1*w5)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)
        match6 = self.match_part_to_full(y2, part_feats)
        p6 = self.latent_up3(match6) 
        t6 = self.up3_temp(temp_emb)
        batch_temp = torch.unique(y2.C[:,0], return_counts=True)[1]
        t6 = torch.repeat_interleave(t6, batch_temp, dim=0)
        w6 = self.latemp_up3(torch.cat((p6,t6),-1))       

        y3 = self.up3[0](y2*w6)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)
        match7 = self.match_part_to_full(y3, part_feats)
        p7 = self.latent_up4(match7) 
        t7 = self.up4_temp(temp_emb)
        batch_temp = torch.unique(y3.C[:,0], return_counts=True)[1]
        t7 = torch.repeat_interleave(t7, batch_temp, dim=0)
        w7 = self.latemp_up4(torch.cat((p7,t7),-1))
        
        y4 = self.up4[0](y3*w7)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)
         
        return self.last(y4.slice(x).F)


class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.last  = nn.Sequential(
            nn.Linear(cs[8], 20),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(20, out_channels),
            nn.Tanh(),
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        return self.last(y4.slice(x).F)


