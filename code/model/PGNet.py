from torch.autograd import Function
import torchvision
import torch.nn as nn
import torch, copy
import numpy as np

class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2 # sum pooling
        out = self.upsample(out)
        return torch.div(x, out + self.epsilon)


class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)


class UpsamplingNorm(nn.Module):
    def __init__(self, in_channels):
        super(UpsamplingNorm, self).__init__()
        self.upscale_factor = 2
        self.in_channels = in_channels
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.in_channels, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)
        
    def forward(self, feature_map, lr):
        out = self.upsampling(feature_map)
        out = self.den_softmax(out)
        out = self.recover(out, lr)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            #nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            #nn.BatchNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.conv_block(x)
    
class ResidualBlock2(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock2, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 9, 4, 4),
            nn.ReLU(),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
        )
        self.conv = nn.Conv2d(in_features, out_features, 3, 4, 1)
    def forward(self, x):
        return self.conv(x) + self.conv_block(x)
    
    
def get_noise(shape, device):
    return torch.autograd.Variable(torch.zeros(shape, dtype=torch.float32, device=device)).normal_()


from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


revgrad = RevGrad.apply


class PGNetGenerator(nn.Module):
    def __init__(self, config):
        super(PGNetGenerator, self).__init__()
        self.config = config
        
        self.cuda_num = config['cuda_num']
        self.device = torch.device('cuda:{}'.format(self.cuda_num))
        self.base_channel = config['base_channel']
        self.input_channel = config['input_channel']
        self.upscale_factor = config['upscale_factor']
        self.augment_length = self.input_channel // 2
        
        self.embed_f = nn.Embedding(48, self.base_channel)
        self.embed_b = nn.Embedding(48, self.base_channel)
        self.c1_embed = nn.Embedding(48, self.base_channel)
        self.c2_embed = nn.Embedding(48, self.base_channel)
        
        self.rnn_f = nn.LSTMCell(self.base_channel, self.base_channel)
        self.rnn_b = nn.LSTMCell(self.base_channel, self.base_channel)
        self.feat_ext_f = nn.Sequential(
            nn.Conv2d(14, self.base_channel, 5, 1, 2),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel))
        self.feat_ext_b = nn.Sequential(
            nn.Conv2d(14, self.base_channel, 5, 1, 2),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel))
        self.generator_f = nn.Sequential(
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            nn.Conv2d(self.base_channel, 1, 5, 1, 2))
        self.generator_b = nn.Sequential(
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            ResidualBlock(self.base_channel),
            nn.Conv2d(self.base_channel, 1, 5, 1, 2))
        
        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)
        
        torch.manual_seed(0) # cpu
        torch.cuda.manual_seed(0) #gpu
        for m in self.modules():
            try: torch.nn.init.normal_(m.weight.data, 0.0, 1e-2)
            except: pass
            
    def forward(self, batch):
        
        pop_hr_48 = batch['pop_hr'][:,int(self.input_channel//2):int(self.input_channel//2+1)].cuda(self.cuda_num)
        poi_hr = batch['poi_hr'].cuda(self.cuda_num)
        pop_lr = batch['pop_lr'].cuda(self.cuda_num)
        t = batch['t'].cuda(self.cuda_num)
        
        pop_list = [pop_hr_48]
        time_fm_1 = self.embed_f(t)[:, 0]
        time_fm_2 = self.embed_b(t)[:, 0]
        poi_fm_f = self.feat_ext_f(poi_hr)
        poi_fm_b = self.feat_ext_b(poi_hr)
        
        c1 = self.c1_embed(t)[:, 0]
        c2 = self.c2_embed(t)[:, 0]
        for iter in range(self.augment_length):
            time_fm_1, c1 = self.rnn_f(self.embed_f(t + 1 + iter)[:, 0], (time_fm_1, c1))
            time_fm_2, c2 = self.rnn_b(self.embed_b(t + 1 + iter)[:, 0], (time_fm_2, c2))
            
            poi_t_fm_f = poi_fm_f * time_fm_1.unsqueeze(-1).unsqueeze(-1).expand_as(poi_fm_f)
            poi_t_fm_b = poi_fm_b * time_fm_2.unsqueeze(-1).unsqueeze(-1).expand_as(poi_fm_b)
            
            pop_sr_1 = pop_list[0] + self.generator_f(poi_t_fm_f)
            pop_sr_2 = pop_list[-1] + self.generator_b(poi_t_fm_b)
            pop_list = [pop_sr_2] + pop_list + [pop_sr_1]
        
        for iter in range(len(pop_list)):
            pop_list[iter] = self.den_softmax(pop_list[iter])
            pop_list[iter] = self.recover(pop_list[iter], pop_lr[:, iter, :, :].unsqueeze(1))
        
        pop_sr = torch.cat(pop_list, dim=1)
        return pop_sr
    
    def assign_cuda(self, cuda_num):
        self.cuda_num = cuda_num
        
        
class PGNetDiscriminator(nn.Module):
    name = 'PGNetDiscriminator'
    def transforms(self, sample_1, sample_2):
        B1 = sample_1.shape[0]
        B2 = sample_2.shape[0]
        sample_1 = sample_1.repeat([1,1,5,5])[:, :, :64, :64]
        sample_2 = sample_2.repeat([1,1,5,5])[:, :, :64, :64]
        # rand = torch.zeros([B1 + B2,1,1,1], dtype=torch.float32, device=self.device).normal_()
        # comb_sample = sample_1 * (rand > 0).to(torch.float32) + sample_2 * (rand < 0).to(torch.float32)
        # use_1 = (rand > 0)[:, 0, 0, 0].to(torch.float32)
        comb_sample = torch.cat([sample_1, sample_2], dim=0)
        use_1 = torch.zeros([B1 + B2, ], dtype=torch.float32, device=self.device)
        use_1[:B1] = 1
        return comb_sample, use_1
        
        
    def __init__(self, config):
        super(PGNetDiscriminator, self).__init__()
        self.config = config
        
        self.cuda_num = config['cuda_num']
        self.device = torch.device('cuda:{}'.format(self.cuda_num))
        self.base_channel = config['base_channel'] if 'base_channel' in config else 1
        self.input_channel = config['input_channel']
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_channel, self.base_channel, 3, 1, 1),
            #ResidualBlock2(self.base_channel, self.base_channel),
            #ResidualBlock2(self.base_channel, self.base_channel),
            ResidualBlock2(self.base_channel, self.base_channel),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.base_channel, 1),
            nn.Sigmoid()
        )
        
        torch.manual_seed(0) # cpu
        torch.cuda.manual_seed(0) #gpu
        for m in self.modules():
            try: torch.nn.init.normal_(m.weight.data, 0.0, 1e-2)
            except: pass
            
    def forward(self, pop_hr, pop_sr):
        pop_hr = pop_hr.cuda(self.cuda_num)
        pop_sr = pop_sr.cuda(self.cuda_num)
        comb_pop, use_hr = self.transforms(pop_hr, pop_sr)
        
        x = self.conv_block(comb_pop)
        x = self.classifier(x[:, :, 0, 0])[:, 0]
        return x, use_hr
    
    def assign_cuda(self, cuda_num):
        self.cuda_num = cuda_num
        
        
        
        
class PADADiscriminator(nn.Module):
    name = 'PADADiscriminator'
    # Pixel Level Discriminator
    def transform2(self, sample_1, sample_2):
        B1, C, W1, H1 = sample_1.shape
        B2, C2, W2, H2 = sample_2.shape
        assert(C == C2)
        sampled_pixels = min([64, W1*H1, W2*H2])
        
        out_1 = sample_1.reshape([B1, C, W1*H1])[..., torch.randperm(W1*H1)[:sampled_pixels]]
        out_2 = sample_2.reshape([B2, C, W2*H2])[..., torch.randperm(W2*H2)[:sampled_pixels]] # [B, C, 32]
        out = torch.cat([out_1, out_2], dim=0).transpose(0, 1).reshape([C, (B1 + B2) * sampled_pixels]).transpose(0, 1) # [(B1 + B2) * 32, C]
        
        use_1 = torch.zeros([B1 + B2, 1, sampled_pixels], dtype=torch.float32, device=self.device)
        use_1[:B1] = 1
        use_1 = use_1.transpose(0, 1).reshape([1, (B1 + B2) * sampled_pixels]).transpose(0, 1) # [(B1 + B2) * 32, 1]
        
        return out, use_1

    def transform(self, sample_1, sample_2):
        B1, C, W1, H1 = sample_1.shape
        B2, C, W2, H2 = sample_1.shape
        
        sample_1 = sample_1.repeat([1,1,5,5])[:, :, :64, :64]
        sample_2 = sample_2.repeat([1,1,5,5])[:, :, :64, :64]
        
        # rand = torch.zeros([B1 + B2,1,1,1], dtype=torch.float32, device=self.device).normal_()
        # comb_sample = sample_1 * (rand > 0).to(torch.float32) + sample_2 * (rand < 0).to(torch.float32)
        # use_1 = (rand > 0)[:, 0, 0, 0].to(torch.float32)
        comb_sample = torch.cat([sample_1, sample_2], dim=0) #[B1 + B2, C, 64, 64]
        use_1 = torch.zeros([B1 + B2, 1, 64, 64], dtype=torch.float32, device=self.device)
        use_1[:B1] = 1
        comb_sample = torch.transpose(comb_sample, 0, 1).reshape([C, (B1 + B2)*64*64]) # [C, (B1 + B2)*64*64]
        use_1 = torch.transpose(use_1, 0, 1).reshape([1, (B1 + B2)*64*64]) # [1, (B1 + B2)*64*64]
        comb_sample = torch.transpose(comb_sample, 0, 1) # [(B1 + B2)*64*64, C]
        use_1 = torch.transpose(use_1, 0, 1) # [(B1 + B2)*64*64, 1]
        return comb_sample, use_1
        
        
    def __init__(self, config):
        super(PADADiscriminator, self).__init__()
        self.config = config
        
        self.cuda_num = config['cuda_num']
        self.device = torch.device('cuda:{}'.format(self.cuda_num))
        self.base_channel = config['base_channel']
        self.input_channel = config['input_channel']
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.input_channel, self.base_channel, 5, 2, 2),
            nn.Sigmoid(),
            #ResidualBlock2(self.base_channel, self.base_channel),
            #ResidualBlock2(self.base_channel, self.base_channel),
            #ResidualBlock(self.base_channel),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.base_channel, self.base_channel*4),
            nn.Sigmoid(),
            nn.Linear(self.base_channel*4, self.base_channel),
            nn.Sigmoid(),
            nn.Linear(self.base_channel, 1),
            nn.Sigmoid()
        )
        
        torch.manual_seed(0) # cpu
        torch.cuda.manual_seed(0) #gpu
        for m in self.modules():
            try: torch.nn.init.normal_(m.weight.data, 0.0, 1e-2)
            except: pass
            
    def forward(self, pop_hr, pop_sr):
        pop_hr = pop_hr.cuda(self.cuda_num)
        pop_sr = pop_sr.cuda(self.cuda_num)
        pop_hr = self.conv_block(pop_hr)
        pop_sr = self.conv_block(pop_sr)
        
        comb_channels, use_hr_channels = self.transform2(pop_hr, pop_sr)
        
        x = self.classifier(comb_channels)
        return x, use_hr_channels
    
    def assign_cuda(self, cuda_num):
        self.cuda_num = cuda_num