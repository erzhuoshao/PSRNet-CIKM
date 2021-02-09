import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from math import floor
import math

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1,bias=False)),

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features


class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features))

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        return self.conv_layer(torch.cat(prev_features, dim=1))


class _DenseBlock(nn.Module):
    _version = 2
    __constants__ = ['layers']
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = ConvLayer(num_input_features + i * growth_rate, growth_rate)
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class UpsamplingNorm(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(UpsamplingNorm, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * (upscale_factor**2), 3, 1, 1),
            nn.BatchNorm2d(self.in_channels * (upscale_factor**2)),
            nn.PixelShuffle(upscale_factor=upscale_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, 1, 9, 1, 4),
            nn.ReLU(inplace=True))
        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)

    def forward(self, feature_map, lr):
        out = self.upsampling(feature_map)
        out = self.den_softmax(out)
        out = self.recover(out, lr)
        return out


class SNet(nn.Module):
    name = 'SNet'
    criterion = nn.MSELoss()
    def __init__(self, args):
        super(SNet, self).__init__()

        self.args = args
        self.cuda_num = args.cuda_num
        self.input_channel = args.input_channel
        self.output_channel = 1
        self.upscale_factor = args.upscale_factor
        self.base_channel = 64
        self.growth_rate = 64
        self.num_layers = 6

        self.name = 'SNet-ic={0}-uf={1}-res={5}->{6}-source={7}-seed={8}'.format(
            self.input_channel, self.upscale_factor,
            args.lr_downscale, args.hr_downscale, args.source, args.seed)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.base_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

        self.dense_blocks = _DenseBlock(num_layers=self.num_layers, num_input_features=self.base_channel, growth_rate=self.growth_rate)

        input_channel = self.base_channel + self.num_layers * self.growth_rate

        upsampling = []
        input_channel = self.base_channel + self.num_layers * self.growth_rate
        for out_features in range(int(math.log(self.upscale_factor, 2))):
            upsampling += [nn.Conv2d(input_channel, self.base_channel * 4, 3, 1, 1),
                           nn.BatchNorm2d(self.base_channel * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
            input_channel = self.base_channel
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = nn.Sequential(nn.Conv2d(self.base_channel, 1, 9, 1, 4), nn.ReLU(inplace=True))

        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)

        self.criterion = None
        self.optimizer = None

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def assign_cuda(self, cuda_num):
        self.cuda_num = cuda_num

    def forward(self, batch):
        pop_lr = batch['pop_lr'][:, -self.input_channel:].cuda(self.cuda_num)

        fm = self.conv1(pop_lr)
        out = self.dense_blocks(fm)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.den_softmax(out)
        out = self.recover(out, pop_lr[:, -1:, :, :])
        return {'pop_sr':out, 'feature_map':fm}


class STNet(nn.Module):
    def __init__(self, args):
        super(STNet, self).__init__()

        self.args = args
        self.input_channel = args.input_channel
        self.upscale_factor = args.upscale_factor
        self.base_channel = args.base_channel
        self.time_channel = args.time_channel
        self.time_stride = args.time_stride
        self.cuda_num = args.cuda_num
        self.num_dense_blocks = self.input_channel // self.time_stride
        self.seed = args.seed

        self.name = 'STNet-ic={0}-uf={1}-bc={2}-tc={3}-ts={4}-res={5}->{6}-source={7}-seed={8}'.format(
            self.input_channel, self.upscale_factor, self.base_channel,
            self.time_channel, self.time_stride,
            args.lr_downscale, args.hr_downscale, args.source, args.seed)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                1, self.base_channel,
                kernel_size = (self.time_stride * 2 + 1, 5, 5),
                padding = (self.time_stride, 2, 2),
                stride = (self.time_stride, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.local_extraction_unit_1 = nn.Sequential(
            nn.Conv2d(1, self.base_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

        self.local_extraction_unit_2 = nn.Sequential(
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channel, self.time_channel, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

        self.dense_blocks = nn.ModuleDict({})
        for iter in range(self.num_dense_blocks):
            self.dense_blocks[str(iter)] = ConvLayer((self.base_channel+self.time_channel)*(iter+1), self.base_channel)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.base_channel * (self.num_dense_blocks + 1), self.base_channel, 5, 1, 2),
            nn.BatchNorm2d(self.base_channel),
            nn.ReLU(inplace=True)
        )

        upsampling = []
        input_channel = self.base_channel * (self.num_dense_blocks + 1)
        for out_features in range(int(math.log(self.upscale_factor, 2))):
            upsampling += [nn.Conv2d(input_channel, self.base_channel * 4, 3, 1, 1),
                           nn.BatchNorm2d(self.base_channel * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
            input_channel = self.base_channel
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = nn.Sequential(nn.Conv2d(self.base_channel, 1, 9, 1, 4), nn.ReLU(inplace=True))
        self.nearest = nn.Upsample(scale_factor=self.upscale_factor, mode='nearest')

        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)

        self.criterion = None
        self.optimizer = None

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('Conv3d') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def assign_cuda(self, cuda_num):
        self.cuda_num = cuda_num

    def forward(self, batch):
        pop_lr = torch.flip(batch['pop_lr'].unsqueeze(1).cuda(self.cuda_num), dims=[2])

        x0 = self.local_extraction_unit_1(pop_lr[:, :, 0])

        x1 = self.conv1(pop_lr) #
        x2 = [self.local_extraction_unit_2(x1[:, :, iter]) for iter in range(self.num_dense_blocks)]

        x3 = [x0]
        for iter in range(self.num_dense_blocks):
            output = self.dense_blocks[str(iter)](x3 + x2[:(iter+1)])
            x3.append(output)

        x3 = torch.cat(x3, dim=1)

        out = self.upsampling(x3)
        out = self.conv3(out)
        out = self.den_softmax(out)
        out = self.recover(out, pop_lr[:, :, 0])
        return {'pop_sr':out, 'feature_map':torch.cat([x0] + x2, dim=1)}
