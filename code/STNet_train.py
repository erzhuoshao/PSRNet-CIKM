import importlib, setproctitle, os, shutil, argparse, sys, tqdm, time
import PIL.Image as Image
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import utils.metrics as metrics
import utils.dataset as dataset
from torch.utils.data import DataLoader, Dataset
import Engine
from model.STNet import STNet

##### Global Config #####
parser = argparse.ArgumentParser()
#parser.add_argument('-test', action="store_true", default=True)
parser.add_argument('-data_path', dest='data_path', default='../data', required=False)
parser.add_argument('-source', dest='source', default=['CITY1'], required=False)
parser.add_argument('-target', dest='target', default=['CITY2'], required=False)
parser.add_argument('-input_channel', dest='input_channel', default=48, type=int, required=False)
parser.add_argument('-hr_downscale', dest='hr_downscale', default=0, type=int, required=False)
parser.add_argument('-lr_downscale', dest='lr_downscale', default=1, type=int, required=False)
parser.add_argument('-batch_size', dest='batch_size', default=16, type=int, required=False)
parser.add_argument('-interval', dest='interval', default=50, type=int, required=False)
parser.add_argument('-seed', dest='seed', default=123, type=int, required=False)
parser.add_argument('-poi_channel', dest='poi_channel', default=14, type=int, required=False)
parser.add_argument('-f', dest='f', default=14, required=False)
##### Model Config #####
parser.add_argument('-cuda_num', dest='cuda_num', default=Engine.GPU_max_free_memory(), required=False)
parser.add_argument('-epoch_num', dest='epoch_num', default=1000, type=int, required=False)
parser.add_argument('-base_channel', dest='base_channel', default=64, type=int, required=False)
parser.add_argument('-time_channel', dest='time_channel', default=16, type=int, required=False)
parser.add_argument('-time_stride', dest='time_stride', default=6, type=int, required=False)
args = parser.parse_args()
args.source = eval(args.source)
args.target = eval(args.target)

args.upscale_factor = 2 ** (args.lr_downscale - args.hr_downscale)
args.f = None
Engine.set_random_seed(args.seed)
print(args)

model = STNet(args).cuda(args.cuda_num)
model_name = '../checkpoint/{0}.pkl'.format(model.name)
print(model_name)

load = lambda dataset, target_key, pop_max, batch, name:{
    'loader':DataLoader(dataset, num_workers=5, batch_size=batch, shuffle=True),
    'target_key':target_key,
    'pop_max':pop_max,
    'name':name}

train_loader_list, valid_loader_list = [], []
for train_name in args.source:
    trainset, validset, testset = dataset.dataset_gen(
        args.data_path, train_name, args.hr_downscale, args.lr_downscale, 0, args.input_channel-1, train_time_list=None, valid_time_list=None)
    train_loader_list.append(load(trainset, 'pop_hr', 1, args.batch_size, 'train:{}'.format(train_name)))
    valid_loader_list.append(load(validset, 'pop_hr', 1, args.batch_size, 'valid:{}'.format(train_name)))

trans_loader_list = []
for trans_name in args.target:
    print('Loading : ' + trans_name)
    _, _, trans1_set = dataset.dataset_gen(
    args.data_path, trans_name, args.hr_downscale, args.lr_downscale, 0, args.input_channel-1, train_time_list=None, valid_time_list=None)
    trans_loader_list.append(load(trans1_set, 'pop_hr', 1, args.batch_size, 'trans:{}'.format(trans_name)))

model.criterion = nn.MSELoss().cuda(args.cuda_num)
model.optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
interval = args.interval

valid_list = [10000]
min_string = ''

for epoch in tqdm.tqdm(range(args.epoch_num), ncols=70, ascii=True):
    Engine.train_one_epoch(model, train_loader_list[0], False, epoch)
    if epoch % interval == interval-1:
        Engine.test_data(model, [*valid_loader_list, *trans_loader_list], True, epoch)
torch.save(model.cpu(), model_name)
