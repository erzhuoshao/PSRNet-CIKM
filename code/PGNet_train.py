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
import model.PGNet as PGNet

##### Global Config #####
parser = argparse.ArgumentParser()
#parser.add_argument('-test', action="store_true", default=True)
parser.add_argument('-data_path', dest='data_path', default='../data', required=False)
parser.add_argument('-source', dest='source', default=['CITY1'], required=False)
parser.add_argument('-source_lr_downscale', dest='source_lr_downscale', default=1, type=int, required=False)
parser.add_argument('-source_hr_downscale', dest='source_hr_downscale', default=0, type=int, required=False)
parser.add_argument('-target', dest='target', default=['CITY2, CITY3'], required=False)
parser.add_argument('-target_lr_downscale', dest='target_lr_downscale', default=1, type=int, required=False)
parser.add_argument('-target_hr_downscale', dest='target_hr_downscale', default=0, type=int, required=False)
parser.add_argument('-input_channel', dest='input_channel', default=48, type=int, required=False)
parser.add_argument('-batch_size', dest='batch_size', default=16, type=int, required=False)
parser.add_argument('-poi_channel', dest='poi_channel', default=14, type=int, required=False)
parser.add_argument('-interval', dest='interval', default=10, type=int, required=False)
parser.add_argument('-seed', dest='seed', default=123, type=int, required=False)
parser.add_argument('-f', dest='f', default=14, required=False)
##### Fine-tune Config #####
parser.add_argument('-finetune_time', dest='finetune_time', default=[48 * 2], required=False)
##### Model Config #####
parser.add_argument('-cuda_num', dest='cuda_num', default=Engine.GPU_max_free_memory(), required=False)
parser.add_argument('-epoch_num', dest='epoch_num', default=100, type=int, required=False)
parser.add_argument('-base_channel', dest='base_channel', default=64, type=int, required=False)
parser.add_argument('-time_channel', dest='time_channel', default=16, type=int, required=False)
parser.add_argument('-time_stride', dest='time_stride', default=6, type=int, required=False)
args = parser.parse_args()
print(args)
args.source = eval(args.source)
args.target = eval(args.target)

if (args.source_hr_downscale == 0) & (args.source_lr_downscale == 1):
    args.augment_frame = 4
    args.generator_channel = 64
elif (args.source_hr_downscale == 0) & (args.source_lr_downscale == 2):
    args.augment_frame = 12
    args.generator_channel = 16
elif (args.source_hr_downscale == 1) & (args.source_lr_downscale == 2):
    args.augment_frame = 12
    args.generator_channel = 32
else:
    print('Undefined Granularities')
    
args.aug_total_frame = args.augment_frame * 2 + 1
args.upscale_factor = 2 ** (args.source_lr_downscale - args.source_hr_downscale)

args.f = None
Engine.set_random_seed(args.seed)
print(args)


load = lambda dataset, target_key, batch, name:{
    'loader':DataLoader(dataset, num_workers=5, batch_size=batch, shuffle=True),
    'target_key':target_key,
    'name':name}

trainset, validset, testset = dataset.dataset_gen(
    args.data_path, args.source[0], args.source_hr_downscale, args.source_lr_downscale,
    args.augment_frame, args.augment_frame, train_time_list=list(range(48*2, 48*26, 48)), valid_time_list=args.finetune_time,
    use_poi=True)
train_loader = load(trainset, 'pop_hr', args.batch_size, '{}_train'.format(args.source[0]))
valid_loader = load(validset, 'pop_hr', args.batch_size, '{}_valid'.format(args.source[0]))
test_loader = load(testset, 'pop_hr', args.batch_size, '{}_valid'.format(args.source[0]))

trans_train_loader_list, trans_valid_loader_list, trans_test_loader_list = [],[],[]
for iter,name in enumerate(args.target):
    temp_trainset, temp_validset, temp_testset = dataset.dataset_gen(
        args.data_path, args.target[iter], args.target_hr_downscale, args.target_lr_downscale,
        args.augment_frame, args.augment_frame, train_time_list=args.finetune_time, valid_time_list=args.finetune_time,
        use_poi=True)

    trans_train_loader_list.append(load(temp_trainset, 'pop_hr', args.batch_size, '{}_train'.format(args.target[iter])))
    trans_valid_loader_list.append(load(temp_validset, 'pop_hr', args.batch_size, '{}_train'.format(args.target[iter])))
    trans_test_loader_list.append(load(temp_testset, 'pop_hr', args.batch_size, '{}_train'.format(args.target[iter])))


for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
    model_name = '../checkpoint/STNet-ic=48-uf={2}-bc=64-tc=16-ts=6-res={0}->{1}-source-{3}.pkl'.format(
        args.source_lr_downscale, args.source_hr_downscale, args.upscale_factor, args.source)
    G = PGNet.PGNetGenerator({'cuda_num':args.cuda_num, 'base_channel':args.generator_channel, 'input_channel':args.aug_total_frame, 'upscale_factor':args.upscale_factor}).cuda(args.cuda_num)
    D = PGNet.PGNetDiscriminator({'cuda_num':args.cuda_num, 'input_channel':args.aug_total_frame}).cuda(args.cuda_num)

    print(args.__dict__)
    G_name = '../checkpoint/PGNetGenerator'
    print(model_name)

    criterion_1 = nn.MSELoss().cuda(args.cuda_num)
    criterion_2 = nn.MSELoss().cuda(args.cuda_num)
    lr = 1e-4
    G.optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    D.optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    mse_list = []
    min_string = ''

    interval = args.epoch_num
    for epoch in range(args.epoch_num):
    #for epoch in range(1):
        print('Epoch = {}\t'.format(epoch), end='')
        G.train()
        D.train()
        mse_batch_list = []
        for batch in train_loader['loader']:
            pop_hr = batch['pop_hr'].cuda(args.cuda_num)
            pop_sr = G(batch)
            pred, target = D(pop_hr, pop_sr)

            G.optimizer.zero_grad()
            loss_G_1 = criterion_1(pred, target*0+0.5)
            loss_G_2 = criterion_2(pop_sr, pop_hr)
            loss_G = loss_G_1 + alpha * loss_G_2
            loss_G.backward(retain_graph=True)
            G.optimizer.step()

            pred, target = D(pop_hr, pop_sr.detach())
            D.optimizer.zero_grad()
            loss_D = criterion_1(pred, target)
            loss_D.backward()
            D.optimizer.step()
            loss_G_1 = np.round(loss_G_1.cpu().detach().numpy(), 4)
            loss_G_2 = np.round(loss_G_2.cpu().detach().numpy(), 4)
            mse_batch_list.append(loss_G_2)

        loss_G_2 = np.mean(loss_G_2)
        print(loss_G_2)
        mse_list.append(loss_G_2)
        if np.mean(loss_G_2) == np.min(mse_list):
            torch.save(G, os.path.join(G_name, 'Generator-epoch-{}.pkl'.format(epoch)))

        if epoch % 5 == 0 and epoch != 0:
            lr /= 2
            G.optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            D.optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            print('Optimizer Updated.')

    print('Load ' + 'Generator-epoch-{}.pkl'.format(np.argmin(mse_list)))
    G = torch.load(os.path.join(G_name, 'Generator-epoch-{}.pkl'.format(np.argmin(mse_list))), map_location='cuda:'+str(args.cuda_num))

    sr_list = []
    hr_list = []

    for loader in trans_train_loader_list:
        sr_list.append([])
        hr_list.append([])
        for batch in loader['loader']:
            sr_list[-1].append(G(batch).detach().cpu().numpy())
            hr_list[-1].append(batch['pop_hr'].numpy())
            break

    for iter in range(len(args.target)):
        sr_list[iter] = np.concatenate(sr_list[iter], 0)
        hr_list[iter] = np.concatenate(hr_list[iter], 0)

    NRMSE_list = []
    for iter, target in enumerate(args.target):
        NRMSE_list.append([])
        for iter2 in range(args.aug_total_frame):
            NRMSE_list[iter].append(metrics.get_NRMSE(sr_list[iter][:, iter2], hr_list[iter][:, iter2]))

        np.savez('../pop_sr/source={0}-sres={1}->{2}-target={3}-tres={4}->{5}-alpha={6}-af={7}.npz'.format(
            args.source, args.source_lr_downscale, args.source_hr_downscale, \
            target, args.target_lr_downscale, args.target_hr_downscale, \
            alpha, args.augment_frame
        ), pop_sr=sr_list[iter], pop_hr=hr_list[iter])
        print(NRMSE_list[iter])
