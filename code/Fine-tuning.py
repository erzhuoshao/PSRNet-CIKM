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

class args:
    data_path = '../data'
    cuda_num = Engine.GPU_max_free_memory()
    frame = 48
    finetune_epochs = 20
    finetune_interval = 2
    finetune_time = [48*2]
    batch_size = 4
    interval = 10
    seed = 123

Engine.set_random_seed(args.seed)

sourse_city_list = np.arange(48, 1440)
np.random.shuffle(sourse_city_list)
train_list = sourse_city_list[:int((1440 - 48)*0.7)]
valid_list = sourse_city_list[int((1440 - 48)*0.7):int((1440 - 48)*0.85)]


load = lambda dataset, target_key, batch, name:{
    'loader':DataLoader(dataset, num_workers=5, batch_size=batch, shuffle=True),
    'target_key':target_key,
    'name':name}

def gen_dataset(
    source_city_name, source_lr_downscale, source_hr_downscale,
    target_city_name, target_lr_downscale, target_hr_downscale,
    augment_frame, center_time_slot=48*2, alpha=1e-3):
    source_dataset, source_validset = dataset.al_dataset_gen(
        args.data_path, source_city_name, source_hr_downscale, source_lr_downscale,
        0, 47, [train_list, valid_list])
    trans_lr_set, trans_testset = dataset.al_dataset_gen(
        args.data_path, target_city_name, target_hr_downscale, target_lr_downscale,
        0, 47, ['all', list(range(center_time_slot+1, center_time_slot+48*7))])

    source_loader = load(source_dataset, 'pop_hr', args.batch_size, '{}_train'.format(source_city_name))
    source_valid_loader = load(source_validset, 'pop_hr', args.batch_size, '{}_train'.format(source_city_name))

    trans_lr_loader = load(trans_lr_set, 'pop_hr', args.batch_size, '{}_train'.format(target_city_name))
    trans_test_loader = load(trans_testset, 'pop_hr', args.batch_size, '{}_valid'.format(target_city_name))

    trans_pop_sr_dict = np.load(
        '../pop_sr/source=[\'{0}\']-sres={1}->{2}-target={3}-tres={4}->{5}-alpha={6}-af={7}.npz'.format(
        source_city_name, source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        alpha, augment_frame))['pop_sr'].transpose([1,0,2,3])
    return source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr_dict

def domain_adaptation(
    gen_path,
    dis_module,
    base_channel,
    augment_frame,
    max_augment_frame,
    city_name,
    source_loader,
    source_valid_loader,
    trans_lr_loader,
    trans_testset,
    trans_test_loader,
    trans_pop_sr, num_iter=50, input_channel=192):
    '''
    gen_path : The path of the checkpoint of generator(STNet).
    dis_module : The module used to distinguish sourse and target cities' feature map. (Regular or Pixel level)
    augment_frame : The number of augmented frames. Input 2 * augment_frame + 1 frames.
    source_loader : The loader used to generate lr image in source city.
    trans_lr_loader : The loader used to generate lr image in target city.
    trans_testset : The loader used to generate testing samples in target city.
    trans_pop_sr : The augmented sr images in target city
    '''
    used_sr_pop = trans_pop_sr[(trans_pop_sr.shape[0] // 2 - augment_frame):(trans_pop_sr.shape[0] // 2 + augment_frame + 1)]
    trans_hr_dataset = dataset.AugmentedDataset(trans_testset, np.arange(48 * 2 - augment_frame, 48 * 2 + augment_frame + 1), args.frame, used_sr_pop)
    trans_hr_loader = load(trans_hr_dataset, 'pop_hr', args.batch_size, '{}_train'.format(city_name))

    G = torch.load(gen_path, map_location='cuda:'+str(args.cuda_num))
    G.cuda_num = args.cuda_num
    D = dis_module({'cuda_num':args.cuda_num, 'input_channel':input_channel, 'base_channel':base_channel}).cuda(args.cuda_num)
    lr = 1e-4
    criterion_1 = nn.MSELoss().cuda(args.cuda_num)
    criterion_2 = nn.MSELoss().cuda(args.cuda_num)
    D.optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    mse_list = []
    min_string = ''
    G.train()

    for iter in tqdm.tqdm(range(50)):
        for batch_1 in source_loader['loader']:
            output_1 = G(batch_1)
            break
        for batch_2 in trans_lr_loader['loader']:
            output_2 = G(batch_2)
            break
        for batch_3 in trans_hr_loader['loader']:
            output_3 = G(batch_3)
            break

        c_pred, c_targ = D(output_1['feature_map'], output_2['feature_map'])
        loss_D_1 = criterion_1(c_pred, c_targ)
        D.optimizer.zero_grad()
        loss_D_1.backward(retain_graph=True)
        D.optimizer.step()

        c_pred, c_targ = D(output_1['feature_map'], output_3['feature_map'])
        loss_D_2 = criterion_1(c_pred, c_targ * 0 + 0.5)
        loss_G_3 = criterion_2(output_3['pop_sr'][:, -1:], batch_3['pop_hr'][:, -1:].cuda(args.cuda_num))
        G.optimizer.zero_grad()
        (loss_D_2 + loss_G_3).backward(retain_graph=True)
        G.optimizer.step()

    Engine.test_data(G, [trans_test_loader], True, 0)

def finetune(
    model_path,
    augment_frame,
    max_augment_frame,
    city_name,
    trans_testset,
    trans_test_loader,
    trans_pop_sr,
    center_time_slot=48*2,
    max_iters=50):

    used_sr_pop = trans_pop_sr[(trans_pop_sr.shape[0] // 2 - augment_frame):(trans_pop_sr.shape[0] // 2 + augment_frame + 1)]
    trans_hr_dataset = dataset.AugmentedDataset(trans_testset, np.arange(center_time_slot-augment_frame, center_time_slot+augment_frame+1), args.frame, used_sr_pop)
    trans_hr_loader = load(trans_hr_dataset, 'pop_hr', args.batch_size, '{}_train'.format(city_name))

    G = torch.load(model_path, map_location='cuda:'+str(args.cuda_num))
    G.assign_cuda(args.cuda_num)

    lr = 1e-4
    criterion_2 = nn.MSELoss().cuda(args.cuda_num)
    mse_list = []
    min_string = ''
    G.train()

    iters_count = 0
    max_iters = 50
    while True:
        for batch_3 in trans_hr_loader['loader']:
            output_3 = G(batch_3)
            loss_G_3 = criterion_2(output_3['pop_sr'][:, -1:], batch_3['pop_hr'][:, -1:].cuda(args.cuda_num))
            G.optimizer.zero_grad()
            loss_G_3.backward(retain_graph=True)
            G.optimizer.step()
            iters_count += 1
            if iters_count == max_iters:
                break
        if iters_count == max_iters:
            break

    Engine.test_data(G, [trans_test_loader], True, 0)


print('\n STNet+PGNet+PADA CITY1 -> CITY2 (X2) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 1, 0
max_augment_frame = 4
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 1):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)


print('\n STNet+PGNet+PADA CITY1 -> CITY3 (X2) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 1, 0
max_augment_frame = 4
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 1):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)


print('\n STNet+PGNet+PADA CITY1 -> CITY2 (X4) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 2, 0
max_augment_frame = 12
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 2):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)


print('\n STNet+PGNet+PADA CITY1 -> CITY3 (X4) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 2, 0
max_augment_frame = 12
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 2):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)
        
        
print('\n STNet+PGNet+PADA CITY1 -> CITY2 (X4) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 3, 0
max_augment_frame = 12
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 2):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)


print('\n STNet+PGNet+PADA CITY1 -> CITY3 (X4) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 3, 0
max_augment_frame = 12
for base_channel in [1,2,4,8]:
    for augment_frame in range(0, max_augment_frame, 2):
        Engine.set_random_seed(args.seed)
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48*2, alpha=1e-3)
        domain_adaptation(
            '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            PGNet.PADADiscriminator,
            base_channel,
            augment_frame, max_augment_frame,
            target_city_name,
            source_loader,
            source_valid_loader,
            trans_lr_loader,
            trans_testset,
            trans_test_loader,
            trans_pop_sr)


print('\n STNet+PGNet CITY1 -> CITY2 (2X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 1, 0
max_augment_frame = 4
for augment_frame in range(0, max_augment_frame, 1):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet CITY1 -> CITY3 (2X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 1, 0
max_augment_frame = 4
for augment_frame in range(0, max_augment_frame, 1):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet CITY1 -> CITY2 (4X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 2, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet CITY1 -> CITY3 (4X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 2, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)
        
        
print('\n STNet+PGNet CITY1 -> CITY2 (8X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 3, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet CITY1 -> CITY3 (8X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY1', 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 3, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n Meta-STNet [CITY1, CITY3, CITY4] -> CITY2 (2X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY3', 'CITY4'], 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 1, 0
max_augment_frame = 4
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n Meta-STNet [CITY1, CITY2, CITY4] -> CITY3 (2X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY2', 'CITY4'], 1, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 1, 0
max_augment_frame = 4
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n Meta-STNet [CITY1, CITY3, CITY4] -> CITY2 (4X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY3', 'CITY4'], 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 2, 0
max_augment_frame = 12
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n Meta-STNet [CITY1, CITY2, CITY4] -> CITY3 (4X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY2', 'CITY4'], 2, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 2, 0
max_augment_frame = 12
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=4-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)
    
    
print('\n Meta-STNet [CITY1, CITY3, CITY4] -> CITY2 (8X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY3', 'CITY4'], 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 3, 0
max_augment_frame = 12
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n Meta-STNet [CITY1, CITY2, CITY4] -> CITY3 (8X) \n')

source_city_name, source_lr_downscale, source_hr_downscale = ['CITY1', 'CITY2', 'CITY4'], 3, 0
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 3, 0
max_augment_frame = 12
for augment_frame in [0]:
    Engine.set_random_seed(args.seed)
    print('af={0} alpha={1}'.format(augment_frame, alpha))
    source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
        source_city_name[0], source_lr_downscale, source_hr_downscale,
        target_city_name, target_lr_downscale, target_hr_downscale,
        max_augment_frame, center_time_slot=48 * 2, alpha=1e-3)
    finetune(
        '../checkpoint/STNet-ic=48-uf=8-bc=64-tc=16-ts=6-res={1}->{2}-source={0}-seed={3}.pkl'.format(
            source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
        augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet+PADA CITY2 (Cross Granularity) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY2', 2, 1
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 1, 0
max_augment_frame = 12
for augment_frame in [12]:
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        for base_channel in [1,2,4,8]:
            Engine.set_random_seed(args.seed)
            print('af={0} alpha={1} channel={2}'.format(augment_frame, alpha, base_channel))
            source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
                source_city_name, source_lr_downscale, source_hr_downscale,
                target_city_name, target_lr_downscale, target_hr_downscale,
                max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
            domain_adaptation(
                '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                    source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
                PGNet.PADADiscriminator,
                base_channel,
                augment_frame, max_augment_frame,
                target_city_name,
                source_loader,
                source_valid_loader,
                trans_lr_loader,
                trans_testset,
                trans_test_loader,
                trans_pop_sr)


print('\n STNet+PGNet+PADA CITY3 (Cross Granularity) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY3', 2, 1
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 1, 0
max_augment_frame = 12
for augment_frame in [12]:
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        for base_channel in [1,2,4,8]:
            Engine.set_random_seed(args.seed)
            print('af={0} alpha={1} channel={2}'.format(augment_frame, alpha, base_channel))
            source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
                source_city_name, source_lr_downscale, source_hr_downscale,
                target_city_name, target_lr_downscale, target_hr_downscale,
                max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
            domain_adaptation(
                '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                    source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
                PGNet.PADADiscriminator,
                base_channel,
                augment_frame, max_augment_frame,
                target_city_name,
                source_loader,
                source_valid_loader,
                trans_lr_loader,
                trans_testset,
                trans_test_loader,
                trans_pop_sr)


print('\n STNet+PGNet CITY2 (Cross Granularity) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY2', 2, 1
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY2', 1, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)


print('\n STNet+PGNet CITY3 (Cross Granularity) \n')

source_city_name, source_lr_downscale, source_hr_downscale = 'CITY3', 2, 1
target_city_name, target_lr_downscale, target_hr_downscale = 'CITY3', 1, 0
max_augment_frame = 12
for augment_frame in range(0, max_augment_frame, 2):
    for alpha in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        Engine.set_random_seed(args.seed)
        print('af={0} alpha={1}'.format(augment_frame, alpha))
        source_loader, source_valid_loader, trans_lr_loader, trans_testset, trans_test_loader, trans_pop_sr = gen_dataset(
            source_city_name, source_lr_downscale, source_hr_downscale,
            target_city_name, target_lr_downscale, target_hr_downscale,
            max_augment_frame, center_time_slot=48 * 2, alpha=alpha)
        finetune(
            '../checkpoint/STNet-ic=48-uf=2-bc=64-tc=16-ts=6-res={1}->{2}-source=[\'{0}\']-seed={3}.pkl'.format(
                source_city_name, source_lr_downscale, source_hr_downscale, args.seed),
            augment_frame, max_augment_frame, target_city_name, trans_testset, trans_test_loader, trans_pop_sr)
