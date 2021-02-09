from utils.metrics import get_MAE, get_MSE, get_NRMSE, get_RMSE, get_MAPE, get_CORR
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils.dataset as dataset
import torch


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy


string = '{0} Epoch: {1}, RMSE: {2:.4f} NRMSE: {3:.4f} MAE: {4:.4f} MAPE: {5:.4f} CORR: {6:.4f}'
def iterate(loader, epoch):
    return tqdm(loader, ncols=70, ascii=True, postfix='Epoch={0}'.format(epoch))


def batch_extract(batch, input_keys, target_keys=[], cuda_num=None):
    input_keys = [input_keys] if type(input_keys) != list else input_keys 
    target_keys = [target_keys] if type(target_keys) != list else target_keys
    input = []
    target = []
    if cuda_num:
        for key in input_keys: input.append(batch[key].cuda(cuda_num))
        for key in target_keys: target.append(batch[key].cuda(cuda_num))
    else:
        for key in input_keys: input.append(batch[key])
        for key in target_keys: target.append(batch[key])
    return input, target


def train_one_batch(model, batch):
    model.train()
    model.optimizer.zero_grad()
    pop_sr = model(batch)['pop_sr'][:, -1:]
    pop_hr = batch['pop_hr'][:, -1:]
    loss = model.criterion(pop_sr, pop_hr.cuda(model.cuda_num))
    loss.backward()
    model.optimizer.step()


def train_one_epoch(model, train_loader, plot, epoch, param=None):
    model.train()
    plot_func = iterate if plot else lambda x, y:x
    for batch in plot_func(train_loader['loader'], epoch):
        model.optimizer.zero_grad()
        pop_sr = model(batch)['pop_sr'][:, -1:]
        pop_hr = batch['pop_hr'][:, -1:]
        loss = model.criterion(pop_sr, pop_hr.cuda(model.cuda_num))
        loss.backward()
        model.optimizer.step()
        
        
def test_loss(model, train_loader):
    loss = 0
    for batch in train_loader['loader']:
        pop_sr = model(batch)['pop_sr'][:, -1:]
        pop_hr = batch['pop_hr'][:, -1:]
        temp_loss = model.criterion(pop_sr, pop_hr.cuda(model.cuda_num))
        loss += temp_loss
    return loss


def test_data(model, test_loader_list, plot, epoch):
    #print('Epoch = {}'.format(epoch))
    plot_func = iterate if plot else lambda x, y:x
    model.eval()
    sr_list, hr_list, MSE_list, RMSE_list, NRMSE_list, MAE_list, MAPE_list, CORR_list = [], [], [], [], [], [], [], []
    for test_loader in test_loader_list:
        sr_temp_list, hr_temp_list = [], []
        for batch in plot_func(test_loader['loader'], epoch):
            #pop_sr = model(batch).cpu().detach().numpy()[:, -1:]
            #pop_hr = batch['pop_hr'].detach().numpy()[:, -1:]
            output = model(batch)
            if type(output) == dict:
                pop_sr = output['pop_sr'].cpu().detach().numpy()[:, -1:]
            else:
                pop_sr = output.cpu().detach().numpy()[:, -1:]
            pop_hr = batch['pop_hr'].detach().numpy()[:, -1:]
            
            sr_temp_list.append(pop_sr)
            hr_temp_list.append(pop_hr)
            
        sr_temp_list = np.concatenate(sr_temp_list, axis=0)
        hr_temp_list = np.concatenate(hr_temp_list, axis=0)
        
        MSE = get_MSE(sr_temp_list, hr_temp_list)
        RMSE = get_RMSE(sr_temp_list, hr_temp_list)
        NRMSE = get_NRMSE(sr_temp_list, hr_temp_list)
        MAE = get_MAE(sr_temp_list, hr_temp_list)
        MAPE = get_MAPE(sr_temp_list, hr_temp_list)
        CORR = get_CORR(sr_temp_list, hr_temp_list)
        
        sr_list.append(sr_temp_list)
        hr_list.append(hr_temp_list)
        MSE_list.append(MSE)
        RMSE_list.append(RMSE)
        NRMSE_list.append(NRMSE)
        MAE_list.append(MAE)
        MAPE_list.append(MAPE)
        CORR_list.append(CORR)
    for iter, test_loader in enumerate(test_loader_list):
        print(string.format(
            test_loader['name'], epoch, 
            RMSE_list[iter], NRMSE_list[iter], MAE_list[iter], MAPE_list[iter], CORR_list[iter]))


def augment_train(path, base_dataset, sr_list, frame, time_center_list, augment_frame, batch_size, cuda_num, num_iters=50):
    model = torch.load(path, map_location='cuda:'+str(cuda_num))
    model.cuda_num = cuda_num
    
    time_list = np.concatenate([np.arange(each - augment_frame, each + augment_frame + 1) for each in time_center_list])
    temp_sr_list = sr_list[:, (sr_list.shape[1] // 2 - augment_frame):(sr_list.shape[1] // 2 + augment_frame + 1)]
    temp_sr_list = np.concatenate([temp_sr_list[iter, :, np.newaxis] for iter in range(temp_sr_list.shape[0])], axis=0)
    
    trans_trainset_ag = dataset.AugmentedDataset(base_dataset, time_list, frame, temp_sr_list)
    trans_train_loader_ag = DataLoader(trans_trainset_ag, num_workers=5, batch_size=batch_size, shuffle=True)
    
    iters_count = 0
    
    while True:
        model.train()
        for batch in trans_train_loader_ag:
            train_one_batch(model, batch)
            iters_count += 1
            if iters_count == num_iters:
                break
        if iters_count == num_iters:
            break
    
    print(path)
    return model


def GPU_max_free_memory():
    import pynvml  
    pynvml.nvmlInit()
    free_list = []
    for iter in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(iter)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_list.append(meminfo.free)
    return np.argmax(free_list)