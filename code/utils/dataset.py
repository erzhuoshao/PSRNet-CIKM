import torch, os, json, time, copy, math
import numpy as np
import PIL.Image as Image
from os.path import exists, join
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def downscale(img, resample):
    B, C, H, W = img.shape
    
    if resample == 'add':
        img2 = img.reshape([B, C, H//2, 2, W//2, 2]).sum(3).sum(4)
    else:
        size2 = [int(img.shape[2]/2), int(img.shape[3]/2)]
        for iter in range(img.shape[0]):
            for channel in range(img.shape[1]):
                temp = Image.fromarray(img[iter, channel].T).resize(size2, resample=Image.BICUBIC)
                img2[iter, channel] = np.asarray(temp).T
    return img2


def upscale(img, resample):
    from scipy import interpolate
    size2 = [img.shape[2] * 2, img.shape[3] * 2]
    img2 = np.zeros([img.shape[0], img.shape[1]] + size2)
    for iter in range(img.shape[0]):
        for channel in range(img.shape[1]):
            temp = Image.fromarray(img[iter, channel].T).resize(size2, resample=resample)
            img2[iter, channel] = np.asarray(temp).T
    return img2


def augment(img, poi, augment_config):
    img2 = []
    poi2 = []
    for t in range(img.shape[0]):
        if augment_config['use_patch']:
            patch_size = augment_config['patch_size']
            stride = augment_config['stride']
            for x in range(0, stride, img.shape[2]-patch_size):
                for y in range(0, stride, img.shape[3]-patch_size):
                    temp_img = img[t:(t+1), :, x:(x+patch_size), y:(y+patch_size)]
                    temp_poi = poi[:, :, x:(x+patch_size), y:(y+patch_size)]
                    for func in augment_config['augment_function']:
                        img2.append(func(temp_img))
                        poi2.append(func(temp_poi))
        else:
            patch_size = min(img.shape[2], img.shape[3])
            temp_img = img[t:(t+1), :, :patch_size, :patch_size]
            temp_poi = poi[:, :, :patch_size, :patch_size]
            for func in augment_config['augment_function']:
                img2.append(func(temp_img))
                poi2.append(func(temp_poi))
    img2 = np.concatenate(img2, axis=0)
    poi2 = np.concatenate(poi2, axis=0)
    return img2, poi2


class ImageFromMatrix(Dataset):
    def __init__(self, config):
        super(ImageFromMatrix, self).__init__()
        
        self.config = config
        self.time_list = config['time_list']
        self.augment_func = config['augment_func']
        self.downscale = 'add'
        self.frame_f = config['frame_f']
        self.frame_b = config['frame_b']
            
        self.pop_hr = config['pop_hr'].astype(float)
        self.pop_lr = config['pop_lr'].astype(float)
        self.iter = 0
        self.pop_sr = config['pop_sr'].astype(float)
        
        if self.config['use_poi']:
            self.poi_hr = config['poi_hr'].astype(float)
            self.poi_lr = config['poi_lr'].astype(float)
        #    self.poi_sr = config['poi_sr'].astype(float)
        if self.config['use_flow']:
            self.flow_hr = config['flow_hr'].astype(float)
            self.flow_lr = config['flow_lr'].astype(float)
            self.flow_sr = config['flow_sr'].astype(float)

    def __getitem__(self, index):
        key = self.time_list[index]
        return_dict = {}
        #return_dict['pop_hr'] = torch.Tensor(self.pop_hr[(key-self.frame_b):(key+self.frame_f+1), 0])
        return_dict['pop_hr'] = torch.Tensor(self.pop_hr[(key-self.frame_b):(key+self.frame_f+1), 0])
        return_dict['pop_lr'] = torch.Tensor(self.pop_lr[(key-self.frame_b):(key+self.frame_f+1), 0])
        return_dict['pop_sr'] = torch.Tensor(self.pop_sr[(key-self.frame_b):(key+self.frame_f+1), 0])
        return_dict['t'] = torch.LongTensor([key % 48])
        return_dict['pop_mean'] = torch.Tensor(self.pop_hr[:48]).mean([1,2,3])
        if self.config['use_poi']:
            return_dict['poi_hr'] = torch.Tensor(self.poi_hr[0])
            return_dict['poi_lr'] = torch.Tensor(self.poi_lr[0])
        #    return_dict['poi_sr'] = torch.Tensor(self.poi_sr[0])
        if self.config['use_flow']:
            return_dict['flow_hr'] = torch.Tensor(self.flow_hr[(key-self.frame_b):(key+self.frame_f+1), 0])
            return_dict['flow_lr'] = torch.Tensor(self.flow_lr[(key-self.frame_b):(key+self.frame_f+1), 0])
            return_dict['flow_sr'] = torch.Tensor(self.flow_sr[(key-self.frame_b):(key+self.frame_f+1), 0])
            return_dict['pop_hr_1f'] = torch.Tensor(self.pop_hr[key:(key+1), 0])
        self.iter += 1
        if self.augment_func != None: return_dict = self.augment_func(return_dict, self.iter)
        return return_dict

    def __len__(self):
        return len(self.time_list)
    
    
class AugmentedDataset(Dataset):
    def __init__(self, dataset, time_list, frame, augmented_pop_hr):
        super(AugmentedDataset, self).__init__()
        self.time_list = time_list
        self.frame = frame
        self.pop_hr = augmented_pop_hr # D, T, H, W
        self.pop_lr = dataset.pop_lr
        self.pop_sr = dataset.pop_sr

    def __getitem__(self, index):
        key = self.time_list[index]
        pop_hr_tensor = torch.Tensor(self.pop_hr[(index):(index+1), 0])
        pop_lr_tensor = torch.Tensor(self.pop_lr[(key-self.frame+1):(key+1), 0])
        pop_sr_tensor = torch.Tensor(self.pop_sr[(key-self.frame+1):(key+1), 0])
        return_dict = {'pop_sr':pop_sr_tensor, 'pop_hr':pop_hr_tensor, 'pop_lr':pop_lr_tensor}
        return return_dict

    def __len__(self):
        return len(self.time_list)



def dataset_gen(data_path, city_name, hr_downscale, lr_downscale, frame_f, frame_b, train_time_list, valid_time_list, use_poi=False, use_flow=False):
    pop_init = np.load(os.path.join(data_path, city_name, 'pop.npy'))
    pop_init = pop_init[pop_init.sum(-1).sum(-1).sum(-1) != 0]
    _, _, H, W = pop_init.shape
    
    pop_hr = pop_init = pop_init[:, :, :H//8*8, :W//8*8]
    
    for iter in range(int(hr_downscale)):
        pop_hr = downscale(pop_hr, resample='add').astype(float)
    pop_lr = pop_hr
    for iter in range(int(lr_downscale - hr_downscale)):
        pop_lr = downscale(pop_lr, resample='add').astype(float)
    pop_sr = pop_lr
    for iter in range(int(lr_downscale - hr_downscale)):
        pop_sr = upscale(pop_sr, resample=Image.BICUBIC).astype(float) / 4
    
    if use_poi:
        poi_init = np.load(os.path.join(data_path, city_name, 'poi.npy'))
        poi_hr = poi_init = poi_init[:, :, :H//8*8, :W//8*8]
        for cate in range(poi_hr.shape[0]):
            poi_hr[:, cate, :, :] /= poi_hr[:, cate, :, :].mean()
            poi_hr[:, cate, :, :] *= pop_hr[0].mean()
        
        for iter in range(int(hr_downscale)):
            poi_hr = downscale(poi_hr, resample='add').astype(float)
        poi_lr = poi_hr
        for iter in range(int(lr_downscale - hr_downscale)):
            poi_lr = downscale(poi_lr, resample='add').astype(float)
        poi_sr = poi_lr
        for iter in range(int(lr_downscale - hr_downscale)):
            poi_sr = upscale(poi_sr, resample=Image.BICUBIC).astype(float) / 4
    else:
        poi_hr, poi_lr, poi_sr = None, None, None
            
    if use_flow:
        flow_hr = copy.deepcopy(pop_hr)
        flow_hr[:-1] = flow_hr[1:] - flow_hr[:-1]
        flow_hr[-1] = 0
    
        temp = flow_hr
        for iter in range(int(math.log(upscale_factor, 2))):
            temp = downscale(temp, resample='add').astype(float)
        flow_lr = temp
        for iter in range(int(math.log(upscale_factor, 2))):
            temp = upscale(temp, resample=Image.BICUBIC).astype(float) / 4
        flow_sr = temp
    else:
        flow_hr, flow_lr, flow_sr = None, None, None
    
    print('Generation time list')
    time_list = np.arange(pop_hr.shape[0])
    time_list = time_list[48:-48]
    
    print('{0} : Total Time Slice = {1}'.format(city_name, len(time_list)))
    train_valid_time_list_temp, test_time_list_temp = train_test_split(time_list, test_size=0.15, shuffle=True, random_state=0)
    train_time_list_temp, valid_time_list_temp = train_test_split(train_valid_time_list_temp, test_size=0.15/0.85, shuffle=True, random_state=0)
    
    test_time_list = test_time_list_temp
    if train_time_list is None:
        train_time_list = train_time_list_temp
    if valid_time_list is None:
        valid_time_list = valid_time_list_temp
    train_time_list, valid_time_list, test_time_list = list(train_time_list), list(valid_time_list), list(test_time_list)
    #for iter in train_time_list:
    #    if iter in valid_time_list: valid_time_list.remove(iter)
        
    print('Train set = {}'.format(len(train_time_list)))
    print('Valid set = {}'.format(len(valid_time_list)))
    print('Test set = {}'.format(len(test_time_list)))
    
    train_dataset = ImageFromMatrix({
        'pop_hr':pop_hr, 'pop_lr':pop_lr, 'pop_sr':pop_sr, 
        'use_poi':use_poi, 'poi_hr':poi_hr, 'poi_lr':poi_lr, 'poi_sr':poi_sr, 
        'use_flow':use_flow, 'flow_hr':flow_hr, 'flow_lr':flow_lr, 'flow_sr':flow_sr,
        'frame_f':frame_f, 'frame_b':frame_b,
        'time_list':train_time_list,'augment_func':None})
    valid_dataset = ImageFromMatrix({
        'pop_hr':pop_hr, 'pop_lr':pop_lr, 'pop_sr':pop_sr, 
        'use_poi':use_poi, 'poi_hr':poi_hr, 'poi_lr':poi_lr, 'poi_sr':poi_sr, 
        'use_flow':use_flow, 'flow_hr':flow_hr, 'flow_lr':flow_lr, 'flow_sr':flow_sr,
        'frame_f':frame_f, 'frame_b':frame_b,
        'time_list':valid_time_list,'augment_func':None})
    test_dataset = ImageFromMatrix({
        'pop_hr':pop_hr, 'pop_lr':pop_lr, 'pop_sr':pop_sr, 
        'use_poi':use_poi, 'poi_hr':poi_hr, 'poi_lr':poi_lr, 'poi_sr':poi_sr, 
        'use_flow':use_flow, 'flow_hr':flow_hr, 'flow_lr':flow_lr, 'flow_sr':flow_sr,
        'frame_f':frame_f, 'frame_b':frame_b,
        'time_list':test_time_list,'augment_func':None})
    return train_dataset, valid_dataset, test_dataset

def time_aggregate(pop_hr, time):
    return pop_hr.reshape([-1, 48, 1, pop_hr.shape[2], pop_hr.shape[3]]).sum(0)

def al_dataset_gen(data_path, city_name, hr_downscale, lr_downscale, frame_f, frame_b, time_list_list):
    pop_hr = np.load(os.path.join(data_path, city_name, 'pop.npy'))
    poi_hr = np.load(os.path.join(data_path, city_name, 'poi.npy'))
    
    pop_hr = pop_hr[pop_hr.sum(-1).sum(-1).sum(-1) != 0]
    _, _, H, W = pop_hr.shape
    
    pop_hr = pop_hr[:, :, :H//8*8, :W//8*8]
    poi_hr = poi_hr[:, :, :H//8*8, :W//8*8]

    #print('Generating low resolution and super resolution POP and POI.')
    for iter in range(int(hr_downscale)):
        pop_hr = downscale(pop_hr, resample='add').astype(float)
        poi_hr = downscale(poi_hr, resample='add').astype(float)
    pop_lr, poi_lr = pop_hr, poi_hr
    for iter in range(int(lr_downscale - hr_downscale)):
        pop_lr = downscale(pop_lr, resample='add').astype(float)
        poi_lr = downscale(poi_lr, resample='add').astype(float)
    pop_sr, poi_sr = pop_lr, poi_lr
    for iter in range(int(lr_downscale - hr_downscale)):
        pop_sr = upscale(pop_sr, resample=Image.BICUBIC).astype(float) / 4
        poi_sr = upscale(poi_sr, resample=Image.BICUBIC).astype(float) / 4
    
    #print('Generation time list')
    time_list = np.arange(pop_hr.shape[0])
    time_list = time_list[48:-48]
    
    #print('{0} : Total Time Slice = {1}'.format(city_name, len(time_list)))
    dataset_list = []
    for iter, time_list_iter in enumerate(time_list_list):
        time_list_iter = time_list if time_list_iter == 'all' else time_list_iter
    #    print('Dataset Len {0} = {1}'.format(iter, len(list(time_list_iter))))
        dataset_list.append(ImageFromMatrix({
            'pop_hr':pop_hr, 'pop_lr':pop_lr, 'pop_sr':pop_sr, 
            'use_poi':False, 'use_flow':False,
            'frame_f':frame_f, 'frame_b':frame_b,
            'time_list':time_list_iter,'augment_func':None}))
    if len(dataset_list) == 1:
        return dataset_list[0]
    else:
        return dataset_list