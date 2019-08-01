from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
from keras.utils import to_categorical
import torch
from torch.utils.data import Dataset
from torchvision import utils

import nibabel as nib
import os
import SimpleITK as sitk #导入itk
import numpy as np
import torch

root_dir = '/data3/Rysander/segment/100178_1_1_3_locnet_stem_crop240_whole/'
train_file = os.path.join(root_dir, "train.csv")
test_file = os.path.join(root_dir, "test.csv")

num_class = 9
means     = np.array([103.399, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 1024, 2048
train_h   = int(h/2)  # 512
train_w   = int(w/2)  # 1024
val_h     = h  # 1024
val_w     = w  # 2048

class SegmentDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=False, flip_rate=0.):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        if phase == 'train':
            self.lenth = [0, 108, 209, 312, 412, 531, 653, 759, 880, 987, 1100, 1205, 1324, 1440, 1566, 1681, 1779, 1882, 1987, 2107, 2229, 2333, 2444, 2550, 2679, 2780, 2889, 3003, 3110, 3215, 3328, 3431, 3558, 3665, 3770, 3891, 3992, 4111, 4225, 4357, 4464, 4577, 4675, 4781, 4879, 4980, 5093, 5190, 5300, 5415, 5530, 5652, 5756, 5864, 5990, 6093, 6202, 6312, 6422, 6531, 6650, 6756, 6876, 6983, 7091, 7220, 7337, 7439, 7565, 7682, 7789, 7897, 8000, 8100, 8209, 8311, 8416, 8523, 8638, 8748, 8849, 8958, 9075, 9176, 9293, 9402, 9517, 9636, 9748, 9853, 9960, 10086, 10212, 10318, 10432, 10543, 10643, 10743, 10847, 10961, 11071, 11177, 11287, 11391, 11507, 11604, 11710, 11817, 11921, 12033, 12132, 12228, 12340, 12459, 12565, 12678, 12782, 12894, 13006, 13118, 13229, 13354, 13455, 13570, 13666, 13778, 13890, 13987, 14083, 14205, 14297, 14404, 14506, 14623, 14738, 14834, 14944, 15064, 15173, 15284, 15394, 15512, 15614, 15728, 15831, 15947, 16047, 16144, 16247, 16351, 16436, 16540, 16659, 16764, 16872, 16983, 17102, 17226, 17327, 17426, 17528, 17640, 17752, 17863, 17964, 18065, 18174, 18281, 18384, 18496, 18590, 18697, 18813, 18914, 19029, 19140, 19241, 19347, 19458, 19559, 19668, 19776, 19887, 20001, 20103, 20232, 20358, 20455, 20581, 20700, 20809, 20908, 21006, 21120, 21235, 21358, 21470, 21575, 21673, 21773, 21887, 21987, 22099, 22213, 22324, 22446, 22559, 22658, 22765, 22876, 22967, 23083, 23204, 23319, 23427, 23542, 23651, 23796, 23899, 24012, 24123, 24227, 24329, 24431, 24541, 24650, 24753, 24868, 24981, 25093, 25202, 25314, 25428, 25521, 25624, 25726, 25840, 25953, 26057, 26164, 26265, 26378, 26486, 26619, 26713, 26824, 26934, 27052, 27165, 27274, 27387, 27514, 27625, 27726, 27832, 27949, 28062, 28178, 28283, 28386, 28503, 28612, 28723, 28832, 28947, 29055, 29153, 29264, 29364, 29483, 29606, 29725, 29839, 29947, 30049, 30152, 30255, 30360, 30477, 30578, 30694, 30808, 30910, 31012, 31125, 31249, 31368, 31495, 31623, 31725, 31840, 31947, 32067, 32174, 32290, 32399, 32519, 32634, 32736, 32833, 32942, 33055, 33155, 33264, 33375, 33474, 33576, 33687, 33792, 33893, 34002, 34112, 34223, 34332, 34439, 34532, 34643, 34750, 34854, 34958, 35065, 35175, 35281, 35381, 35493, 35604, 35703, 35800, 35912, 36027, 36138, 36248, 36363, 36487, 36588, 36694, 36799, 36904, 37002, 37107, 37213, 37315, 37434, 37538, 37649, 37757, 37866, 37984, 38093, 38177, 38293, 38404, 38510, 38637, 38743, 38846, 38959, 39059, 39171, 39288, 39410, 39532, 39647, 39750, 39852, 39960, 40058, 40168, 40285, 40392, 40506, 40623]
        elif phase == 'test':
            self.lenth = [0, 122, 232, 344, 450, 566, 686, 796, 910, 1026, 1143, 1257, 1370, 1456, 1564, 1649, 1762, 1868, 1974, 2082]
        # finding lenth	
        # self.lenth = [int(0)]
        # # print(type(self.lenth))
        # for j in range(len(self.data)):
        #     img_name = self.data.ix[j,0]
        #     img = sitk.ReadImage(root_dir+'data/'+img_name)
        #     img = sitk.GetArrayFromImage(img)
        #     # img = img(:,1,1)
        #     # print(len(img))
        #     # print(type(len(img)))
        #     self.lenth.append(self.lenth[-1]+len(img))
        #     # print(type(self.lenth))
        #     # print(lenth)
        #     # self.plenth.append(lenth)
        #     # print(img.shape)
        # print(self.lenth,'images read')

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train_h
            self.new_w = train_w
        print('loading data')


    def __len__(self):
        kkk = self.lenth
        return kkk[-1]
        # return len(self.data)*68

    def __getitem__(self, idx):
        # print(idx)
        # print(self.plenth)
        for num in range(len(self.lenth)-1):
            # print(num,leng,type(num),type(leng))
            if int(idx) >= int(self.lenth[num]):
                if int(idx) < int(self.lenth[num+1]):
                    i1 = num
                    i2 = idx - self.lenth[num]
                    # print(i1,i2)

        # img_name   = self.data.ix[int((idx+1)/70), 0]#idx按照len作迭代
        img_name   = self.data.ix[i1, 0]#idx按照len作迭代
        img        = sitk.ReadImage(root_dir+'data/'+img_name) #读取数据
        #  numpyimage = sitk.GetArrayFromImage(itkimage) #转为numpy
        #  tensorimage = torch.from_numpy(numpyimage ).type(torch.FloatTensor) #转为tensor
        # label_name = self.data.ix[int((idx+1)/70), 1]
        label_name = self.data.ix[i1, 1]

        label      = sitk.ReadImage(root_dir+'label/'+label_name)


        #if self.crop:
        #    h, w, _ = img.shape
        #    top   = random.randint(0, h - self.new_h)
        #    left  = random.randint(0, w - self.new_w)
        #    img   = img[top:top + self.new_h, left:left + self.new_w]
        #   label = label[top:top + self.new_h, left:left + self.new_w]

        #if random.random() < self.flip_rate:
        #    img   = np.fliplr(img)
        #    label = np.fliplr(label)

        # reduce mean
        #img = img[:, :, ::-1]  # switch to BGR
        #img = np.transpose(img, (2, 0, 1)) / 255.
        #img[0] -= self.means[0]
        #img[1] -= self.means[1]
        #img[2] -= self.means[2]

        # convert to tensor
        # 强行剪枝至81
        
        img = sitk.GetArrayFromImage(img) #转为numpy
        # print(img.shape)
        # print(img.size)
        # print(np.max(img))
        # img = img[-(idx+2-int((idx+1)/70)*70),8:-8,8:-8]
        img = img[i2,8:-8,8:-8]
        # print('max',np.max(img),'min',np.min(img))
        norm = ((np.max(img)-np.min(img))/255)
        img = (img-np.min(img))/norm

        # print(idx-int((idx+1)/70)*70)
        # print(np.max(img))
        img = np.array([img,img,img])
        # print(np.max(img))
        # label = sitk.GetArrayFromImage(label[idx-int((idx+1)/70)*70,8:-8,8:-8]) #转为numpy
        label = sitk.GetArrayFromImage(label) #转为numpy
        # print(np.max(np.max(label[:,8:-8,8:-8],axis = 1),axis = 1))
        # print(np.max(np.max(label,axis = 1),axis = 1).shape)
        # print(label[1,])
        # print(label[1].shape)
        # print(label.shape)
        # print(np.sum(label[idx-int((idx+1)/70)*70] == 1),11)
        # label = label[1,:,:]
        # label = np.delete(label,[1],axis = 0)
        # idx+2-int((idx+1)/70)*70

        # print(np.max(label))
        # label = label[-(idx+2-int((idx+1)/70)*70),8:-8,8:-8]
        label = label[i2,8:-8,8:-8]
        # 
        valuation = label

        num_classes = 9
        label = to_categorical(label,num_classes = 9)
        # print(label.shape)
        la = np.empty(shape = [label.shape[-1],label.shape[0],label.shape[1]])
        # print(la.shape)
        for i in range(num_classes):
        	la[i,:,:] = label[:,:,i]
        label = la

        # if np.max(np.max(label,axis = 1),axis = 0) != 0:
        # print(np.max(np.max(label,axis = 0),axis = 0))
        # print(np.max(np.max(label,axis = 0),axis = 0).shape)
        # label = label[[[idx-int((idx+1)/70)*70]]]
        # print(label.shape)
        # print(np.max(label))
        # label = label[[:,[8:-8]]]
        # label = label[[[:,:,[8:-8]]]
        # a = label
        # print(np.max(label))
        # print(np.sum(a==1),1,np.sum(a==2),2)
        # print(np.sum(a==3),3,np.sum(a==4),4)
        # print(np.sum(a==5),5,np.sum(a==6),6)
        # print(np.sum(a==7),7,np.sum(a==0),0)
        # if np.max(label) != 0:
        # print(label.shape)
        # print(np.max(label[:,:]))
        # print(np.max(label[:,50]))
        # print(np.max(label))
        # label = np.array([label,label,label])
        
        
        #img = np.delete(img,np.arange(1,delt),axis=0)
        #label = np.delete(label,np.arange(1,delt),axis=0)
        # print(img.shape,label.shape)
        #imag = torch.from_numpy(numpyimage).type(torch.FloatTensor) #转为tensor
        # label = label.reshape(N, h, w)
        # print(label.shape)
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()#这个之后改long可以节省空间
        valuation = torch.from_numpy(valuation.copy()).long()
        # create one-hot encoding
        #h, w = label.size()
        #target = torch.zeros(self.n_class, h, w)
        #for c in range(self.n_class):
        #    target[c][label == c] = 1

        #sample = {'X': img, 'Y': target, 'l': label}
        sample = {'X': img, 'l': label, 'v': valuation}

        return sample


#def show_batch(batch):
#    img_batch = batch['X']
#    batch_size = len(img_batch)#
#
#    grid = utils.make_grid(img_batch)
#    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))#
#
#    plt.title('Batch from dataloader')


if __name__ == "__main__":

    train_data = SegmentDataset(csv_file=test_file, phase='test')

    # show a batch
    batch_size = 10
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['l'].size())


    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    for i, bartch in enumerate(dataloader, 0):
        print(i,bartch['X'].size(),bartch['l'].size())

    # for i, batch in enumerate(dataloader):
       # print(i, batch)
    
        # observe 4th batch
    #    if i == 3:
    #        plt.figure()
    #        show_batch(batch)
    #        plt.axis('off')
    #        plt.ioff()
    #        plt.show()
    #        break

# import nibabel as nib
# import os
# import SimpleITK as sitk #导入itk
# import numpy as np
# import torch

# file_dir = '/data3/Rysander/segment/100181_1_1_3_locnet_stem_crop240_whole/data/'

# file_dir2 = '/data3/Rysander/segment/100181_1_1_3_locnet_stem_crop240_whole/label/'

# def file_names(file_dir):
#     L = []
#     for root,dirs,files in os.walk(file_dir):
#         for file in files:
#             if file[-7:] == '.nii.gz':
#                 L.append(file)
#     return L

# def read_file(file_dir):#,file_dir2):
#     L = file_names(file_dir)
#     #L2 = file_names(file_dir2)
#     i = 0
#     while i < len(L):
#         itkimage = sitk.ReadImage(file_dir+L[i]) #读取数据
#         numpyimage = sitk.GetArrayFromImage(itkimage) #转为numpy
#         tensorimage = torch.from_numpy(numpyimage ).type(torch.FloatTensor) #转为tensor
#         yield tensorimage
#         i = i + 1

# file_names(file_dir)

# file_names(file_dir2)
# p = read_file(file_dir)
# print(p)
# next(p)
# print(p)
# #,'1.2.276.0.7230010.3.1.4''_data.nii.gz')]





# # -*- coding: utf-8 -*-

# from __future__ import print_function

# from collections import namedtuple
# from matplotlib import pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import scipy.misc
# import random
# import os


# #############################
#     # global variables #
# #############################
# root_dir = '/data3/Rysander/segment/100181_1_1_3_locnet_stem_crop240_whole/'
# train_dir = os.path.join(root_dir, "train.csv")
# test_dir = os.path.join(root_dir, "test.csv")

# label_dir = os.path.join(root_dir, "gtFine")
# train_dir = os.path.join(label_dir, "train")
# val_dir   = os.path.join(label_dir, "val")
# test_dir  = os.path.join(label_dir, "test")

# # create dir for label index
# label_idx_dir = os.path.join(root_dir, "Labeled_idx")
# train_idx_dir = os.path.join(label_idx_dir, "train")
# val_idx_dir   = os.path.join(label_idx_dir, "val")
# test_idx_dir  = os.path.join(label_idx_dir, "test")
# for dir in [train_idx_dir, val_idx_dir, test_idx_dir]:
#     if not os.path.exists(dir):
#         os.makedirs(dir)

# train_file = os.path.join(root_dir, "train.csv")
# val_file   = os.path.join(root_dir, "val.csv")
# test_file  = os.path.join(root_dir, "test.csv")

# color2index = {}

# # Label = namedtuple('Label', [
# #                    'name', 
# #                    'id', 
# #                    'trainId', 
# #                    'category', 
# #                    'categoryId', 
# #                    'hasInstances', 
# #                    'ignoreInEval', 
# #                    'color'])

# # labels = [
# #     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
# #     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# #     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# #     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# #     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# #     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# #     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
# #     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
# #     Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# #     Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
# #     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
# #     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
# #     Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
# #     Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
# #     Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
# #     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
# #     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
# #     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
# #     Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
# #     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
# #     Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
# #     Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
# #     Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
# #     Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
# #     Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
# #     Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
# #     Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
# #     Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# #     Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
# #     Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
# #     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
# #     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
# #     Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
# #     Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
# #     Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
# #     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# # ]


# def parse_label():
#     # change label to class index
#     color2index[(0,0,0)] = 0  # add an void class 
#     for obj in labels:
#         if obj.ignoreInEval:
#             continue
#         idx   = obj.trainId
#         label = obj.name
#         color = obj.color
#         color2index[color] = idx

#     # parse train, val, test data    
#     for label_dir, index_dir, csv_file in zip([train_dir, val_dir, test_dir], [train_idx_dir, val_idx_dir, test_idx_dir], [train_file, val_file, test_file]):
#         f = open(csv_file, "w")
#         f.write("img,label\n")
#         for city in os.listdir(label_dir):
#             city_dir = os.path.join(label_dir, city)
#             city_idx_dir = os.path.join(index_dir, city)
#             data_dir = city_dir.replace("gtFine", "leftImg8bit")
#             if not os.path.exists(city_idx_dir):
#                 os.makedirs(city_idx_dir)
#             for filename in os.listdir(city_dir):
#                 if 'color' not in filename:
#                     continue
#                 lab_name = os.path.join(city_idx_dir, filename)
#                 img_name = filename.split("gtFine")[0] + "leftImg8bit.png"
#                 img_name = os.path.join(data_dir, img_name)
#                 f.write("{},{}.npy\n".format(img_name, lab_name))

#                 if os.path.exists(lab_name + '.npy'):
#                     print("Skip %s" % (filename))
#                     continue
#                 print("Parse %s" % (filename))
#                 img = os.path.join(city_dir, filename)
#                 img = scipy.misc.imread(img, mode='RGB')
#                 height, weight, _ = img.shape
        
#                 idx_mat = np.zeros((height, weight))
#                 for h in range(height):
#                     for w in range(weight):
#                         color = tuple(img[h, w])
#                         try:
#                             index = color2index[color]
#                             idx_mat[h, w] = index
#                         except:
#                             # no index, assign to void
#                             idx_mat[h, w] = 19
#                 idx_mat = idx_mat.astype(np.uint8)
#                 np.save(lab_name, idx_mat)
#                 print("Finish %s" % (filename))


# '''debug function'''
# def imshow(img, title=None):
#     try:
#         img = mpimg.imread(img)
#         imgplot = plt.imshow(img)
#     except:
#         plt.imshow(img, interpolation='nearest')

#     if title is not None:
#         plt.title(title)
    
#     plt.show()


# if __name__ == '__main__':
#     parse_label()