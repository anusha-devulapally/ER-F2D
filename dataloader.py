import torch
import os
from os.path import join
import random
import numpy as np
from skimage import io
from torch.utils.data import ConcatDataset
from data_loader.dataset import *
#from data_loader_uni_norm.dataset import *
import bisect
""" DATASET Structure
eventscape_dataset
- train
  -- Town01
      -- sequence0
          -- rgb
          -- events
          -- depth
  -- Town02
  -- Town03
- val
  -- Town05
- test
  -- Town05
"""
class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
        
def concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=None, proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1,
                           clip_distance=100.0, every_x_rgb_frame=1, normalize=True, scale_factor=1.0,
                           use_phased_arch=False, baseline=False, loss_composition=False, reg_factor=3.7, dataset_idx_flag=False, recurrency=True):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """
    print("entered concatenate_subfolders")
    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))

    train_datasets = []
    #for dataset_name in subfolders:
      #print(dataset_name, len(os.listdir(base_folder+"/"+dataset_name)))
        #for sequences in os.listdir(base_folder+"/"+dataset_name):
          #print("sequences :", sequences)
    train_datasets.append(eval(dataset_type)(base_folder=base_folder, event_folder=event_folder,
                                                 depth_folder=depth_folder,
                                                 frame_folder=frame_folder, 
                                                 sequence_length=sequence_length,
                                                 transform=transform,
                                                 proba_pause_when_running=proba_pause_when_running,
                                                 proba_pause_when_paused=proba_pause_when_paused,
                                                 step_size=step_size,
                                                 clip_distance=clip_distance,
                                                 every_x_rgb_frame=every_x_rgb_frame,
                                                 normalize=normalize,
                                                 scale_factor=scale_factor,
                                                 use_phased_arch=use_phased_arch,
                                                 baseline=baseline,
                                                 reg_factor=reg_factor, recurrency=recurrency))
          
          #mean_stds.append(mean_std)
    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(train_datasets)
        #means_concat = ConcatDataset(mean_stds)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(train_datasets)
        #means_concat = ConcatDatasetCustom(mean_stds)
    return concat_dataset#, means_concat

event_path = "events/voxels"  # for accumulated events. To use voxels path = "events/voxel"
rgb_path = "rgb/davis"
gt_path = "depth/data"
#
#def rgb2gray(rgb):
#  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
            
def build_dataset(set, transform, args):
   if(set=='train'):
     print("entered build_dataset")
     dataset = concatenate_subfolders(join(args.data_path, set),
                                           "SequenceSynchronizedFramesEventsDataset",
                                           event_path,
                                           gt_path,
                                           rgb_path,
                                           sequence_length=1,
                                           transform=transform,
                                           proba_pause_when_running=0.0,
                                           proba_pause_when_paused=0.0,
                                           step_size=args.train_step_size,
                                           clip_distance=args.clip_distance,
                                           every_x_rgb_frame=1,
                                           normalize='True',
                                           scale_factor=1,
                                           use_phased_arch="False",
                                           baseline="False",
                                           loss_composition = ['image','event0'],
                                           reg_factor=args.reg_factor,
                                           recurrency = "False"
                                           )
   elif(set=='validation'):
     dataset = concatenate_subfolders(join(args.data_path, set),
                                           "SequenceSynchronizedFramesEventsDataset",
                                           event_path,
                                           gt_path,
                                           rgb_path,
                                           sequence_length=1,
                                           transform=transform,
                                           proba_pause_when_running=0.0,
                                           proba_pause_when_paused=0.0,
                                           step_size=args.val_step_size,
                                           clip_distance=args.clip_distance,
                                           every_x_rgb_frame=1,
                                           normalize='True',
                                           scale_factor=1,
                                           use_phased_arch="False",
                                           baseline="False",
                                           loss_composition = ['image','event0'],
                                           reg_factor=args.reg_factor,
                                           recurrency = "False"
                                           )

   return dataset