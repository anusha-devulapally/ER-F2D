import torch.nn.functional as F
import torch
import os
import numpy as np
import math
import torch.nn as nn
from utils import *
from torchvision import utils
from kornia.filters.sobel import spatial_gradient, sobel
from utils.path_utils import ensure_dir
int_path = os.path.join('analysis_exps/exp_24', 'intermediate_results')
print(int_path)

ensure_dir(int_path)


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    # if img.is_cuda:
    weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    # if img.is_cuda:
    weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    is_nan = torch.isnan(img)
    #print("yx",is_nan)
    grad_y, grad_x = imgrad(img)
    #print("grads",grad_y, grad_x)
    #print(torch.isnan(grad_y), torch.isnan(grad_x))
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, grad_fake, grad_real):
        is_nan = torch.isnan(grad_real)
        #print("norm",is_nan)
        a_real = grad_fake[:,:,None,:]
        a_fake = grad_real[:,:,:,None]
        
        #print(grad_real, grad_fake)
        prod = ( a_fake[~is_nan[:,:,:,None]] @ a_real[~is_nan[:,:,None,:]] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake[~is_nan]**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real[~is_nan]**2, dim=-1 ) )
        
        return 1 - torch.mean( prod/(fake_norm*real_norm) )

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
    log_diff = y_input - y_target
    is_nan = torch.isnan(log_diff)
    #print("scale", is_nan)
    return weight * ((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))
 

class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]
        
        loss_value = 0
        loss_value_2 = 0
        diff = prediction - target
        _,_,H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                loss_value += torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
                # * batch size * 2 (because kornia spatial product has two outputs).
                # replaces the following line to be able to deal with nan's.
                # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        if preview:
            return record
        else:
            return (loss_value/self.num_scales)


multi_scale_grad_loss_fn = MultiScaleGradient()


def multi_scale_grad_loss(prediction, target, preview = False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)
    

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
                     
def make_preview(rgb, event,gt, target_pred,preview_idx,epoch,mode):
  if(rgb.size()[1]==3):
    #print("rgb size", rgb.size())
    gray = torch.squeeze(rgb)
    #print("rgb size", gray.size())
    gray = gray.permute(1,2,0)
    #print("rgb size", gray.size())
    gray = rgb2gray(gray)
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=0)
    #print("rgb size", gray.size())
    rgb = torch.from_numpy(gray)
    #print("rgb size", gray.size())
    #rgb = rgb.permute(0,1,1)
  if(event.size()[1]==5):
    event = torch.squeeze(event)
    event = event[4]
    event = torch.unsqueeze(event, dim=0)
    event = torch.unsqueeze(event, dim=0)
  #print("sizes", rgb.size(), event.size(), gt.size(), target_pred.size())
  grid = utils.make_grid(torch.cat([rgb.to('cpu'), event.to('cpu'),gt.to('cpu'), target_pred.to('cpu')], dim=0),normalize=False, scale_each=True, nrow=1)
  utils.save_image(grid,int_path+"/"+"%s_preview_idx_%d_epoch_%d.png" % (mode, preview_idx, epoch), normalize=False)
  return grid                            

def loading_weights_from_eventscape(model_state_dict,p):
  main = model_state_dict
  main_keys = main.keys()
  #print(main_keys, p.keys())
  for k,v in main.items():
    #if('patch_embed_rgb' in k):# or 'conv_skip_rgb' in k):
    #  continue
    c='module.'+k
    if c in p.keys():
      
      #print("modified keys", k)
      #print("before", main[k])
      main[k]=p[c]
      #print("after",main[k])
    #else:
    #  print("skipped keys:", k)
  
  return main
  
  
def loading_weights_from_vit(model_state_dict, p):
  main = model_state_dict
  main_keys = main.keys()
  #print(main_keys)
  #print(main_keys)
  for k,v in main.items():
    if k in p.keys():
    
      #print("modified keys", k)
      #print("before", main[k])
      main[k]=p[k]
      #print("after",main[k])
    #else:
      #print("skipped keys:", k)
  
  return main