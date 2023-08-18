import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn import init

class PatchEmbed(nn.Module):
  def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = int((img_size[0] * img_size[1]) / (patch_size ** 2))

        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

  def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2) 
        return x

class PatchEmbedEvents(nn.Module):
  def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = int((img_size[0] * img_size[1]) / (patch_size ** 2))
        self.embed_dim = 768
        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

  def forward(self, x):
        seq_len = x.shape[1]
        concat=[]
        #print(concat.size())
        for t in range(seq_len):
            x_t = x[:,t, :, :].unsqueeze(1)
            #print(x_t.size())
            x_t = self.proj(x_t) 
            x_t = x_t.flatten(2)  
            x_t = x_t.transpose(1, 2) 
            concat.append(x_t)
        result = torch.zeros(x.shape[0],5*self.n_patches, self.embed_dim)
        for i, tensor in enumerate(concat):
          result[:, i*196: (i+1)*196, :] = tensor
        return result.to(x.device)
        
class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        #print("qkv",qkv.size())
        #print(n_samples, n_tokens, dim, self.n_heads,self.head_dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, y, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        kv = self.kv(x) 
        qkv = torch.cat((y,kv), dim=-1)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) 
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) 
        dp = (q @ k_t) * self.scale 
        attn = dp.softmax(dim=-1)  
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v  
        weighted_avg = weighted_avg.transpose(1, 2) 
        weighted_avg = weighted_avg.flatten(2) 
        x = self.proj(weighted_avg) 
        x = self.proj_drop(x)  
        return x
        

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x) 
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Dec_Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.cross_attn = Cross_Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, y, x):
        y = y + self.attn(self.norm1(y))
        y=self.norm1(y)
        y = y + self.cross_attn(y,x)
        y = y + self.mlp(self.norm2(y))

        return y
                
class Enc_Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

 
class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
    
    def forward(self, x):
        inputs = x
        
        out = self.b1(inputs)
        inputs = torch.cat([inputs, out], 1)
        
        out = self.b2(inputs)
        inputs = torch.cat([inputs, out], 1)
        
        out = self.b3(inputs)
        inputs = torch.cat([inputs, out], 1)
        
        out = self.b4(inputs)
        inputs = torch.cat([inputs, out], 1)
        
        out = self.b5(inputs)
        
        
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters))#, DenseResidualBlock(filters), DenseResidualBlock(filters))

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
                
class Recurrent2ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(Recurrent2ConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
            #print("enter convlstm")
        else:
            #print("enter convgru")
            RecurrentBlock = ConvGRU

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state

class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden.to(input_.device)), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate.to(input_.device) * prev_cell.to(input_.device)) + (in_gate.to(input_.device) * cell_gate.to(input_.device))
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell
                
class ConvLSTMSequenceModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1, padding=0, recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(ConvLSTMSequenceModel, self).__init__()
        self.recurrent_layer = Recurrent2ConvLayer(
            in_channels=1,
            out_channels=64,#hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            recurrent_block_type=recurrent_block_type,
            activation=activation,
            norm=norm,
            BN_momentum=BN_momentum
        )
        self.recurrent_layer2 = Recurrent2ConvLayer(
            in_channels=64,#hidden_channels,
            out_channels=64,#hidden_channels*2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            recurrent_block_type=recurrent_block_type,
            activation=activation,
            norm=norm,
            BN_momentum=BN_momentum
        )
        self.conv_layer = ConvLayer(
            in_channels=64,#hidden_channels,
            out_channels=1,#hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=None,
            norm=None,
        )

    def forward(self, x):
        seq_len = x.shape[1]
        state = None
        for t in range(seq_len):
            x_t = x[:,t, :, :].unsqueeze(1)
            x_t2, state = self.recurrent_layer(x_t, state)
            x_t2 = x_t2+x_t
            x_t2, state = self.recurrent_layer2(x_t2, state)
        x_t2 = self.conv_layer(x_t2)
        return x_t2
 
