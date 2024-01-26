import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        
        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        # print('x:',x_position)
        print("pe",pe.shape)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x,xmf,ymf):
        """
        Args:
            x: [N, C, H, W]
        """
        # print('x:',x.size(2),x.size(3)) #6080
        # print("x",x.shape) #x torch.Size([1, 256, 60, 80])  60 80 是啥？
        # print('pe.shape',self.pe)
        # print('pe.size',self.pe.size)
        length=256

        
        # method 1
        y0 = x.size(2)-int(ymf)
        y1 = 2*x.size(2)-int(ymf)
        x0 = x.size(3)-int(xmf)
        x1 = 2*x.size(3)-int(xmf)
        return x + self.pe[:,:,y0:y1,x0:x1]
        print('y0y1x0x1',y0,y1,x0,x1)


        # method2
        # y0 = (length-int(ymf))% length
        # y1 = (length+x.size(2)-int(ymf))% length
        # x0 = (length-int(xmf))% length
        # x1 = (length+x.size(3)-int(xmf))% length

        # if y0 < y1:
        #     if x0 < x1:
        #         return x + self.pe[:, :, y0:y1, x0:x1]
        #     else:
        #         x_slice = torch.cat([self.pe[:, :, y0:y1, x0:], self.pe[:, :, y0:y1, :x1]], dim=3)
        #         return x + x_slice
        # else:
        #     if x0 < x1:
        #         y_slice = torch.cat([self.pe[:, :, y0:, x0:x1], self.pe[:, :, :y1, x0:x1]], dim=2)
        #         return x + y_slice
        #     else:
        #         y_slice = torch.cat([torch.cat([self.pe[:, :, y0:, x0:], self.pe[:, :, :y1, x0:]], dim=2),torch.cat([self.pe[:, :, y0:, :x1], self.pe[:, :, :y1, :x1]], dim=2)],dim=3)
        #         return x + y_slice

        # loftr
        # return x + self.pe[:, :, :x.size(2),: x.size(3)]