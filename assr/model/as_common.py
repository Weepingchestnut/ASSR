import math

import torch
import torch.nn as nn


class MetaUpscale(nn.Module):
    def __init__(self, scale: list, planes: int, act_mode: str = 'relu',
                 use_affine: bool = True):
        self.scale = scale

    def input_matrix_wpn(self, inH, inW, scale, add_scale=True):
        """
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        """
        outH, outW = int(scale * inH), int(scale * inW)

        # mask records which pixel is invalid, 1 valid or o invalid,
        # h_offset and w_offset calculate the offset to generate the input matrix
        # （mask 记录哪些像素无效，1有效 or 0无效，h_offset 和 w_offset 计算偏移量以生成输入矩阵）
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH, scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1, 1)
            scale_mat[0, 0] = 1.0 / scale
            # res_scale = scale_int - scale
            # scale_mat[0,scale_int-1]=1-res_scale
            # scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat] * (inH * inW * (scale_int ** 2)), 0)  # (inH*inW*scale_int**2, 4)

        # projection coordinate and calculate the offset
        # （投影坐标和计算偏移量）
        h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        # flag for number for current coordinate LR image
        # （标记当前LR图像坐标的编号）
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag, 0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        # the size is scale_int * inH * (scal_int * inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        #
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1, 2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2)

        return pos_mat, mask_mat  # outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
