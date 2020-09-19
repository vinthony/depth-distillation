# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import reduce
from .blocks import *

class ResNextDecoderAtt(nn.Module):

    def __init__(self, pretrained_net, n_class=1, type='res', decoder=UpsampleSKConvPlus3, side=SideClassifer,att=MergeAndConv):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        if type == 'res':
            inlist = [2048,1024,512,256,64]
            olist = [1024,512,256,64,32]
        else:
            inlist = [512,512,256,128,64]
            olist = [512,256,128,64,32]

        self.deconv1 = decoder(inlist[0],olist[0])
        self.deconv2 = decoder(inlist[1],olist[1])
        self.deconv3 = decoder(inlist[2],olist[2])
        self.deconv4 = decoder(inlist[3],olist[3])
        self.deconv5 = decoder(inlist[4],olist[4])

        self.classifier1 = side(olist[0], n_class, kernel_size=1)
        self.classifier2 = side(olist[1], n_class, kernel_size=1)
        self.classifier3 = side(olist[2], n_class, kernel_size=1)
        self.classifier4 = side(olist[3], n_class, kernel_size=1)
        self.classifier5 = side(olist[4], n_class, kernel_size=1)

        self.sideatt1 = att(2, 1)
        self.sideatt2 = att(2, 1)
        self.sideatt3 = att(2, 1)
        self.sideatt4 = att(2, 1)

        self.fusion = nn.Conv2d(5,n_class,kernel_size=1)
        self.fusiond = nn.Conv2d(5,n_class,kernel_size=1)


    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5)  # size=(N, 512, x.H/16, x.W/16)
        side1,d1 = self.classifier1(score) # utilized the features as the att of the main stream
        score = score * self.sideatt1(torch.cat([side1, d1],1))

        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score)  # size=(N, 256, x.H/8, x.W/8)
        side2,d2 = self.classifier2(score)
        score = score * self.sideatt2(torch.cat([side2, d2],1))

        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        side3,d3 = self.classifier3(score)
        score = score * self.sideatt3(torch.cat([side3, d3],1))

        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        side4,d4 = self.classifier4(score)
        score = score * self.sideatt4(torch.cat([side4, d4],1))

        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score) # size=(N, 32, x.H, x.W)
        side5,d5 = self.classifier5(score)  # size=(N, n_class, x.H/1, x.W/1)

        sides = torch.cat([ F.interpolate(i, size=(x.size(2), x.size(3)), mode='bilinear') for i in (side1, side2, side3, side4, side5)], 1)
        depths = torch.cat([ F.interpolate(i, size=(x.size(2), x.size(3)), mode='bilinear') for i in (d1,d2,d3,d4,d5)], 1)

        score = self.fusion(sides)
        ds = self.fusiond(depths)

        return (score, side1, side2, side3, side4, side5),(ds, d1, d2, d3, d4, d5)  # size=(N, n_class, x.H/1, x.W/1)

    
if __name__ == "__main__":
    batch_size, n_class, h, w = 10, 20, 160, 160
