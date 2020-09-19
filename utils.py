import glob
import random
import os
import numpy as np
import torch

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from skimage import transform
# import pydensecrf.densecrf as dcrf

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def to_numpy(tesor):
    return tesor.cpu().data.numpy()

def to_tensor(npy):
    return torch.from_numpy(npy)


def get_cm(impredictm,maskgt):
    # torch to numpy
    maskgt_ravel = np.where(maskgt>128,np.full_like(maskgt,255),np.zeros_like(maskgt)).ravel()
    impred_ravel = np.where(impredictm>128,np.full_like(impredictm,255),np.zeros_like(impredictm)).ravel()
    return confusion_matrix(maskgt_ravel,impred_ravel, labels=[0,255] ).ravel()

def resize_to_match(fm,to):
    # just use interpolate
    # [1,3] = (h,w)
    return F.interpolate(fm,to.size()[-2:],mode='bilinear')

def cal_ber(fps,tps,tns,fns):
    
    FP = np.sum(fps)
    TP = np.sum(tps)
    TN = np.sum(tns)
    FN = np.sum(fns)

    BER_NS = FP/(TN+FP)
    BER_S = FN/(FN+TP)
    BER = 0.5*(BER_S + BER_NS)

    return BER,BER_S,BER_NS


def save_checkpoint(generator,dataset_name,epoch,is_best=False,per='generator'):
    torch.save(generator.state_dict(), "saved_models/%s/%s_%d.pth" % (dataset_name, per, epoch))



def add_image(writer,name,tens,iter):

    tens = tens.squeeze()

    if len(tens.size()) == 2:
        tens = tens.view((1,tens.size(0),tens.size(1))).repeat((3,1,1))
    
    writer.add_image(name,tens,iter)

    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
