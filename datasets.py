import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DefocusDataset(Dataset):
    def __init__(self, root='./datasets/CUHKDefocus/', mode="train"):

        self.root = root
        self.is_train = True if mode == 'train' else False
        
        if 'CUHKDefocus' in self.root:
            with open(os.path.join(root,'test.txt')) as f:
                test_images = [s.strip() for s in f.readlines()]

        if self.is_train:
            self.images = sorted([ os.path.join(root,'image',x) for x in os.listdir(os.path.join(root,'image')) if x not in test_images ])
        else:
            if 'CUHKDefocus' in self.root:
                self.images = [ os.path.join(root,'image',x) for x in test_images ]
            else:
                self.images = sorted([ os.path.join(root,'image',x) for x in os.listdir(os.path.join(root,'image'))])


        self.transform = transforms.Compose([
            transforms.Resize((320, 320), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print('Dataset:%s'%(len(self.images)))

    def __getitem__(self, index):

        imgo = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.images[index].replace('image','gt').replace('jpg','png')).convert('L')
      
        img = self.transform(imgo)
        mask = np.asarray(mask)/255.
        imgo = np.array(imgo)

        if 'CUHKDefocus' in self.root:
            mask = 1 - mask

        return {"A": img, 'Ao':np.array(imgo), 'M': mask, 'name':self.images[index].split('/')[-1] }

    def __len__(self):
        return len(self.images)

