from models.resnext.resnext101_regular import ResNeXt101
from models.vgg import VGGNet
from models.fcn import ResNextDecoderAtt

def resskspp():
    return ResNextDecoderAtt(pretrained_net=ResNeXt101(), type='res')

def vggskspp():
    return ResNextDecoderAtt(pretrained_net=VGGNet(),type='vgg')

if __name__ == '__main__':
    
    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    print(model_names)