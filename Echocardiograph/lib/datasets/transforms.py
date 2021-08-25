"""

    To transform the input tensor

"""
from torchvision import transforms


def make_transforms(cfg, is_train):

    if cfg.data.DataTransType == 'Norm_224':
        vision_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train is True:
            transform = transforms.Compose(
                [transforms.Resize(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 vision_normalize]
            )
        else:
            transform = transforms.Compose(
                [transforms.Resize(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 vision_normalize]
            )

    else:
        transform = transforms.Compose([transforms.ToTensor()])
        # warning transforms.ToTensor will turn a ( x y z ) array as (z x y) tensor!!!!!

    return transform
