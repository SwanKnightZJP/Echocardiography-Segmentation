import os
import torch

from termcolor import colored


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:   # train from existed model
        os.system('rm -rf {}'.format(model_dir))  # delete the saved model
        return 0

    if not os.path.exists(model_dir):  # if model folder is not existed
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]  # if no file in this folder
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:    # keep the number of the saved model is less than 200
        return
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODE LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1
