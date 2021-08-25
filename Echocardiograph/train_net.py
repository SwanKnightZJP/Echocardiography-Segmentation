from lib.config import cfg, args
from lib.net import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch


def train(cfg, network):
    trainer = make_trainer(cfg, network)  # --- to instantiate (lib.train.trainers.Task.py  *@(BScanSeg.py))
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)  # --- to instantiate (lib.evaluators.Task.Task.py  *@(BScanSeg.py))
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)  # lib.train.trainers.trainer// Train.train(BScanSeg.py)
        scheduler.step()  # equal to the opt.step()
        if (epoch + 1) % cfg.save_ep == 0:  # save this model every save_ep times
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)
        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)


def test(cfg, network):
    # since the test code and train code use the same trainer code with different tags
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)  # run Class Evaluation alone
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    if args.test:
        network = make_network(cfg, is_train=False)  # =ft
        test(cfg, network)
    else:
        network = make_network(cfg, is_train=True)   # lib.net.Task.Network.py -- lib.net.Task.Network.get_network
        train(cfg, network)


if __name__ == "__main__":
    main()
