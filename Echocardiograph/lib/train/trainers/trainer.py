import time
import tqdm
import datetime
import torch
import torch.nn as nn
import numpy as np


class Trainer(object):
    def __init__(self, network):
        # network.half()
        network = nn.DataParallel(network).cuda()
        self.network = network  # --- lib.trian.trainer.BScanSeg // NetworkWrapper

    def reduce_loss_state(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}  # k:key & v:specific_value
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        # mean_loss = []
        self.network.train()  # train tag
        end = time.time()
        # for iteration, (inputs, mask, target) in enumerate(data_loader):
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1
            '''
                output:  pred of the net
                loss: the loss value 
                loss_stats: the loss dictionary (if multiple loss contains)
                img_stats: {} empty dictionary
            '''
            output, loss, loss_stats, image_state = self.network(batch)  # @ BScanSeg.NetworkWrapper.loss_calculate

            loss = loss.mean()
            # mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)  # clip the grad between +- 40
            optimizer.step()

            loss_stats = self.reduce_loss_state(loss_stats)  # mean loss in the loss_dictionary
            recorder.update_loss_stats(loss_stats)   # here recorder ?

            batch_time = time.time() - end
            end = time.time()

            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):  # insert the recorder
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                lr = optimizer.param_groups[0]['lr']  # optimizer_list[0]['lr']: current learning rate
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = ' '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                recorder.update_loss_stats(image_state)
                recorder.record('train')
        # mean_loss = sum(mean_loss) / len(mean_loss)

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):  # load data from val folder which have 5522 2d files
            for k in batch:
                # the batch returned by the data loader is the clipped mini patch and it's shape = 112,112,160
                # so as the ground truth
                # and the validated dice score is the dice of the mini region, to evaluate the entire input array
                # the val part should be rewritten, how to joint the segmentation results ?

                # this should be considered in the next time
                if k != 'meta':
                    batch[k] = batch[k].cuda()  # if not meta then turn each tensor to GPU
            with torch.no_grad():  # for val()
                output, loss, loss_stats, image_stats = self.network(batch)  # output = predict,
                # gt = batch['label']  # B 1 W H
                # gt_array = gt.detach().cpu().numpy()
                #
                # tip = np.max(np.max(gt_array))
                #
                # if tip > 0:
                if evaluator is not None:
                    evaluator.evaluate(output, batch)  # output = (B C W H) batch = (B 2 512 512) append the dict_ann

            loss_stats = self.reduce_loss_state(loss_stats)
            for k, v in loss_stats.items():      #  key & value in the loss_state:  --- here we got the dice_loss and its value
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []  # include 'dice_loss'

        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size      # data_size equal the length of the val dataset

            if k == 'loss':
                loss_state.append('{}: {:.4f}'.format('Val_loss', val_loss_stats[k]))
            else:
                loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            #  need to be changed
            result = evaluator.summarize()  # calculate the acc IoU hit_rate or some other dict
            val_loss_stats.update(result)  # add those new results

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)  # show them in the curves
