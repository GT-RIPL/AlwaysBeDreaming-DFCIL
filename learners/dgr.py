from __future__ import print_function
import torch
import torch.nn as nn
import models
from utils.metric import AverageMeter, Timer
from .dgr_helper import Scholar
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import numpy as np

class Generative_Replay(NormalNN):

    def __init__(self, learner_config):
        super(Generative_Replay, self).__init__(learner_config)
        self.generator = self.create_generator()
        self.generative_replay = False
        self.previous_scholar = None
        self.generator.recon_criterion = nn.BCELoss(reduction="none")
        self.dw = self.config['DW']

        # generator optimizor
        self.generator.optimizer, self.generator_scheduler = self.new_optimizer(self.generator)

        # repeat call for generator network
        if self.gpu:
            self.cuda_gen()
        
    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if need_train:
            if self.reset_optimizer:  # Reset optimizer before learning each task
                self.log('Optimizer is reset!')
                self.init_optimizer()
                self.generator.optimizer, self.generator_scheduler = self.new_optimizer(self.generator)

            # Evaluate the performance of current task
            self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=0,total=self.config['schedule'][-1]))
            if val_loader is not None:
                self.validation(val_loader)
        
            losses = AverageMeter()
            acc = AverageMeter()
            gen_losses = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0:
                    self.scheduler.step()
                    self.generator_scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()
                    self.generator.train()

                    # send data to gpu
                    if self.gpu:
                        x =x.cuda()
                        y = y.cuda()

                    # data replay
                    if not self.generative_replay:
                        x_replay = None   #-> if no replay
                    else:
                        allowed_predictions = list(range(self.last_valid_out_dim))
                        x_replay, y_replay, y_replay_hat = self.previous_scholar.sample(len(x), allowed_predictions=allowed_predictions,
                                                            return_scores=True)
                    
                    # if KD
                    if self.generative_replay:
                        y_hat = self.previous_scholar.generate_scores(x, allowed_predictions=allowed_predictions)
                        _, y_hat_com = self.combine_data(((x, y_hat),(x_replay, y_replay_hat)))
                    else:
                        y_hat_com = None

                    # combine inputs and generated samples for classification
                    if self.generative_replay:
                        x_com, y_com = self.combine_data(((x, y),(x_replay, y_replay)))
                    else:
                        x_com, y_com = x, y

                    # dgr data weighting
                    mappings = torch.ones(y_com.size(), dtype=torch.float32)
                    if self.gpu:
                        mappings = mappings.cuda()
                    rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
                    mappings[:self.last_valid_out_dim] = rnt
                    mappings[self.last_valid_out_dim:] = 1-rnt
                    dw_cls = mappings[y_com.long()]

                    # model update
                    loss, output= self.update_model(x_com, y_com, y_hat_com, dw_force = dw_cls, kd_index = np.arange(len(x), len(x_com)))

                    # generator update
                    loss_gen = self.generator.train_batch(x_com, dw_cls, list(range(self.valid_out_dim)))

                    # measure elapsed time
                    batch_time.update(batch_timer.toc()) 

                    # measure accuracy and record loss
                    y_com = y_com.detach()
                    accumulate_acc(output, y_com, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y_com.size(0)) 
                    gen_losses.update(loss_gen, y_com.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                self.log(' * Gen Loss {loss.avg:.3f}'.format(loss=gen_losses))

                # Evaluate the performance of current task
                if val_loader is not None:
                    self.validation(val_loader)

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
                gen_losses = AverageMeter()

        self.model.eval()
        self.generator.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        # new scholar
        scholar = Scholar(generator=self.generator, solver=self.model)
        self.previous_scholar = copy.deepcopy(scholar)
        self.generative_replay = True

        try:
            return batch_time.avg
        except:
            return None

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    def save_model(self, filename):
        
        model_state = self.generator.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving generator model to:', filename)
        torch.save(model_state, filename + 'generator.pth')
        super(Generative_Replay, self).save_model(filename)

    def load_model(self, filename):
        
        self.generator.load_state_dict(torch.load(filename + 'generator.pth'))
        if self.gpu:
            self.generator = self.generator.cuda()
        self.generator.eval()
        super(Generative_Replay, self).load_model(filename)

    def create_generator(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = models.__dict__[cfg['gen_model_type']].__dict__[cfg['gen_model_name']]()
        return generator

    def print_model(self):
        super(Generative_Replay, self).print_model()
        self.log(self.generator)
        self.log('#parameter of generator:', self.count_parameter_gen())
    
    def reset_model(self):
        super(Generative_Replay, self).reset_model()
        self.generator.apply(weight_reset)

    def count_parameter_gen(self):
        return sum(p.numel() for p in self.generator.parameters())

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.count_parameter_gen() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda_gen(self):
        self.generator = self.generator.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.generator= torch.nn.DataParallel(self.generator, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y