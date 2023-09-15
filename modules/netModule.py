# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:32:44 2020

@author: Jerry
"""

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from modules.networks.losses import get_loss
import numpy as np
# https://github.com/ipython/ipython/issues/10627
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
class NetModule(object):
    def __init__(self, net, loss_config, device = torch.device("cpu")):
        # Set network structure
        self.device = device
        # self.net = getAENET(net_config)
        self.net = net
        self.net.to(device)
        self.criterion = get_loss(loss_config)
        self.continueTraining = False
    
    def train(self, training_dataset, training_config, valid_dataset = None, expPath = None, nni = False, logger = None, finetune = False):
        # Create dataloader        
        training_dataloader = DataLoader(training_dataset, batch_size=training_config['batch_size'],
                        shuffle=True)
        #alldata = torch.from_numpy(training_dataset.data).to(self.device, dtype = torch.float)
        #alllabels = torch.from_numpy(training_dataset.labels).to(self.device, dtype = torch.float)
        
        # Set Optimizer
        if self.continueTraining == False:
            if finetune:
                # print('finetune!')
                self.optimizer = torch.optim.Adam(self.net.netFinetune.parameters(), lr=training_config['learning_rate'],
                                weight_decay=1e-5)
            else:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=training_config['learning_rate'],
                                weight_decay=1e-5)

        # Save valid image if needed
        ifValid = training_config.get('valid_check', False) and valid_dataset is not None
        
        # Initalize if not continue previosu training
        if self.continueTraining == False:
            if finetune:
                self.net.netFinetune.apply(weights_init)
            else:
                self.net.apply(weights_init)
        
        # Training process
        start_time = time.time()
        loss_history = np.zeros([0])
        validLoss_history = np.zeros([0])        
        for epoch in range(1, training_config['epochs_num']  + 1):
            epoch_grad_norm = 0
            for data in training_dataloader: 
                if training_config.get('contrastive', False):
                    img0  = data['data0'].to(self.device, dtype = torch.float)
                    img1  = data['data1'].to(self.device, dtype = torch.float)
                    label = data['label'].to(self.device, dtype = torch.float)
                    output0, output1 = self.net(img0, img1)
                    loss = self.criterion(output0, output1, label)
                else:                        
                    imgs = data['data'].to(self.device, dtype = torch.float)
                    #labels = data['label'].to(self.device, dtype = torch.long)
                    labels = data['label'].to(self.device, dtype = torch.float)
                    labelmask = 1 if data['labelmask'] is None else data['labelmask'].to(self.device, dtype = torch.float)
                    
                    # ===================forward=====================
                    output = self.net(imgs)
                    # print('output shape: ', output.shape)
                    # print('label shape: ', labels.shape)
                    # print('mask shape: ', labelmask.shape)
                    loss = self.criterion(output*labelmask, labels*labelmask) / training_config["batch_size"]
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                batch_grad_norm = 0
                for p in net.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    batch_grad_norm += param_norm.item() ** 2
                batch_grad_norm = batch_grad_norm ** 0.5
                epoch_grad_norm += batch_grad_norm
                self.optimizer.step()
            print('Epoch Gradient Norm', epoch_grad_norm)
            if ifValid and (not training_config.get('contrastive', False)):
                #validPred, validLoss = self.pred(valid_dataset, labels = valid_dataset.labels)
                validPred, validLoss = self.pred(valid_dataset.data, labels = valid_dataset.labels, avg = True)
                # validLoss = validLoss
                validLossStr = f'lossVa:{validLoss:.4E}, '
                if nni:
                    import nni
                    nni.report_intermediate_result(validLoss)
                    logger.debug('test accuracy %g', validLoss)
                    logger.debug('Pipe send intermediate result done.')
            else:
                validLoss = np.nan
                validLossStr = ''
                
            # ===================log========================            
            report_epochs_num = training_config.get('report_per_epochs', 10)            
            if epoch % report_epochs_num == 0:                
                #allloss = self.pred(alldata, alllabels, avg = True)[1]
                # Report Time and Statistics
                loss_history = np.append(loss_history, loss.to(torch.device('cpu')).detach().numpy())
                validLoss_history = np.append(validLoss_history, validLoss)
                past_time = time.time() - start_time
                time_per_epoch_min = (past_time / epoch) / 60
                print(f'epoch [{epoch}/{training_config["epochs_num"]}], '+
                        f'lossTr:{loss:.4E}, '+
                        validLossStr +
                        f'used: {past_time / 60:.1f} mins, ' +
                        f'finish in:{(training_config["epochs_num"] - epoch)*time_per_epoch_min:.0f} mins')                    
                
                # Save sample image in training set
#                if training_config.get('save_training_img', False):
#                    net_sample_test(self.net)
                
                # Save sample image in validation set
                #if ifValid:
                #    valid_save_filename = expPath + f'/valid_img/valid_epoch_{str(epoch).zfill(prtDigitalLen)}.png'
                #    self.valid(valid_img, valid_save_filename, training_config['valid_check'])
                    
                # Save loss history
                #lossh_save_filename = expPath + f'/loss_log.png'
                #self.saveLossHistory(loss_history, lossh_save_filename, report_epochs_num)
                
        print(f'Traing finished with {training_config["epochs_num"]} epochs and {past_time/3600} hours')
        return loss_history, validLoss_history, past_time
    
    def pred(self, data, labels = None, avg = True):
        if type(data) == np.ndarray:
            data = torch.from_numpy(data.copy())
        if type(labels) == np.ndarray:
            labels = torch.from_numpy(labels.copy())
        data = data.to(self.device, dtype = torch.float)            
        
        predictions = self.net(data).to(torch.device('cpu')).detach().numpy()
        
        if labels is not None:
            labels = labels.to(self.device, dtype = torch.float)
            loss = self.criterion(self.net(data), labels)
            loss = loss.to(torch.device('cpu')).detach().numpy()            
            if avg:
                loss = loss / data.shape[0]
        else:
            loss = None
        return predictions, loss
    
    
    def predOLD(self, dataset, labels = None, avg = True):
        # Load new data and let go through network
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions = np.zeros(dataset.labels.shape)
        for dataIdx, data in enumerate(dataloader):
            # img = data.to(self.device, dtype = torch.float)
            imgs = data['data'].to(self.device, dtype = torch.float)
            prediction = np.moveaxis(self.net(imgs).to(torch.device('cpu')).detach().numpy(),0,-1)            
#            prediction = self.net(img).to(torch.device('cpu')).detach().numpy()
            try:
                predictions[dataIdx,:] = prediction#.flatten()
            except:
                predictions[dataIdx,:] = prediction.flatten()
        if labels is not None:
            #loss = self.criterion(predictions, labels)
            loss = self.criterion(self.net(torch.from_numpy(dataset.data).to(self.device, dtype = torch.float)), 
                                  torch.from_numpy(labels).to(self.device, dtype = torch.float))
            loss = loss.to(torch.device('cpu')).detach().numpy()
            if avg:
                loss = loss / len(dataset)
        else:
            loss = None
        return predictions, loss
    
    def pred_seg(self, dataset, labels = None):
        # Load new data and let go through network
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        N, C, H, W = dataset.labels.shape
        predictions = np.zeros((N, 1, H, W))
        for dataIdx, data in enumerate(dataloader):
            imgs = data['data'].to(self.device, dtype = torch.float)
            #prediction = np.moveaxis(self.net(imgs).to(torch.device('cpu')).detach().numpy(),0,-1)
            prediction = np.argmax(self.net(imgs).to(torch.device('cpu')).detach().numpy(), axis = 1)
            predictions[dataIdx,:] = prediction#.flatten()
        if labels is not None:
            #print(predictions.shape, labels.shape)
            loss = self.criterion(self.net(torch.from_numpy(dataset.data).to(self.device, dtype = torch.float)), torch.from_numpy(labels).to(self.device, dtype = torch.long))            
        else:
            loss = None
        return predictions, loss
    
    def saveLossHistory(self, loss_history, save_filename, report_epochs_num):
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib        
        plt.ioff()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(np.arange(1,(len(loss_history)+1))*report_epochs_num, np.log(loss_history))
        fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
        plt.close(fig)        
    
    def saveLossHistories(self, loss_histories, save_filename, report_epochs_num = 1, legends = None):
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib        
        plt.ioff()
        fig, axe = plt.subplots( nrows=1, ncols=1 )
        
        for idx, loss_history in enumerate(loss_histories):
            line,  = axe.plot(np.arange(1,(len(loss_history)+1))*report_epochs_num, loss_history)
            if legends is not None:
                line.set_label(legends[idx])
        axe.legend()
        #axe.plot(np.arange(1,(len(loss_history)+1))*report_epochs_num, np.log(loss_history))
        fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
        plt.close(fig)        
    
    def valid(self, img, save_filename, config):
        # 1. Go through network
        img = torch.from_numpy(img).to(self.device, dtype = torch.float)
        outimg = self.net(img).to(torch.device('cpu')).detach().numpy()
        
        # 2. Take slice as an image
#        if len(np.shape(img)) == 4:
#            # if images are 2D image and img has shape [N,C,H,W]
#            img_sample = outimg[config.get('index', 0), :]            
#        elif len(np.shape(img)) == 5:
#            # if images are 3D image and img has shape [N,C,D,H,W]
#            img_sample_3D = outimg[config.get('index', 0), :]
#            slice_axis = config.get('slice_axis',2)
#            slice_index = config.get('slice_index',0)
#            if slice_index == 'middle': slice_index = int(np.shape(img_sample_3D)[slice_axis]/2)
#            if slice_axis == 0:             
#                img_sample = img_sample_3D[:,slice_index,:,:]
#            elif slice_axis == 1:
#                img_sample = img_sample_3D[:,:,slice_index,:]
#            elif slice_axis == 2:
#                img_sample = img_sample_3D[:,:,:,slice_index]
#        else:
#            raise ValueError(f'Wrong image dimension. \
#                             Should be 4 ([N,C,H,W]) for 2d images \
#                             and 5 ([N,C,D,H,W]) for 3D images, \
#                             but got {len(np.shape(img))}')
        img_sample = slice_img(outimg, config)
        
        # 3. Save slice
        plt.imsave(save_filename, np.squeeze(img_sample), cmap='gray')
    
    def test(self, dataset):
        # If classification/regression, do pred() and report error
        pass
    
    def save(self, filename_full):
        # Save trained net parameters to file
        # torch.save(self.net.state_dict(), f'../model/{name}.pth')
        # torch.save(self.net.state_dict(), filename_full)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename_full)

    
    def load(self, checkpoint_path, map_location = None):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        # Load saved parameters
        self.continueTraining = True
        checkpoint = torch.load(checkpoint_path,map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.eval()
        # self.net.load_state_dict(torch.load(model_path))
        # self.net.eval()
    
def weights_init(m):
    # https://discuss.pytorch.org/t/reset-model-weights/19180/3
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    # torch.nn.init.xavier_uniform(m.weight)
        
