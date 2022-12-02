import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch,torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler
from Code import Datasets
from WorkSpace import *
from numpy import Inf
import pickle
import torch.nn as nn
from Code import Models
#from FeatureLearningRotNet.architectures.FCRNRot import FCRN_simclr
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gc

class Supervised_Learning():

    def __init__(self,args,hyperparams):

        self.architecture = args.architect
        self.hyperparams = hyperparams
        self.datasets = args.datasets
        self.criterion = args.loss
        self.affine = args.affine
        print(self.criterion)
        self.targets = args.target
        self.datasets_path = args.datasets_path
        self.wrkspace = ManageWorkSpace(datasets=self.datasets)
        self.save_path_prefix = '../models/' + self.wrkspace.map_dict['Supervised_Learning'] + '/'
        self.save_dir = self.save_path_prefix+self.architecture+args.savedir
        self.log_path = '../Logging/{}/{}/'.format(self.wrkspace.map_dict['Supervised_Learning'],self.architecture)
        self.experiment_name_postfix = args.name
        self.createDirs()
        self.joint = args.joint
        self.num_unlabeled = args.num_unlabeled
    def initModel(self):
        return  Models.FCRN(in_channels=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False) \
            if self.architecture=='FCRN' \
            else Models.UNet(n_class=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False)

    def createDirs(self):
        k_shot = [1,3,5,7,10]
        self.wrkspace.create_dir([self.save_dir,
                                  self.log_path + 'Pre-trained/Losses/',
                                  self.log_path + 'Trained/Losses/'])


    def getDataLoader(self,batchsize,dataset_collection,split=True,
                      datasets_path='',split_type='train_valid',shuffle=True,target=None):


        kwargs = {'num_workers':2, 'pin_memory': True}
        if split==True:
            traindataset_labelled = Datasets.CustomDatasetJointOpt(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                                    split_type='train_labelled', train_valid='train',
                                                                    target=target)


            traindataset_unlabelled = Datasets.CustomDatasetJointOpt(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                                    split_type='train_unlabelled', train_valid='train',
                                                                    num_unlabelled = self.num_unlabeled,
                                                                    target=target)
            print(self.num_unlabeled)

            trainloader_labelled = DataLoader(traindataset_labelled, batch_size=batchsize,shuffle=shuffle)#,**kwargs)
            trainloader_unlabelled = DataLoader(traindataset_unlabelled, batch_size=batchsize,shuffle=shuffle,**kwargs)



            val_dataset = Datasets.CustomDatasetJointOpt(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                 train_valid='valid', target=target)

            val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=shuffle)

            return trainloader_labelled,trainloader_unlabelled,val_loader


    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            try:
                ratio = float(neg_weight.item()/pos_weight.item())
            except ZeroDivisionError:
                ratio = 1
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor


    def train_joint(self, model,
              trainloader_labelled,
              trainloader_unl,
              experiment_name,
              save_every=5,save_dir=None,writer=None):


        optimizer1 = optim.Adam(params=list(model.encoder.parameters())+
                                list(model.decoder1.parameters()), lr=self.hyperparams['model_lr'],
                                weight_decay=self.hyperparams['optimizer']['weight_decay'])

        optimizer2 = optim.Adam(params=list(model.encoder.parameters())+
                                list(model.decoder2.parameters()), lr=self.hyperparams['model_lr'],
                                weight_decay=self.hyperparams['optimizer']['weight_decay'])

        train_loss = 0
        train_loss_cvs = 0
        train_iou = 0


        train_loss_epoch = []
        epochs = self.hyperparams['epochs']

        for e in range(epochs):
            model.train()
            start = time.time()

            labelled_iter = iter(trainloader_labelled)

            i = 0
            for images, _,cvs in trainloader_unl:
                images,cvs = images.cuda(), cvs.cuda()

                optimizer1.zero_grad()
                optimizer2.zero_grad()


                try:
                    images_labelled, labels_labelled,_ = next(labelled_iter)
                except StopIteration:
                    labelled_iter = iter(trainloader_labelled)
                    images_labelled, labels_labelled,_ = next(labelled_iter)
                 
               

                images_labelled, labels_labelled = images_labelled.cuda(), labels_labelled.cuda()
                output,_ = model(images_labelled)
                iou_temp, acc_temp = self.intersection_over_union(nn.Sigmoid()(output), labels_labelled)
                
                loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels_labelled))(output,labels_labelled)
                loss.backward()
                optimizer1.step()


                output_cvs,_ = model.forward(images,out_idx=2)
                loss_cvs = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(cvs))(output_cvs, cvs)

                loss_cvs.backward()
                optimizer2.step()
                train_loss_cvs += loss_cvs.item() * images.size(0)
                train_loss += loss.item() * images_labelled.size(0)
                train_iou+=iou_temp * images_labelled.size(0)
                i +=1
                torch.cuda.empty_cache()

            if (e+1)%10==0:
                image_grid = torchvision.utils.make_grid(images_labelled[:3].squeeze(), nrow=3)
                writer.add_images("Images", image_grid.unsqueeze(dim=1), e)
                output_image_grid = torchvision.utils.make_grid((nn.Sigmoid()(output[:3]).squeeze() >= 0.5).type(torch.IntTensor),
                                                                nrow=3)
                output_cvs_grid = torchvision.utils.make_grid((nn.Sigmoid()(output_cvs[:3]).squeeze() >= 0.5).type(torch.IntTensor),
                                                              nrow=3)
                groundtruth_image_grid = torchvision.utils.make_grid(labels_labelled[:3].squeeze().type(torch.IntTensor), nrow=3)
                groundtruth_cvs_grid = torchvision.utils.make_grid(cvs[:3].squeeze().type(torch.IntTensor), nrow=3)
                writer.add_images("Prediction/Binary Seg ", (output_image_grid.unsqueeze(dim=1)), e)
                writer.add_images("Prediction/Canny ",
                                  (output_cvs_grid.unsqueeze(dim=1)), e)
                writer.add_images("Groundtruth/Binary Seg ", (groundtruth_image_grid.unsqueeze(dim=1)), e)
                writer.add_images("Groundtruth/Canny ",
                                  (groundtruth_cvs_grid.unsqueeze(dim=1)), e)
                torch.save(model.state_dict(), save_dir + experiment_name + '_' + str(e + 1) + '_state_dict.pt')

            print('Epoch:{}//{}\tTrain loss: {:.4f}\tTrain IOU: {:.4f}\n'.format(e + 1, epochs,  train_loss,  train_iou))
            print("Time per epoch: ", (time.time() - start) / 60)
            writer.add_scalar("Loss/Train",train_loss,e)
            writer.add_scalar("Loss/CVS", train_loss_cvs, e)

            train_loss = 0
            train_loss_cvs = 0

            if e % save_every == 0 or e + 1 == epochs:
                checkpoint = {'epoch': e,
                              'train_loss': train_loss_epoch,
                              'optimizer': optimizer1.state_dict(),
                              'state_dict': model.state_dict()}
                torch.save(checkpoint, save_dir+experiment_name+ '_checkpoint.pt')

        return train_loss_epoch


    def train_labelled(self, model,
                    trainloader_labelled,
                    experiment_name,
                    save_every=5, save_dir=None, writer=None):

        optimizer1 = optim.Adam(params=list(model.encoder.parameters()) +
                                       list(model.decoder1.parameters()), lr=self.hyperparams['model_lr'],
                                weight_decay=self.hyperparams['optimizer']['weight_decay'])



        model.cuda()

        train_loss = 0
        train_iou = 0
        train_acc = 0

        train_loss_epoch = []
        epochs = self.hyperparams['epochs']

        for e in range(epochs):
            model.train()
            start = time.time()
            num_samples = 0
            i = 0


            for images_labelled,labels_labelled,_ in trainloader_labelled:

                optimizer1.zero_grad()

                images_labelled, labels_labelled = images_labelled.cuda(), labels_labelled.cuda()

                output, _ = model(images_labelled)
                iou_temp, acc_temp = self.intersection_over_union(nn.Sigmoid()(output), labels_labelled)

                loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels_labelled))(output, labels_labelled)
                loss.backward()
                optimizer1.step()


                train_loss += loss.item() * images_labelled.size(0)
                train_iou += iou_temp * images_labelled.size(0)
                num_samples += images_labelled.size(0)
                i += 1
                torch.cuda.empty_cache()

            if (e + 1) % 10 == 0:
                image_grid = torchvision.utils.make_grid(images_labelled[:3].squeeze(), nrow=3)
                writer.add_images("Images", image_grid.unsqueeze(dim=1), e)
                output_image_grid = torchvision.utils.make_grid(
                    (nn.Sigmoid()(output[:3]).squeeze() >= 0.5).type(torch.IntTensor),
                    nrow=3)

                groundtruth_image_grid = torchvision.utils.make_grid(
                    labels_labelled[:3].squeeze().type(torch.IntTensor), nrow=3)
                writer.add_images("Prediction/Binary Seg ", (output_image_grid.unsqueeze(dim=1)), e)

                writer.add_images("Groundtruth/Binary Seg ", (groundtruth_image_grid.unsqueeze(dim=1)), e)

                torch.save(model.state_dict(), save_dir + experiment_name + '_' + str(e + 1) + '_state_dict.pt')

            train_loss = train_loss / num_samples
            train_iou = train_iou.item() / num_samples
            print('Epoch:{}//{}\tTrain loss: {:.4f}\tTrain IOU: {:.4f}\n'.format(e + 1, epochs, train_loss, train_iou))

            print("Time per epoch: ", (time.time() - start) / 60)

            train_loss_epoch.append(train_loss)


            writer.add_scalar("Loss/Train", train_loss, e)


            train_loss = 0
            train_iou = 0
            if e % save_every == 0 or e + 1 == epochs:
                checkpoint = {'epoch': e,
                              'train_loss': train_loss_epoch,
                              'optimizer': optimizer1,
                              'state_dict': model.state_dict()}
                torch.save(checkpoint, save_dir + experiment_name + '_checkpoint.pt')

        return train_loss_epoch
    def supervised_train(self):
        for target in self.targets:

            datasets = [set for set in self.datasets if set != target]


            print("Source Datasets: {}\tTarget Datasets: {}".format(datasets,target))

            save_dir_dataset = self.save_dir+'Target_'+target+'/'
            self.wrkspace.create_dir([save_dir_dataset])
            experiment_name = 'Supervised_Learning_'+str(self.hyperparams['model_lr'])+'_modellr_'+\
                               str(self.hyperparams['epochs'])+'_epochs_'+target+self.experiment_name_postfix

            model = self.initModel()
            writer = SummaryWriter(
                log_dir='../Logging/Supervised_Pretraining/' + self.architecture + '/' + experiment_name + '/')
            trainloader_labelled, trainloader_unl, val_loader = self.getDataLoader(
                batchsize=self.hyperparams['batchsize'],
                dataset_collection=datasets, datasets_path=self.datasets_path, target=target)
            if self.joint:
                print("Joint Training with CannyEdge")
                train_loss_epoch = self.train_joint(model=model,
                                                  trainloader_labelled=trainloader_labelled,
                                                  trainloader_unl=trainloader_unl,
                                                  save_dir=save_dir_dataset,experiment_name=experiment_name,writer=writer)




            else:
                print("Labelled only")
                train_loss_epoch = self.train_labelled(model=model,
                                                        trainloader_labelled=trainloader_labelled,
                                                        save_dir=save_dir_dataset,
                                                        experiment_name=experiment_name, writer=writer)

            writer.close()
            result = [train_loss_epoch]
            self.save_results(result,descr='Pre-trained',target=target,
                              filename=experiment_name+'_pretrain_loss_Target_')


    def test(self,model,testloader):
        iou = 0
        foreground_acc = 0
        total_foreground = 0
        test_loss = 0
        model.cuda()
        model.eval()
        for child in model.children():
            if type(child) == nn.Sequential:
                for ii in range(len(child)):
                    if type(child[ii]) == nn.BatchNorm2d:
                        child[ii].track_running_stats = False

        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output,_ = model(images)

            loss = nn.BCELoss()(output, labels)
            test_loss += loss.item() * images.size(0)
            iou_temp, acc_temp= self.intersection_over_union(output, labels)
            iou+=iou_temp
            foreground_acc+=acc_temp
            total_foreground+=torch.sum(labels==1).item()
        test_loss = test_loss / len(testloader.dataset)
        iou = iou.item() / len(testloader.dataset)
        foreground_acc = torch.sum(foreground_acc).item()/total_foreground
        print('Test Loss: {:.6f} \tTest IOU={:.6f}\tTest FCA={:.6f}\n'.format(test_loss, iou,foreground_acc))
        return test_loss,iou,foreground_acc



    def intersection_over_union(self, tensor, labels, device=torch.device("cuda:0")):
        iou = 0
        foreground_acc = 0

        labels_tens = labels.type(torch.BoolTensor)
        ones_tens = torch.ones_like(tensor, device=device)
        zeros_tens = torch.zeros_like(tensor, device=device)
        if tensor.shape[0] > 1:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum(
                (1, 2))

            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum((1, 2))
            iou += torch.mean((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens
        else:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum()
            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum()
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens

        del temp_tens
        del labels_tens
        del ones_tens
        del zeros_tens
        torch.cuda.empty_cache()
        total_iou = iou
        return total_iou, foreground_acc
    def save_results(self,results,target,descr='',filename=''):
        f_csv = open(self.log_path+descr+'/Losses/'+ filename + target + '.csv', 'w')
        f_pickle = open(self.log_path + descr + '/Losses/' + filename + target + '.pickle', 'wb')
        pickle.dump(results,f_pickle)
        df = pd.DataFrame(results)
        df.to_csv(f_csv,header=False,index=False)
        f_csv.close()
        f_pickle.close()


