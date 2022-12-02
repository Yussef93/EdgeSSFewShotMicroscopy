
import sys
import os
from WorkSpace import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pickle
from Code import Models, Datasets
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import sklearn
class Evaluation():

    def __init__(self, args,evaluation_config):

        self.wrkspace = ManageWorkSpace(datasets=evaluation_config['targets'])
        self.supervised_params = args.supervised_params
        self.lr_method = 'Supervised_Learning'
        self.state_dict_epoch = args.statedictepoch
        self.evaluation_config = evaluation_config
        self.architecture = args.architect
        self.criterion = args.finetune_loss
        self.affine = args.affine
        self.switch_affine = args.switchaffine
        self.few_shot_target_dir = '../Datasets/FewShot/Microscopy/Target/'
        self.pretrained_experiment_name_postfix = args.pretrained_name
        self.ft_experiment_name_postfix = args.finetune_name

        self.createFineTuneDirs()
    def createFineTuneDirs(self):
        """
        create main dirs for storing fine-tuned models and results

        """
        map_lr_method = self.wrkspace.map_dict[self.lr_method]
        models_save_dir = '../models/' + map_lr_method + '/' + self.architecture + '/'
        logging_save_dir = '../Logging/' + map_lr_method + '/' + self.architecture + '/'
        self.wrkspace.create_dir([models_save_dir + 'Fine-tuned/',
                                  logging_save_dir + 'Fine-tuned/'])



    def createEvaluationTransferDirs(self):
        """
        create sub dirs for saving transfer learning models and results

        """

        model_save_dir = '../models/Supervised-models/' + self.architecture + '/'
        logging_save_dir = '../Logging/Supervised-models/' + self.architecture + '/'

        for k_shot in self.evaluation_config['k-shot']:
            for target in self.evaluation_config['targets']:
                self.wrkspace.create_dir([model_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/Target_' + target + '/',
                                          logging_save_dir + 'Fine-tuned/' + str(
                                              k_shot) + '-shot/Target_' + target + '/FT_Loss_IoU/',
                                          logging_save_dir + 'Fine-tuned/' + str(
                                              k_shot) + '-shot/Target_' + target + '/Test_Loss_IoU/'])
        return model_save_dir, logging_save_dir

    def initModel(self):
        """
        creates an instance of deep neural network acrhitecture (FCRN or UNet)
        :return: randomly initilaized network
        """
        return Models.FCRN(in_channels=1, affine=self.affine, sigmoid=True if self.criterion == 'bce' else False) \
            if self.architecture == 'FCRN' \
            else Models.UNet(n_class=1, affine=self.affine, sigmoid=True if self.criterion == 'bce' else False)

    def load_model_state_dict(self, state_dict_path,epoch=None):
        #state_dict_path = '/home/dawoud/PycharmProjects/MICCAI/models/Meta-models/FCRN/Pre-trained/BCE/Target_TNBC/Meta_Learning_BCE_1.0meta_lr_0.001modellr_700meta_epochs_30inner_epochs_5shot_TNBCnew_source_preprocessing_batchnorm_normal_init_relu_act_weightedbce_metaupdatecorrect_adam_opt/State_Dict/Meta_Learning_BCE_1.0meta_lr_0.001modellr_700meta_epochs_30inner_epochs_5shot_TNBCnew_source_preprocessing_batchnorm_normal_init_relu_act_weightedbce_metaupdatecorrect_adam_opt_300_state_dict.pt'
        #state_dict_path = '/home/dawoud/PycharmProjects/MICCAI/models/Meta-models/FCRN/Pre-trained/BCE/Target_TNBC/Meta_Learning_BCE_1.0meta_lr_0.001modellr_700meta_epochs_30inner_epochs_5shot_TNBCnew_source_preprocessing_batchnorm_normal_init_relu_act_weightedbce_metaupdatecorrect_adam_opt/State_Dict/Meta_Learning_BCE_1.0meta_lr_0.001modellr_700meta_epochs_30inner_epochs_5shot_TNBCnew_source_preprocessing_batchnorm_normal_init_relu_act_weightedbce_metaupdatecorrect_adam_opt'
        """
        :param state_dict_path: path to saved pre-trained parameters
        :param epoch: load saved pre-trained parameters from training epoch #
        :return: pre-trained model
        """
        model = self.initModel()

        if epoch==None:
            model.load_state_dict(torch.load(state_dict_path + '_state_dict.pt'))
        else:
            model.load_state_dict(torch.load(state_dict_path +'_' +str(epoch) +'_state_dict.pt'))

        return model

    def load_model_checkpoint(self,state_dict_path):
        model = self.initModel()
        model.load_state_dict(torch.load(state_dict_path + '_checkpoint.pt')['state_dict'])
        return model
    def getFTandTestLoader(self, selection_ft_path,
                           selection_test_path,
                           selection_aug_path,
                           batchsize_ftset,
                           batchsize_testset, dataset):

        """

        :param selection_ft_path: path to fine-tune (few-shot) samples from target selection #
        :param selection_test_path: path to test samples from target selection #
        :param batchsize_ftset: batch size of fine-tuning set
        :param batchsize_testset: batch size of test set
        :param dataset: target data set
        :return: finetuneloader, testloader
        """
        finetune_set = Datasets.CustomDataset(root_dir=selection_ft_path, dataset_selection=[dataset])
        testset = Datasets.CustomDataset(root_dir=selection_test_path, dataset_selection=[dataset])
        finetuneloader = DataLoader(finetune_set, batch_size=batchsize_ftset, shuffle=True)
        testloader = DataLoader(testset, batch_size=batchsize_testset[dataset])
        testloader_aug = None
        return finetuneloader, testloader, testloader_aug

    def getExperimentName(self, k_shot='', target='', descr='finetuned',
                          selection='', lr_method=None):


        prefix = lr_method + '_' + descr + '_' + self.supervised_params['model_lr'] + 'model_lr_' + \
                 self.supervised_params['epochs'] + '_epochs_' + self.ft_experiment_name_postfix


        if descr == 'finetuned':
            prefix = prefix + '_' + str(self.evaluation_config['ft_epochs']) + '_ft_epochs_' + \
                     str(self.evaluation_config['ft_lr']) + 'ft_lr'



        experiment_name = prefix + str(k_shot) + 'shot_' + target + '_Selection_' + str(selection)

        return experiment_name, prefix



    def getTargetandFTDir(self, selection_dir, k_shot):
        target_ft_dir = selection_dir + 'FinetuneSamples/' + str(k_shot) + '-shot/preprocessed/'
        target_test_dir = selection_dir + 'TestSamples/' + str(k_shot) + '-shot/'
        target_test_cropped = selection_dir + 'TestSamples_cropped/' + str(k_shot) + '-shot/'
        return target_ft_dir, target_test_dir,target_test_cropped

    def swithBatchNormAffine(self,model):

        for k,m in enumerate(model.encoder.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                model.encoder[k-1] = nn.BatchNorm2d(m.num_features, affine=True)

        for k, m in enumerate(model.decoder1.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                model.decoder1[k-1] = nn.BatchNorm2d(m.num_features, affine=True)

        for k, m in enumerate(model.decoder2.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                model.decoder2[k-1] = nn.BatchNorm2d(m.num_features, affine=True)

        for k,m in enumerate(model.encoder.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                Models.init.constant_(m.weight, 0.1)
                Models.init.constant_(m.bias, 0)

        for k,m in enumerate(model.decoder1.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                Models.init.constant_(m.weight, 0.1)
                Models.init.constant_(m.bias, 0)

        for k,m in enumerate(model.decoder2.modules(),0):
            if isinstance(m, nn.BatchNorm2d):
                Models.init.constant_(m.weight, 0.1)
                Models.init.constant_(m.bias, 0)

        return model



    def evaluate_transfer_learning(self):
        """
           Evaluate supervised-trained models by fine-tuning on samples from target and then testing
           :return: average IoU over selections
         """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        model_save_dir, logging_save_dir = self.createEvaluationTransferDirs()

        pre_trained_model_target_dir = model_save_dir + 'Pre-trained/Target_'
        pre_train_lr, pre_train_epochs = self.supervised_params['model_lr'], self.supervised_params['epochs']
        logger.info("Network:{}".format(self.architecture))

        for selection in self.evaluation_config['selections']:
            selection_dir = self.few_shot_target_dir + 'Selection_' + str(selection) + '/'
            for target in self.evaluation_config['targets']:

                for k_shot in self.evaluation_config['k-shot']:
                    model_pre_train_state_dict_dir = pre_trained_model_target_dir + target + '/Supervised_Learning_' + \
                                                     pre_train_lr + '_modellr_' + pre_train_epochs + '_epochs_' + target
                    model_pretrained = self.load_model_state_dict(state_dict_path=model_pre_train_state_dict_dir +
                                                                                  self.pretrained_experiment_name_postfix,
                                                                  epoch=self.state_dict_epoch)

                    if self.evaluation_config['Finetune'] and self.switch_affine:
                        logger.info("Switch Affine to True")
                        model_pretrained = self.swithBatchNormAffine(model_pretrained)

                        # print(model_pretrained)
                    else:
                        logger.info("Switch Affine to False")
                    logger.info(
                        '{} Evaluation\nTransfer Learning\nSelection: {}\tTarget: {}\tFine-tune Shots: {}'.format(
                            self.lr_method, selection, target, k_shot))
                    target_ft_dir, target_test_dir, target_test_cropped = self.getTargetandFTDir(k_shot=k_shot,
                                                                                                 selection_dir=selection_dir)

                    finetuneloader, testloader, \
                    testloader_aug = self.getFTandTestLoader(selection_ft_path=target_ft_dir,
                                                             selection_test_path=target_test_dir,
                                                             selection_aug_path=target_test_cropped,
                                                             batchsize_ftset=self.evaluation_config[
                                                                 'batchsize_ftset'],
                                                             batchsize_testset=self.evaluation_config[
                                                                 'batchsize_testset'],
                                                             dataset=target)


                    experiment_name, _ = self.getExperimentName(k_shot=k_shot, target=target,selection=selection,
                                                                lr_method='Transfer_Learning')

                    writer = SummaryWriter(log_dir=os.getcwd()+'/Logging/Transfer_Learning/' + self.architecture + '/' +experiment_name + '/')
                    model_finetuned_state_dict_dir = model_save_dir + '/Fine-tuned/' + str(k_shot) + \
                                                     '-shot/' + 'Target_' + target + '/' + experiment_name

                    logging_ft_prefix = logging_save_dir + 'Fine-tuned/' + str(k_shot) + \
                                                        '-shot/Target_' + target + '/'


                    if self.evaluation_config['Finetune']:

                        finetune_loss,finetune_iou = self.finetune(finetuneloader=finetuneloader, model=model_pretrained,
                                                                   save_path=model_finetuned_state_dict_dir, logger=logger,
                                                                   writer=writer,testloader=testloader)
                        self.save_result(logging_dir=logging_ft_prefix + 'FT_Loss_IoU/',
                                         result=[finetune_loss, finetune_iou],
                                         experiment_name=experiment_name)

                    if self.evaluation_config['Test_Finetuned']:
                        if self.switch_affine:
                            self.affine = True
                        model = self.load_model_state_dict(state_dict_path=model_finetuned_state_dict_dir)
                        logging_save_model_target_dir = logging_ft_prefix+'Test_Loss_IoU/'

                        print("---Testing---")
                        test_loss, test_iou, test_acc = self.test(model, testloader)
                        test_result = [test_loss, test_iou, test_acc]
                        if self.switch_affine:
                            self.affine = False
                        self.save_result(result=test_result, logging_dir=logging_save_model_target_dir,
                                         experiment_name=experiment_name)

            time.sleep(2)

        _, prefix = self.getExperimentName(lr_method='Transfer_Learning')

    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            ratio = float(neg_weight.item()/pos_weight.item())
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor

    def finetune(self, finetuneloader, model, save_path, logger=None,writer=None ):
        finetune_loss = 0
        iou_finetune = 0
        acc_finetune = 0
        total_foreground = 0
        finetune_loss_epoch = []
        finetune_iou_epoch = []
        num_samples = 0


        ft_epochs = self.evaluation_config['ft_epochs']
        temp = self.evaluation_config['ft_lr']


        optimizer1 = optim.Adam(list(model.encoder.parameters())+list(model.decoder1.parameters()),lr=self.evaluation_config['ft_lr'],
                                weight_decay=self.evaluation_config['optimizer']['weight_decay'])


        model.cuda()
        for e in range(ft_epochs):
            model.train()
            for images, labels,cvs in finetuneloader:
                images, labels,cvs = images.cuda(), labels.cuda(), cvs.cuda()
                optimizer1.zero_grad()
                output, _ = model(images)

                iou_temp, intersection_temp,union_temp,acc_temp = self.intersection_over_union(nn.Sigmoid()(output), labels)
                if self.criterion =='bce':
                    loss = nn.BCELoss()(output, labels)
                else:
                    loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output, labels)

                loss.backward()
                optimizer1.step()

                finetune_loss += loss.item() * images.size(0)
                iou_finetune += iou_temp.item()* images.size(0)
                acc_finetune += torch.sum(acc_temp).item()

                total_foreground += torch.sum(labels == 1).item()
                num_samples += images.size(0)
                torch.cuda.empty_cache()
            finetune_loss = finetune_loss / len(finetuneloader.dataset)
            iou_finetune = iou_finetune / len(finetuneloader.dataset)
            acc_finetune = acc_finetune / total_foreground
            logger.info('Epoch:{}//{} \tTrain loss: {:.4f}\tTrain IOU: {:.4f}\t FCA: {:.4f}'.format(e + 1, ft_epochs,
                                                                                                   finetune_loss,iou_finetune,
                                                                                                   acc_finetune))

            if writer !=None:
                writer.add_scalar('Finetune/Loss ',finetune_loss,e)
                writer.add_scalar('Finetune/IoU ' ,iou_finetune, e)
                writer.add_scalar('Finetune/FCA ', acc_finetune, e)

            finetune_loss_epoch.append(finetune_loss)
            finetune_iou_epoch.append(iou_finetune)
            finetune_loss = 0
            acc_finetune = 0
            if (e+1)%20==0:
                output_image_grid = torchvision.utils.make_grid((output[:3].squeeze()>=0.5).type(torch.IntTensor),nrow=3)

                groundtruth_image_grid = torchvision.utils.make_grid(labels[:3].squeeze().type(torch.IntTensor),nrow=3)
                groundtruth_cvs_grid = torchvision.utils.make_grid(cvs[:3].squeeze().type(torch.IntTensor),nrow=3)
                writer.add_images("Prediction/Binary Seg ",(output_image_grid.unsqueeze(dim=1)),e)

                writer.add_images("Groundtruth/Binary Seg ", (groundtruth_image_grid.unsqueeze(dim=1)),e)
                writer.add_images("Groundtruth/Canny ",
                                  (groundtruth_cvs_grid.unsqueeze(dim=1)), e)
            total_foreground = 0
            num_samples = 0
            iou_finetune = 0
            #logger.info('Epoch:{}//{} \tBest test IOU: {:.4f}\t '.format(best_ft_epoch, ft_epochs, test_iou_best))
            if e + 1 == ft_epochs:
                torch.save(model.state_dict(), save_path + '_state_dict.pt')
        self.evaluation_config['ft_lr']=temp
        return finetune_loss_epoch, finetune_iou_epoch




    def test(self, model, testloader):
        iou = 0
        acc = 0

        total_foreground = 0
        test_loss = 0
        model.eval()
        model.cuda()
        for child in model.children():
            if type(child) == nn.Sequential:
                for ii in range(len(child)):
                    if type(child[ii]) == nn.BatchNorm2d:
                        child[ii].track_running_stats = False

        test_start = time.time()
        for images, labels,_ in testloader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output, _ = model(images)
            output=nn.Sigmoid()(output) if self.criterion!='bce' else output
            loss = nn.BCELoss()(output, labels)
            test_loss += loss.item() * images.size(0)
            iou_temp, intersection_temp, union_temp, acc_temp = self.intersection_over_union(output, labels)
            iou += iou_temp.item()* images.size(0)

            acc += torch.sum(acc_temp).item()
            total_foreground += torch.sum(labels == 1).item()
        test_end = time.time()
        test_loss = test_loss / len(testloader.dataset)
        iou = iou / len(testloader.dataset)

        acc = acc / total_foreground
        print('Test Loss: {:.4f} \tTest IOU: {:.4f}\tFCA: {:.4f}\tTest Time: {:.3f} min\n'.format(test_loss, iou, acc,
                                                                                                  (test_end - test_start) / 60))

        return test_loss, iou, acc


    def intersection_over_union(self, tensor, labels, device=torch.device("cuda:0")):
        iou = 0
        foreground_acc = 0

        labels_tens = labels.type(torch.BoolTensor)
        ones_tens = torch.ones_like(tensor, device=device)
        zeros_tens = torch.zeros_like(tensor, device=device)
        if tensor.shape[0] > 1:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum((1, 2))

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
        return total_iou,torch.sum(intersection_tens).item(),torch.sum(union_tens).item(), foreground_acc

    def save_result(self, logging_dir, result, experiment_name):
        f_pickle = open(logging_dir + experiment_name + '.pickle', 'wb')
        f_csv = open(logging_dir + experiment_name + '.csv', 'w')
        pickle.dump(result, f_pickle)
        df = pd.DataFrame(result)
        df.to_csv(f_csv, header=False, index=False)
        f_pickle.close()
        f_csv.close()



