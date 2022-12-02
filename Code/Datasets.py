from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import cv2

class CustomDataset(Dataset):
    def __init__(self, root_dir,
                 dataset_selection, split=False, train_valid='train', target=None,k_shot=None,
                 split_type='train_valid',transform=None):
        self.root_dir = root_dir
        self.selection = dataset_selection
        self.target = target
        self.k_shot = k_shot
        self.transform = transform
        self.split = split
        self.split_type = split_type
        self.train_valid = train_valid
        self.ground_truth_train = []
        self.ground_truth_valid = []
        self.ground_truth_test = []
        self.images_train = []
        self.images_valid = []
        self.images_test = []

        for set in dataset_selection:
            ground_truth_prefix = set+'/Groundtruth/'
            image_prefix = set+'/Image/'

            if self.split == True:

                if self.split_type == 'train_valid':
                    file_names_g = sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
                    file_names_i = sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(file_names_g)]

                    f = open('/path/to/FewShot/Microscopy/Train_Valid_1000/valid_ids_{}.pickle'.format(set), 'rb')
                    valid_samples = pickle.load(f)
                    f.close()
                    for i in range(len(file_names_g)):
                        if i in valid_samples:
                            self.ground_truth_valid.append(file_names_g[i])
                            self.images_valid.append(file_names_i[i])

                        else:
                            self.ground_truth_train.append(file_names_g[i])
                            self.images_train.append(file_names_i[i])

                elif self.split_type == 'train_valid_test':
                    self.ground_truth_train =  sorted([self.root_dir + 'Train/' + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + 'Train/' + ground_truth_prefix) if f[0] != '.'])
                    self.images_train = sorted([self.root_dir + 'Train/' +image_prefix + f for f in os.listdir(self.root_dir +'Train/' + image_prefix)
                                                if f[0] != '.'])[:len(self.ground_truth_train)]

                    self.ground_truth_valid = sorted([self.root_dir + 'Valid/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Valid/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_valid = sorted([self.root_dir + 'Valid/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Valid/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_valid)]

                    self.ground_truth_test = sorted([self.root_dir + 'Test/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Test/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_test = sorted([self.root_dir + 'Test/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Test/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_test)]


            else:
                self.ground_truth_train +=  sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
                self.images_train +=  sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(self.ground_truth_train)]


    def selectKshots(self, splitFactor=1):


        dataset_size = len(self.ground_truth_train)
        indicies = list(range(0, len(self.ground_truth_train)))
        np.random.shuffle(indicies)
        samples = np.random.choice(indicies,self.k_shot if self.k_shot!=None else dataset_size)
        dataSplit = int(np.floor((1-splitFactor) * len(samples)))
        _,trainIdx = samples[:dataSplit], samples[dataSplit:]


        return trainIdx


    def __len__(self):
        if self.split:
            if self.train_valid == 'train':
                return(len(self.ground_truth_train))
            elif self.train_valid == 'valid':
                return(len(self.ground_truth_valid))
            elif self.train_valid == 'test':
                return(len(self.ground_truth_test))
        else:
            return(len(self.ground_truth_train))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split:
            if self.train_valid == 'train':
                image = Image.open(self.images_train[idx])
                ground_truth = Image.open(self.ground_truth_train[idx])
            elif self.train_valid == 'valid':
                image = Image.open(self.images_valid[idx])
                ground_truth = Image.open(self.ground_truth_valid[idx])
            elif self.train_valid == 'test':
                image = Image.open(self.images_test[idx])
                ground_truth = Image.open(self.ground_truth_test[idx])

        else:
            image = Image.open(self.images_train[idx])
            ground_truth = Image.open(self.ground_truth_train[idx])


        cvs = canny_edge_detector(image)

        if self.transform:
            image = self.transform(image)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],std=[0.5])])
        ground_truth = transforms.ToTensor()(ground_truth)
        cvs = transforms.ToTensor()(cvs)
        image = transform(image)



        return image,ground_truth,cvs


class CustomDatasetJointOpt(Dataset):
    def __init__(self, root_dir, dataset_selection, split=False, train_valid='train',
                 target=None,k_shot=None,num_unlabelled=0.3, consistency=False,
                 split_type='',transform=None):
        self.root_dir = root_dir
        self.selection = dataset_selection
        self.consistency = consistency
        self.target = target
        self.k_shot = k_shot
        self.transform = transform
        self.split = split
        self.split_type = split_type
        self.train_valid = train_valid
        self.ground_truth_train_unlabelled = []
        self.ground_truth_train_labelled = []
        self.ground_truth_train = []
        self.ground_truth_valid = []
        self.images_train_labelled = []
        self.images_train_unlabelled = []
        self.images_train = []
        self.images_valid = []


        for set in dataset_selection:
            ground_truth_prefix = set+'/Groundtruth/'
            image_prefix = set+'/Image/'


            file_names_g = sorted([self.root_dir + ground_truth_prefix + f for f in
                                   os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
            file_names_i = sorted(
                [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                 f[0] != '.'])[:len(file_names_g)]

            f = open('../Datasets/FewShot/Microscopy/Train_Valid_1000/valid_ids_{}.pickle'.format(set), 'rb')
            valid_samples = pickle.load(f)
            f.close()
            for i in range(len(file_names_g)):
                if i in valid_samples:
                    self.ground_truth_valid.append(file_names_g[i])
                    self.images_valid.append(file_names_i[i])

                else:
                    self.ground_truth_train.append(file_names_g[i])
                    self.images_train.append(file_names_i[i])

            if os.path.isfile('../Datasets/FewShot/Microscopy/Train_Valid_1000/10percent_ids_{}.pickle'.format(set)):
                f = open('../Datasets/FewShot/Microscopy/Train_Valid_1000/10percent_ids_{}.pickle'.format(set), 'rb')
                indices = pickle.load(f)
                f.close()
            else:

                indices = list(range(0,len(self.ground_truth_train)))
                np.random.shuffle(indices)

                f = open('../Datasets/FewShot/Microscopy/Train_Valid_1000/10percent_ids_{}.pickle'.format(set), 'wb')
                pickle.dump((indices), f)
                f.close()
            labelled_samples = indices
            unlabelled_samples = indices[:int(num_unlabelled*(len(self.ground_truth_train)))]
            for i in range(len(self.ground_truth_train)):
                if i in labelled_samples:
                    self.ground_truth_train_labelled.append(self.ground_truth_train[i])
                    self.images_train_labelled.append(self.images_train[i])
                if i in unlabelled_samples:
                    self.ground_truth_train_unlabelled.append(self.ground_truth_train[i])
                    self.images_train_unlabelled.append(self.images_train[i])
            self.ground_truth_train = []
            self.images_train = []



    def __len__(self):

        if self.train_valid == 'train':
            if self.split_type=='train_labelled':
                return(len(self.ground_truth_train_labelled))
            elif self.split_type=='train_unlabelled':
                return (len(self.ground_truth_train_unlabelled))

        elif self.train_valid == 'valid':
            return(len(self.ground_truth_valid))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train_valid == 'train':
            if self.split_type=='train_labelled':
                image = Image.open(self.images_train_labelled[idx])
                ground_truth = Image.open(self.ground_truth_train_labelled[idx])
            elif self.split_type =='train_unlabelled':
                image = Image.open(self.images_train_unlabelled[idx])
                ground_truth = Image.open(self.ground_truth_train_unlabelled[idx])

        elif self.train_valid == 'valid':
            image = Image.open(self.images_valid[idx])
            ground_truth = Image.open(self.ground_truth_valid[idx])


        cvs = canny_edge_detector(image)
        ground_truth = transforms.ToTensor()(ground_truth)
        cvs = transforms.ToTensor()(cvs)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])

        image = transform(image)

        return image,ground_truth,cvs



def canny_edge_detector(image):

    gray = np.array(image)

    gray_blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    filtered_blurred = cv2.Canny(gray_blurred, 30, 100)
    _, label = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)
    label[label > 0] = 255
    return Image.fromarray(label)


