

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code import Supervised_Learning
import torch
import numpy as np
def addLearningArgs():

    parser = argparse.ArgumentParser(description="Train with a Learning method")
    parser.add_argument("--loss", type=str, default="weightedbce",
                        help="standard BCE Loss function or weighted BCE Loss")
    parser.add_argument('--savedir', type=str, default='/Pre-trained/' )
    parser.add_argument('--datasets', type=str, nargs="*", default=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                        help="Combination of B5,B39,TNBC,ssTEM,EM")
    parser.add_argument('--architect', type=str, default='FCRN', help="Enter FCRN")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--num_unlabeled", type=float, default=0.3)
    parser.add_argument("--joint", type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100, help="# of Training epochs in supervised-learning")
    parser.add_argument('--target', type=str, nargs="*", default=['TNBC'],
                        help="Define Target dataset in leave-out-one-dataset cross validation")

    parser.add_argument('--name',type=str,default='test')

    return parser

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    parser = addLearningArgs()
    args = parser.parse_args()
    print(args)

    hyperparams = {'model_lr': args.lr,
                   'epochs': args.epochs,
                   'batchsize': 64,
                   'optimizer': {'weight_decay': 0.0005,
                                 'momentum': 0.9}
                   }


    args.datasets_path = '../Datasets/FewShot/Microscopy/Source/'


    supervised_learn = Supervised_Learning.Supervised_Learning(args=args,hyperparams=hyperparams)

    supervised_learn.supervised_train()

if __name__ =='__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()
