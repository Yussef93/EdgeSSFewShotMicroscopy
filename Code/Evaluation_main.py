

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code import Evaluation

import torch
import numpy as np

def addEvaluationArgs():
    parser = argparse.ArgumentParser(description="Evaluation Arguments")
    parser.add_argument("--finetune", type=int, default=1)
    parser.add_argument("--testfinetune", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--switchaffine", type=int, default=0)
    parser.add_argument("--targets",type=str,nargs="*",default=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                        help="Combination of B5,B39,TNBC,ssTEM,EM")
    parser.add_argument("--architect",type=str,default='FCRN',help="Enter FCRN or UNet")

    parser.add_argument("--eval-selections",type=int,nargs="*",default=list(range(1,11)),
                        help="Up to 10 selections")


    parser.add_argument("--lr", type=float, default=0.001,
                        help="Pre-trained learning rate")

    parser.add_argument("--finetune-lr", type=float, default=0.1,
                        help="Finetune learning rate")
    parser.add_argument("--finetune-loss", type=str, default="bce",
                        help="Binary Cross entropy Loss (BCE) function or Weighted BCE (weightedbce)")
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--statedictepoch', type=int, default=None,help="Load saved parameters from pre-training epoch #")
    parser.add_argument('--num-shots', type=int,nargs="*",default=[1,3,5,7,10])
    parser.add_argument("--pretrained-name", type=str, default='',
                        help="model name to be finetuned and evaluated")

    parser.add_argument("--finetune-name", type=str, default='',
                        help="finetuned model name")
    return parser



def main():
   torch.manual_seed(0)
   np.random.seed(0)
   torch.backends.cudnn.deterministic = True
   parser = addEvaluationArgs()
   args = parser.parse_args()
   print(args)


   args.supervised_params = {'model_lr': str(args.lr),
                        'epochs': '51'}

   batchsize_testset = {'TNBC': 32,
                        'B39': 32,
                        'ssTEM': 32,
                        'EM': 20,
                        'B5': 32}

   evaluation_config = {'targets': args.targets,
                        'selections': args.eval_selections,
                        'k-shot': args.num_shots,
                        'batchsize_ftset': 64,
                        'batchsize_testset': batchsize_testset,
                        'ft_lr': args.finetune_lr,
                        'ft_epochs': args.finetune_epochs,
                        'optimizer': {'weight_decay': 0.0005,
                                      'momentum': 0.9, },
                        'Finetune': bool(args.finetune),
                        'Test_Finetuned':bool(args.testfinetune)}


   evaluation = Evaluation.Evaluation(args=args, evaluation_config=evaluation_config)
   print("Evaluating on Selections: ",args.eval_selections)

   evaluation.evaluate_transfer_learning()




if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()

