'''
the main function of fraud detection module.
1. select used model and set parameters of model here
'''

import argparse
from train import ModelTrainer, EnsembleModels
import warnings


def main(args):
    # TODO add GNNexplainer
    # TODO tranfer ges graph to dgl

    if args['ensemble']:
        trainer = EnsembleModels(args)
        trainer.ensemble_learning(n_jobs=args['n_jobs'])
        acc, f1, recall = trainer.predict()
        print('Final, ACC:{}, F1:{}, Recall:{}'.format(acc, f1, recall))
    else:
        trainer = ModelTrainer(args)
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate of model')
    parser.add_argument('--num_layer_gnn', type=int, default=3, help='the number of layers for gnn model (including '
                                                                     'input and output layers)')

    parser.add_argument('--gpu', type=int, default=0, help='the used gpu device')
    parser.add_argument('--model_name', type=str, default='gcn',
                        choices=['gcn', 'gcn_vae', 'random forest', 'decision tree', 'lr'],
                        help='the model name [gcn, gcn_vae, random forest, decision tree, gcn_with_fs, lr, lr_with_fs]')
    parser.add_argument('--num_layer_mlp', type=int, default=3, help='the number of layers for mlp')
    parser.add_argument('--graph_explainer', type=bool, default=False, help='explaine results predicted from the model')
    parser.add_argument('--saved_path', type=str, default='ckpt/', help='explaine results predicted from the model')
    parser.add_argument('--num_class', type=int, default=4, help='the number of classes for this task')
    parser.add_argument('--class_weight', type=int, default=4, help='the weight of classes')
    parser.add_argument('--max_depth', type=int, default=None, help='the number of classes for this task')
    parser.add_argument('--dataset', type=str, default='dblp', help='dataset')
    parser.add_argument('--num_tree', type=int, default=100, help='the number of tree for tree-based model')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer of neural network')
    parser.add_argument('--ensemble', type=bool, default=True, help='whether using the ensemble models')
    parser.add_argument('--group_models', type=list, default=['gcn', 'decision_tree', 'mlp'], help='the ensemble models')
    parser.add_argument('--self_loop', type=bool, default=True, help='self loop edge of graph')
    parser.add_argument('--early_stop', type=int, default=-1, help='set the number of epoch for early stop')
    parser.add_argument('--epochs', type=int, default=1000, help='set the number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden dim of node feature')
    parser.add_argument('--preprocess', type=int, default=300, help='pre-process the data')
    parser.add_argument('--eval-step', type=int, default=5, help='evaluate step')
    parser.add_argument('--batch-size', type=int, default=64, help='evaluate step')
    parser.add_argument('--parallel', type=bool, default=True, help='evaluate step')
    parser.add_argument('--n-jobs', type=int, default=3, help='evaluate step')
    parser.add_argument('--feat-selection', type=bool, default=True, help='evaluate step')
    # parser.add_argument('')
    args = parser.parse_args().__dict__
    main(args)
