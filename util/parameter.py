import os
from datetime import datetime

def add_default_args(parser):
    parser.add_argument("--seed", default=39, type=int)
    parser.add_argument("--train", default=False, action='store_true')
    parser.add_argument("--dev", default=False, action='store_true')
    parser.add_argument("--test", default=False, action='store_true')
    
    
    parser.add_argument("--criterion", default='bce', type=str)
    parser.add_argument("--direct", default=False, action='store_true')
    
    parser.add_argument("--model-path", default='', type=str)

    return parser

def add_model_args(parser):
    parser.add_argument("--vocab_size", default=46098, type=int)
    parser.add_argument("--emb_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--memory_dim", default=512, type=int)
    parser.add_argument("--word_drop_rate", type=float, default=0.2)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--d_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.001)

    return parser


def add_optim_args(parser):

    #############
    # optimizer #
    #############
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    return parser



def add_trainer_args(parser):

    #############
    # iteration #
    #############
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--batchsize", default=4, type=int)
    parser.add_argument("--accumulate", default=8, type=int)
    parser.add_argument("--lr_step", default=10, type=int)
    parser.add_argument("--lr_gamma", default=0.8, type=float)
    parser.add_argument("--grad_clip", default=10, type=float)
    parser.add_argument("--save-path", required=True, type=str)
    
    
    return parser


def add_dataset_args(parser):
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--pair-path", required=True, type=str)
    parser.add_argument("--pre", required=True, type=str)
    parser.add_argument("--vocab", required=True, type=str)
    parser.add_argument("--total", default=False, action='store_true')

    return parser
