import time
import numpy as np
import os
from sklearn.metrics import f1_score
import argparse
import collections
import random
import sys

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataloader import persuasiveDataset, persuasive_collate_fn
from model.model import DTDMN
from util.parameter import (add_default_args, add_model_args, add_optim_args, add_trainer_args, add_dataset_args)

from util.loss import (hinge)
import util.criterion as criterions
#from optim import RAdam, Ranger


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    parser = add_model_args(parser)
    parser = add_optim_args(parser)
    parser = add_trainer_args(parser)
    parser = add_dataset_args(parser)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

def check_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)

def build_data(args, persuasive_path, vocab):
    batch_size = args.batchsize

    # need to check
    num_worker=6
    persuasive = []
    data_path, pair_path = persuasive_path
    if(args.total):
        print('use full data to train')
        temp = [('', '_train', True, batch_size), ('', '_dev', False, batch_size*4), ('_test', '_test', False, batch_size*4)]
    else:
        temp = [('', '_train', True, batch_size), ('', '_dev', False, batch_size*4), ('_test', '_test', False, batch_size*4)]

    for dtype, pair_type, shuffle, b in temp:
        dataset = persuasiveDataset(data_path=data_path+dtype, pair_path=pair_path+pair_type, vocab=vocab['bow'], train=False)
        dataloader = DataLoader(dataset, batch_size=b, shuffle=shuffle, num_workers=num_worker, collate_fn=persuasive_collate_fn)
        persuasive.append(dataloader)
    
    return persuasive

def build_vocab(args):
    vocab = np.load(args.vocab, allow_pickle=True).item()
    args.vocab_size = len(vocab)
    return vocab

def convert(data, device):
    if(isinstance(data, dict)):
        temp = {}
        for key in data:
            try:
                temp[key] = data[key].to(device)
            except:
                pass
        return temp
    elif(isinstance(data, list)):
        return [_.to(device) for _ in data]

def update(buffer=None, in_data=None, out_data=None, criterion=None, alpha=0, dtype=None):
    if((in_data is not None) and (out_data is not None)):
        temp = collections.defaultdict(list)
        for i, out in enumerate(out_data):
            vae_d_resp, vae_t_resp, dec_out, _ = out

            temp['recon'].append(dec_out.softmax(-1).cpu())
            temp['qd'].append(vae_d_resp.x_out.softmax(-1).cpu())
            temp['qt'].append(vae_t_resp.c_out.softmax(-1).cpu())
            temp['qd_logits'].append(vae_d_resp.qd_logits.softmax(-1).cpu())
            temp['bow'].append( in_data[i]['bow'].cpu() )
            temp['t_mu'].append(vae_t_resp.z_mu.cpu())
            temp['t_logvar'].append(vae_t_resp.z_logvar.cpu())

        temp['persuasive'].append( (out_data[1][-1]-out_data[0][-1]).cpu().view(-1) )

        for key in ['recon', 't_mu', 't_logvar', 'qd', 'qt', 'qd_logits', 'bow', 'persuasive']:
            for _ in temp[key]:
                buffer[key].append(_.detach().cpu())

    
    if(dtype=='update'):
        return 

    if(dtype=='stat'):
        temp = buffer
    
    sent_bow = torch.cat(temp['bow'], dim=0)
    

    # vae_d_kl
    qd_logits = torch.cat(temp['qd_logits'], dim=0)
    batch_size = qd_logits.shape[0]
    avg_log_qd_logits = torch.log(qd_logits.mean(0) + 1e-15)
    log_uniform_d = (torch.ones(1) / avg_log_qd_logits.shape[-1]).log()
    vae_d_kl = criterion['cat_kl_loss'](avg_log_qd_logits, log_uniform_d, batch_size, unit_average=True)

    # vae_t_kl
    t_mu = torch.cat(temp['t_mu'], dim=0)
    t_logvar = torch.cat(temp['t_logvar'], dim=0)
    vae_t_kl = criterion['kl_loss'](t_mu, t_logvar, batch_size, unit_average=True)

    # gen loss
    
    # vae_d_nll
    qd = torch.cat(temp['qd'], dim=0)
    log_qd = qd.log()
    vae_d_nll = criterion['nll_loss'](log_qd, sent_bow, batch_size, unit_average=True)

    # vae_t_nll
    qt = torch.cat(temp['qt'], dim=0)
    log_qt = qt.log()
    vae_t_nll = criterion['nll_loss_filtered'](log_qt, sent_bow, batch_size, unit_average=True)
    
    # div_kl
    div_kl = - criterion['cat_kl_loss'](log_qd, log_qt, batch_size, unit_average=True)  # maximize the kl loss

    # decoder loss
    recon = torch.cat(temp['recon'], dim=0)
    log_dec = recon.log()
    dec_nll = criterion['nll_loss'](log_dec, sent_bow, batch_size, unit_average=True)

    persuasive = torch.cat(temp['persuasive'], dim=0)
    per_loss = criterion['persuasive'](persuasive, torch.ones_like(persuasive))
    ##############################
    stat = {
        'vae':{
            'vae_d_kl':vae_d_kl,
            'vae_t_kl':vae_t_kl,
            'vae_d_nll':vae_d_nll,
            'vae_t_nll':vae_t_nll,
            'div_kl':div_kl,
            'dec_nll':dec_nll
        },  
        'persuasive':{
            'per_loss':per_loss,
            'acc':(persuasive>0).float().mean(),
            'diff':persuasive.mean()
        }
    }
    loss = vae_d_kl + vae_t_kl + vae_d_nll + vae_t_nll + alpha*div_kl + dec_nll + per_loss
    return loss, stat

    

def train(args, persuasive_data_iter, model, criterion, device):
    # initial for training 
    model.train()
    # build up optimizer
    if(args.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'Ranger'):
        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'Radam'):
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr*1000, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    grad_clip = args.grad_clip
    save_path = args.save_path
    accumulate = args.accumulate
    print_every = 100*accumulate
    eval_every = 25*accumulate
    #print_every, eval_every = 1, 1

    total_epoch = args.epoch*len(persuasive_data_iter[0])
    print('total training step:', total_epoch)
    persuasive_datas = iter(persuasive_data_iter[0])

    alpha = args.alpha
    best_acc = [0, 0]


    # start training
    model.zero_grad()
    t = time.time()
    persuasive_preds = collections.defaultdict(list)
    for count in range(1, total_epoch+1):
        try:
            datas = next(persuasive_datas)
        except:
            persuasive_datas = iter(persuasive_data_iter[0])
            datas = next(persuasive_datas)
        
        outputs = []
        for data in datas:
            data = convert(data, device)
            out = model(**data)
            outputs.append(out)

        loss, stat = update(buffer=persuasive_preds, in_data=datas, out_data=outputs, criterion=criterion, alpha=alpha, dtype='loss')
        loss.backward()
        
        if(count%accumulate==0):
            utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        if(count % eval_every==0):
            loss, stat = update(buffer=persuasive_preds, criterion=criterion, alpha=alpha, dtype='stat')

            nt = time.time()
            print('now:{}, time: {:.4f}s'.format(count, nt - t))
            print('vae:', end=' ')
            for key, val in stat['vae'].items():
                print('{}: {:.4f}'.format(key, val), end=' ')
            print('persuasive:', end=' ')
            for key, val in stat['persuasive'].items():
                print('{}: {:.4f}'.format(key, val), end=' ')
            print('', flush=True)

            t = nt
            persuasive_preds = collections.defaultdict(list)
            scheduler.step()
       
        if(count % print_every == 0):
            dev_acc = test('dev {}'.format(count), persuasive_data_iter[1], model, criterion, device, alpha, save_path=save_path)
            test_acc = test('test {}'.format(count), persuasive_data_iter[2], model, criterion, device, alpha, save_path=save_path)
            model.train()
            if(dev_acc>best_acc[0]):
                best_acc = [dev_acc, test_acc]
            torch.save(model.state_dict(), save_path+'/check_{}.pt'.format(count))   
    print('all finish with acc:', best_acc)


def test(epoch, persuasive_iter, model, criterion, device, alpha=[0, 0], save_path=''):
    model.eval()
    t = time.time()

    total_preds = []
    with torch.no_grad():
        for i, datas in enumerate(persuasive_iter):
            outputs = []
            for data in datas:
                data = convert(data, device)
                out = model(**data)
                outputs.append(out)
            
            total_preds.append((outputs[1][-1]-outputs[0][-1]).view(-1))

            #if(i==2):   break
    total_preds = torch.cat(total_preds, dim=-1)
    torch.save(total_preds, save_path+'{}.pt'.format(epoch))

    stat = [criterion['persuasive'](total_preds, torch.ones_like(total_preds)), total_preds.mean(), (total_preds>0).float().mean()]
    
    nt = time.time()
    print(epoch,'time: {:.4f}s'.format(nt - t),
        '\npersuasive: [loss: {:.4f}, diff: {:.4f}, acc: {:.4f}]'.format(stat[0], stat[1], stat[2]),
        flush=True)
    
    return stat[-1]
    
    """
    nt = time.time()
    loss, stat = update(buffer=total_preds, criterion=criterion, alpha=alpha, dtype='stat')
    print(epoch,'time: {:.4f}s'.format(nt - t))
    print('vae:', end=' ')
    for key, val in stat['vae'].items():
        print('{}: {:.4f}'.format(key, val), end=' ')
    print('persuasive:', end=' ')
    for key, val in stat['persuasive'].items():
        print('{}: {:.4f}'.format(key, val), end=' ')
    print('', flush=True)
    """
    return stat['persuasive']['acc']

def main():
    args = parse_args()
    if(not os.path.isdir(args.save_path)):
        os.makedirs(args.save_path)
    else:
        print('file exist', file=sys.stderr)
        #raise ValueError('file exist')
    check_seed(args)
    
    print('finish build vocab')
    vocab = build_vocab(args)


    # Model and optimizer
    model = DTDMN(args)
    print(model, flush=True)
    if(args.cuda):
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('finish build model')
    print('*'*10)
    print(args)
    print('*'*10)
    


    # Load data
    persuasive_path = (args.data_path, args.pair_path)
    persuasive = build_data(args, persuasive_path, vocab)

    print('finish build data')
    # Train model
    t_total = time.time()
    print(args.criterion) 
    print('accumulate: ', args.accumulate)
    criterion = {}
    if(args.criterion =="bce" ):
        criterion['persuasive'] = nn.BCEWithLogitsLoss()
    elif(args.criterion == 'hinge'):
        criterion['persuasive'] = hinge
    else:
        raise ValueError('no this loss function')
    
    criterion['nll_loss'] = criterions.PPLLoss(args)
    criterion['nll_loss_filtered'] = criterions.PPLLoss(args, vocab=vocab['bow'], ignore_vocab=vocab['stopword'])
    criterion['kl_loss'] = criterions.GaussianKLLoss()
    criterion['cat_kl_loss'] = criterions.CatKLLoss()
        
    train(args, persuasive, model, criterion, device)   
    torch.save(model.state_dict(), args.save_path+'/check_last.pt')  
    print("Optimization Finished!")
    print('dev:')
    test('End', persuasive[1], model, criterion, device, alpha= args.alpha, save_path=args.save_path)
    print('test:')
    test('End', persuasive[2], model, criterion, device, alpha= args.alpha, save_path=args.save_path)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing

if(__name__ == '__main__'):
    main()
