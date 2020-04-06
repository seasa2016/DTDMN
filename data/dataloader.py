import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import collections
import copy
import h5py
import numpy as np
import json
import random
import os

class persuasiveDataset(Dataset):
    def __init__(self, data_path, pair_path, vocab, train=True):
        self.train = train
        self.vocab_len = len(vocab)
        
        total_path = data_path+'.pt'
        if(os.path.isfile(total_path)):
            self.data = torch.load(total_path)
        else:
            with open(data_path) as f:
                data = json.loads(f.readline())
                self.data = {}
                for pair_id, pair in data.items():
                    
                    bow = torch.zeros(self.vocab_len)
                    for _ in pair['op_info']['word_id']:
                        bow[_] = 1
                    pair['op_info']['bow'] = bow

                    for side in range(2):
                        for path_id in range(len(pair['content'][side])):
                            for reply_id in range(len(pair['content'][side][path_id])):
                                bow = torch.zeros(self.vocab_len)
                                for _ in pair['content'][side][path_id][reply_id]['word_id']:
                                    bow[_] = 1
                                pair['content'][side][path_id][reply_id]['bow'] = bow
                    self.data[int(pair_id)] = pair
            torch.save(self.data, total_path)

        with open(pair_path) as f:
            self.pair = json.loads(f.readline())
        if(train):
            mapping = {'cat':{}, 'list':{}}
            temp = []
            for index, pos, neg in self.pair:
                if(index not in mapping['cat']):
                    mapping['cat'][index] = len(temp)
                    mapping['list'][ len(temp) ] = index
                    temp.append([])
                    
                index = mapping['cat'][index]
                temp[index].append((pos, neg))
            for _ in temp:
                random.shuffle(_)
                
            self.pair = temp
            self.count = [0] * len(self.pair)
            self.mapping = mapping
    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        if(self.train):
            index = self.mapping['list'][idx]
            t = self.count[idx]
            self.count[idx] = ((self.count[idx]+1) % len(self.pair[idx]))
            neg_post_index, pos_post_index = self.pair[idx][t]
        else:
            index, neg_post_index, pos_post_index = self.pair[idx]

        data_pair = []

        for side, post_index in enumerate([neg_post_index, pos_post_index]):
            #print(self.data[index])
            sample = {
                'word_id':[self.data[index]['op_info']['word_id']],
                'sent_lens':[self.data[index]['op_info']['sent_lens']],
                'bow':[self.data[index]['op_info']['bow']],
                'turn_lens':len(self.data[index]['content'][side][post_index])+1
            }
            
            for _ in self.data[index]['content'][side][post_index]:
                sample['word_id'].append(_['word_id'])
                sample['sent_lens'].append(_['sent_lens'])
                sample['bow'].append(_['bow'])

            data_pair.append(copy.deepcopy(sample))
        return data_pair

def persuasive_collate_fn(src):
    def padding(data, dtype):
        # first find max in every dimension
        size = len(data[0].shape)

        temp_len = np.array( [ _.shape for _ in data] )
        max_len = [len(data)] + temp_len.max(axis=0).tolist()

        temp = torch.zeros(max_len, dtype=dtype)
        if(size == 4):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1], :temp_len[i][2], :temp_len[i][3]] = data[i]
        elif(size == 3):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1], :temp_len[i][2]] = data[i]
        elif(size == 2):
            for i in range(len(data)):
                temp[i, :temp_len[i][0], :temp_len[i][1]] = data[i]
        elif(size==1):
            for i in range(len(data)):
                temp[i, :temp_len[i][0]] = data[i]
        else:
            raise ValueError('no this size {size}')
        return temp

    # convert
    outputs = []

    for side in range(2):
        data = dict()
        for key in src[0][side]:
            data[key] = [ _[side][key] for _ in src]
        # deal with bow, sent_lens, context_id, turn_lens
        output = {}
        word_id = []
        for post in data['word_id']:
            for sent in post:
                word_id.append(torch.tensor(sent, dtype=torch.long))
        output['word_id'] = padding(word_id, dtype=torch.long)
        
        for key in ['sent_lens']:
            temp = []
            for _ in data[key]:
                temp.extend(_)
            output[key] = torch.tensor(temp, dtype=torch.long)    
        
        for key in ['bow']:
            temp = []
            for _ in data[key]:
                temp.extend(_)
            output[key] = torch.stack(temp)

        output['turn_lens'] = torch.tensor([_ for _ in data['turn_lens']], dtype=torch.long)
        outputs.append(output)

    return outputs


if(__name__ == '__main__'):
    data_path = '/nfs/nas-5.1/kyhuang/preprocess/cmv_raw_origin_full_final/train/baseline/data'
    pair_path = '/nfs/nas-5.1/kyhuang/preprocess/cmv_raw_origin_full_final/train/baseline/graph_pair'
    vocab = np.load('/nfs/nas-5.1/kyhuang/preprocess/cmv_raw_origin_full_final/baseline/vocab.npy', allow_pickle=True).item()
    
    dataloader = persuasiveDataset(data_path=data_path, pair_path=pair_path, vocab=vocab['bow'])
    print('----------')
    for key in dataloader[0][0]:
        try:
            print(key, dataloader[0][0][key].shape)
        except:
            print(dataloader[0][0][key])

    print('----------')
    batch_size = 64
    train_dataloader = DataLoader(dataloader, batch_size=batch_size,shuffle=False, num_workers=4,collate_fn=persuasive_collate_fn)
    for i, datas in enumerate(train_dataloader):
        #if(i%1000==0):
        if(i==0):
            #torch.save(data, './test_data')
            for key in datas[0]:
                try:
                    print(key, datas[0][key].shape)
                except:
                    print(datas[0][key])

        pass
