import torch
import torch.nn as nn
import torch.nn.functional as F
import model.module as module
import sys
import os
import numpy as np

class DTDMN(nn.Module):
    def __init__(self, args):
        super(DTDMN, self).__init__()
        self.args = args
        # if pretrain embedding exist, load from npy file
        if(os.path.isfile(args.pre)):
            pretrain = np.load(args.pre)
            args.vocab_size, args.emb_dim = pretrain.shape
            self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
            self.word_emb.weight.data = torch.tensor(pretrain, dtype=torch.float)
            self.word_emb.weight.data[0] = 0.
            del pretrain
        else:
            self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        #self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        self.vocab_size = args.vocab_size

        # sequence encoder
        self.word_encoder = module.gru(args.emb_dim, args.hidden_dim)
        self.word_drop = nn.Dropout(args.word_drop_rate)

        args.atten_in = args.hidden_dim*2
        self.attn = module.single_attention(args)

        self.topic_discourse = module.Topic_Discourse(args)
        self.memory = module.Memory(args)

        self.persuasive_pred = module.gru(args.memory_dim, args.memory_dim, bidirectional=False)
        self.fc = nn.Linear(args.memory_dim, 1, bias=False)


    def trunc(self, outs, turn_lens):
        device = outs.device
        _, hidden_dim = outs.shape

        temp = []
        start, end = 0, 0
        max_len = turn_lens.max().item()
        
        for l in turn_lens:
            l = l.item()
            end+=l
            temp.append(
                torch.cat([outs[start:end], torch.zeros(max_len-l, hidden_dim).to(device)], dim=0)
            )
            
            start = end
        
        return torch.stack(temp)
    def forward(self, word_id, bow, sent_lens, turn_lens):
        # initial
        device = word_id.device

        # sequence encoder
        word_emb = self.word_emb(word_id)
        
        word_emb, _ = self.word_encoder(word_emb, sent_lens)
        sent_emb = self.attn.length_extract(word_emb, sent_lens)

        vae_d_resp, vae_t_resp, dec_out, memory_scale = self.topic_discourse(bow)
        outs = self.memory(sent_emb, memory_scale)

        # truncate
        outs = self.trunc(outs, turn_lens)

        last = torch.stack([ out[l-1]   for out, l in zip(outs, turn_lens)])
        score = self.fc(last)

        return vae_d_resp, vae_t_resp, dec_out, score
