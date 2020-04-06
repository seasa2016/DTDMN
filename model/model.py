import torch
import torch.nn as nn
import torch.nn.functional as F
import model.module as module
import sys
import os
import numpy as np

class DTDMN(BaseModel):
    def __init__(self, args):
        super(DTDMN, self).__init__(args)
        self.args = args
        # if pretrain embedding exist, load from npy file
        if(os.path.isfile(args.pre)):
            pretrain = np.load(args.pre)
            args.vocab_size, args.emb_dim = pretrain.shape
            self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
            self.word_emb.weight.data = torch.tensor(pretrain)
            self.word_emb.weight.data[0] = 0
        else:
            self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)

        self.vocab_size = args.vocab_size

        # sequence encoder
        self.word_encoder = module.gru(args.emb_dim, args.hidden_dim)
        self.word_drop = nn.Dropout(args.word_drop_rate)

        args.atten_in = args.hidden_dim
        self.attn = module.single_attention(args)

        # build mode here
        # x is for discourse
        self.x_encoder = module.MultiFC(self.vocab_size, args.hidden_size, args.d,
                                 num_hidden_layers=1, short_cut=True)

        self.x_generator = module.MultiFC(args.d, args.d, args.d,
                                   num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(args.d, self.vocab_size, bias=False)

        # context encoder
        # ctx is for topic
        self.ctx_encoder = module.MultiFC(self.vocab_size, args.hidden_size, args.hidden_size,
                                   num_hidden_layers=1, short_cut=True)
        self.q_z_mu = nn.Linear(args.hidden_size, args.k)
        self.q_z_logvar = nn.Linear(args.hidden_size, args.k)

        
        self.ctx_generator = module.MultiFC(args.k, args.k, args.k, num_hidden_layers=0, short_cut=False)

        # decoder
        self.ctx_dec_connector = nn.Linear(args.k, args.k, bias=True)
        self.x_dec_connector = nn.Linear(args.d, args.d, bias=True)
        self.ctx_decoder = nn.Linear(args.k, self.vocab_size)

        self.decoder = nn.Linear(args.k + args.d, self.vocab_size, bias=False)

        # connector
        self.cat_connector = GumbelConnector()

        # memory 
        self.memory = nn.Parameter(torch.rand(args.d+args.k, args.memory_dim))
        self.eraser_layer = nn.Linear(args.hidden_dim, args.memory_dim)
        self.update_layer = nn.Linear(args.hidden_dim, args.memory_dim)

        self.persuasive_pred = module.gru(args.memory_dim, args.memory_dim, bidirectional=False)
        self.fc = nn.Linear(args.memory_dim, 1, bias=False)

    def qdx_forward(self, tar_utts):
        qd_logits = self.x_encoder(tar_utts).view(-1, self.args.d)
        qd_logits_multi = qd_logits.repeat(self.args.d_size, 1, 1)
        sample_d_multi, d_ids_multi = self.cat_connector(qd_logits_multi, 1.0,
                                                         self.use_gpu, return_max_id=True)
        sample_d = sample_d_multi.mean(0)
        d_ids = d_ids_multi.view(self.args.d_size, -1).transpose(0, 1)

        return module.Pack(qd_logits=qd_logits, sample_d=sample_d, d_ids=d_ids)

    def pxy_forward(self, results):
        gen_d = self.x_generator(results.sample_d)
        x_out = self.x_decoder(gen_d)

        results['gen_d'] = gen_d
        results['x_out'] = x_out

        return results

    def qzc_forward(self, ctx_utts):
        ctx_out = F.tanh(self.ctx_encoder(ctx_utts))
        z_mu = self.q_z_mu(ctx_out)
        z_logvar = self.q_z_logvar(ctx_out)

        sample_z = self.reparameterize(z_mu, z_logvar)
        return module.Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def pcz_forward(self, results):
        gen_c = self.ctx_generator(results.sample_z)
        c_out = self.ctx_decoder(gen_c)

        results['gen_c'] = gen_c
        results['c_out'] = c_out

        return results

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, word_id, sent_bow, sent_lens, turn_lens):
        # initial
        device = word_id.device

        # sequence encoder
        word_emb = self.word_emb(word_id)
        word_emb = self.word_encoder(word_emb, sent_lens)
        sent_emb = self.attn(word_emb)

        # vae here
        vae_d_resp = self.pxy_forward(self.qdx_forward(sent_bow))
        vae_t_resp = self.pcz_forward(self.qzc_forward(sent_bow))

        # prior network (we can restrict the prior to stopwords and emotional words)

        # combine context topic and x discourse
        sample_d, d_ids = vae_d_resp.sample_d.detach(), vae_d_resp.d_ids.detach()
        sample_z = vae_t_resp.sample_z.detach()
        memory_scale = torch.cat([sample_d, sample_z], dim=1)

        gen = torch.cat([self.x_dec_connector(sample_d), self.ctx_dec_connector(sample_z)], dim=1)
        dec_out = self.decoder(gen)

        # sentence encode part
        memory = self.update_memory(sent_emb, memory_scale)

        outs = torch.matmul(memory_scale.unsqueeze(1), memory).squeeze(1)

        # truncate
        outs = self.trunc(outs, turn_lens)

        last = torch.stack([ out[l-1]   for out, l in zip(outs, turn_lens)])
        score = self.fc(last)

        return vae_d_resp, vae_t_resp, dec_out,score

class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, use_gpu, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, use_gpu)
        _, y_hard = torch.max(y, dim=-1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, use_gpu)
            y_onehot.scatter_(-1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y










