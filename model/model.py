import torch
import torch.nn as nn
import torch.nn.functional as F
import model.module as module

class DTDMN(nn.Module):
    def __init__(self, args):
        self.args = args
        self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim)
        self.word_encoder = module.gru(args.emb_dim, args.hidden_dim)
        self.word_drop = nn.Dropout(args.word_drop_rate)

        args.atten_in = args.hidden_dim
        self.attn = module.single_attention(args)
        
        # context encoder
        # x is for discourse
        self.x_encoder = nn.Linear(args.vocab_size, args.d)

        # ctx is for topic
        self.ctx_encoder = nn.Linear(args.vocab_size, args.vae_hidden_dim)
        self.q_z_mu = nn.Linear(args.vae_hidden_dim, args.k)
        self.q_z_logvar = nn.Linear(args.vae_hidden_dim, args.k)

        # decoder
        self.x_decoder = nn.Linear(args.d, self.vocab_size)
        self.ctx_decoder = nn.Linear(args.k, self.vocab_size)

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

        return qd_logits, sample_d, d_ids

    def qzc_forward(self, ctx_utts):
        ctx_out = F.tanh(self.ctx_encoder(ctx_utts))
        z_mu = self.q_z_mu(ctx_out)
        z_logvar = self.q_z_logvar(ctx_out)

        sample_z = self.reparameterize(z_mu, z_logvar)
        return sample_z, z_mu, z_logvar

    def update_memory(self, sent_emb, memory_scale):
        batchsize, _  = memory_scale.shape
        dic_dim, memory_dim = self.memory.shape

        erase = self.eraser_layer(sent_emb).sigmoid()
        update = self.update_layer(sent_emb).tanh()

        memory = self.memory.repeat(batchsize, 1, 1)

        memory_scale = memory_scale.unsqueeze(-1).repeat(1, 1, memory_dim).view()
        erase = erase.repeat(1, dic_dim).view(batchsize, dic_dim, memory_scale)
        update = update.repeat(1, dic_dim).view(batchsize, dic_dim, memory_scale)

        erase = erase*memory_scale
        update = update*memory_scale
        memory = memory*(1-erase)+update
        
        return memory

    def trunc(outs, lengths):
        start, end = 0, 0
        
        _, hidden_din = outs.shape
        device = outs.device
        max_len = lengths.max().item()

        temp = []
        for l in lengths:
            end += l
            temp.append(
                torch.cat(
                    [outs[start:end], torch.zeros(max_len-l, hidden_din).to(device)], dim=-1
                )
            )
            start = end
        return torch.stack(temp)

    def forward(self, word_id, sent_bow, sent_lens, turn_lens):
        # initial
        device = word_id.device

        # sequence encoder
        word_emb = self.word_emb(word_id)
        word_emb = self.word_encoder(word_emb, sent_lens)
        sent_emb = self.attn(word_emb)

        # topic vae part
        # discourse
        qd_logits, sample_d, d_ids = self.qdx_forward(sent_bow)
        vae_d_resp = self.x_decoder(sample_d)

        # topic
        sample_t, t_mu, t_logvar = self.qzc_forward(sent_bow)
        vae_t_resp = self.ctx_decoder(sample_t.softmax(-1))

        # prior network (we can restrict the prior to stopwords and emotional words)
        # combine context topic and x discourse
        sample_d, d_ids = sample_d.detach(), d_ids.detach()
        sample_t = sample_t.detach()
        memory_scale = torch.cat([sample_d, sample_t], dim=1)
        
        recon = vae_d_resp+vae_t_resp

        # sentence encode part
        memory = self.update_memory(sent_emb, memory_scale)

        outs = torch.matmul(memory_scale.unsqueeze(1), memory).squeeze(1)

        # truncate
        outs = self.trunc(outs, turn_lens)

        last = torch.stack([ out[l-1]   for out, l in zip(outs, turn_lens)])
        score = self.fc(last)
        
        return recon, score

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










