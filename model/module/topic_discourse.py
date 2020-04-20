import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import MultiFC
from .util import Pack



class Topic_Discourse(nn.Module):
    def __init__(self, args):
        super(Topic_Discourse, self).__init__()
        self.vocab_size = args.vocab_size
        self.args = args

        # build mode here
        # x is for discourse
        self.x_encoder = MultiFC(self.vocab_size, args.hidden_dim, args.d,
                                 num_hidden_layers=1, short_cut=True)

        self.x_generator = MultiFC(args.d, args.d, args.d,
                                   num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(args.d, self.vocab_size, bias=False)

        # context encoder
        # ctx is for topic
        self.ctx_encoder = MultiFC(self.vocab_size, args.hidden_dim, args.hidden_dim,
                                   num_hidden_layers=1, short_cut=True)
        self.q_z_mu = nn.Linear(args.hidden_dim, args.k)
        self.q_z_logvar = nn.Linear(args.hidden_dim, args.k)

        
        self.ctx_generator = MultiFC(args.k, args.k, args.k, num_hidden_layers=0, short_cut=False)

        # decoder
        self.ctx_dec_connector = nn.Linear(args.k, args.k, bias=True)
        self.x_dec_connector = nn.Linear(args.d, args.d, bias=True)
        self.ctx_decoder = nn.Linear(args.k, self.vocab_size)

        self.decoder = nn.Linear(args.k + args.d, self.vocab_size, bias=False)

        # connector
        self.cat_connector = GumbelConnector()

    def qdx_forward(self, tar_utts):
        qd_logits = self.x_encoder(tar_utts).view(-1, self.args.d)
        qd_logits_multi = qd_logits.repeat(self.args.d_size, 1, 1)
        sample_d_multi, d_ids_multi = self.cat_connector(qd_logits_multi, 1.0, return_max_id=True)
        sample_d = sample_d_multi.mean(0)
        d_ids = d_ids_multi.view(self.args.d_size, -1).transpose(0, 1)

        return Pack(qd_logits=qd_logits, sample_d=sample_d, d_ids=d_ids)

    def pxy_forward(self, results):
        gen_d = self.x_generator(results.sample_d)
        x_out = self.x_decoder(gen_d)

        results['gen_d'] = gen_d
        results['x_out'] = x_out

        return results

    def qzc_forward(self, ctx_utts):
        ctx_out = self.ctx_encoder(ctx_utts).tanh()
        z_mu = self.q_z_mu(ctx_out)
        z_logvar = self.q_z_logvar(ctx_out)

        sample_z = self.reparameterize(z_mu, z_logvar)
        return Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def pcz_forward(self, results):
        gen_c = self.ctx_generator(results.sample_z)
        c_out = self.ctx_decoder(gen_c)

        results['gen_c'] = gen_c
        results['c_out'] = c_out

        return results

    def reparameterize(self, mu, logvar):
        if(self.training):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, bow):
        # vae here
        vae_d_resp = self.pxy_forward(self.qdx_forward(bow))
        vae_t_resp = self.pcz_forward(self.qzc_forward(bow))

        # prior network (we can restrict the prior to stopwords and emotional words)

        # combine context topic and x discourse
        sample_d, d_ids = vae_d_resp.sample_d.detach(), vae_d_resp.d_ids.detach()
        sample_z = vae_t_resp.sample_z.detach()
        memory_scale = torch.cat([sample_d, sample_z], dim=1)

        gen = torch.cat([self.x_dec_connector(sample_d), self.ctx_dec_connector(sample_z)], dim=1)
        dec_out = self.decoder(gen)

        return vae_d_resp, vae_t_resp, dec_out, memory_scale


class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, eps=1e-20):
        device = logits.device
        u = torch.rand(logits.size())
        sample = -torch.log(-torch.log(u + eps) + eps).to(device)
        
        return sample

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        device = logits.device

        y = self.gumbel_softmax_sample(logits, temperature)
        _, y_hard = torch.max(y, dim=-1, keepdim=True)
        if hard:
            y_onehot = torch.zeros(y.size(), dtype=torch.float).to(device)
            y_onehot.scatter_(-1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y