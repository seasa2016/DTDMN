import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()

        # memory 
        self.memory = nn.Parameter(torch.rand(args.d+args.k, args.memory_dim))
        self.eraser_layer = nn.Linear(args.hidden_dim*2, args.memory_dim)
        self.update_layer = nn.Linear(args.hidden_dim*2, args.memory_dim)

    def update_memory(self, sent_emb, memory_scale):
        erase = self.eraser_layer(sent_emb).sigmoid()
        update = self.update_layer(sent_emb).tanh()
        
        batchsize, memory_dim = erase.shape
        batchsize, num_memory = memory_scale.shape

        memory = self.memory.repeat(batchsize, 1, 1)
        
        erase = erase.repeat(1, num_memory).view(batchsize, num_memory, memory_dim)
        update = update.repeat(1, num_memory).view(batchsize, num_memory, memory_dim)
        memory_scale = memory_scale.unsqueeze(-1).repeat(1, 1, memory_dim).view(batchsize, num_memory, memory_dim)

        memory = memory*(1-memory_scale*erase)+update
        return memory

    def forward(self, sent_emb, memory_scale):
        # sentence encode part
        memory_emb = self.update_memory(sent_emb, memory_scale)

        outs = torch.matmul(memory_scale.unsqueeze(1), memory_emb).squeeze(1)

        return outs