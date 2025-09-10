import torch.nn as nn
import torch
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
# this code is just for the random initialised model for comparing with the actual model. the actual pretrained model is on hugging face, but the architecture is here

class SummaryMixing(nn.Module): # a single attention head, learns local and global summaries
    def __init__(self, input_dim, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        
        self.local_norm = nn.LayerNorm(dimensions_f)
        self.summary_norm = nn.LayerNorm(dimensions_s)

        self.s = nn.Linear(input_dim, dimensions_s)
        self.f = nn.Linear(input_dim, dimensions_f)
        self.c = nn.Linear(dimensions_s + dimensions_f, dimensions_c)

    def forward(self, x):

        local_summ = torch.nn.GELU()(self.local_norm(self.f(x))) # local encoding of each tile
        time_summ = self.s(x) # summary encoding over the whole image   
        time_summ = torch.nn.GELU()(self.summary_norm(torch.mean(time_summ, dim=1)))
        time_summ = time_summ.unsqueeze(1).repeat(1, x.shape[1], 1)
        out = torch.nn.GELU()(self.c(torch.cat([local_summ, time_summ], dim=-1))) # local and global (image) features are combined

        return out
    

class MultiHeadSummary(nn.Module): # combines multiple heads
    def __init__(self, nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection):
        super().__init__()

        self.mixers = nn.ModuleList([])
        for _ in range(nheads): # each head is looped through with x passed through it
            self.mixers.append(SummaryMixing(input_dim=input_dim, dimensions_f=dimensions_f, dimensions_s=dimensions_s, dimensions_c=dimensions_c)) # each head gives an output of [batch, n_tiles, dimensions_c]

        self.projection = nn.Linear(nheads*dimensions_c, dimensions_projection) # outputs for all heads are stitched up

    def forward(self, x):

        outs = []
        for mixer in self.mixers:
            outs.append(mixer(x))
        
        outs = torch.cat(outs, dim=-1)
        out = self.projection(outs) # output dimensions are reduced

        return out
    

class FeedForward(nn.Module): # feed forward block
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class SummaryTransformer(nn.Module): # stacks the multi-head attentio stuff with feedforward network
    def __init__(self, input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadSummary(nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection=input_dim),
                FeedForward(input_dim, input_dim)
            ])) # custom attention block then mlp
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x                 # dimensions_projection needs to be equal to input_dim
            x = ff(x) + x                   # output_dim of feedforward needs to be equal to input_dim
        return x


class ViS(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_outputs, input_dim, depth, nheads, 
                        dimensions_f, dimensions_s, dimensions_c, 
                        num_clusters=100, device='cuda:0'):
        # num_outputs: number of genes to predict per sample
        # input_dim: feature size for each patch
        # depth: number of transformer blocks
        # nheads: number of SummaryMixing heads per block
        # dimensions_f: hidden size for patch-level features
        # dimensions_s: hidden size for slide-level features
        # dimensions_c: size of concatenated local+ summary features before projection
        # num_clusters: number of patches per slide
        super().__init__()

        self.pos_emb1D = nn.Parameter(torch.randn(num_clusters, input_dim))

        self.transformer = SummaryTransformer(input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_outputs)
        ) # maps tile features, which have already been reduced to one vector, into gene expression values
        self.device = device
        
    def forward(self, x):
        #pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + self.pos_emb1D # gets 1D embeddings from tile feature vector
        x = self.transformer(x)
        x = x.mean(dim = 1) # tile vectors are averaged so only one vector for the whole slide

        x = self.to_latent(x)
        return self.linear_head(x)
