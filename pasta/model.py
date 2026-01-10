import torch
import torch.nn as nn
import torch.nn.functional as F
from pasta.model_utils import Interpolate
from pasta.encoder import _make_encoder, forward_encoder, FeatureFusionBlock

class FreqEmbedder:
    def __init__(self, multi_freq, include_input=True, input_dims=1, log_sampling=True):
        self.multi_freq = multi_freq
        self.input_dims = input_dims
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]

        self.embed_fns = None
        self.out_dim = None
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = min(self.multi_freq - 1, 10) 
        N_freqs = self.multi_freq

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class DPT(torch.nn.Module):
    def __init__(
        self,
        head,
        model_name='UNI',
        features=256,
        freq_flag = True,
        readout="project",
        channels_last=False,
        use_bn=False,
        pathway_dim=6,
        enable_attention_hooks=False,
        N = 128,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.model_name = model_name
        
        self.pretrained, self.scratch = _make_encoder(
            features,
            groups=1,
            expand=False,
            enable_attention_hooks=enable_attention_hooks,
            model_name=model_name
        )

        self.scratch.refinenet1 = FeatureFusionBlock(features,nn.ReLU(False),deconv=False,bn=use_bn,expand=False,align_corners=True,)
        self.scratch.refinenet2 = FeatureFusionBlock(features,nn.ReLU(False),deconv=False,bn=use_bn,expand=False,align_corners=True,)
        self.scratch.refinenet3 = FeatureFusionBlock(features,nn.ReLU(False),deconv=False,bn=use_bn,expand=False,align_corners=True,)
        # For non-PLIP models, add refinenet4
        if model_name != 'PLIP':
            # Because of patch_size=32 in PLIP, 256=32*2^3, we need to change the number of hook layers to 3
            self.scratch.refinenet4 = FeatureFusionBlock(features,nn.ReLU(False),deconv=False,bn=use_bn,expand=False,align_corners=True,)

        self.scratch.output_conv = head

        if model_name=='UNI':
            self.embed_dim = 1024
        elif model_name=='UNIv2':
            self.embed_dim = 1536
        elif model_name == 'Phikon':
            self.embed_dim = 768
        elif model_name=='Phikonv2':
            self.embed_dim = 1024
        elif model_name=='Virchow':
            self.embed_dim = 1280
        elif model_name=='Virchow2':
            self.embed_dim = 1280
        elif model_name=='CONCH':
            self.embed_dim = 768
        elif model_name=='gigapath':
            self.embed_dim = 1536
        elif model_name=='H-optimus-0':
            self.embed_dim = 1536
        elif model_name=='H-optimus-1':
            self.embed_dim = 1536
        elif model_name=='Kaiko-B':
            self.embed_dim = 768
        elif model_name=='Kaiko-L':
            self.embed_dim = 1024
        elif model_name=='H-optimus-0':
            self.embed_dim = 1536
        elif model_name=='Hibou-B':
            self.embed_dim = 768
        elif model_name=='Hibou-L':
            self.embed_dim = 1024
        elif model_name=='PLIP':
            self.embed_dim = 768
        # self.i_start_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  # learnable start token
        # self.g_start_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  
        # self.end_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.register_buffer('i_start_token', torch.zeros(1, 1, self.embed_dim))  # fixed start token
        self.register_buffer('g_start_token', torch.ones(1, 1, self.embed_dim) * 0.5)  
        self.register_buffer('end_token', torch.ones(1, 1, self.embed_dim)) 
        if freq_flag:
            self.freqemb = FreqEmbedder(N)
            # for gene information projection
            self.fc_g = nn.Sequential(
                nn.Linear(2*N+1, self.embed_dim), 
                nn.LeakyReLU(),  
                nn.Dropout(0.1) 
            )

    def forward(self, x, g=None):
        b = x.shape[0]
        i_start_token = self.i_start_token.expand(b, -1, -1)
        if g is None:
            g = self.end_token.expand(b, -1, -1) 
        else:
            g = g.unsqueeze(2)
            g = self.freqemb.embed(g)
            g = self.fc_g(g) 
            g = g.expand(b, -1, -1)

            g_start_token = self.g_start_token.expand(b, -1, -1)
            g_end_token = self.end_token.expand(b, -1, -1)
            g = torch.cat((g_start_token, g, g_end_token), dim=1)
            # Concat start_token, gene_token, end_token [64, 1, 1280] for no gene, [64, 16, 1280] for ST gene

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = forward_encoder(self.pretrained, x, g, i_start_token)
        
        if self.model_name == 'PLIP':
            layer_1, layer_2, layer_3 = layers
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            path_3 = self.scratch.refinenet3(layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        else:
            layer_1, layer_2, layer_3, layer_4 = layers
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            
        out = self.scratch.output_conv(path_1)

        return out
    
class PASTA(DPT):
    def __init__(
        self, model_name, non_negative=False, pathway_dim=14, scale=1.0, **kwargs
    ):
        features = 256

        self.scale = scale

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, pathway_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, model_name=model_name, pathway_dim=pathway_dim, **kwargs)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.recon_conv = ConvNet(pathway_dim)

    def forward(self, x, g=None):
        feat = super().forward(x,g).squeeze(dim=1)
        feat = self.scale * feat
        feat_mean = self.global_avg_pool(feat).squeeze(dim=[2,3])
        feat_recon = self.recon_conv(feat)
        return feat, feat_mean, feat_recon


class ConvNet(nn.Module):
    def __init__(self,pathway_dim=14):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(pathway_dim, 12, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(12, 3, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x