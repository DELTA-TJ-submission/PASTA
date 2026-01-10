# Inspired from DPT(https://github.com/isl-org/DPT) with modifications, thanks for their great work.

import os
import types
import timm
from timm.layers import SwiGLUPacked
import torch
import torch.nn as nn
from pasta.model_utils import Transpose, ProjectReadout, forward_flex, _resize_pos_embed
from pasta.model_transform import Coca_transform, Dinov2_transform, phikon_transform, plip_transform


def get_readout_oper(vit_features, features, start_index=1):
    readout_oper = [
        ProjectReadout(vit_features, start_index) for out_feat in features
    ]
    return readout_oper

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


def _load_conch_model():
    """
    Load CONCH model directly from Hugging Face Hub without conch package dependency.
    This function downloads the model weights and converts them to a timm-compatible format.
    
    Returns:
        model: A timm ViT model with CONCH weights loaded
    """
    from huggingface_hub import hf_hub_download
    
    # Download CONCH model weights from Hugging Face
    checkpoint_path = hf_hub_download(
        repo_id="MahmoodLab/conch",
        filename="pytorch_model.bin",
        cache_dir=None
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    class SimpleCoCaWrapper(nn.Module):
        """Temporary wrapper to hold CONCH state dict for conversion"""
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict
        
        def state_dict(self):
            return self._state_dict
    
    coca_wrapper = SimpleCoCaWrapper(state_dict)
    
    # Convert to timm ViT format
    model = Coca_transform(coca_wrapper, "vit_base_patch16_224")
    
    return model


def _make_encoder(
    features,
    groups=1,
    expand=False,
    hooks=None,
    enable_attention_hooks=False,
    model_name='UNI'
):
    pretrained = _load_pretrained_model(
        hooks=hooks,
        enable_attention_hooks=enable_attention_hooks,
        model_name=model_name
    )
    if model_name == 'UNI':
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif model_name == 'UNIv2':
        scratch = _make_scratch(
            [384, 384, 1536, 1536], features, groups=groups, expand=expand
        )
    elif model_name == 'Phikon':
        scratch = _make_scratch(
            [192, 384, 768, 768] , features, groups=groups, expand=expand
        )
    elif model_name == 'Phikonv2':
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif model_name == 'Virchow2':
        scratch = _make_scratch(
            [320, 640, 1280, 1280] , features, groups=groups, expand=expand
        )  
    elif model_name == 'Virchow':
        scratch = _make_scratch(
            [320, 640, 1280, 1280] , features, groups=groups, expand=expand
        )  
    elif model_name == 'CONCH':
        scratch = _make_scratch(
            [192, 384, 768, 768] , features, groups=groups, expand=expand
        )
    elif model_name == 'gigapath':
        scratch = _make_scratch(
            [384, 384, 1536, 1536] , features, groups=groups, expand=expand
        )
    elif model_name == 'Kaiko-B':
        scratch = _make_scratch(
            [192, 384, 768, 768] , features, groups=groups, expand=expand
        )
    elif model_name == 'Kaiko-L':
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif model_name == 'H-optimus-0':
        scratch = _make_scratch(
            [384, 384, 1536, 1536] , features, groups=groups, expand=expand
        )
    elif model_name == 'H-optimus-1':
        scratch = _make_scratch(
            [384, 384, 1536, 1536] , features, groups=groups, expand=expand
        )
    elif model_name == 'Hibou-B':
        scratch = _make_scratch(
            [192, 384, 768, 768] , features, groups=groups, expand=expand
        )
    elif model_name == 'Hibou-L':
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif model_name == 'PLIP':
        scratch = _make_scratch(
            [192, 384, 768], features, groups=groups, expand=expand
        )
    return pretrained, scratch


def _load_pretrained_model(hooks=None, enable_attention_hooks=False, size=[384, 384], freeze_vit=True, model_name='UNI',
):
    from transformers import AutoModel, ViTModel,CLIPModel
    # The following settings of models are from their original paper.
    if model_name=='UNI':
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        hooks = [5, 11, 17, 23] if hooks == None else hooks # ViT-L
        vit_features = 1024; features=[256, 512, 1024, 1024]
    elif model_name=='CONCH':
        # Load CONCH model directly from Hugging Face without conch package dependency
        model = _load_conch_model()
        hooks = [2, 5, 8, 11] if hooks == None else hooks
        vit_features=768; features=[192, 384, 768, 768]
    elif model_name=='Virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        hooks = [7, 15, 23, 31] if hooks == None else hooks
        vit_features = 1280; features=[320, 640, 1280, 1280] 
    elif model_name=='Virchow2':
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        hooks = [7, 15, 23, 31] if hooks == None else hooks # ViT-H
        vit_features = 1280; features=[320, 640, 1280, 1280] 
    elif model_name=='gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        hooks = [9, 19, 29, 39] if hooks == None else hooks
        vit_features = 1536; features=[384, 384, 1536, 1536]
    elif model_name=='Phikon':
        model = AutoModel.from_pretrained('owkin/phikon',add_pooling_layer=False)
        vit_features=768; features=[192, 384, 768, 768]
        model = phikon_transform(model)
        hooks = [2, 5, 8, 11] if hooks == None else hooks
    elif model_name=='Phikonv2':
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        model = Dinov2_transform(model, "vit_large_patch16_224")
        hooks = [5, 11, 17, 23] if hooks == None else hooks # ViT-L
        vit_features = 1024; features=[256, 512, 1024, 1024]
    elif model_name=='Kaiko-B':
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8", trust_repo=True)
        vit_features=768; features=[192, 384, 768, 768]
        hooks = [2, 5, 8, 11] if hooks == None else hooks
    elif model_name=='Kaiko-L':
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True)
        hooks = [5, 11, 17, 23] if hooks == None else hooks # ViT-L
        vit_features = 1024; features=[256, 512, 1024, 1024]
    elif model_name=='H-optimus-0':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        hooks = [9, 19, 29, 39] if hooks == None else hooks
        vit_features = 1536; features=[384, 384, 1536, 1536]
    elif model_name=='H-optimus-1':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        hooks = [9, 19, 29, 39] 
        vit_features = 1536; features=[384, 384, 1536, 1536]
    elif model_name == 'Hibou-B':
        model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        vit_features=768; features=[192, 384, 768, 768]
        hooks = [2, 5, 8, 11] if hooks == None else hooks
        model = phikon_transform(model)
    elif model_name == 'Hibou-L':
        model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
        vit_features=1024; features=[256, 512, 1024, 1024]
        model = phikon_transform(model)
        hooks = [5, 11, 17, 23] if hooks == None else hooks
    elif model_name == 'PLIP':
        model = CLIPModel.from_pretrained("vinid/plip")
        model = plip_transform(model, "vit_base_patch32_224")
        vit_features=768; features=[192, 384, 768, 768]
        hooks = [2, 5, 8] if hooks == None else hooks  # PLIP uses 3 layers, because patch_size=32 in PLIP, 256=32*2^3
    elif model_name == 'UNIv2':
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",pretrained=True, **timm_kwargs)
        hooks = [5, 11, 17, 23] 
        vit_features = 1536; features=[384, 384, 1536, 1536]
    else:
        raise NotImplementedError

    if freeze_vit:
        for name, param in model.named_parameters():
            param.requires_grad = False
    print(f'Load model {model_name} successfully!')

    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    if len(hooks) > 3:
        pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    pretrained.activations = activations
    
    readout_oper = get_readout_oper(vit_features, features, start_index=1)
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0], Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])), # concatenate
        nn.Conv2d( 
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ), # resample: Conv2d + ConvTranspose2d
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )
    
    pretrained.model.start_index = 1 
    if model_name == 'Virchow2':
        pretrained.model.start_index = 5
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'Virchow':
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'UNI':
        pretrained.model.patch_size = [16, 16]
    elif model_name == 'UNIv2':
        pretrained.model.start_index = 0
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'Phikon':
        pretrained.model.patch_size = [16, 16]
    elif model_name == 'Phikonv2':
        pretrained.model.patch_size = [16, 16]
    elif model_name == 'CONCH':
        pretrained.model.patch_size = [16, 16]
    elif model_name == 'gigapath':
        pretrained.model.patch_size = [16, 16]
    elif model_name == 'Kaiko-B':
        pretrained.model.patch_size = [8, 8]
    elif model_name == 'Kaiko-L':
        pretrained.model.start_index = 0
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'H-optimus-0':
        pretrained.model.start_index = 0
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'H-optimus-1':
        pretrained.model.start_index = 0
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'Hibou-B':
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'Hibou-L':
        pretrained.model.patch_size = [14, 14]
    elif model_name == 'PLIP':
        pretrained.model.patch_size = [32, 32]
        # Because of patch_size=32 in PLIP, 256=32*2^3, we need to change the number of hook layers to 3

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )
    return pretrained

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    num_layers = len(in_shape)
    out_shapes = [out_shape * (2**i if expand else 1) for i in range(num_layers)]
    
    scratch.layer1_rn = nn.Conv2d(in_shape[0],out_shapes[0],kernel_size=3,stride=1,padding=1,bias=False,groups=groups,)
    scratch.layer2_rn = nn.Conv2d(in_shape[1],out_shapes[1],kernel_size=3,stride=1,padding=1,bias=False,groups=groups,)
    scratch.layer3_rn = nn.Conv2d(in_shape[2],out_shapes[2],kernel_size=3,stride=1,padding=1,bias=False,groups=groups,)
    if num_layers > 3:
        scratch.layer4_rn = nn.Conv2d(in_shape[3],out_shapes[3],kernel_size=3,stride=1,padding=1,bias=False,groups=groups,)

    return scratch


class ResConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features,out_features,kernel_size=1,stride=1,padding=0,bias=True,groups=1,)
        self.resConfUnit1 = ResConvUnit(features, activation, bn)
        self.resConfUnit2 = ResConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional() 

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)

        output = nn.functional.interpolate( 
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def forward_encoder(pretrained, x, g,i_start_token):
    b, c, h, w = x.shape
    glob = pretrained.model.forward_flex(x,g,i_start_token)
    n = g.shape[1] 
    layer_1 = pretrained.activations["1"][:,1:-n,:] # [B, 257+n, L] 
    layer_2 = pretrained.activations["2"][:,1:-n,:]
    layer_3 = pretrained.activations["3"][:,1:-n,:]
    
    layer_1 = pretrained.act_postprocess1[0:2](layer_1).contiguous()
    layer_2 = pretrained.act_postprocess2[0:2](layer_2).contiguous()
    layer_3 = pretrained.act_postprocess3[0:2](layer_3).contiguous()

    unflatten = nn.Sequential(
        nn.Unflatten(2,torch.Size([h // pretrained.model.patch_size[1],w // pretrained.model.patch_size[0],])))

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    
    # For models with 4 layers (non-PLIP)
    if "4" in pretrained.activations:
        layer_4 = pretrained.activations["4"][:,1:-n,:]
        layer_4 = pretrained.act_postprocess4[0:2](layer_4).contiguous()
        if layer_4.ndim == 3:
            layer_4 = unflatten(layer_4)
        layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)
        return layer_1, layer_2, layer_3, layer_4
    else:
        # For PLIP (3 layers)
        return layer_1, layer_2, layer_3