import timm
from timm.layers import resample_abs_pos_embed
import torch
import torch.nn as nn

def _convert_openai_clip(state_dict, model):
    out_dict = {}
    swaps = [
        ('visual.', ''), ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'),
        ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith('visual.'):
            continue
        if 'contrast' in k or 'caption' in k:
            continue
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k.startswith('trunk.'):
            k = k.replace('trunk.', '')
        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            if v.shape[1] != model.pos_embed.shape[1]:
                num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
                v = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                    v,
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    interpolation='bicubic',
                    antialias=False,
                    verbose=True,
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(state_dict, model):
    out_dict = {}
    swaps = [
        ('encoder.', ''), ('position_embeddings', 'pos_embed'),('projection', 'proj'),
        ('layer.', 'blocks.'), ('patch_embeddings', 'patch_embed'),
        ("attention.attention.query.weight", "attn.qkv.weight"),
        ("attention.attention.query.bias", "attn.qkv.bias"),
        ("attention.attention.key.weight", "attn.qkv.weight"),
        ("attention.attention.key.bias", "attn.qkv.bias"),
        ("attention.attention.value.weight", "attn.qkv.weight"),
        ("attention.attention.value.bias", "attn.qkv.bias"),
        ("attention.output.dense.weight", "attn.proj.weight"),
        ("attention.output.dense.bias", "attn.proj.bias"),
        ('layernorm', 'norm'), ('layer_scale1.lambda1', 'ls1.gamma'), ('layer_scale2.lambda1', 'ls2.gamma'), 
    ]
    for k, v in state_dict.items():
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k.startswith('embeddings.'):
            k = k.replace('embeddings.', '')
        if k.endswith('mask_token'):
            continue
        if 'contrast' in k or 'caption' in k:
            continue
        
        if "attn.qkv.weight" in k:
            block_id = int(k.split(".")[1])
            q_weight = state_dict.get(f"encoder.layer.{block_id}.attention.attention.query.weight")
            k_weight = state_dict.get(f"encoder.layer.{block_id}.attention.attention.key.weight")
            v_weight = state_dict.get(f"encoder.layer.{block_id}.attention.attention.value.weight")
            v = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        if "attn.qkv.bias" in k:
            block_id = int(k.split(".")[1])
            q_bias = state_dict.get(f"encoder.layer.{block_id}.attention.attention.query.bias")
            k_bias = state_dict.get(f"encoder.layer.{block_id}.attention.attention.key.bias")
            v_bias = state_dict.get(f"encoder.layer.{block_id}.attention.attention.value.bias")
            v = torch.cat([q_bias, k_bias, v_bias], dim=0)
        if k in model.state_dict():
            out_dict[k] = v
    return out_dict

def _convert_clip(clip_state_dict, model):
    new_state_dict = {}
    swaps = [
                ('vision_model.', ''), ('embeddings.patch_embedding.weight', 'patch_embed.proj.weight'),('embeddings.position_embedding.weight', 'pos_embed'),
                ('embeddings.class_embedding', 'cls_token'),
                ('encoder.layers.', 'blocks.'), ('self_attn.', 'attn.'),
                ("q_proj.weight", "qkv.weight"),
                ("q_proj.bias", "qkv.bias"),
                ("k_proj.weight", "qkv.weight"),
                ("k_proj.bias", "qkv.bias"),
                ("v_proj.weight", "qkv.weight"),
                ("v_proj.bias", "qkv.bias"),
                ('pre_layrnorm.', 'norm_pre.'), ('post_layernorm.', 'norm.'), 
                ('layer_norm1.', 'norm1.'), ('layer_norm2.', 'norm2.'),
                ("self_attn.", "attn."),  ('attn.out_proj', 'attn.proj')
            ]
    for key, value in clip_state_dict.items():
        if key.startswith("vision_model."):
            key = key.replace('vision_model.', '')
            for sp in swaps:
                key = key.replace(sp[0], sp[1])
            if "attn.qkv.weight" in key:
                block_id = int(key.split(".")[1])
                q_weight = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.q_proj.weight")
                k_weight = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.k_proj.weight")
                v_weight = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.v_proj.weight")
                value = torch.cat([q_weight, k_weight, v_weight], dim=0)

            if "attn.qkv.bias" in key:
                block_id = int(key.split(".")[1])
                q_bias = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.q_proj.bias")
                k_bias = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.k_proj.bias")
                v_bias = clip_state_dict.get(f"vision_model.encoder.layers.{block_id}.self_attn.v_proj.bias")
                value = torch.cat([q_bias, k_bias, v_bias], dim=0)
            if key == "pos_embed" and value.ndim == 2:
                value = value.unsqueeze(0)  # Add batch dimension
            if key == "cls_token" and value.ndim == 1:
                value = value.view(1,1,-1)
            if key in model.state_dict():
                new_state_dict[key] = value
            if "patch_embed.proj.bias" in model.state_dict():
                new_state_dict["patch_embed.proj.bias"] = torch.zeros_like(model.state_dict()["patch_embed.proj.bias"])
    return new_state_dict

def Coca_transform(coca_model, timm_model_name="vit_base_patch16_224"):
    sample_model = timm.create_model(timm_model_name, pretrained=False, num_classes=0)
    map_weigh = _convert_openai_clip(coca_model.state_dict(), sample_model)
    sample_model.load_state_dict(map_weigh)
    return sample_model

def Dinov2_transform(dinov2_model, timm_model_name="vit_large_patch16_224"):
    sample_model =  timm.create_model(timm_model_name, 
                                img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    map_weigh = _convert_dinov2(dinov2_model.state_dict(), sample_model)
    sample_model.load_state_dict(map_weigh)
    return sample_model

def plip_transform(plip_model, timm_model_name='vit_base_patch32_224'):
    sample_model = timm.create_model(timm_model_name, pretrained=False, num_classes=0)
    map_weigh = _convert_clip(plip_model.state_dict(), sample_model)
    sample_model.load_state_dict(map_weigh)
    return sample_model

def phikon_transform(model):
    model.blocks = model.encoder.layer 
    model.pos_embed = model.embeddings.position_embeddings; model.cls_token = model.embeddings.cls_token
    model.patch_embed = model.embeddings.patch_embeddings; model.patch_embed.proj = model.patch_embed.projection
    model.pos_drop = nn.Dropout(p=0.0, inplace=False) # Keep consistent with timm
    model.norm = model.layernorm
    def _hook_return_first_output(module, input, output):
        if isinstance(output, tuple):
            return output[0]
    for block in model.blocks:
        block.register_forward_hook(_hook_return_first_output)
    del model.encoder.layer  
    return model
