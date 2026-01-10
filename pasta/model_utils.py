import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Val Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if isinstance(model,dict):
            for i in model:
                torch.save(model[i].state_dict(), f'{ckpt_name}_i.pt')
        else:
            torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def validate_file(file_path, error_message):
    if not os.path.exists(file_path):
        raise FileNotFoundError(error_message)
    
def get_disk_mask(radius, line=224, boundary_width=None):
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-line//2, line//2),
            np.arange(-line//2, line//2),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin

def post_collate_fn(batch):
    """
    Post collate function to clean up batch
    """
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    if 'imm_score' in batch.keys():
        if batch["imm_score"].dim() == 3:
            assert batch["imm_score"].size(0) == 1
            batch["imm_score"] = batch["imm_score"].squeeze(0)
    if 'mask' in batch.keys():
        if batch["mask"].dim() == 5:
            assert batch["mask"].size(0) == 1
            batch["mask"] = batch["mask"].squeeze(0)    
    return batch


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : 1],
        posemb[0, self.start_index :],
    )
    # posemb_tok, posemb_grid = (
    #     posemb[:, : self.start_index],
    #     posemb[0, self.start_index :],
    # )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2).contiguous()
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).contiguous().reshape(1, gs_h * gs_w, -1).contiguous()

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

def forward_flex(self, x, g, i_start_token):
    b, c, h, w = x.shape
    if hasattr(self, 'patch_size'):
        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
        )
    else:
        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_embed.patch_size[1], w // self.patch_embed.patch_size[0]
        )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2).contiguous()

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  
        dist_token = self.dist_token.expand(B, -1, -1) # Check dist_token attribute. If exists, it means this is a distillation model, and extra dist_token is used.
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  
        x = torch.cat((cls_tokens, x), dim=1) # [64, 1+256=257, embed_dim]

    x = x + pos_embed
    x = self.pos_drop(x)
    # concatenate start_token, gene_token, end_token
    x = torch.cat((i_start_token, x,g), dim=1)  # [64, 1+257+3/1, embed_dim]

    # control g at input, if no gene information is needed, set g to an empty tensor of shape [b,0, embed_dim]
    for blk in self.blocks:
        x = blk(x)
        # if isinstance(x, tuple):
        #     x = x[0] 

    x = self.norm(x)
    return x


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            return [base_lr for base_lr in self.base_lrs]
        

class Interpolate(nn.Module):
    """Interpolation module."""
    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
    

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1).contiguous()
        return x
    

class ProjectReadout(nn.Module): 
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


def get_img_transforms(model_name):
    """
    Get image transforms based on model name.
    
    Args:
        model_name (str): Name of the model (e.g., 'PLIP', 'CONCH', 'gigapath', 'UNI', etc.)
    
    Returns:
        torchvision.transforms.Compose: Image transformation pipeline
    """
    if model_name in ['CONCH', 'PLIP']:
        img_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.481, 0.458, 0.408),
                                        std=(0.269, 0.261, 0.276)),
                ]
            )
    elif model_name=='gigapath':
        img_transforms = transforms.Compose([
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
    elif 'H-optimus' in model_name:
        img_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.707223, 0.578729, 0.703617), 
                        std=(0.211883, 0.230117, 0.177517)),
                ]
            )
    elif 'Kaiko' in model_name:
        img_transforms = v2.Compose([
                    v2.ToImage(),
                    v2.Resize(size=224),
                    v2.CenterCrop(size=224),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)),
                ]
            )
    elif 'Hibou' in model_name:
        img_transforms = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.7068, 0.5755, 0.722),
                                            std=(0.195, 0.2316, 0.1816)),
                    ]
                )
    else:
        img_transforms = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225)),
                    ]
                )
    return img_transforms


def load_model_weights(model, trainable_path=None):
    """
    Load model weights from separate backbone and trainable checkpoints.
    
    Args:
        model: The model to load weights into
        trainable_path: Path to trained custom layer weights
    
    Returns:
        model: Model with loaded weights
    """
    # state_dict = {}  
    # if backbone_path and os.path.exists(backbone_path):
    #     backbone_state = torch.load(backbone_path, map_location='cpu')
    #     state_dict.update(backbone_state)
    #     print(f"✓ Loaded backbone weights from: {backbone_path}")
    
    if trainable_path and os.path.exists(trainable_path):
        trainable_state = torch.load(trainable_path, map_location='cpu',weights_only=True)
        print(f"✓ Loaded trainable weights from: {trainable_path}")
    
        _ = model.load_state_dict(trainable_state, strict=False)
        print(f"✓ Model weights loaded successfully")
    
    return model
