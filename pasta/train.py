from tqdm import tqdm
import os
import gc
import json
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from pasta.dataset import H5TileDataset
from pasta.model_utils import *
from pasta.model import PASTA
from pasta.utils import setup_huggingface
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
OUT_RESIZE_MODEL = ['Virchow', 'Virchow2','Kaiko-B','H-optimus-0','H-optimus-1','Kaiko-L','Hibou-B','Hibou-L','PLIP','UNIv2']


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def prepare_tile_dataset(train_df, i, img_transforms, batch_size, cfg, mask=True, sample=None, augment=False):
    """Prepare tile dataset with paths from config."""
    info_type = cfg['pathway']['info_type']
    
    sample_id = train_df.loc[i, 'id']
 
    tile_h5_path = f"{cfg['data']['tile_h5_base']}/{sample_id}.h5"
    info_path = f"{cfg['data']['info_base']}/{info_type}/{sample_id}.csv"
    
    validate_file(tile_h5_path, f"File {tile_h5_path} not exist!")
    validate_file(info_path, f"File {info_path} not exist!")
    
    tile_dataset = H5TileDataset(tile_h5_path, 
                                    info_path, 
                                    chunk_size=batch_size, 
                                    img_transform=img_transforms,
                                    mask=mask,
                                    sample_ratio=sample,
                                    augment=augment
                                    )
    return tile_dataset


def main(rank, world_size, cfg):
    """Main training function using DDP."""
 
    info_type = cfg['pathway']['info_type']
    num_pathways = cfg['pathway']['num_pathways']
    alpha = cfg['training']['alpha']
    random_seed = cfg['data']['random_seed']
    model_name = cfg['model']['backbone_model']
    mask = cfg['model']['mask']
    train_test_split_flg = cfg['data'].get('train_test_split', True)

    out_model_dir = cfg['output']['output_dir']
    os.makedirs(out_model_dir, exist_ok=True)
    meta = pd.read_csv(cfg['data']['meta_path'], index_col=0) 
    print(f'Total {meta.shape[0]} files')

    if train_test_split_flg:
        train_df, test_df = train_test_split(meta, test_size=1-cfg['data']['train_ratio'], random_state=random_seed) 
        test_df, selected_val = train_test_split(test_df, test_size=0.2, random_state=random_seed) 
        
        train_df.to_csv(os.path.join(out_model_dir, 'train_df.csv')); test_df.to_csv(os.path.join(out_model_dir, 'test_df.csv')); selected_val.to_csv(os.path.join(out_model_dir, 'val_df.csv'))
    else:
        train_df = meta
        selected_val = None
        test_df = None     
    
    # Get image transforms based on model name
    img_transforms = get_img_transforms(model_name)
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize PASTA model
    model = PASTA(
        model_name=model_name,
        pathway_dim=num_pathways,
        non_negative=cfg['model']['non_negative'],
        enable_attention_hooks=cfg['model']['enable_attention_hooks'],
        scale=cfg['model']['scale']
    ).to(device)
    
    if cfg['model']['warm_weight_path']:
        model.load_state_dict(torch.load(cfg['model']['warm_weight_path']))
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # initialize stop signal
    stop_signal = torch.tensor([0], device=device)

    epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['learning_rate']
    drop_ratio = cfg['model']['drop_ratio']
    sample_ratio = cfg['training']['sample_ratio']
    warmup_steps = cfg['training']['warmup_steps']
    augment = cfg['training']['augment']

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps, lr)
    step_scheduler = StepLR(optimizer, step_size=cfg['scheduler']['step_size'], gamma=cfg['scheduler']['gamma'])
    criterion = nn.HuberLoss()
    recon_criterion = nn.MSELoss()

    if rank == 0:
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        writer = SummaryWriter(os.path.join(out_model_dir, f'runs/{current_time}/'))
        early_stopper = EarlyStopping(
            patience=cfg['early_stopping']['patience'],
            stop_epoch=cfg['early_stopping']['stop_epoch'],
            verbose=True
        )
    
    for epoch in range(epochs):
        train_loss_tot = 0.0; recon_loss_tot = 0.0; reg_loss_tot = 0.0
        test_loss_tot = 0.0; test_recon_loss_tot = 0.0; test_reg_loss_tot = 0.0
        if rank == 0:
            print(f'Epoch:{epoch+1}')
            start_time = time.time()
            epoch_iterator = tqdm(train_df.index, desc=f"Epoch {epoch+1}")
            epoch_iterator_test = tqdm(selected_val.index, desc="Validation") if selected_val is not None else []
        else:
            epoch_iterator = train_df.index
            epoch_iterator_test = selected_val.index if selected_val is not None else []

        for i in epoch_iterator:
            train_loss = 0.0; train_recon_loss = 0.0; train_reg_loss = 0.0
            tile_dataset = prepare_tile_dataset(train_df, i, img_transforms, batch_size, cfg, mask=mask, sample=sample_ratio, augment=augment)

            if tile_dataset is None:
                continue
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(tile_dataset, num_replicas=world_size, rank=rank)
            tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                                      batch_size=1, 
                                                      shuffle=False,
                                                      num_workers=2,
                                                      sampler=train_sampler)
            for batch_idx, batch in enumerate(tile_dataloader):
                batch = post_collate_fn(batch)
                imgs_raw = batch['imgs']; imm_score = batch['imm_score']

                imgs_raw = imgs_raw.to(device); imm_score = imm_score.to(device)
                if mask:
                    mask_ = batch['mask']
                    mask_ = mask_.to(device)
                    imgs = imgs_raw*mask_
                else:
                    imgs = imgs_raw
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if torch.rand(1).item() > drop_ratio:
                        _, pred_mean, pred_recon = model(imgs,imm_score)
                    else:
                        _, pred_mean, pred_recon = model(imgs)
                    reg_loss = criterion(pred_mean, imm_score)
                    if model_name in OUT_RESIZE_MODEL:
                        pred_recon = F.interpolate(pred_recon, size=(224, 224), mode="bilinear")
                    recon_loss = recon_criterion(imgs_raw, pred_recon)
                    loss = reg_loss + 0.5*recon_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                train_loss = train_loss + loss.item(); train_recon_loss = train_recon_loss + recon_loss.item(); train_reg_loss = train_reg_loss + reg_loss.item()

            train_loss_tot = train_loss_tot + train_loss / len(tile_dataloader)
            recon_loss_tot = recon_loss_tot + train_recon_loss / len(tile_dataloader)
            reg_loss_tot = reg_loss_tot + train_reg_loss / len(tile_dataloader)
            del tile_dataloader, tile_dataset
            torch.cuda.empty_cache()        
            
        if epoch < warmup_steps:
            warmup_scheduler.step()
        else:
            step_scheduler.step()
        for i in epoch_iterator_test:
            test_loss = 0.0; test_recon_loss = 0.0; test_reg_loss = 0.0
            tile_dataset = prepare_tile_dataset(selected_val, i, img_transforms, 64, cfg, mask=mask, augment=False)

            if tile_dataset is None:
                continue
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(tile_dataset, num_replicas=world_size, rank=rank)
            tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                                      batch_size=1, 
                                                      shuffle=False,
                                                      num_workers=2,
                                                      sampler=train_sampler)
            for batch_idx, batch in enumerate(tile_dataloader):
                batch = post_collate_fn(batch)
                imgs_raw = batch['imgs']; imm_score = batch['imm_score']

                imgs_raw = imgs_raw.to(device); imm_score = imm_score.to(device)
                if mask:
                    mask_ = batch['mask']
                    mask_ = mask_.to(device)
                    imgs = imgs_raw*mask_
                else:
                    imgs = imgs_raw
                
                with torch.inference_mode(): 
                    # drop gene info to prevent shortcut
                    _, pred_mean, pred_recon = model(imgs)
                    reg_loss = criterion(pred_mean, imm_score)
                    if model_name in OUT_RESIZE_MODEL:
                        pred_recon = F.interpolate(pred_recon, size=(224, 224), mode="bilinear")
                    recon_loss = recon_criterion(imgs_raw, pred_recon)
                    loss = reg_loss + alpha*recon_loss
                test_loss = test_loss + loss.item(); test_recon_loss = test_recon_loss + recon_loss.item(); test_reg_loss = test_reg_loss + reg_loss.item()
            test_loss_tot = test_loss_tot + test_loss/len(tile_dataloader)
            test_recon_loss_tot = test_recon_loss_tot + test_recon_loss/len(tile_dataloader)
            test_reg_loss_tot = test_reg_loss_tot + test_reg_loss/len(tile_dataloader)
            del tile_dataloader, tile_dataset
            gc.collect()
            torch.cuda.empty_cache() 
        if rank == 0:
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f'Epoch {epoch+1}/{epochs}, Time:{epoch_duration:.2f} seconds')
            writer.add_scalar('Train Loss', float(train_loss_tot/len(train_df.index)), epoch) 
            writer.add_scalar('Recon Loss', float(recon_loss_tot/len(train_df.index)), epoch) 
            writer.add_scalar('Reg Loss', float(reg_loss_tot/len(train_df.index)), epoch) 
            if epoch >= cfg['output']['save_start_epoch'] and (epoch+1) % cfg['output']['save_interval'] == 0:
                torch.save(model.module.state_dict(),  os.path.join(out_model_dir, f'model_state_dict_{epoch+1}.pth')) 
                print(f'Epoch {epoch+1} model saved!') 
            early_stopper(epoch, train_loss_tot/len(train_df.index), model, ckpt_name=os.path.join(out_model_dir, f'best_model.pt'))
            if early_stopper.early_stop:
                print("Early stopping")
                stop_signal.fill_(1) 
        dist.broadcast(stop_signal, src=0)  # broadcast stop signal
        if stop_signal.item() == 1:
            break  
            
    cleanup()
    print('Done')


if __name__=='__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    with open('pasta/configs/train.json', 'r') as f:
        cfg = json.load(f)
    
    setup_huggingface(cfg)
    main(rank, world_size, cfg)

    # shell cmd:
    # CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_addr="localhost" train.py &> logs/train.log
    