# Before running this script, make sure you have finished:
# 1. Training related models and saved weights in /workspace/results/train/[model_name]/
# 2. Aggregating Xenium or VisiumHD data into h5 files with different granularity(64µm, 32µm, 16µm), save to /workspace/data/benchmark/st_agg/[spot_size]um/

import openslide
import torch
import torch.nn.functional as F
import pandas as pd
import os
import gc
from tqdm import tqdm
import pickle
import h5py
import numpy as np
import scanpy as sc
from scipy.sparse import spmatrix
from sklearn.metrics import mean_squared_error

from pasta.model import PASTA
from pasta.dataset import H5TileDataset, H5TileDataset_infer
from pasta.model_utils import post_collate_fn, get_img_transforms, weight_map_make, weight_matrix_make

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def min_max_scale(x, min_val, max_val):
    """Scale values to [0, 1] range."""
    return (x - min_val) / (max_val - min_val)


def structural_similarity(im1, im2, M=1):
    """Calculate SSIM between two 1D arrays."""
    im1, im2 = im1 / im1.max(), im2 / im2.max()
    mu1, mu2 = im1.mean(), im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1, C2, C3 = (k1 * L) ** 2, (k2 * L) ** 2, ((k2 * L) ** 2) / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return l12 * c12 * s12


def calculate_metrics(pred, gt):
    """Calculate PCC, RMSE, and SSIM metrics between predictions and ground truth."""
    correlation_list, rmse_list, ssim_list = [], [], []
    
    for i in range(gt.shape[1]):
        gt_, pred_ = gt[:, i], pred[:, i]
        
        # PCC
        correlation_list.append(np.corrcoef(gt_, pred_)[0, 1])
        
        # Scale for RMSE and SSIM
        combined = np.concatenate((gt_, pred_), axis=0)
        min_val, max_val = combined.min(), combined.max()
        gt_scaled = min_max_scale(gt_, min_val, max_val)
        pred_scaled = min_max_scale(pred_, min_val, max_val)

        #RMSE
        mse = mean_squared_error(gt_scaled, pred_scaled)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        # gt_2D = np.zeros((num_rows, num_cols))
        # for index, value in zip(zip(rows, cols), gt_scaled):
        #     row, col = index
        #     gt_2D[row, col] = value

        # pred_2D = np.zeros((num_rows, num_cols))
        # for index, value in zip(zip(rows, cols), pred_scaled):
        #     row, col = index
        #     pred_2D[row, col] = value.item()
        ssim = structural_similarity(gt_scaled, pred_scaled, M=1)
        ssim_list.append(ssim)
    return correlation_list, rmse_list, ssim_list


def load_pasta_model(model_path, model_name, pathway_dim, device='cuda:0'):
    """Load PASTA model from checkpoint."""
    model = PASTA(
        non_negative=False,
        model_name=model_name,
        enable_attention_hooks=False,
        pathway_dim=pathway_dim
    )
    state_dict = torch.load(model_path, map_location='cpu')
    
    if 'best_model' in model_path or any('module.' in key for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def access_bins(sample_id, spot_um, adata_dir, gene_names):
    """Load ground truth data for a specific spatial resolution."""
    adata = sc.read_h5ad(f'{adata_dir}/{spot_um}um/{sample_id}.h5ad')
    if not adata.var.index.is_unique:
        adata = adata[:, ~adata.var.index.duplicated(keep='first')]
    
    bin_coords = adata.obsm['spatial']
    common_genes = [gene for gene in gene_names if gene in adata.var_names]
    common_indices = [gene_names.index(gene) for gene in common_genes]
    
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes found between gene_names and adata.var_names.")
    
    gt = adata[:, common_genes].X
    return bin_coords, gt, common_indices


def aggregate_spot_predictions(tile_dataloader, model, sample_id, weight_matrix, weight_map,
                               patch_size, mpp, pathway_dim, spot_um_list, adata_dir, 
                               gene_names, device='cuda:0', save_dir=None):  # pylint: disable=dangerous-default-value
    """Aggregate patch-level predictions to spot-level for multiple resolutions."""
    pred_spot_dict, gt_spot_dict = {}, {}
    common_indices_dict, bin_coords_dict = {}, {}
    negative_coord_flag = False
    
    r1 = patch_size // 2
    r2 = patch_size // 2 if weight_matrix.shape[0] % 2 == 0 else patch_size // 2 + 1
    
    if not torch.is_tensor(weight_matrix):
        weight_matrix = torch.from_numpy(weight_matrix).float()
    weight_matrix = weight_matrix.to(device)
    
    for spot_um in spot_um_list:
        bin_coords, gt, common_indices = access_bins(sample_id, spot_um, adata_dir, gene_names)
        gt_spot_dict[spot_um] = gt
        pred_spot_dict[spot_um] = np.zeros((bin_coords.shape[0], pathway_dim))
        common_indices_dict[spot_um] = common_indices
        bin_coords_dict[spot_um] = bin_coords
    
    with torch.inference_mode():
        for batch in tqdm(tile_dataloader, total=len(tile_dataloader), desc="Inference & Aggregation"):
            batch = post_collate_fn(batch)
            imgs = batch['imgs'].to(device)
            coords_ = batch['coords'].numpy()
            
            pred, _, _ = model(imgs)
            pred_resized = F.interpolate(pred, size=(patch_size, patch_size),
                                       mode='bicubic', align_corners=False).squeeze(0)
            weighted_pred = (pred_resized * weight_matrix).cpu()
            
            del pred, pred_resized
            torch.cuda.empty_cache()
            
            for c_idx, coord in enumerate(coords_):
                j, i = int(coord[0]), int(coord[1])
                if (i - r1 < 0) or (j - r1 < 0):
                    negative_coord_flag = True
                    continue
                
                pred_normalized = weighted_pred[c_idx, ...] / weight_map[0, i - r1:i + r2, j - r1:j + r2]
                
                for spot_um in spot_um_list:
                    diff = np.abs(coord - bin_coords_dict[spot_um])
                    current_bin_size = spot_um / mpp
                    mask = ((diff[:, 0] <= current_bin_size / 2 + patch_size / 2) &
                           (diff[:, 1] <= current_bin_size / 2 + patch_size / 2))
                    contain_bins = bin_coords_dict[spot_um][mask]
                    contain_index = np.where(mask)[0]
                    
                    if len(contain_bins) > 0:
                        for idx, (bj, bi) in enumerate(contain_bins):
                            left = int(max(i - patch_size / 2, bi - current_bin_size / 2) - i + patch_size / 2)
                            right = int(min(i + patch_size / 2, bi + current_bin_size / 2) - i + patch_size / 2)
                            high = int(max(j - patch_size / 2, bj - current_bin_size / 2) - j + patch_size / 2)
                            low = int(min(j + patch_size / 2, bj + current_bin_size / 2) - j + patch_size / 2)
                            pred_spot_value = pred_normalized[:, left:right, high:low].sum(axis=(-1, -2))
                            pred_spot_dict[spot_um][contain_index[idx]] += pred_spot_value
    
    for spot_um in spot_um_list:
        pred_spot_dict[spot_um] = pred_spot_dict[spot_um][:, common_indices_dict[spot_um]] / (224 * 224)
        
        if negative_coord_flag:
            zero_row_indices = np.where(np.all(pred_spot_dict[spot_um] == 0, axis=1))[0]
            pred_spot_dict[spot_um] = np.delete(pred_spot_dict[spot_um], zero_row_indices, axis=0)
            gt_spot_dict[spot_um] = np.delete(gt_spot_dict[spot_um], zero_row_indices, axis=0)
            bin_coords_dict[spot_um] = np.delete(bin_coords_dict[spot_um], zero_row_indices, axis=0)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.savez_compressed(f"{save_dir}/{sample_id}_{spot_um}.npz", data=pred_spot_dict[spot_um])
    
    return pred_spot_dict, gt_spot_dict, bin_coords_dict


def evaluate_fine_grained(model, sample_list, model_name, h5_dir, wsi_dir, adata_dir,
                          gene_names, spot_um_list=None, device='cuda:0', 
                          pathway_dim=100, save_dir=None):
    """Evaluate model at fine-grained multi-resolution level."""
    if spot_um_list is None:
        spot_um_list = [64, 32, 16]
    tot_results = {}
    
    for sample_id in sample_list:
        tile_h5_path = f'{h5_dir}/{sample_id}.h5'
        wsi_path = f'{wsi_dir}/{sample_id}.tif'
        
        with h5py.File(tile_h5_path, 'r') as f:
            coords = f['coords'][:]
            img_attrs = dict(f['img'].attrs)
        
        patch_size = img_attrs.get('patch_size_src', img_attrs.get('patch_size'))
        if patch_size is None:
            raise ValueError("patch_size not found in h5 attributes")
        mpp = img_attrs['pixel_size']
        
        wsi = openslide.open_slide(wsi_path)
        H, W = wsi.dimensions
        weight_matrix = weight_matrix_make(overlap=patch_size // 2, patch_size=patch_size)
        weight_map = weight_map_make(0, 0, W, H, weight_matrix, coords)
        
        tile_dataset = H5TileDataset_infer(tile_h5_path, chunk_size=64, 
                                          img_transform=get_img_transforms(model_name))
        tile_dataloader = torch.utils.data.DataLoader(tile_dataset, batch_size=1, 
                                                      shuffle=False, num_workers=1)
        
        print(f'Start - {sample_id} - Inference')
        
        pred_spot_dict, gt_spot_dict, bin_coords_dict = aggregate_spot_predictions(
            tile_dataloader, model, sample_id, weight_matrix, weight_map, patch_size, mpp,
            pathway_dim, spot_um_list, adata_dir, gene_names, device, save_dir
        )
        
        sample_dict = tot_results.setdefault(sample_id, {})
        
        for spot_um in spot_um_list:
            if isinstance(gt_spot_dict[spot_um], spmatrix):
                gt_spot_dict[spot_um] = gt_spot_dict[spot_um].toarray()
            
            correlation_list, rmse_list, ssim_list = calculate_metrics(
                pred_spot_dict[spot_um], gt_spot_dict[spot_um]
            )
            print(f'{sample_id}-{spot_um}um: PCC={np.mean(correlation_list):.3f}, '
                  f'RMSE={np.mean(rmse_list):.3f}, SSIM={np.mean(ssim_list):.3f}')
            
            bin_dict = sample_dict.setdefault(spot_um, {"PCC": [], "SSIM": [], "RMSE": []})
            bin_dict["PCC"] = correlation_list
            bin_dict["SSIM"] = ssim_list
            bin_dict["RMSE"] = rmse_list
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return tot_results


def evaluate_spot_level(model, sample_list, model_name, h5_dir, info_dir, 
                       device='cuda:0'):
    """Evaluate model at spot level with direct comparison."""
    tot_results = {'PCC': {}, 'SSIM': {}, 'RMSE': {}}
    
    for sample_id in sample_list:
        tile_h5_path = f'{h5_dir}/{sample_id}.h5'
        info_path = f'{info_dir}/{sample_id}.csv'
        
        print(f'Processing {sample_id}')
        
        tile_dataset = H5TileDataset(tile_h5_path, info_path, chunk_size=64,
                                     img_transform=get_img_transforms(model_name))
        tile_dataloader = torch.utils.data.DataLoader(tile_dataset, batch_size=1,
                                                      shuffle=False, num_workers=1)
        
        pred_list, info_values_list, coords_list = [], [], []
        
        with torch.inference_mode():
            for batch in tqdm(tile_dataloader, total=len(tile_dataloader)):
                batch = post_collate_fn(batch)
                imgs = batch['imgs'].to(device)
                info_values = batch['info_values']
                
                info_values_list.append(info_values)
                coords_list.append(batch['coords'])
                
                _, pred_mean, _ = model(imgs)
                pred_list.append(pred_mean.detach().cpu())
        
        pred_out = torch.cat(pred_list, dim=0)
        info_values_tensor = torch.cat(info_values_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        
        correlation_list, rmse_list, ssim_list = calculate_metrics(
            pred_out.numpy(), info_values_tensor.numpy()
        )
        
        tot_results['PCC'][sample_id] = correlation_list
        tot_results['RMSE'][sample_id] = rmse_list
        tot_results['SSIM'][sample_id] = ssim_list
        
        print(f'{sample_id}: PCC={np.mean(correlation_list):.3f}, '
              f'RMSE={np.mean(rmse_list):.3f}, SSIM={np.mean(ssim_list):.3f}')
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return tot_results


def run_fine_grained_benchmark(model_name, model_path, meta_csv, h5_dir, wsi_dir, 
                               adata_dir, gene_names_file, output_dir, 
                               spot_um_list=None, device='cuda:0', pathway_dim=100):
    """Run fine-grained benchmark evaluation."""
    if spot_um_list is None:
        spot_um_list = [64, 32, 16]
    
    with open(gene_names_file, 'r', encoding='utf-8') as f:
        gene_names = [line.strip() for line in f]
    
    meta = pd.read_csv(meta_csv, index_col=0)
    meta = meta[(meta.datatype == 'ST') & (meta.st_technology.isin(['Visium HD', 'Xenium']))]
    sample_list = meta.id.tolist()
    
    model = load_pasta_model(model_path, model_name, pathway_dim, device)
    
    save_dir = os.path.join(output_dir, 'predictions')
    tot_results = evaluate_fine_grained(model, sample_list, model_name, h5_dir, wsi_dir,
                                       adata_dir, gene_names, spot_um_list, device, 
                                       pathway_dim, save_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'fine_grained_metrics.pkl'), 'wb') as f:
        pickle.dump(tot_results, f)
    
    print(f'Fine-grained benchmark for {model_name} completed!')
    return tot_results


def run_spot_level_benchmark(model_name, model_path, sample_list, h5_dir, info_dir,
                             output_dir, device='cuda:0', pathway_dim=100):
    """Run spot-level benchmark evaluation."""
    model = load_pasta_model(model_path, model_name, pathway_dim, device)
    
    tot_results = evaluate_spot_level(model, sample_list, model_name, h5_dir, 
                                     info_dir, device)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'spot_level_metrics.pkl'), 'wb') as f:
        pickle.dump(tot_results, f)
    
    print(f'Spot-level benchmark for {model_name} completed!')
    return tot_results


def main():
    """Benchmark evaluation"""
    # Configuration
    model_list = ['UNI', 'UNIv2', 'Virchow2', 'Virchow', 'Phikonv2', 'Phikon', 
                  'gigapath', 'Kaiko-L', 'Kaiko-B', 'Hibou-B', 'Hibou-L', 
                  'CONCH', 'PLIP', 'H-optimus-0', 'H-optimus-1']
    
    # Example: Fine-grained evaluation
    for model_name in model_list:
        model_path = f'/workspace/results/train/{model_name}/best_model.pt'
        run_fine_grained_benchmark(
            model_name=model_name,
            model_path=model_path,
            meta_csv='/workspace/data/benchmark/meta.csv',
            h5_dir='/workspace/data/benchmark/patches',
            wsi_dir='/workspace/data/benchmark/wsis',
            adata_dir='/workspace/data/benchmark/st_agg',
            gene_names_file='pasta/configs/gene_names/rep_gene_names.txt',
            output_dir=f'/workspace/results/train/{model_name}/',
            spot_um_list=[64, 32, 16],
            device='cuda:0',
            pathway_dim=100
        )


if __name__ == '__main__':
    main()
