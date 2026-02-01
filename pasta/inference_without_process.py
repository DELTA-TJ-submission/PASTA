import torch
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import openslide
import numpy as np
import h5py
import scanpy as sc
import glob
import cv2
from scipy.sparse import csr_matrix
import torch.nn.functional as F

from pasta.model import PASTA
from pasta.dataset import H5TileDataset_infer, H5TileDataset_runtime
from pasta.model_utils import post_collate_fn, get_img_transforms, load_model_weights, weight_map_make, weight_matrix_make
from pasta.utils import generate_he_mask_not_white, get_pathway_config, prepare_pathway_prediction, load_config, npy_to_adata, save_prediction_for_qupath, setup_huggingface


def generate_large_images(tile_dataloader, patch_size_pxl, model, H_, W_, start_x, start_y, weight_map, pathway_idx=None, is_tls=False, tls_indices=None, loss_model=False, device='cuda:0', downsample_size=1):
    H_down = H_ // downsample_size
    W_down = W_ // downsample_size
    large_images = torch.zeros((1, H_down, W_down), dtype=torch.float16)
    
    patch_size_down = patch_size_pxl // downsample_size
    weight_matrix = weight_matrix_make(patch_size_down//2, patch_size_down)
    if weight_matrix.shape[0] % 2 == 0:
        r1 = patch_size_down // 2; r2 = patch_size_down // 2
    else:
        r1 = patch_size_down // 2; r2 = patch_size_down // 2 + 1
        
    if not torch.is_tensor(weight_matrix):
        weight_matrix = torch.tensor(weight_matrix, dtype=torch.float16)
    weight_matrix = weight_matrix.to(device)

    model = model.to(device)
    with torch.inference_mode():
        for batch in tqdm(tile_dataloader, total=len(tile_dataloader), desc="Pixel-level Inference"):
            batch = post_collate_fn(batch)
            imgs = batch['imgs'].to(device)
            coords_ = batch['coords'].numpy()
            if loss_model:
                pred, _, _,_,_ = model(imgs)
            else:
                pred, _, _ = model(imgs)
                
            if is_tls and tls_indices:
                tls_pred = torch.stack([pred[:, idx:idx+1, ...] for idx in tls_indices]).mean(dim=0)
                pred_resized = F.interpolate(tls_pred, size=(patch_size_down, patch_size_down),
                                         mode='bicubic', align_corners=False).squeeze(0)
            elif pathway_idx is not None:
                pred_resized = F.interpolate(pred[:,pathway_idx:pathway_idx+1, ...], size=(patch_size_down, patch_size_down),
                             mode='bicubic', align_corners=False).squeeze(0)
            else:
                raise ValueError("Either pathway_idx or (is_tls and tls_indices) must be provided")

            weighted_pred = pred_resized * weight_matrix
            weighted_pred_cpu = weighted_pred.cpu() 

            del pred, pred_resized, weighted_pred 
            torch.cuda.empty_cache()

            for c_idx, coord in enumerate(coords_):
                j, i = coord
                j = j-start_x; i=i-start_y
                # Convert to downsampled coordinates
                i_down = int(i // downsample_size)
                j_down = int(j // downsample_size)
                if (i_down - r1<0) or (j_down - r1<0):
                    negative_coord_flag=True
                    continue
                try:
                    # weight_map is already in downsampled space
                    pred_normlized = weighted_pred_cpu[c_idx, ...] / weight_map[0, i_down - r1 : i_down + r2, j_down - r1 : j_down + r2]
                    large_images[0, i_down-r1:i_down+r2, j_down-r1:j_down+r2] += pred_normlized.squeeze()
                except Exception as e:
                    breakpoint()
    return large_images


def predict_pixel_level(tile_h5_path, model, img_transforms, wsi_path, edge_info_path, pathway_info, patch_size, output_path='/workspace/results/predictions', save_h5ad=False, save_tiffs=False, downsample_size=10, figsize=10, device='cuda:0', file_type='.svs', draw_images=True, cmap='turbo', use_mask=False, blank=3000, use_runtime_extraction=False, num_workers=1): 
    '''High-resolution pixel-level prediction with optional visualization and h5ad file saving.
    Args:
        tile_h5_path: the path of the tile h5 file
        model: the PASTA model
        img_transforms: image transformations
        wsi_path: the path of the wsi file
        edge_info_path: the path of the edge info file
        pathway_info: dict from prepare_pathway_prediction() with 'to_predict' and 'tls_indices'
        patch_size: the size of the patch
        output_path: the path of the output file
        save_h5ad: whether to save h5ad file
        downsample_size: the size of the downsample
        device: device for inference
        file_type: the type of the file
        draw_images: whether to draw the images
        cmap: colormap for visualization
        use_mask: whether h5 files are generated from mask processing
        blank: blank space around the image (default: 3000)
        use_runtime_extraction: if True, extract images from WSI on-the-fly instead of reading from H5
    '''
    sample_id = os.path.basename(tile_h5_path).removesuffix('.h5')
    if use_mask:
        wsi_base_name = sample_id.split('_')[0]  # e.g., "mask_name_roi" -> "mask_name"
    else:
        wsi_base_name = sample_id
    
    os.makedirs(os.path.join(output_path, sample_id),exist_ok=True)
    os.makedirs(os.path.join(output_path, f'{sample_id}/plots'),exist_ok=True)
    os.makedirs(os.path.join(output_path, f'{sample_id}/plots_overlay'),exist_ok=True)
    
    print(f'Start {sample_id}')

    # Choose dataset based on runtime extraction mode
    if use_runtime_extraction:
        wsi_file_path = f"{wsi_path}/{wsi_base_name}{file_type}"
        print(f"Using runtime extraction from WSI: {wsi_file_path}")
        tile_dataset = H5TileDataset_runtime(tile_h5_path, wsi_file_path, patch_size=patch_size, 
                                            chunk_size=64, img_transform=img_transforms)
    else:
        tile_dataset = H5TileDataset_infer(tile_h5_path, chunk_size=64, img_transform=img_transforms)
    
    tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                              batch_size=1, 
                                              shuffle=False,
                                              num_workers=num_workers)

    slide = openslide.open_slide(f"{wsi_path}/{wsi_base_name}{file_type}") 
    H,W = slide.dimensions   
    
    # Load coordinates from appropriate h5 file
    if use_mask:
        # For mask h5 files, get coords from the original h5 file
        original_h5 = os.path.join(os.path.dirname(tile_h5_path).replace('_masks', ''), f'{wsi_base_name}.h5')
        coords_h5_file = original_h5 if os.path.exists(original_h5) else tile_h5_path
    else:
        coords_h5_file = tile_h5_path
    
    with h5py.File(coords_h5_file, 'r') as f:
        coords = f['coords'][:]
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)    
    x_origin = min_x - blank//2   
    y_origin = min_y - blank//2  
    w_origin = max_x - min_x + blank  
    h_origin = max_y - min_y + blank 

    level = slide.get_best_level_for_downsample(downsample_size)        
    factor = slide.level_downsamples[level]               
    w_lv, h_lv = int(w_origin / factor), int(h_origin / factor)
    weight_matrix = weight_matrix_make(patch_size//2, patch_size)
    weight_map = weight_map_make(y_origin,x_origin,h_origin,w_origin,weight_matrix,coords,downsample_size=downsample_size)

    thumb = slide.read_region((x_origin, y_origin), level, (w_lv, h_lv)).convert("RGB")
    thumb = thumb.resize((w_origin//downsample_size, h_origin//downsample_size))
    gray = cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2GRAY) 
    
    # Generate tissue mask
    img_slide_arr = np.array(thumb)
    tissue_mask = generate_he_mask_not_white(img_slide_arr)
    pathway_names = [name for name, _, _ in pathway_info['to_predict']]

    for pathway_name, original_idx, is_tls in pathway_info['to_predict']:
        print(f'Start {sample_id}-{pathway_name}')

        if os.path.exists(f'{output_path}/{sample_id}/{pathway_name.replace("/","_")}_downsample_{downsample_size}.npz'):
            print(f'File exists: {output_path}/{sample_id}/{pathway_name.replace("/","_")}_downsample_{downsample_size}.npz, skip.')
            continue

        if is_tls:
            large_image = generate_large_images(
                tile_dataloader, patch_size, model, h_origin, w_origin, x_origin, y_origin, weight_map,
                is_tls=True, tls_indices=pathway_info['tls_indices'],
                downsample_size=downsample_size, device=device
            )
        else:
            large_image = generate_large_images(
                tile_dataloader, patch_size, model, h_origin, w_origin, x_origin, y_origin, weight_map,
                pathway_idx=original_idx, downsample_size=downsample_size, device=device
            )
        large_image_small = large_image[0].cpu().numpy() 
        np.savez_compressed(f'{output_path}/{sample_id}/{pathway_name.replace("/","_")}_downsample_{downsample_size}.npz', data=large_image_small)
        print(f'Save {pathway_name} npz file')
        if save_tiffs:
            save_prediction_for_qupath(large_image_small, f'{output_path}/{sample_id}/{pathway_name.replace("/","_")}_downsample_{downsample_size}.tiff')

        if draw_images:
            plt.switch_backend('Agg')
            large_image_small_norm = (large_image_small - np.min(large_image_small)) / (np.max(large_image_small) - np.min(large_image_small))        
            vmin, vmax = np.percentile(large_image_small_norm, (2, 98))
            large_image_small_norm = large_image_small_norm[:tissue_mask.shape[0], :tissue_mask.shape[1]]
            image_tensor_erosed = large_image_small_norm*(tissue_mask/255)
            valid_mask = tissue_mask.astype(bool)
            image_nan = np.where(valid_mask, image_tensor_erosed.astype(float), np.nan)
            plt.figure(figsize=(figsize,figsize), dpi=300)
            plt.axis('off')
            plt.imshow(image_nan, cmap=cmap, interpolation='nearest', vmin=vmin
                       #norm=PowerNorm(gamma=0.7,vmin=vmin)
                      )
            plt.savefig(f'{output_path}/{sample_id}/plots/{pathway_name}_turbo.png', bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(figsize,figsize), dpi=300)
            plt.imshow(gray,cmap='gray',interpolation='nearest')
            plt.imshow(image_nan,cmap='jet',interpolation='nearest',vmin=vmin, alpha=0.35)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{output_path}/{sample_id}/plots_overlay/{pathway_name}_turbo_overlay.png', bbox_inches='tight', dpi=300)
            plt.close()
            print(f'Saved {output_path}/{sample_id}/{pathway_name} images')
    
    if save_h5ad:
        adata = npy_to_adata(os.path.join(output_path, sample_id))
        adata.write_h5ad(f'{output_path}/{sample_id}/{sample_id}_downsample_{downsample_size}.h5ad')


def predict_spot_level(tile_h5_path, model, img_transforms, pathway_names, output_path='/workspace/results/predictions_h5ad', device='cuda:0', wsi_path=None, file_type='.svs', use_mask=False, blank=3000, downsample_size=10, filter_background=True, use_runtime_extraction=False, patch_size=224, num_workers=1):
    '''Low-resolution spot-level prediction for quick prediction and large datasets, output as h5ad format.
    
    Args:
        tile_h5_path: path to the H5 file with patches
        model: PASTA model
        img_transforms: image transformations
        pathway_names: list of pathway names
        output_path: output directory for h5ad files
        device: device for inference
        wsi_path: path to WSI files directory (required if filter_background=True)
        file_type: WSI file extension (default: '.svs')
        use_mask: whether h5 files are generated from mask processing
        blank: blank space around the image (default: 3000)
        downsample_size: downsample size for generating tissue mask (default: 10)
        filter_background: whether to filter out background spots using tissue mask (default: True)
        use_runtime_extraction: if True, extract images from WSI on-the-fly instead of reading from H5
        patch_size: patch size for runtime extraction (default: 224)
    '''
    
    model = model.to(device)
    os.makedirs(output_path, exist_ok=True)
    sample_id = os.path.basename(tile_h5_path).removesuffix('.h5')
    print(f'Start spot-level prediction: {sample_id}')
    
    # Determine WSI base name
    if use_mask:
        wsi_base_name = sample_id.split('_')[0]  # e.g., "mask_name_roi" -> "mask_name"
    else:
        wsi_base_name = sample_id
    
    # Choose dataset based on runtime extraction mode
    if use_runtime_extraction:
        if wsi_path is None:
            raise ValueError("wsi_path must be provided when use_runtime_extraction=True")
        wsi_file_path = f"{wsi_path}/{wsi_base_name}{file_type}"
        print(f"Using runtime extraction from WSI: {wsi_file_path}")
        tile_dataset = H5TileDataset_runtime(tile_h5_path, wsi_file_path, patch_size=patch_size,
                                            chunk_size=64, img_transform=img_transforms)
    else:
        tile_dataset = H5TileDataset_infer(tile_h5_path, chunk_size=64, img_transform=img_transforms)
    
    tile_dataloader = torch.utils.data.DataLoader(
        tile_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers
    )
    
    # Load coordinates from h5 file
    if use_mask:
        # For mask h5 files, get coords from the original h5 file
        original_h5 = os.path.join(os.path.dirname(tile_h5_path).replace('_masks', ''), f'{wsi_base_name}.h5')
        coords_h5_file = original_h5 if os.path.exists(original_h5) else tile_h5_path
    else:
        coords_h5_file = tile_h5_path
    
    with h5py.File(coords_h5_file, 'r') as f:
        coords_from_h5 = f['coords'][:]
    
    with torch.inference_mode():
        pred_out_mean_list = []
        coords_list = []
        
        for batch_idx, batch in tqdm(enumerate(tile_dataloader), total=len(tile_dataloader), desc="Spot-level inference"):
            batch = post_collate_fn(batch)
            imgs = batch['imgs'].to(device)
            coords_list.append(batch['coords'])
            
            _, pred_mean, _ = model(imgs)
            pred_out_mean_list.append(pred_mean.detach().cpu())
        
        pred_out_mean = torch.vstack(pred_out_mean_list)
        coords = torch.vstack(coords_list)
    
    # Filter background spots using tissue mask if requested
    if filter_background and wsi_path is not None:
        print(f'Filtering background spots using tissue mask...')
        try:
            # Load WSI and generate tissue mask
            slide = openslide.open_slide(f"{wsi_path}/{wsi_base_name}{file_type}")
            H, W = slide.dimensions
            
            # Calculate bounding box
            min_x, min_y = coords_from_h5.min(axis=0)
            max_x, max_y = coords_from_h5.max(axis=0)
            x_origin = min_x - blank // 2
            y_origin = min_y - blank // 2
            w_origin = max_x - min_x + blank
            h_origin = max_y - min_y + blank
            
            # Read thumbnail for mask generation
            level = slide.get_best_level_for_downsample(downsample_size)
            factor = slide.level_downsamples[level]
            w_lv, h_lv = int(w_origin / factor), int(h_origin / factor)
            
            thumb = slide.read_region((x_origin, y_origin), level, (w_lv, h_lv)).convert("RGB")
            thumb = thumb.resize((w_origin // downsample_size, h_origin // downsample_size))
            
            # Generate tissue mask
            img_slide_arr = np.array(thumb)
            tissue_mask = generate_he_mask_not_white(img_slide_arr)
            tissue_mask_bool = tissue_mask.astype(bool)
            
            # Filter spots based on tissue mask
            valid_spots = []
            coords_np = coords.numpy()
            
            for i, coord in enumerate(coords_np):
                x_pixel, y_pixel = coord[0], coord[1]
                # Convert pixel coordinates to mask coordinates
                x_mask = int((x_pixel - x_origin) // downsample_size)
                y_mask = int((y_pixel - y_origin) // downsample_size)
                
                # Check if coordinate is within mask bounds and in tissue
                if (0 <= x_mask < tissue_mask_bool.shape[0] and 
                    0 <= y_mask < tissue_mask_bool.shape[1] and
                    tissue_mask_bool[x_mask, y_mask]):
                    valid_spots.append(i)
            
            if len(valid_spots) == 0:
                print(f'Warning: No valid tissue spots found for {sample_id}. Saving all spots.')
                valid_spots = list(range(len(coords)))
            else:
                print(f'Filtered {len(coords) - len(valid_spots)} background spots, keeping {len(valid_spots)} tissue spots.')
            
            # Filter predictions and coordinates
            pred_out_mean = pred_out_mean[valid_spots]
            coords = coords[valid_spots]
            
        except Exception as e:
            print(f'Warning: Failed to filter background spots for {sample_id}: {e}')
            print('Saving all spots without background filtering.')
    
    adata = sc.AnnData(X=csr_matrix(pred_out_mean.numpy()))
    
    coords_df = pd.DataFrame(
        coords.numpy(), 
        index=[f'spot_{i}' for i in range(len(pred_out_mean))], 
        columns=['raw_row', 'raw_col']
    )
    coords_df['array_row'] = coords_df['raw_row'].rank(method='dense').astype(int) - 1
    coords_df['array_col'] = coords_df['raw_col'].rank(method='dense').astype(int) - 1
    
    adata.obs["x_array"] = coords_df['array_row'].values
    adata.obs["y_array"] = coords_df['array_col'].values
    adata.obs['x_pixel'] = coords[:, 0].numpy()
    adata.obs['y_pixel'] = coords[:, 1].numpy()
    adata.obsm['spatial'] = adata.obs[['x_pixel', 'y_pixel']].values
    
    spot_ids = [f'spot_{i}' for i in range(len(pred_out_mean))]
    adata.obs_names = pd.Index(spot_ids)
    adata.var_names = pd.Index(pathway_names)
    
    output_file = os.path.join(output_path, f"{sample_id}.h5ad")
    adata.write(output_file)
    print(f'Saved spot-level prediction: {output_file} (n_spots={len(adata)})')
    
    return adata


def run_inference_pipeline(cfg):
    """
    Run inference pipeline from config.
    
    Args:
        cfg: inference config prediction.json or demo.json['inference'] block
    """
    
    # Load pathway config from separate file
    pathway_config = get_pathway_config(cfg['pathway_file'], cfg['pathway_config'])
    pathway_info = prepare_pathway_prediction(
        pathway_config,
        selected_pathways=cfg.get('selected_pathways'),
        include_tls=cfg.get('include_tls', False)
    )
    
    print(f"Predicting pathways: {[name for name, _, _ in pathway_info['to_predict']]}")
    
    # Initialize model
    img_transforms = get_img_transforms(cfg['backbone_model_name'])
    model = PASTA(
        model_name=cfg['backbone_model_name'], 
        pathway_dim=len(pathway_info['pathway_names']),
        non_negative=False
    )
    print(f"Initializing model with pathway_dim={len(pathway_info['pathway_names'])}")
    model = load_model_weights(model, cfg['model_path'])
    model.eval()
    
    # Process files
    pat_list = glob.glob(f"{cfg['h5_path']}/*.h5")
    print(f"Found {len(pat_list)} H5 files to process")
    
    # Auto-detect if runtime extraction is needed
    use_runtime_extraction = cfg.get('use_runtime_extraction', None)
    if use_runtime_extraction is None and len(pat_list) > 0:
        # Check first H5 file to see if it contains image data
        with h5py.File(pat_list[0], 'r') as f:
            has_img_data = ('img' in f) or ('imgs' in f)
            use_runtime_extraction = not has_img_data
            if use_runtime_extraction:
                print("Auto-detected: H5 files contain only coordinates (from create_patches_fp)")
                print("Enabling runtime extraction mode - images will be extracted from WSI on-the-fly")
            else:
                print("Auto-detected: H5 files contain pre-extracted images (from process_patch_h5)")
    elif use_runtime_extraction:
        print("Runtime extraction mode enabled via config")
    else:
        print("Using pre-extracted images from H5 files")
    
    # Get prediction mode
    prediction_mode = cfg.get('prediction_mode', 'pixel')
    if prediction_mode == 'pixel':
        print(f"Running pixel-level (high-resolution) prediction...")
        for tile_h5_path in pat_list:
            predict_pixel_level(
                tile_h5_path, model, img_transforms, 
                cfg['wsi_path'], cfg['edge_info_path'], 
                pathway_info, 
                patch_size=cfg['patch_size'],
                output_path=cfg.get('output_path', 'results/predictions'),
                save_h5ad=cfg.get('save_h5ad', False),
                save_tiffs=cfg.get('save_tiffs', False),
                downsample_size=cfg.get('downsample_size', 10),
                figsize=cfg.get('figsize', 10),
                device=cfg.get('device', 'cuda:0'),
                file_type=cfg.get('file_type', '.svs'),
                draw_images=cfg.get('draw_images', True),
                cmap=cfg.get('cmap', 'turbo'),
                use_mask=cfg.get('use_mask', False),
                blank=cfg.get('blank', 3000),
                use_runtime_extraction=use_runtime_extraction
            )
    
    elif prediction_mode == 'spot':
        print(f"Running spot-level (low-resolution) prediction...")
        for tile_h5_path in pat_list:
            predict_spot_level(
                tile_h5_path, model, img_transforms,
                pathway_names=pathway_config['names'],
                output_path=cfg.get('output_path', 'results/predictions_h5ad'),
                device=cfg.get('device', 'cuda:0'),
                wsi_path=cfg.get('wsi_path'),
                file_type=cfg.get('file_type', '.svs'),
                use_mask=cfg.get('use_mask', False),
                blank=cfg.get('blank', 3000),
                downsample_size=cfg.get('downsample_size', 10),
                filter_background=cfg.get('filter_background', True),
                use_runtime_extraction=use_runtime_extraction,
                patch_size=cfg['patch_size']
            )
    
    else:
        raise ValueError(f"Invalid prediction_mode: {prediction_mode}. Must be 'pixel' or 'spot'.")

    
if __name__ == "__main__":
    config = load_config('code/configs/inference.json')
    if config['huggingface']['endpoint']:
        setup_huggingface(config['huggingface'])
    run_inference_pipeline(config['inference'])
