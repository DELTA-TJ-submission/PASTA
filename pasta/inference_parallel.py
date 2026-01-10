#!/usr/bin/env python3
"""
Parallel inference script for PASTA model using multiple GPUs.

This script enables parallel processing of multiple WSI slides across multiple GPUs, speeding up inference for large datasets.

Usage:
    # Process all slides in h5_path using 4 GPUs
    python inference_parallel.py --config pasta/configs/inference.json --num_gpus 4
    
    # Process specific slides
    python inference_parallel.py --config pasta/configs/inference.json --num_gpus 2 --h5_files slide1.h5 slide2.h5
    
    # Use specific GPU IDs
    python inference_parallel.py --config pasta/configs/inference.json --gpu_ids 0 2 3
"""

import multiprocessing as mp
import concurrent.futures
import glob
import os
import sys
import argparse
import torch
from typing import List, Optional, Tuple

from pasta.inference_without_process import predict_pixel_level, predict_spot_level
from pasta.model import PASTA
from pasta.model_utils import get_img_transforms, load_model_weights
from pasta.utils import (
    load_config, 
    setup_huggingface, 
    get_pathway_config, 
    prepare_pathway_prediction
)
import h5py


def process_single_slide(args: Tuple[str, dict, int, int, bool]) -> str:
    """
    Process a single slide on a specific GPU.
    
    Args:
        args: Tuple of (h5_file_path, config_dict, gpu_id, slide_index, use_runtime_extraction)
        
    Returns:
        Status message string
    """
    h5_file, config, gpu_id, slide_idx, use_runtime_extraction = args
    slide_name = os.path.basename(h5_file)
    
    # Initialize model variable for cleanup
    model = None
    
    try:
        print(f"[GPU {gpu_id}] [{slide_idx}] Starting: {slide_name}")
        
        cfg = config['inference']
        device = f'cuda:{gpu_id}'
        
        # Set the current CUDA device
        torch.cuda.set_device(gpu_id)
        
        # Load pathway configuration
        pathway_config = get_pathway_config(cfg['pathway_file'], cfg['pathway_config'])
        pathway_info = prepare_pathway_prediction(
            pathway_config,
            selected_pathways=cfg.get('selected_pathways'),
            include_tls=cfg.get('include_tls', False)
        )
        
        # Initialize model
        img_transforms = get_img_transforms(cfg['backbone_model_name'])
        model = PASTA(
            model_name=cfg['backbone_model_name'], 
            pathway_dim=len(pathway_info['pathway_names']),
            non_negative=False
        )
        model = load_model_weights(model, cfg['model_path'])
        model = model.to(device)  # Move model to GPU
        model.eval()
        
        # Get prediction mode
        prediction_mode = cfg.get('prediction_mode', 'pixel')
        
        # Use no_grad context to prevent gradient accumulation
        with torch.no_grad():
            if prediction_mode == 'pixel':
                # High-resolution pixel-level prediction
                predict_pixel_level(
                    h5_file, 
                    model, 
                    img_transforms,
                    cfg['wsi_path'],
                    cfg['edge_info_path'],
                    pathway_info,
                    patch_size=cfg['patch_size'],
                    output_path=cfg.get('output_path', 'results/predictions'),
                    save_h5ad=cfg.get('save_h5ad', False),
                    save_tiffs=cfg.get('save_tiffs', False),
                    downsample_size=cfg.get('downsample_size', 10),
                    figsize=cfg.get('figsize', 10),
                    device=device,
                    file_type=cfg.get('file_type', '.svs'),
                    draw_images=cfg.get('draw_images', True),
                    cmap=cfg.get('cmap', 'turbo'),
                    use_mask=cfg.get('use_mask', False),
                    blank=cfg.get('blank', 3000),
                    use_runtime_extraction=use_runtime_extraction,
                    num_workers=0  # Must be 0 in multiprocessing context
                )
            elif prediction_mode == 'spot':
                # Low-resolution spot-level prediction
                predict_spot_level(
                    h5_file,
                    model,
                    img_transforms,
                    pathway_names=pathway_config['names'],
                    output_path=cfg.get('output_path', 'results/predictions_h5ad'),
                    device=device,
                    wsi_path=cfg.get('wsi_path'),
                    file_type=cfg.get('file_type', '.svs'),
                    use_mask=cfg.get('use_mask', False),
                    blank=cfg.get('blank', 3000),
                    downsample_size=cfg.get('downsample_size', 10),
                    filter_background=cfg.get('filter_background', True),
                    use_runtime_extraction=use_runtime_extraction,
                    patch_size=cfg['patch_size'],
                    num_workers=0  # Must be 0 in multiprocessing context
                )
            else:
                raise ValueError(f"Invalid prediction_mode: {prediction_mode}")
        
        print(f"[GPU {gpu_id}] [{slide_idx}] ✓ Completed: {slide_name}")
        return f"SUCCESS: {slide_name} (GPU {gpu_id})"
        
    except Exception as e:
        error_msg = f"ERROR: {slide_name} (GPU {gpu_id}) - {str(e)}"
        print(f"[GPU {gpu_id}] [{slide_idx}] ✗ Failed: {slide_name}")
        print(f"[GPU {gpu_id}] Error details: {str(e)}")
        return error_msg
    
    finally:
        # Critical: Clean up GPU memory after each slide
        if model is not None:
            # Move model to CPU and delete
            model.cpu()
            del model
        
        # Clear any cached tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize(device=gpu_id)


def get_available_gpus(gpu_ids: Optional[List[int]] = None) -> List[int]:
    """
    Get list of available GPU IDs.
    
    Args:
        gpu_ids: Optional list of specific GPU IDs to use
        
    Returns:
        List of GPU IDs
    """
    if gpu_ids is not None:
        return gpu_ids
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available!")
    
    return list(range(num_gpus))


def filter_completed_slides(h5_files: List[str], config: dict) -> List[str]:
    """
    Filter out slides that have already been processed.
    
    Args:
        h5_files: List of H5 file paths
        config: Configuration dictionary
        
    Returns:
        List of H5 files that need processing
    """
    cfg = config['inference']
    output_path = cfg.get('output_path', 'results/predictions')
    downsample_size = cfg.get('downsample_size', 10)
    prediction_mode = cfg.get('prediction_mode', 'pixel')
    
    # Get pathway info to check all expected outputs
    pathway_config = get_pathway_config(cfg['pathway_file'], cfg['pathway_config'])
    pathway_info = prepare_pathway_prediction(
        pathway_config,
        selected_pathways=cfg.get('selected_pathways'),
        include_tls=cfg.get('include_tls', False)
    )
    
    remaining_files = []
    
    for h5_file in h5_files:
        sample_id = os.path.basename(h5_file).removesuffix('.h5')
        
        if prediction_mode == 'pixel':
            # Check if all pathway predictions exist
            all_exist = True
            for pathway_name, _, _ in pathway_info['to_predict']:
                pathway_file = f"{output_path}/{sample_id}/{pathway_name.replace('/', '_')}_downsample_{downsample_size}.npz"
                if not os.path.exists(pathway_file):
                    all_exist = False
                    break
            
            if not all_exist:
                remaining_files.append(h5_file)
            else:
                print(f"Skipping (already processed): {sample_id}")
        
        elif prediction_mode == 'spot':
            output_file = os.path.join(output_path, f"{sample_id}.h5ad")
            if not os.path.exists(output_file):
                remaining_files.append(h5_file)
            else:
                print(f"Skipping (already processed): {sample_id}")
    
    return remaining_files


def main():
    parser = argparse.ArgumentParser(
        description='Parallel inference for PASTA model across multiple GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='pasta/configs/inference_Cohort.json',
        help='Path to inference configuration JSON file'
    )
    
    parser.add_argument(
        '--num_gpus', 
        type=int, 
        default=None,
        help='Number of GPUs to use (default: all available)'
    )
    
    parser.add_argument(
        '--gpu_ids', 
        type=int, 
        nargs='+',
        default=None,
        help='Specific GPU IDs to use (e.g., --gpu_ids 0 2 3)'
    )
    
    parser.add_argument(
        '--h5_files', 
        type=str, 
        nargs='+',
        default=None,
        help='Specific H5 files to process (default: all files in h5_path)'
    )
    
    parser.add_argument(
        '--skip_completed',
        action='store_true',
        help='Skip slides that have already been processed'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be processed without actually running'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup HuggingFace if needed
    if config.get('huggingface', {}).get('endpoint'):
        setup_huggingface(config['huggingface'])
    
    # Get H5 files to process
    if args.h5_files:
        h5_files = args.h5_files
    else:
        h5_path = config['inference']['h5_path']
        h5_files = sorted(glob.glob(f"{h5_path}/*.h5"))
    
    if not h5_files:
        print("ERROR: No H5 files found!")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Filter completed slides if requested
    if args.skip_completed:
        print("\nChecking for already processed slides...")
        h5_files = filter_completed_slides(h5_files, config)
        print(f"Remaining files to process: {len(h5_files)}")
    
    if not h5_files:
        print("All slides already processed!")
        sys.exit(0)
    
    # Auto-detect if runtime extraction is needed
    use_runtime_extraction = config['inference'].get('use_runtime_extraction', None)
    if use_runtime_extraction is None and len(h5_files) > 0:
        # Check first H5 file to see if it contains image data
        with h5py.File(h5_files[0], 'r') as f:
            has_img_data = ('img' in f) or ('imgs' in f)
            use_runtime_extraction = not has_img_data
            if use_runtime_extraction:
                print("\n[Auto-detected] H5 files contain only coordinates (from create_patches_fp)")
                print("[Auto-detected] Enabling runtime extraction mode - images will be extracted from WSI on-the-fly")
            else:
                print("\n[Auto-detected] H5 files contain pre-extracted images (from process_patch_h5)")
    elif use_runtime_extraction:
        print("\n[Config] Runtime extraction mode enabled")
    else:
        print("\n[Config] Using pre-extracted images from H5 files")
    
    # Get GPU configuration
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    elif args.num_gpus:
        gpu_ids = list(range(args.num_gpus))
    else:
        gpu_ids = get_available_gpus()
    
    num_gpus = len(gpu_ids)
    print(f"\nUsing {num_gpus} GPUs: {gpu_ids}")
    print(f"Processing {len(h5_files)} slides")
    print(f"Prediction mode: {config['inference'].get('prediction_mode', 'pixel')}")
    print(f"Output path: {config['inference'].get('output_path', 'results/predictions')}")
    print(f"Runtime extraction: {use_runtime_extraction}")
    
    # Print file list
    print("\nFiles to process:")
    for i, h5_file in enumerate(h5_files, 1):
        print(f"  {i:3d}. {os.path.basename(h5_file)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without processing")
        sys.exit(0)
    
    # Prepare tasks: assign each file to a GPU in round-robin fashion
    tasks = []
    for i, h5_file in enumerate(h5_files):
        gpu_id = gpu_ids[i % num_gpus]
        tasks.append((h5_file, config, gpu_id, i + 1, use_runtime_extraction))
    
    print(f"\n{'='*80}")
    print("Starting parallel processing...")
    print(f"{'='*80}\n")
    
    # Create process pool and run
    # Use 'spawn' to avoid CUDA initialization issues
    mp.set_start_method('spawn', force=True)
    
    results = []
    executor = None
    
    try:
        # Use ProcessPoolExecutor with proper timeout and error handling
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=mp.get_context('spawn')
        )
        
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_slide, task): task 
            for task in tasks
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            h5_file = task[0]
            slide_name = os.path.basename(h5_file)
            
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per slide
                results.append(result)
            except concurrent.futures.TimeoutError:
                error_msg = f"ERROR: {slide_name} - Timeout (exceeded 1 hour)"
                print(f"⏱ Timeout: {slide_name}")
                results.append(error_msg)
            except Exception as e:
                error_msg = f"ERROR: {slide_name} - {str(e)}"
                print(f"✗ Exception: {slide_name} - {str(e)}")
                results.append(error_msg)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Keyboard interrupt detected! Cleaning up processes...")
        if executor:
            # Shutdown executor and kill all running processes
            executor.shutdown(wait=False, cancel_futures=True)
        print("✓ Cleanup complete. Exiting...")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n✗ Unexpected error in main process: {str(e)}")
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    
    finally:
        # Ensure executor is properly closed
        if executor:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception as e:
                print(f"Warning: Error during executor shutdown: {str(e)}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    error_count = sum(1 for r in results if r.startswith("ERROR"))
    
    print(f"Summary:")
    print(f"  Total slides: {len(results)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {error_count}")
    
    if error_count > 0:
        print(f"\nFailed slides:")
        for result in results:
            if result.startswith("ERROR"):
                print(f"  {result}")
        sys.exit(1)
    else:
        print("\n✓ All slides processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
