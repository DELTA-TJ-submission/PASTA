#!/usr/bin/env python3
"""A quick demo for PASTA"""
import os
import sys
import glob
import traceback
from tqdm import tqdm

from pasta.utils import load_config, setup_huggingface

def run_patch_extraction(config):
    """Extract patches from WSI."""
    from pasta.create_patches_fp import seg_and_patch
    cfg = config['patch_extraction']

    for key in ['patches', 'masks', 'stitches']:
        os.makedirs(os.path.join(cfg['save_dir'], key), exist_ok=True)
    
    if not os.path.exists(cfg['source']):
        raise FileNotFoundError(f"WSI image source not found: {cfg['source']}")
    
    # Directly pass all parameters from config
    seg_and_patch(**cfg)


def process_h5_file(config):
    """Process H5 file."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pasta.process_patch_h5 import process_file

    cfg = config['h5_processing']
    file_list = glob.glob(f'{cfg["h5_path"]}/*.h5')
    
    with ThreadPoolExecutor(max_workers=cfg['max_workers']) as executor:
        futures = {
            executor.submit(
                process_file, f, cfg['patch_size'], cfg['slide_path'], cfg['file_type'], 
                cfg['mask_path'], cfg.get('save_edge', False), cfg.get('edge_info_path'), 
                cfg.get('blank', 3000)
            ): f for f in file_list
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())


def run_model_inference(config):
    """Run model inference on patches."""
    from pasta.inference import run_inference_pipeline

    cfg = config['inference']
    run_inference_pipeline(cfg)


def main():
    # Load config
    config = load_config('pasta/configs/demo.json')
    print(f"Config loaded: {config.get('description', 'PASTA Demo')}")
    if config['huggingface']['endpoint']:
        setup_huggingface(config['huggingface'])
    # Run pipeline
    try:
        print("🚀 Starting demo...")
        model_path = config['inference']['model_path']
        if not os.path.exists(model_path):
            print("\n [0/3] Downloading PASTA model weights...")
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="mengflz/pasta-tumor",
                filename=os.path.basename(model_path),
                local_dir=os.path.dirname(model_path) or ".",
                local_dir_use_symlinks=False
            )
            os.rename(downloaded_path, model_path)
            print(f"✓ Model weights downloaded to: {model_path}")
        
        print("\n[1/3] Extracting patches...")
        run_patch_extraction(config)
        print("✓ Patch extraction complete")
        
        print("\n[2/3] Processing H5 files...")
        process_h5_file(config)
        print("✓ H5 processing complete")
        
        print("\n[3/3] Running inference...")
        run_model_inference(config)
        print("✓ Inference complete")
        
        print("\n✓ Demo complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

