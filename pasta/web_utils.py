#!/usr/bin/env python3
"""Utility functions for web interface."""

import os
import glob
import shutil
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import scanpy as sc

from pasta.utils import load_pathway_config


def scan_model_files(model_dir: str = "model") -> Dict[str, List[str]]:
    """
    Scan model directory for available model files (recursively).
    
    Args:
        model_dir: Directory containing model files (can have subdirectories)
        
    Returns:
        Dictionary mapping backbone names to model file paths
    """
    if not os.path.exists(model_dir):
        return {}
    
    # Recursively scan for .pt and .pth files in all subdirectories
    # Supports structures like:
    #   model/
    #     pasta-neuro/
    #       UNI_neuro3_no_pos.pt
    #       UNIv2_neuro1_pos.pt
    #     pasta-tumor/
    #       ...
    model_files = glob.glob(os.path.join(model_dir, "**/*.pt"), recursive=True) + \
                  glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
    
    model_files = [os.path.abspath(f) for f in model_files]
    model_mapping = {}
    
    backbone_patterns = {
        "UNI": ["uni", "UNI"],
        "UNIv2": ["univ2", "UNI2", "uni2"],
        "CONCH": ["conch", "CONCH"],
        "Virchow": ["virchow", "Virchow"],
        "Virchow2": ["virchow2", "Virchow2"],
        "gigapath": ["gigapath", "Gigapath"],
        "Phikon": ["phikon", "Phikon"],
        "Phikonv2": ["phikonv2", "phikon-v2", "Phikonv2"],
        "Kaiko-B": ["kaiko-b", "kaiko_b", "KaikoB"],
        "Kaiko-L": ["kaiko-l", "kaiko_l", "KaikoL"],
        "H-optimus-0": ["H-optimus-0", "h_optimus_0", "Hoptimus0"],
        "H-optimus-1": ["H-optimus-1", "h_optimus_1", "Hoptimus1"],
        "Hibou-B": ["Hibou-B", "hibou_b", "HibouB"],
        "Hibou-L": ["Hibou-L", "hibou_l", "HibouL"],
        "PLIP": ["plip", "PLIP"],
    }
    
    # Sort backbones by maximum pattern length (descending) to match longer patterns first
    # This ensures "phikonv2" matches before "phikon", "univ2" before "uni", etc.
    def get_max_pattern_length(patterns):
        return max(len(p) for p in patterns)
    
    sorted_backbones = sorted(
        backbone_patterns.items(),
        key=lambda x: get_max_pattern_length(x[1]),
        reverse=True
    )
    
    for model_file in model_files:
        basename = os.path.basename(model_file)
        basename_lower = basename.lower()
        
        matched = False
        for backbone, patterns in sorted_backbones:
            sorted_patterns = sorted(patterns, key=len, reverse=True)
            
            for pattern in sorted_patterns:
                if pattern.lower() in basename_lower:
                    if backbone not in model_mapping:
                        model_mapping[backbone] = []
                    model_mapping[backbone].append(model_file)
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            if "Unknown" not in model_mapping:
                model_mapping["Unknown"] = []
            model_mapping["Unknown"].append(model_file)
    
    return model_mapping


def get_pathway_choices(pathway_file: str = "pasta/configs/pathways.json") -> List[Tuple[str, str]]:
    """
    Get available pathway configurations.
    
    Args:
        pathway_file: Path to pathways configuration file
        
    Returns:
        List of (display_name, config_key) tuples
    """
    try:
        pathways = load_pathway_config(pathway_file)
        choices = []
        
        for key, config in pathways.items():
            if key != "description":
                display_name = config.get("description", key)
                choices.append((f"{display_name} ({key})", key))
        
        return choices
    except Exception as e:
        print(f"Error loading pathway configs: {e}")
        return [("default_14", "default_14")]


def get_pathway_names(pathway_file: str, pathway_config: str) -> List[str]:
    """
    Get pathway names for a given configuration.
    
    Args:
        pathway_file: Path to pathways configuration file
        pathway_config: Configuration key
        
    Returns:
        List of pathway names
    """
    try:
        pathways = load_pathway_config(pathway_file)
        config = pathways.get(pathway_config, {})
        
        names = config.get("names", [])
        if isinstance(names, str):
            # It's a file path
            with open(names, 'r') as f:
                names = [line.strip() for line in f]
        
        return names
    except Exception as e:
        print(f"Error loading pathway names: {e}")
        return []


def create_temp_config(
    wsi_path: str,
    task_output_dir: str,
    backbone_model: str,
    model_path: str,
    pathway_config: str,
    selected_pathways: Optional[List[str]],
    prediction_mode: str,
    downsample_size: int,
    include_tls: bool,
    patch_size: int,
    step_size: int,
    figsize: int,
    device: str,
    file_type: str,
    save_tiffs: bool,
    cmap: str,
    hf_token: Optional[str] = None,
    hf_endpoint: Optional[str] = None
) -> Dict:
    """
    Create temporary configuration for prediction pipeline.
    
    Args:
        Various configuration parameters
        
    Returns:
        Configuration dictionary
    """
    # Determine file type from WSI path
    if not file_type:
        ext = os.path.splitext(wsi_path)[1]
        file_type = ext if ext else ".tif"
    
    wsi_dir = os.path.dirname(wsi_path)
    wsi_filename = os.path.basename(wsi_path)
    
    final_hf_token = hf_token if hf_token and hf_token.strip() else os.environ.get("HF_TOKEN")
    final_hf_endpoint = hf_endpoint if hf_endpoint and hf_endpoint.strip() else os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    if not model_path:
        model_path = "model/Phikonv2_14_no_pos.pt"
    
    config = {
        "description": "Web UI temporary configuration",
        "huggingface": {
            "token": final_hf_token,
            "endpoint": final_hf_endpoint
        },
        "patch_extraction": {
            "source": wsi_dir,  
            "save_dir": task_output_dir,
            "patch_save_dir": os.path.join(task_output_dir, "patches"),
            "mask_save_dir": os.path.join(task_output_dir, "masks"),
            "stitch_save_dir": os.path.join(task_output_dir, "stitches"),
            "patch_size": patch_size,
            "step_size": step_size,
            "patch_level": 0,
            "auto_patch_size": False,
            "use_default_params": False,
            "seg": True,
            "patch": True,
            "stitch": False,
            "save_mask": True,
            "auto_skip": False,
            "process_list": None,
            "seg_params": {
                "seg_level": -1,
                "sthresh": 5,
                "mthresh": 1,
                "close": 4,
                "use_otsu": False,
                "keep_ids": "none",
                "exclude_ids": "none"
            },
            "filter_params": {
                "a_t": 2,
                "a_h": 4,
                "max_n_holes": 8
            },
            "vis_params": {
                "vis_level": -1,
                "line_thickness": 100
            },
            "patch_params": {
                "use_padding": True,
                "contour_fn": "four_pt"
            }
        },
        "h5_processing": {
            "h5_path": os.path.join(task_output_dir, "patches"),
            "slide_path": os.path.dirname(wsi_path),
            "file_type": file_type,
            "mask_path": None,
            "max_workers": 8,
            "save_edge": True,
            "edge_info_path": os.path.join(task_output_dir, "edge_info"),
            "blank": 3000,
            "patch_size": patch_size  # Required for process_file function
        },
        "inference": {
            "backbone_model_name": backbone_model,
            "model_path": model_path,
            "h5_path": os.path.join(task_output_dir, "patches"),
            "wsi_path": os.path.dirname(wsi_path),
            "edge_info_path": os.path.join(task_output_dir, "edge_info"),
            "output_path": os.path.join(task_output_dir, "predictions"),
            "file_type": file_type,
            "pathway_file": "pasta/configs/pathways.json",
            "pathway_config": pathway_config,
            "selected_pathways": selected_pathways,
            "include_tls": include_tls,
            "prediction_mode": prediction_mode,
            "patch_size": patch_size,
            "downsample_size": downsample_size,
            "figsize": figsize,
            "device": device,
            "draw_images": True,
            "save_h5ad": True,
            "save_tiffs": save_tiffs,
            "cmap": cmap,
            "use_mask": False,
            "blank": 3000
        }
    }
    return config


def format_results_for_display(result_paths: Dict[str, str]) -> Tuple[List[str], List[str], Optional[str], str]:
    """
    Format prediction results for display in Gradio.
    
    Args:
        result_paths: Dictionary of result file paths
        
    Returns:
        Tuple of (heatmap_images, overlay_images, h5ad_path, summary_text)
    """
    heatmaps = []
    overlays = []
    h5ad_path = None
    summary_lines = []
    
    if "prediction_dir" in result_paths:
        pred_dir = result_paths["prediction_dir"]
        
        # Find sample directories
        if os.path.exists(pred_dir):
            for sample_dir in os.listdir(pred_dir):
                sample_path = os.path.join(pred_dir, sample_dir)
                if os.path.isdir(sample_path):
                    # Look for heatmaps
                    plots_dir = os.path.join(sample_path, "plots")
                    if os.path.exists(plots_dir):
                        plot_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")))
                        heatmaps.extend(plot_files)
                    
                    # Look for overlays
                    overlay_dir = os.path.join(sample_path, "plots_overlay")
                    if os.path.exists(overlay_dir):
                        overlay_files = sorted(glob.glob(os.path.join(overlay_dir, "*.png")))
                        overlays.extend(overlay_files)
                    
                    h5ad_files = glob.glob(os.path.join(sample_path, "*.h5ad"))
                    if h5ad_files:
                        h5ad_path = h5ad_files[0]
                    
                    summary_lines.append(f"## Sample: {sample_dir}")
                    summary_lines.append(f"- Heatmaps generated: {len(plot_files) if 'plot_files' in locals() else 0}")
                    summary_lines.append(f"- Overlay images: {len(overlay_files) if 'overlay_files' in locals() else 0}")
                    if h5ad_path:
                        file_size = os.path.getsize(h5ad_path) / (1024 * 1024)
                        summary_lines.append(f"- H5AD file size: {file_size:.2f} MB")
    
    summary_text = "\n".join(summary_lines) if summary_lines else "No results found"
    
    return heatmaps, overlays, h5ad_path, summary_text


def cleanup_old_tasks(base_dir: str = "results/web_tasks", max_age_hours: int = 24):
    """
    Clean up old task directories.
    
    Args:
        base_dir: Base directory for task outputs
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of directories cleaned up
    """
    if not os.path.exists(base_dir):
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned = 0
    
    for task_dir in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_dir)
        if os.path.isdir(task_path):
            # Check directory age
            dir_mtime = os.path.getmtime(task_path)
            if (current_time - dir_mtime) > max_age_seconds:
                try:
                    shutil.rmtree(task_path)
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to clean up {task_path}: {e}")
    
    return cleaned


def get_result_statistics(h5ad_path: str) -> pd.DataFrame:
    """
    Extract statistics from h5ad file for display.
    
    Args:
        h5ad_path: Path to h5ad file
        
    Returns:
        DataFrame with statistics
    """
    try:
        adata = sc.read_h5ad(h5ad_path)
        
        # Extract basic statistics
        stats = []
        for var_name in adata.var_names[:20]: 
            values = adata[:, var_name].X
            if hasattr(values, 'toarray'):
                values = values.toarray()
            values = np.array(values).flatten()
            
            stats.append({
                "Pathway": var_name,
                "Mean": f"{np.mean(values):.4f}",
                "Std": f"{np.std(values):.4f}",
                "Min": f"{np.min(values):.4f}",
                "Max": f"{np.max(values):.4f}",
            })
        
        return pd.DataFrame(stats)
    except Exception as e:
        print(f"Error extracting statistics: {e}")
        return pd.DataFrame({"Error": [str(e)]})

