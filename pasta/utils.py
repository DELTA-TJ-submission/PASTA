import json
import numpy as np
import cv2
import os
import scanpy as sc
import anndata as ad
import pandas as pd
import glob
import math
from typing import Any, List, Sequence, Union
import tifffile

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Merge common params into each step
    if 'common' in config:
        for key in config:
            if key not in ['description', 'version', 'common'] and isinstance(config[key], dict):
                config[key] = {**config['common'], **config[key]}
    
    return config

def setup_huggingface(hf_cfg):
    """Setup HuggingFace authentication from config or environment."""
    token = hf_cfg.get('token')
    endpoint = hf_cfg.get('endpoint')
    
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if token:
        os.environ["HF_TOKEN"] = token
        from huggingface_hub import login
        login(token=token)
        
def generate_he_mask_not_white(image_np: np.ndarray, white_threshold = 0.8, black_threshold = 0.15, kernel_size = 17) -> np.ndarray:
    '''Generate the mask of the HE image'''
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    img_float = image_np.astype(np.float32) / 255.0

    r = img_float[..., 0]
    g = img_float[..., 1]
    b = img_float[..., 2]

    white_background = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
    black_background = (r < black_threshold) & (g < black_threshold) & (b < black_threshold)
    
    background_mask = white_background | black_background
    foreground_mask = ~background_mask
    
    mask = (foreground_mask * 255).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def load_pathway_config(pathway_file: str) -> dict:
    """Load pathway configurations from separate JSON file."""
    with open(pathway_file, 'r') as f:
        return json.load(f)


def get_pathway_config(pathway_file: str, config_name: str) -> dict:
    """Get specific pathway configuration by name."""
    pathways = load_pathway_config(pathway_file)
    if config_name not in pathways:
        raise ValueError(f"Pathway config '{config_name}' not found in {pathway_file}")
    return pathways[config_name]


def prepare_pathway_prediction(pathway_config: dict, selected_pathways=None, include_tls: bool = False) -> dict:
    """
    Prepare pathways for prediction.
    
    Args:
        pathway_config: dict with 'names' and 'tls_components'
        selected_pathways: None (all), list of names, or list of indices
        include_tls: whether to compute TLS
    
    Returns:
        dict with:
        - 'pathway_names': original full list
        - 'to_predict': list of (name, original_idx, is_tls) tuples
        - 'tls_indices': list of indices for TLS components if needed
    """
    if isinstance(pathway_config['names'], list):
        all_pathways = pathway_config['names']
    elif isinstance(pathway_config['names'], str):
        with open(pathway_config['names'], mode='r') as f:
            all_pathways = [line.strip() for line in f] 
    else:
        raise ValueError("pathway_config['names'] must be a list or a filepath string.")

    if selected_pathways is None:
        pathways_to_predict = all_pathways.copy()
    else:
        # Support both names and indices
        pathways_to_predict = []
        for p in selected_pathways:
            if isinstance(p, int):
                pathways_to_predict.append(all_pathways[p])
            else:
                pathways_to_predict.append(p)
    
    # Build prediction list
    to_predict = []
    for name in pathways_to_predict:
        original_idx = all_pathways.index(name)
        to_predict.append((name, original_idx, False))
    
    # Add TLS if requested
    tls_indices = None
    if include_tls:
        tls_components = pathway_config.get('tls_components', ['B_cells', 'T_cells'])
        tls_indices = [all_pathways.index(c) for c in tls_components]
        to_predict.append(('TLS', None, True))
    
    return {
        'pathway_names': all_pathways,
        'to_predict': to_predict,
        'tls_indices': tls_indices
    }


def npy_to_adata(npz_dir: str) -> sc.AnnData:
    '''Transfer .npy files to .h5ad'''
    files = sorted(glob.glob(f"{npz_dir}/*.npz"))
    gene_names = [os.path.basename(item).removesuffix('.npz').split('_')[0] for item in files]

    arrays = [np.load(f)['data'] for f in files]
    height, width = arrays[0].shape
    n_genes = len(arrays)
    
    # stack gene exp：(height, width, genes) -> (spots, genes)
    stacked = np.stack(arrays, axis=2)  # shape: (height, width, genes)
    expression_matrix = stacked.reshape(-1, n_genes)  # shape: (spots, genes)
    
    y, x = np.mgrid[0:height, 0:width]
    coords = np.column_stack([x.ravel(), y.ravel()])
    
    obs = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]})
    obs.index = [f"spot_{i}" for i in range(len(obs))]
    
    var = pd.DataFrame(index=gene_names or [f"Gene_{i+1}" for i in range(n_genes)])
    
    adata = ad.AnnData(X=expression_matrix, obs=obs, var=var)
    adata.obsm['spatial'] = coords
    
    return adata


def _move_channel_axis_to_last(data: np.ndarray) -> np.ndarray:
    """Ensure channel axis is last when possible."""

    if data.ndim != 3:
        return data

    if data.shape[0] in {1, 3, 4} and data.shape[-1] not in {1, 3, 4}:
        return np.moveaxis(data, 0, -1)

    return data


def _convert_to_supported_dtype(data: np.ndarray) -> np.ndarray:
    """Convert array into uint8 or uint16 as required by QuPath."""

    if np.issubdtype(data.dtype, np.floating):
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            raise ValueError("Array contains no finite values, cannot be converted to image data.")

        finite_values = data[finite_mask]
        min_val = float(finite_values.min())
        max_val = float(finite_values.max())

        if math.isclose(max_val, min_val):
            scaled = np.zeros_like(data, dtype=np.uint8)
        else:
            scaled = (data - min_val) / (max_val - min_val)
            scaled = np.clip(scaled, 0.0, 1.0) * 255.0

        return scaled.astype(np.uint8)

    if np.issubdtype(data.dtype, np.integer):
        min_val = int(data.min())
        max_val = int(data.max())

        if min_val >= 0 and max_val <= 255:
            return data.astype(np.uint8, copy=False)

        if min_val >= 0 and max_val <= 65535:
            return data.astype(np.uint16, copy=False)

        clipped = np.clip(data, 0, 65535)
        return clipped.astype(np.uint16)

    # Fallback: normalise any other dtype into uint8
    data = np.asarray(data, dtype=np.float32)
    data -= data.min()
    if data.max() > 0:
        data /= data.max()
    return (data * 255).astype(np.uint8)


def _to_grayscale(data: np.ndarray) -> np.ndarray:
    """Convert channel-last RGB arrays to single channel grayscale."""

    if data.ndim == 2:
        return data

    if data.shape[-1] == 1:
        return data[..., 0]

    if data.shape[-1] >= 3:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray = np.tensordot(data[..., :3], weights, axes=([2], [0]))
        return gray.astype(data.dtype)

    raise ValueError("Unknown number of channels in the array.")


def _to_rgb(data: np.ndarray) -> np.ndarray:
    """Ensure the array has three colour channels in the last dimension."""
    if data.ndim == 2:
        return np.repeat(data[..., None], 3, axis=2)
    if data.ndim == 3 and data.shape[-1] == 1:
        return np.repeat(data, 3, axis=2)
    if data.ndim == 3 and data.shape[-1] == 3:
        return data

    raise ValueError("RGB mode requires array with shape [H, W, 3] or [H, W].")


def _calc_pyramid_levels(height: int, width: int, tile_size: int) -> int:
    """Estimate how many pyramid levels are required when halving per level."""

    levels = 0
    h, w = height, width
    while min(h, w) > tile_size and h > 1 and w > 1:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)
        levels += 1

    return max(levels, 0)


def _downsample_by_two(data: np.ndarray) -> np.ndarray:
    """Downsample array by factor of 2 using average pooling."""

    h, w = data.shape[:2]
    pad_h = (-h) % 2
    pad_w = (-w) % 2

    if data.ndim == 2:
        pad_cfg = ((0, pad_h), (0, pad_w))
    else:
        pad_cfg = ((0, pad_h), (0, pad_w), (0, 0))

    if pad_h or pad_w:
        padded = np.pad(data, pad_cfg, mode="edge")
    else:
        padded = data

    new_h = padded.shape[0] // 2
    new_w = padded.shape[1] // 2

    if data.ndim == 2:
        down = padded.reshape(new_h, 2, new_w, 2)
        down = down.astype(np.float32).mean(axis=(1, 3))
    else:
        down = padded.reshape(new_h, 2, new_w, 2, padded.shape[2])
        down = down.astype(np.float32).mean(axis=(1, 3))

    if data.dtype == np.uint16:
        return np.clip(np.round(down), 0, 65535).astype(np.uint16)

    return np.clip(np.round(down), 0, 255).astype(np.uint8)


def _build_pyramid(data: np.ndarray, tile_size: int, max_levels: int = 10) -> List[np.ndarray]:
    """Construct pyramid levels (base level first)."""

    levels: List[np.ndarray] = [data]
    current = data

    for _ in range(max_levels):
        if min(current.shape[:2]) <= tile_size:
            break
        current = _downsample_by_two(current)
        levels.append(current)

        if min(current.shape[:2]) <= tile_size:
            break

    return levels


def save_prediction_for_qupath(
    data: np.array,
    output_path: str,
    pixel_size_um: float = 1.0,
    compression: str = "deflate",
    bigtiff: bool | None = None,
    pyramid: bool = True,
    grayscale: bool = True,
    tile_size: int = 512,
) -> None:
    """Save a large array as a QuPath compatible (OME-)TIFF pyramid.

    The function normalises the input data to ``uint8``/``uint16`` automatically,
    enforces channel-last layout, pads odd dimensions when pyramid levels are
    requested, and writes a tiled pyramidal TIFF that QuPath can open directly.
    """

    data = np.squeeze(data)

    if data.ndim < 2:
        raise ValueError("Array must have at least two dimensions (H, W).")

    if data.ndim > 3:
        raise ValueError("Only 2D or 3D (with channels) arrays are supported.")

    data = _move_channel_axis_to_last(data)

    if grayscale:
        data = _to_grayscale(data)
    else:
        data = _to_rgb(data)

    data = _convert_to_supported_dtype(data)
    data = np.ascontiguousarray(data)

    if bigtiff is None:
        bigtiff = data.nbytes >= 2**32 or pyramid

    height, width = data.shape[:2]
    max_dim = max(height, width)

    if tile_size <= 0 or tile_size > max_dim:
        tile_size = 512

    photometric = "minisblack" if data.ndim == 2 else "rgb"
    axes = "YX" if data.ndim == 2 else "YXS"

    levels = _calc_pyramid_levels(height, width, tile_size) if pyramid else 0
    pad_factor = 2 ** levels if levels else 1

    if pyramid and pad_factor > 1:
        pad_h = (pad_factor - height % pad_factor) % pad_factor
        pad_w = (pad_factor - width % pad_factor) % pad_factor

        if pad_h or pad_w:
            pad_config = ((0, pad_h), (0, pad_w)) if data.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
            data = np.pad(data, pad_config, mode="edge")
            height, width = data.shape[:2]

    metadata = {
        "axes": axes,
        "PhysicalSizeX": float(pixel_size_um),
        "PhysicalSizeY": float(pixel_size_um),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
    }

    if photometric == "rgb":
        metadata["SignificantBits"] = 8 if data.dtype == np.uint8 else 16

    compression_arg = None if compression in {None, "none", ""} else compression

    if pyramid:
        pyramid_series = _build_pyramid(data, tile_size)

        base_resolution = 1e4 / pixel_size_um

        with tifffile.TiffWriter(output_path, bigtiff=bigtiff) as tif:
            for idx, level_data in enumerate(pyramid_series):
                level_factor = 2 ** idx
                level_resolution = (base_resolution / level_factor, base_resolution / level_factor)

                write_kwargs = {
                    "compression": compression_arg,
                    "photometric": photometric,
                    "metadata": metadata if idx == 0 else None,
                    "resolution": level_resolution,
                    "resolutionunit": "CENTIMETER",
                    "subfiletype": 0 if idx == 0 else 1,
                }

                if max(level_data.shape[:2]) >= tile_size:
                    write_kwargs["tile"] = (tile_size, tile_size)

                tif.write(level_data, **write_kwargs)
    else:
        write_kwargs = {
            "bigtiff": bigtiff,
            "compression": compression_arg,
            "photometric": photometric,
            "metadata": metadata,
            "resolution": (1e4 / pixel_size_um, 1e4 / pixel_size_um),
            "resolutionunit": "CENTIMETER",
        }

        if max_dim > tile_size:
            write_kwargs["tile"] = (tile_size, tile_size)

        tifffile.imwrite(output_path, data, **write_kwargs)

    print(
        f"Saved to {output_path} | Shape: {data.shape} | Dtype: {data.dtype} | "
        f"Mode: {'grayscale' if photometric == 'minisblack' else 'RGB'} | "
        f"Pyramid: {pyramid}"
    )
