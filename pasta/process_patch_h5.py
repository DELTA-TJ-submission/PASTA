import glob
import h5py
import numpy as np
import openslide
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import sys
import argparse
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely import affinity
import rasterio.features
import traceback

def get_effective_bounds(slide):
    """get slide valid bound info"""
    bounds_x = slide.properties.get('openslide.bounds-x')
    bounds_y = slide.properties.get('openslide.bounds-y') 
    bounds_w = slide.properties.get('openslide.bounds-width')
    bounds_h = slide.properties.get('openslide.bounds-height')
    
    if bounds_x and bounds_y and bounds_w and bounds_h:
        bounds_x, bounds_y = int(bounds_x), int(bounds_y)
        bounds_w, bounds_h = int(bounds_w), int(bounds_h)
        print(f"bounds detected: offset=({bounds_x}, {bounds_y}), size=({bounds_w}, {bounds_h})")
        return bounds_x, bounds_y, bounds_w, bounds_h
    else:
        W, H = slide.dimensions
        print(f"default bounds: offset=(0, 0), size=({W}, {H})")
        return 0, 0, W, H

def read_annotations(mask_path, bounds_x, bounds_y, H, W):
    '''process mask file from Qupath'''
    masks = {}    
    if os.path.exists(mask_path):
        gdf = gpd.read_file(mask_path)                        
        for name in gdf.index:
            geoms = gdf.loc[gdf.index == name, 'geometry']
            if geoms.empty:
                print(f'no polygon found for {name}')
                continue

            shapes = ((mapping(geom), 1) for geom in geoms)

            mask = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(H, W),       
                fill=0,                 
                default_value=1,        
                dtype='uint8',
                all_touched=True        
            ).astype(bool)

            masks[name] = mask
    else:
        raise FileNotFoundError(f'{mask_path} not exist!')
    return masks


def save_edge_json(sample_name, coords, edge_info_path, blank=3000):
    """Save edge information (bounding box) as JSON."""
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    edge_info = {
        'min_x': int(min_x),
        'min_y': int(min_y),
        'max_x': int(max_x),
        'max_y': int(max_y),
        'x_origin': int(min_x - blank // 2),
        'y_origin': int(min_y - blank // 2),
        'w_origin': int(max_x - min_x + blank),
        'h_origin': int(max_y - min_y + blank)
    }
    
    os.makedirs(edge_info_path, exist_ok=True)
    
    with open(f"{edge_info_path}/{sample_name}.json", "w") as f:
        json.dump(edge_info, f)
    
    print(f"Saved edge info: {edge_info_path}/{sample_name}.json")


def create_dataset(wsi, h5filepath, coords, num_patches, patch_size, save_edge=False, blank=3000):
    """Create H5 dataset with images from WSI coordinates."""
    img_attrs = {
        'patch_size': patch_size,
        'num_patches': num_patches,
    }
    
    with h5py.File(h5filepath, 'w') as h5file:
        img_dataset = h5file.create_dataset(
                        'img',
                        shape=(num_patches, patch_size, patch_size, 3),
                        dtype=np.uint8
                    )
        coords_dataset = h5file.create_dataset('coords', data=coords, dtype='int32')
        
        for idx, coord in enumerate(coords):
            img = wsi.read_region(
                    coord-patch_size//2, 0, (patch_size, patch_size) 
                ).convert("RGB")
            img_np = np.array(img)
            img_dataset[idx] = img_np  
            del img_np, img
        
        for key, value in img_attrs.items():
            h5file['img'].attrs[key] = value
    
    

def process_file(f, patch_size, slide_path, file_type, mask_path, save_edge=True, edge_info_path='results/edge_info', blank=3000):
    """Process H5 file: extract images from WSI and optionally save edge info."""
    try:
        with h5py.File(f, 'r') as h5file:
            coords = h5file['coords'][:]

        file_name = os.path.basename(f).removesuffix(".h5")
        num_patches = len(coords)

        if save_edge and edge_info_path:
            save_edge_json(file_name, coords, edge_info_path, blank)

        wsi = openslide.open_slide(os.path.join(slide_path, f'{file_name+file_type}'))
        W,H = wsi.dimensions
        bounds_x, bounds_y, bounds_w, bounds_h = get_effective_bounds(wsi)

        if mask_path:
            output_dir = os.path.join(os.path.dirname(f), f'{file_name}_masks')
            os.makedirs(output_dir, exist_ok=True)
            
            file_list = glob.glob(os.path.join(mask_path, f'{file_name}*.geojson'))
            file_num = len(file_list)
            if not file_list:
                fallback = os.path.join(mask_path, f'{file_name}.geojson')
                if os.path.exists(fallback):
                    file_list = [fallback]
            
            for mask_file_path in file_list:
                mask_name = os.path.basename(mask_file_path).removesuffix('.geojson')
                masks = read_annotations(mask_file_path, bounds_x, bounds_y, bounds_h, bounds_w)
                print(f"{mask_name}: Read {len(masks)} masks.")
                subcoords = {}; subidx = {}
                for m_name, mask in masks.items():
                    subcoords[m_name] = []; subidx[m_name] = []
                    for idx, coord in enumerate(coords):
                        valid_x = coord[0]-bounds_x; valid_y = coord[1]-bounds_y
                        if mask[valid_y, valid_x]:
                            subcoords[m_name].append(coord)
                            subidx[m_name].append(idx)
                    subcoords[m_name] = np.vstack(subcoords[m_name]); subidx[m_name] = np.vstack(subidx[m_name])
                    print(f"Found {subcoords[m_name].shape[0]} patches for {m_name}")
                for m_name, subcoord in subcoords.items():
                    sub_num_patches = subcoord.shape[0]
                    if sub_num_patches == 0:
                        continue
                    else:
                        new_h5file = os.path.join(output_dir, f'{mask_name}_{m_name}.h5')
                        create_dataset(wsi, new_h5file, subcoord, sub_num_patches, patch_size, save_edge)
                        print(f"Created h5 file for {m_name} in {new_h5file}")     
        else:
            create_dataset(wsi, f, coords, num_patches, patch_size, save_edge)
            print(f"Created h5file without masks: {f}")     
        wsi.close()
        
    except Exception as e:
        traceback.print_exc()
    finally:
        gc.collect() 
        return f"Processed {f}"

parser = argparse.ArgumentParser(description='process h5 files')
parser.add_argument('--patch_size', type=int, default=224, help='patch size')
parser.add_argument('--slide_path', type=str, default='data', help='wsi image files path')
parser.add_argument('--file_type', choices=['.svs', '.mrxs', '.tiff', '.tif'], help='file extension')
parser.add_argument('--h5_path', type=str, default='results/patches', help='h5 files path')
parser.add_argument('--mask_path', type=str, default=None, help='geojson mask files path')
parser.add_argument('--max_workers', type=int, default=8, help='max workers for parallel processing')
parser.add_argument('--save_edge', action='store_true', help='save edge information as JSON')
parser.add_argument('--edge_info_path', type=str, default='results/edge_info', help='edge info output path')
parser.add_argument('--blank', type=int, default=3000, help='blank border size for edge calculation')
parser.add_argument('--sample_list', type=str, nargs='+', default=None, help='list of sample IDs to process')

if __name__ == '__main__':
    args = parser.parse_args()
    
    file_list = glob.glob(f'{args.h5_path}/*.h5')
    if args.sample_list:
        print(f"Filtering samples: {args.sample_list}")
        filtered_list = []
        for f in file_list:
            sample_name = os.path.basename(f).removesuffix('.h5')
            if sample_name in args.sample_list:
                filtered_list.append(f)
        file_list = filtered_list
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_file, 
                f, 
                args.patch_size, 
                args.slide_path, 
                args.file_type, 
                args.mask_path,
                args.save_edge,
                args.edge_info_path,
                args.blank
            ): f for f in file_list
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())

# python -m code.process_patch_h5 \
#   --patch_size 256 \
#   --h5_path results/patches \
#   --slide_path data/wsis \
#   --file_type .svs \
#   --sample_list sample001 sample002 sample003 \
#   --save_edge \
#   --edge_info_path results/edge_info