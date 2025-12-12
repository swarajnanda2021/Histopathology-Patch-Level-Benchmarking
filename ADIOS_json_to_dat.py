#!/usr/bin/env python3
"""
Convert ADIOS nuclei channel benchmark JSON files to pgfplots .dat format.

Usage:
    python json_to_dat.py

Outputs:
    data/adios_pannuke.dat
    data/adios_monuseg.dat
"""

import json
import os
from pathlib import Path

# Paths
UNET_JSON = "/data1/vanderbc/nandas1/ADIOS-UNet/visualizations/nuclei_channel_benchmark.json"
VIT_JSON = "/data1/vanderbc/nandas1/ADIOS-CellViT/visualizations/nuclei_channel_benchmark.json"
OUTPUT_DIR = Path(__file__).parent / "data"


def load_json(path):
    """Load JSON file, return empty dict if not found."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def merge_and_write(unet_data, vit_data, dataset_key, output_path):
    """
    Merge UNet and ViT results for a single dataset, write to .dat file.
    
    Columns: iteration  unet_mean  unet_std  vit_mean  vit_std
    Missing values written as NaN (pgfplots handles these).
    """
    # Build lookup dictionaries
    unet_lookup = {}
    if unet_data and 'iterations' in unet_data:
        for i, it in enumerate(unet_data['iterations']):
            unet_lookup[it] = (
                unet_data[f'{dataset_key}_mean'][i],
                unet_data[f'{dataset_key}_std'][i]
            )
    
    vit_lookup = {}
    if vit_data and 'iterations' in vit_data:
        for i, it in enumerate(vit_data['iterations']):
            vit_lookup[it] = (
                vit_data[f'{dataset_key}_mean'][i],
                vit_data[f'{dataset_key}_std'][i]
            )
    
    # Get all unique iterations, sorted
    all_iterations = sorted(set(unet_lookup.keys()) | set(vit_lookup.keys()))
    
    if not all_iterations:
        print(f"  No data for {dataset_key}")
        return
    
    # Write .dat file
    with open(output_path, 'w') as f:
        # Header (pgfplots ignores lines starting with #, but reads column names)
        f.write("iteration unet_mean unet_std vit_mean vit_std\n")
        
        for it in all_iterations:
            unet_mean, unet_std = unet_lookup.get(it, (float('nan'), float('nan')))
            vit_mean, vit_std = vit_lookup.get(it, (float('nan'), float('nan')))
            
            f.write(f"{it} {unet_mean:.6f} {unet_std:.6f} {vit_mean:.6f} {vit_std:.6f}\n")
    
    print(f"  Written: {output_path} ({len(all_iterations)} rows)")


def main():
    print("Loading JSON files...")
    unet_data = load_json(UNET_JSON)
    vit_data = load_json(VIT_JSON)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Generate .dat files
    print("Generating .dat files...")
    merge_and_write(unet_data, vit_data, 'pannuke', OUTPUT_DIR / 'adios_pannuke.dat')
    merge_and_write(unet_data, vit_data, 'monuseg', OUTPUT_DIR / 'adios_monuseg.dat')
    
    print("Done.")


if __name__ == '__main__':
    main()
