#!/usr/bin/env python3
"""
Convert image-caption pairs to WebDataset format.
This script converts images (PNG/JPG/JPEG) and their corresponding TXT captions
from the datasets folder into WebDataset tar archives.
"""

import os
import argparse
import tarfile
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


def collect_pairs(dataset_dir):
    """
    Collect all image-caption pairs from the dataset directory.
    
    Args:
        dataset_dir: Path to the directory containing image and text files
        
    Returns:
        List of tuples (base_name, png_path, txt_path)
    """
    dataset_path = Path(dataset_dir)
    
    # Group files by their base name
    files_by_base = defaultdict(dict)
    
    for file_path in dataset_path.iterdir():
        if file_path.is_file():
            base_name = file_path.stem
            extension = file_path.suffix.lower()
            
            if extension in ['.png', '.jpg', '.jpeg']:
                files_by_base[base_name]['image'] = file_path
            elif extension == '.txt':
                files_by_base[base_name]['txt'] = file_path
    
    # Create pairs list
    pairs = []
    for base_name, files in files_by_base.items():
        if 'image' in files and 'txt' in files:
            pairs.append((base_name, files['image'], files['txt']))
        else:
            print(f"Warning: Missing pair for {base_name}")
    
    return sorted(pairs)


def create_webdataset(pairs, output_dir, shard_size=1000, prefix="dataset"):
    """
    Create WebDataset tar archives from image-caption pairs.
    
    Args:
        pairs: List of (base_name, png_path, txt_path) tuples
        output_dir: Directory to save the tar files
        shard_size: Number of samples per shard
        prefix: Prefix for the tar file names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_shards = (len(pairs) + shard_size - 1) // shard_size
    
    print(f"Creating {total_shards} shards with up to {shard_size} samples each")
    print(f"Total samples: {len(pairs)}")
    
    for shard_idx in range(total_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(pairs))
        shard_pairs = pairs[start_idx:end_idx]
        
        # Create tar file name with zero-padded shard number
        tar_filename = f"{prefix}-{shard_idx:06d}.tar"
        tar_path = output_path / tar_filename
        
        print(f"\nCreating {tar_filename} ({len(shard_pairs)} samples)...")
        
        with tarfile.open(tar_path, 'w') as tar:
            for idx, (base_name, image_path, txt_path) in enumerate(tqdm(shard_pairs, desc=f"Shard {shard_idx+1}/{total_shards}")):
                # Use simple sequential naming within each shard
                sample_id = f"{idx:06d}"
                
                # Get original image size
                with Image.open(image_path) as img:
                    orig_width, orig_height = img.size
                
                # Add image file with original extension
                image_ext = image_path.suffix.lower()
                image_arcname = f"{sample_id}{image_ext}"
                tar.add(image_path, arcname=image_arcname)
                
                # Add caption file
                txt_arcname = f"{sample_id}.txt"
                tar.add(txt_path, arcname=txt_arcname)
                
                # Add JSON metadata with original size
                # Keys must match WDS_JSON_WIDTH and WDS_JSON_HEIGHT in training script
                json_data = {
                    "width": orig_width,
                    "height": orig_height
                }
                json_content = json.dumps(json_data).encode('utf-8')
                
                # Create tarinfo for JSON file
                json_arcname = f"{sample_id}.json"
                json_tarinfo = tarfile.TarInfo(name=json_arcname)
                json_tarinfo.size = len(json_content)
                
                import io
                tar.addfile(json_tarinfo, io.BytesIO(json_content))
        
        print(f"Created {tar_path} ({os.path.getsize(tar_path) / (1024**2):.2f} MB)")
    
    print(f"\n✓ WebDataset creation complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Total shards: {total_shards}")
    print(f"  Total samples: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image-caption pairs to WebDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (1000 samples per shard)
  python convert_webdataset.py
  
  # Specify custom shard size
  python convert_webdataset.py --shard-size 500
  
  # Specify custom output directory
  python convert_webdataset.py --output-dir ./my_webdatasets
  
  # Custom prefix for tar files
  python convert_webdataset.py --prefix my_dataset
        """
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets',
        help='Directory containing image and caption files (default: datasets)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='web_datasets',
        help='Directory to save WebDataset tar files (default: web_datasets)'
    )
    
    parser.add_argument(
        '--shard-size',
        type=int,
        default=1000,
        help='Number of samples per shard (default: 1000)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='dataset',
        help='Prefix for tar file names (default: dataset)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset directory
    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist")
        return
    
    # Collect pairs
    print(f"Scanning {args.dataset_dir}...")
    pairs = collect_pairs(args.dataset_dir)
    
    if not pairs:
        print("Error: No valid image-caption pairs found")
        return
    
    print(f"Found {len(pairs)} image-caption pairs")
    
    # Create WebDataset
    create_webdataset(
        pairs=pairs,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
