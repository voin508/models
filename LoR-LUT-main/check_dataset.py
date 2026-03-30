#!/usr/bin/env python3
"""
Dataset diagnostic tool - checks train/val splits and identifies potential issues
"""
import os, argparse, glob
from PIL import Image
import numpy as np

def check_dataset(root, in_dir="input", gt_dir="gt", exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    """Check dataset structure and statistics"""
    print(f"\n🔍 Checking dataset at: {root}\n")
    
    splits = ["train", "val"]
    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.exists(split_path):
            print(f"❌ {split}/ directory not found at {split_path}")
            continue
        
        in_path = os.path.join(split_path, in_dir)
        gt_path = os.path.join(split_path, gt_dir)
        
        if not os.path.exists(in_path):
            print(f"❌ {split}/{in_dir}/ not found")
            continue
        if not os.path.exists(gt_path):
            print(f"❌ {split}/{gt_dir}/ not found")
            continue
        
        # List files
        in_files = []
        gt_files = []
        for ext in exts:
            in_files += glob.glob(os.path.join(in_path, f"*{ext}"))
            gt_files += glob.glob(os.path.join(gt_path, f"*{ext}"))
        
        in_names = set([os.path.basename(f) for f in in_files])
        gt_names = set([os.path.basename(f) for f in gt_files])
        
        # Find matches
        matched = sorted(in_names & gt_names)
        in_only = sorted(in_names - gt_names)
        gt_only = sorted(gt_names - in_names)
        
        print(f"📁 {split.upper()}")
        print(f"   Input files: {len(in_files)}")
        print(f"   GT files: {len(gt_files)}")
        print(f"   ✅ Matched pairs: {len(matched)}")
        
        if in_only:
            print(f"   ⚠️  Input-only (no GT): {len(in_only)}")
            if len(in_only) <= 5:
                for f in in_only:
                    print(f"      - {f}")
        
        if gt_only:
            print(f"   ⚠️  GT-only (no input): {len(gt_only)}")
            if len(gt_only) <= 5:
                for f in gt_only:
                    print(f"      - {f}")
        
        # Check a sample image
        if len(matched) > 0:
            sample_name = matched[0]
            in_sample = os.path.join(in_path, sample_name)
            gt_sample = os.path.join(gt_path, sample_name)
            
            try:
                in_img = Image.open(in_sample)
                gt_img = Image.open(gt_sample)
                
                in_arr = np.array(in_img)
                gt_arr = np.array(gt_img)
                
                print(f"\n   📸 Sample check ({sample_name}):")
                print(f"      Input: {in_img.size} {in_img.mode} range=[{in_arr.min()},{in_arr.max()}]")
                print(f"      GT: {gt_img.size} {gt_img.mode} range=[{gt_arr.min()},{gt_arr.max()}]")
                
                if in_img.size != gt_img.size:
                    print(f"      ⚠️  Size mismatch! (will be center-cropped during training)")
                
                if in_img.mode != gt_img.mode:
                    print(f"      ⚠️  Mode mismatch!")
                
                # Check if images are too similar (identical)
                if in_arr.shape == gt_arr.shape:
                    diff = np.abs(in_arr.astype(float) - gt_arr.astype(float)).mean()
                    if diff < 1.0:
                        print(f"      ⚠️  Input and GT are nearly identical (diff={diff:.4f})!")
                
            except Exception as e:
                print(f"      ❌ Error loading sample: {e}")
        
        print()
    
    # Final summary
    print("="*60)
    print("✅ Diagnostic complete. Check for warnings above.")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--in_dir", type=str, default="input")
    parser.add_argument("--gt_dir", type=str, default="gt")
    args = parser.parse_args()
    
    check_dataset(args.root, args.in_dir, args.gt_dir)


