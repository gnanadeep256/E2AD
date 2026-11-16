import os
import hashlib
import json
import csv
import time
from PIL import Image
from collections import defaultdict

def audit_dataset(oct_root):
    print(f"Starting exhaustive dataset audit on: {oct_root}")
    start_time = time.time()
    
    splits = {
        "train": ["NORMAL"],
        "test": ["NORMAL", "CNV", "DME", "DRUSEN"]
    }
    
    manifest_rows = []
    hash_to_paths = defaultdict(list)
    corrupted_files = []
    dimensions_count = defaultdict(int)
    formats_count = defaultdict(int)
    modes_count = defaultdict(int)
    split_class_counts = defaultdict(int)
    
    total_scanned = 0
    
    for split, classes in splits.items():
        for cls in classes:
            folder = os.path.join(oct_root, split, cls)
            if not os.path.isdir(folder):
                print(f"Warning: Directory missing: {folder}")
                continue
            
            filenames = sorted(os.listdir(folder))
            print(f"Scanning {split}/{cls}: {len(filenames)} files found...")
            
            for fname in filenames:
                fpath = os.path.join(folder, fname)
                if not os.path.isfile(fpath):
                    continue
                
                rel_path = os.path.relpath(fpath, oct_root).replace('\\', '/')
                file_size = os.path.getsize(fpath)
                
                # SHA-256 computation
                hasher = hashlib.sha256()
                try:
                    with open(fpath, "rb") as f:
                        while chunk := f.read(65536):
                            hasher.update(chunk)
                    file_hash = hasher.hexdigest()
                except Exception as e:
                    corrupted_files.append({"path": rel_path, "error": f"Read error: {str(e)}"})
                    continue
                
                # Image verification
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    # Reopen for info (verify closes image)
                    with Image.open(fpath) as img:
                        w, h = img.size
                        fmt = img.format or "UNKNOWN"
                        mode = img.mode or "UNKNOWN"
                        
                    is_valid = True
                    err_msg = ""
                except Exception as e:
                    is_valid = False
                    err_msg = str(e)
                    corrupted_files.append({"path": rel_path, "error": err_msg})
                    w, h = -1, -1
                    fmt = "CORRUPT"
                    mode = "CORRUPT"
                
                if is_valid:
                    dimensions_count[f"{w}x{h}"] += 1
                    formats_count[fmt] += 1
                    modes_count[mode] += 1
                    split_class_counts[f"{split}/{cls}"] += 1
                    hash_to_paths[file_hash].append((split, cls, rel_path))
                
                manifest_rows.append({
                    "split": split,
                    "class": cls,
                    "relative_path": rel_path,
                    "filename": fname,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "mode": mode,
                    "size_bytes": file_size,
                    "sha256": file_hash,
                    "is_valid": is_valid
                })
                
                total_scanned += 1
                if total_scanned % 10000 == 0:
                    print(f"  Processed {total_scanned} images... ({time.time() - start_time:.1f}s)")
    
    elapsed = time.time() - start_time
    print(f"\nExhaustive scan complete in {elapsed:.1f}s. Total scanned: {total_scanned}")
    
    # Analyze duplicates and train/test leakage
    exact_duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    train_test_overlap = []
    
    for h, paths in exact_duplicates.items():
        splits_in_hash = set(p[0] for p in paths)
        if "train" in splits_in_hash and "test" in splits_in_hash:
            train_test_overlap.append({
                "sha256": h,
                "instances": [{"split": p[0], "class": p[1], "path": p[2]} for p in paths]
            })
    
    # Save dataset_manifest.csv
    manifest_csv_path = "dataset_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "class", "relative_path", "filename", "width", "height",
            "format", "mode", "size_bytes", "sha256", "is_valid"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved dataset manifest: {manifest_csv_path}")
    
    # Save dataset_integrity_report.json
    integrity_report = {
        "oct_root": os.path.abspath(oct_root),
        "total_scanned": total_scanned,
        "total_valid": total_scanned - len(corrupted_files),
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "split_class_counts": dict(split_class_counts),
        "unique_content_hashes": len(hash_to_paths),
        "exact_duplicate_hash_count": len(exact_duplicates),
        "train_test_overlap_count": len(train_test_overlap),
        "train_test_overlap_instances": train_test_overlap,
        "formats_distribution": dict(formats_count),
        "modes_distribution": dict(modes_count),
        "top_dimensions": sorted(dimensions_count.items(), key=lambda x: x[1], reverse=True)[:10],
        "audit_elapsed_sec": round(elapsed, 2),
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("dataset_integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)
    print("Saved integrity report: dataset_integrity_report.json")
    
    # Copy to E2AD_RCAR_OCT2017_CORRECTED
    import shutil
    shutil.copy("dataset_manifest.csv", "E2AD_RCAR_OCT2017_CORRECTED/manifests/dataset_manifest.csv")
    shutil.copy("dataset_integrity_report.json", "E2AD_RCAR_OCT2017_CORRECTED/audit/dataset_integrity_report.json")
    print("Copied reports to E2AD_RCAR_OCT2017_CORRECTED/ directories.")

if __name__ == "__main__":
    audit_dataset("OCT2017")
