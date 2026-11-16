import os
import zipfile

def package_archive():
    target_dir = "E2AD_RCAR_OCT2017_CORRECTED"
    zip_path = "E2AD_RCAR_OCT2017_CORRECTED.zip"
    
    print(f"Packaging {target_dir} into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # Skip massive raw datasets if present
                if "OCT2017" in full_path and ("train" in full_path or "test" in full_path):
                    continue
                arcname = os.path.relpath(full_path, target_dir)
                zipf.write(full_path, arcname)
                
    # Integrity check
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        bad_file = zipf.testzip()
        assert bad_file is None, f"Corrupted zip archive at {bad_file}"
        file_list = zipf.namelist()
        size_mb = os.path.getsize(zip_path) / (1024**2)
        
    print("=" * 70)
    print("ZIP ARCHIVE INTEGRITY VERIFIED")
    print("=" * 70)
    print(f"Destination Archive : {zip_path}")
    print(f"Archive Size        : {size_mb:.2f} MB")
    print(f"Total Packaged Files: {len(file_list)}")
    print("Sample Included Files:")
    for f in file_list[:12]:
        print(f"  - {f}")
    print("=" * 70)

if __name__ == "__main__":
    package_archive()
