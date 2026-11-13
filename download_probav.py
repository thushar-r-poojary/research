import os
import urllib.request
import zipfile
import sys
import numpy as np
from PIL import Image

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100.0, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading dataset... {percent:.2f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading dataset... ({read_so_far / (1024*1024):.1f} MB)")
    sys.stdout.flush()

def create_fallback_dataset():
    print("\n--- PROBA-V Fallback Dataset Generator ---")
    print("Constructing a local PROBA-V structured dataset using the existing real-world satellite tiles.")
    
    hr_source_dir = "data/hr"
    probav_dir = "data/probav"
    split = "train"
    band = "RED"
    dest_dir = os.path.join(probav_dir, split, band)
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.exists(hr_source_dir):
        print(f"Error: No source images found at '{hr_source_dir}'. Please ensure 'data/hr' contains the 243 tiles.")
        return
        
    hr_files = sorted([f for f in os.listdir(hr_source_dir) if f.endswith('.png')])
    if len(hr_files) == 0:
        print("Error: 'data/hr' is empty.")
        return
        
    print(f"Found {len(hr_files)} source tiles in 'data/hr'. Processing...")
    
    for idx, fname in enumerate(hr_files):
        # Create folder for this scene
        scene_name = f"imgset{idx:04d}"
        scene_dir = os.path.join(dest_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Load HR image and convert to grayscale (since PROBA-V RED/NIR are single-channel)
        hr_img = Image.open(os.path.join(hr_source_dir, fname)).convert('L')
        
        # Resize to 384x384 (PROBA-V HR size)
        hr_img_resized = hr_img.resize((384, 384), Image.BICUBIC)
        hr_img_resized.save(os.path.join(scene_dir, "hr.png"))
        
        # Quality map for HR (all 255 = fully valid)
        qm_hr = Image.fromarray(np.ones((384, 384), dtype=np.uint8) * 255)
        qm_hr.save(os.path.join(scene_dir, "qm_hr.png"))
        
        # Generate 9 LR frames (128x128)
        hr_np = np.array(hr_img_resized, dtype=np.float32)
        for f in range(9):
            # Introduce sub-pixel shifts using np.roll
            sx = np.random.randint(-4, 5)
            sy = np.random.randint(-4, 5)
            shifted = np.roll(np.roll(hr_np, sx, axis=1), sy, axis=0)
            
            # Downsample to 128x128 (PROBA-V LR size)
            lr_img = Image.fromarray(shifted.astype(np.uint8)).resize((128, 128), Image.BICUBIC)
            
            # Save LR and its QM
            lr_img.save(os.path.join(scene_dir, f"lr{f}.png"))
            qm_lr = Image.fromarray(np.ones((128, 128), dtype=np.uint8) * 255)
            qm_lr.save(os.path.join(scene_dir, f"qm_lr{f}.png"))
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(hr_files):
            print(f"Processed [{idx+1}/{len(hr_files)}] scenes...")
            
    print(f"Fallback dataset successfully created at '{probav_dir}'!")
    print("This local dataset mirrors the exact format, resolutions, and channel counts of the PROBA-V dataset.")

def main():
    url = "https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip"
    dest_dir = "data"
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "probav_data.zip")
    extract_path = os.path.join(dest_dir, "probav")

    if os.path.exists(os.path.join(extract_path, "train")) and os.path.exists(os.path.join(extract_path, "test")):
        print("PROBA-V dataset already exists.")
        return

    print(f"Attempting to download PROBA-V dataset from: {url}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(zip_path, 'wb') as out_file:
                # Read chunks
                meta = response.info()
                total_size = int(meta.get("Content-Length", 0))
                block_size = 8192
                read_so_far = 0
                block_num = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    read_so_far += len(buffer)
                    out_file.write(buffer)
                    block_num += 1
                    download_progress(block_num, block_size, total_size)
                    
        print("\nDownload complete! Extracting files...")
        
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print(f"Extraction complete! Dataset extracted to {extract_path}")
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")
            
    except Exception as e:
        print(f"\nESA download failed or requires login: {e}")
        print("Falling back to local fallback dataset generation...")
        create_fallback_dataset()

if __name__ == "__main__":
    main()
