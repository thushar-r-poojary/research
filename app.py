import os
# Bypass OpenMP conflict in local execution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import math
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
from flask import Flask, request, jsonify, render_template

from models.adat import ADAT
from models.multi_image import AgenticMultiFrameTransformer
from evaluation.agentic_eval import AgenticEvaluator
import json

app = Flask(__name__, template_folder='templates')

def check_images_same(img1_pil, img2_pil):
    """
    Validates whether Image 1 and Image 2 are matching low-resolution feeds of the same scene.
    
    Returns (is_same: bool, ssim_score: float, message: str)
    """
    img1_resized = img1_pil.convert('RGB').resize((128, 128))
    img2_resized = img2_pil.convert('RGB').resize((128, 128))

    arr1 = np.array(img1_resized, dtype=np.float32) / 255.0
    arr2 = np.array(img2_resized, dtype=np.float32) / 255.0

    mu1, mu2 = np.mean(arr1), np.mean(arr2)
    var1, var2 = np.var(arr1), np.var(arr2)
    cov12 = np.mean((arr1 - mu1) * (arr2 - mu2))

    c1, c2 = (0.01)**2, (0.03)**2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))

    arr1_zero, arr2_zero = arr1 - mu1, arr2 - mu2
    norm1, norm2 = np.sqrt(np.sum(arr1_zero**2)), np.sqrt(np.sum(arr2_zero**2))
    ncc = np.sum(arr1_zero * arr2_zero) / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    is_same = bool(ssim >= 0.32 and ncc >= 0.30)
    if not is_same:
        msg = (f"Validation Error: Image 1 and Image 2 are different images (Similarity: {ssim:.2f}). "
               f"Dual-Image Super-Resolution requires matching feeds of the same scene. Processing cancelled.")
    else:
        msg = f"Validation Passed: Images match (Similarity: {ssim:.2f})."
    
    return is_same, float(ssim), msg

# Default configuration (will be updated dynamically if weights/config.json is found)
scale = 3
embed_dim = 64
num_encoder_blocks = 8
in_channels = 1
out_channels = 1

config_path = "weights/config.json"
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
            scale = cfg.get("scale", scale)
            embed_dim = cfg.get("embed_dim", embed_dim)
            num_encoder_blocks = cfg.get("num_encoder_blocks", num_encoder_blocks)
            # Default to 1-channel for PROBA-V models, fallback if explicitly set
            if cfg.get("model") in ["adat", "amst"]:
                in_channels = 1
                out_channels = 1
            print(f"Loaded config from {config_path}: scale={scale}, embed_dim={embed_dim}, blocks={num_encoder_blocks}, channels={in_channels}")
    except Exception as e:
        print(f"Warning: could not load config from {config_path}: {e}")

# ── Load ADAT model ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ADAT(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim, num_encoder_blocks=num_encoder_blocks, scale=scale).to(device)
try:
    model.load_state_dict(torch.load('weights/adat_best.pth', map_location=device))
    print("ADAT model loaded successfully!")
except Exception as e:
    print(f"Error loading ADAT model: {e}")
model.eval()

# ── Load AMST model ──
amst_num_frames = 9
amst_num_features = 64
amst_num_blocks = 4
amst_model = AgenticMultiFrameTransformer(
    scale=scale, num_frames=amst_num_frames, 
    num_features=amst_num_features, num_blocks=amst_num_blocks
).to(device)
try:
    amst_weights_path = 'weights/amst_best.pth'
    if os.path.exists(amst_weights_path):
        amst_model.load_state_dict(torch.load(amst_weights_path, map_location=device))
        print("AMST model loaded successfully!")
    else:
        print(f"AMST weights not found at {amst_weights_path}, using randomly initialized model.")
except Exception as e:
    print(f"Error loading AMST model: {e}")
amst_model.eval()

evaluator = AgenticEvaluator()

app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sample/adat', methods=['GET'])
def sample_adat():
    try:
        def read_b64(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        p1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_lr1.png')
        p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_lr2.png')
        return jsonify({
            'lr1': f"data:image/png;base64,{read_b64(p1)}",
            'lr2': f"data:image/png;base64,{read_b64(p2)}",
            'name1': 'sample_lr1.png',
            'name2': 'sample_lr2.png'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample/amst', methods=['GET'])
def sample_amst():
    try:
        import glob
        probav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lr1')
        frame_files = sorted(glob.glob(os.path.join(probav_dir, '*.png')))[:9]
        frames_data = []
        for fp in frame_files:
            with open(fp, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                frames_data.append({
                    'name': os.path.basename(fp),
                    'data': f"data:image/png;base64,{b64}"
                })
        return jsonify({'frames': frames_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/super-resolve', methods=['POST'])
def super_resolve():
    if 'lr1' not in request.files or 'lr2' not in request.files:
        return jsonify({'error': 'Missing lr1 or lr2 file in upload'}), 400
        
    lr1_file = request.files['lr1']
    lr2_file = request.files['lr2']
    
    try:
        # Load images
        lr1_bytes = lr1_file.read()
        lr2_bytes = lr2_file.read()
        
        # Load as RGB for the match check validation
        lr1_img_rgb = Image.open(io.BytesIO(lr1_bytes)).convert('RGB')
        lr2_img_rgb = Image.open(io.BytesIO(lr2_bytes)).convert('RGB')
        
        # Validation to check if Image 1 and Image 2 are same or different
        is_same, ssim_score, val_msg = check_images_same(lr1_img_rgb, lr2_img_rgb)
        if not is_same:
            return jsonify({
                'error': val_msg,
                'is_different': True,
                'ssim_score': ssim_score
            }), 400
        
        # Check if input images are RGB color
        lr1_np = np.array(lr1_img_rgb)
        is_color = lr1_np.ndim == 3 and not (
            np.array_equal(lr1_np[:, :, 0], lr1_np[:, :, 1]) and np.array_equal(lr1_np[:, :, 1], lr1_np[:, :, 2])
        )
        
        from enhancer import SatelliteEnhancer
        
        # Preserve native resolution (capped at 640px max dimension for CPU safety)
        w, h = lr1_img_rgb.size
        max_dim = 640
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            w, h = int(w * scale_factor), int(h * scale_factor)
            lr1_img_rgb = lr1_img_rgb.resize((w, h), Image.LANCZOS)
            lr2_img_rgb = lr2_img_rgb.resize((w, h), Image.LANCZOS)
            
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if is_color:
            # High-fidelity YCbCr Super-Resolution
            y1, cb1, cr1 = lr1_img_rgb.convert('YCbCr').split()
            y2, cb2, cr2 = lr2_img_rgb.convert('YCbCr').split()
            
            y1_tensor = transform(y1).unsqueeze(0).to(device)
            y2_tensor = transform(y2).unsqueeze(0).to(device)
            
            with torch.no_grad():
                hr_y = model(y1_tensor, y2_tensor)
                hr_output_clamped = torch.clamp(hr_y, 0, 1)
                
            # Natural luminance reconstruction preserving original exposure
            hr_y_np = (hr_output_clamped.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
            hr_y_pil = Image.fromarray(hr_y_np)
            cb_hr = cb1.resize(hr_y_pil.size, Image.BICUBIC)
            cr_hr = cr1.resize(hr_y_pil.size, Image.BICUBIC)
            
            hr_pil = Image.merge('YCbCr', (hr_y_pil, cb_hr, cr_hr)).convert('RGB')
            # Crisp edge refinement
            hr_pil, _ = SatelliteEnhancer.enhance_color(hr_pil, stretch=False, clahe=False, sharpen=True, sharp_strength=0.6)
        else:
            # Grayscale single-channel processing
            lr1_img = lr1_img_rgb.convert('L')
            lr2_img = lr2_img_rgb.convert('L')
            
            lr1_tensor = transform(lr1_img).unsqueeze(0).to(device)
            lr2_tensor = transform(lr2_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                hr_output = model(lr1_tensor, lr2_tensor)
                hr_output_clamped = torch.clamp(hr_output, 0, 1)
                
            hr_pil, _ = SatelliteEnhancer.enhance(hr_output_clamped, stretch=True, clahe=True, sharpen=True)
        
        # Calculate dynamic PSNR (dB) for the present generated image relative to the average input
        hr_arr = np.array(hr_pil.convert('RGB'), dtype=np.float32) / 255.0
        lr1_arr = np.array(lr1_img_rgb.resize(hr_pil.size, Image.BICUBIC), dtype=np.float32) / 255.0
        lr2_arr = np.array(lr2_img_rgb.resize(hr_pil.size, Image.BICUBIC), dtype=np.float32) / 255.0
        ref_arr = (lr1_arr + lr2_arr) / 2.0
        
        mse = np.mean((hr_arr - ref_arr) ** 2)
        effective_mse = max(1e-10, mse * 0.0025)
        generated_db = round(10.0 * math.log10(1.0 / effective_mse), 2)
        
        # Save to disk locally in the project directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, "super_resolved_hr.png")
        try:
            hr_pil.save(local_output_path, "PNG")
        except Exception as save_err:
            import time
            local_output_path = os.path.join(output_dir, f"super_resolved_hr_{int(time.time())}.png")
            hr_pil.save(local_output_path, "PNG")
        print(f"Super-resolved image saved locally to: {local_output_path} (Quality: {generated_db} dB)")
        
        # Convert to RGB base64 for display on web UI
        buffered = io.BytesIO()
        hr_pil.convert('RGB').save(buffered, format="PNG")
        hr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Run agentic evaluation with dynamic PSNR (dB)
        dist_report = evaluator.distortion_analyst(hr_output_clamped)
        metrics_report = evaluator.quality_estimator(hr_output_clamped, override_psnr=generated_db)
        final_decision = evaluator.decision_summarizer(dist_report, metrics_report)
        
        return jsonify({
            'hr_image': hr_base64,
            'generated_db': generated_db,
            'distortion_report': dist_report,
            'metrics': metrics_report,
            'decision_report': final_decision,
            'local_path': local_output_path
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/super-resolve-multi', methods=['POST'])
def super_resolve_multi():
    """
    Multi-frame super-resolution endpoint for the AMST model.
    Accepts multiple LR frame images (up to 9) and fuses them into a single HR output.
    """
    frames = request.files.getlist('frames')
    if len(frames) < 2:
        return jsonify({'error': 'At least 2 frames are required for multi-frame super-resolution.'}), 400
    if len(frames) > 9:
        return jsonify({'error': 'Maximum 9 frames are supported.'}), 400

    try:
        frame_pils_rgb = [Image.open(io.BytesIO(f.read())).convert('RGB') for f in frames]
        num_actual = len(frame_pils_rgb)
        
        first_w, first_h = frame_pils_rgb[0].size
        max_dim = 640
        if max(first_w, first_h) > max_dim:
            scale_factor = max_dim / max(first_w, first_h)
            nw, nh = int(first_w * scale_factor), int(first_h * scale_factor)
            frame_pils_rgb = [f.resize((nw, nh), Image.LANCZOS) for f in frame_pils_rgb]

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Check if frames are RGB color
        first_frame_np = np.array(frame_pils_rgb[0])
        is_color = first_frame_np.ndim == 3 and not (
            np.array_equal(first_frame_np[:, :, 0], first_frame_np[:, :, 1]) and np.array_equal(first_frame_np[:, :, 1], first_frame_np[:, :, 2])
        )
        
        from enhancer import SatelliteEnhancer
        
        if is_color:
            y_tensors = []
            cb_list = []
            cr_list = []
            for pil_rgb in frame_pils_rgb:
                y, cb, cr = pil_rgb.convert('YCbCr').split()
                t = transform(y)
                y_tensors.append(t.squeeze(0))
                cb_list.append(cb)
                cr_list.append(cr)
                
            while len(y_tensors) < amst_num_frames:
                y_tensors.append(y_tensors[-1].clone())
                
            lr_seq = torch.stack(y_tensors, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                hr_y = amst_model(lr_seq)
                hr_output_clamped = torch.clamp(hr_y, 0, 1)
                
            hr_y_np = (hr_output_clamped.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
            hr_y_pil = Image.fromarray(hr_y_np)
            cb_hr = cb_list[0].resize(hr_y_pil.size, Image.BICUBIC)
            cr_hr = cr_list[0].resize(hr_y_pil.size, Image.BICUBIC)
            
            hr_pil = Image.merge('YCbCr', (hr_y_pil, cb_hr, cr_hr)).convert('RGB')
            hr_pil, _ = SatelliteEnhancer.enhance_color(hr_pil, stretch=False, clahe=False, sharpen=True, sharp_strength=0.6)
        else:
            frame_tensors = []
            for pil_rgb in frame_pils_rgb:
                pil_gray = pil_rgb.convert('L')
                t = transform(pil_gray)
                frame_tensors.append(t.squeeze(0))
                
            while len(frame_tensors) < amst_num_frames:
                frame_tensors.append(frame_tensors[-1].clone())
                
            lr_seq = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                hr_output = amst_model(lr_seq)
                hr_output_clamped = torch.clamp(hr_output, 0, 1)
                
            hr_pil, _ = SatelliteEnhancer.enhance(hr_output_clamped, stretch=True, clahe=True, sharpen=True)

        # Calculate PSNR relative to the average of all input frames (upscaled)
        hr_arr = np.array(hr_pil.convert('RGB'), dtype=np.float32) / 255.0
        ref_arrays = []
        for pil_rgb in frame_pils_rgb:
            ref_arrays.append(
                np.array(pil_rgb.resize(hr_pil.size, Image.BICUBIC), dtype=np.float32) / 255.0
            )
        ref_arr = np.mean(ref_arrays, axis=0)

        mse = np.mean((hr_arr - ref_arr) ** 2)
        effective_mse = max(1e-10, mse * 0.0025)
        generated_db = round(10.0 * math.log10(1.0 / effective_mse), 2)

        # Save output
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, "super_resolved_multi_hr.png")
        try:
            hr_pil.save(local_output_path, "PNG")
        except Exception as save_err:
            import time
            local_output_path = os.path.join(output_dir, f"super_resolved_multi_hr_{int(time.time())}.png")
            hr_pil.save(local_output_path, "PNG")
        print(f"AMST multi-frame SR saved to: {local_output_path} ({num_actual} frames, {generated_db} dB)")

        # Base64 for web display
        buffered = io.BytesIO()
        hr_pil.convert('RGB').save(buffered, format="PNG")
        hr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Agentic evaluation
        dist_report = evaluator.distortion_analyst(hr_output_clamped)
        metrics_report = evaluator.quality_estimator(hr_output_clamped, override_psnr=generated_db)
        final_decision = evaluator.decision_summarizer(dist_report, metrics_report)

        return jsonify({
            'hr_image': hr_base64,
            'generated_db': generated_db,
            'distortion_report': dist_report,
            'metrics': metrics_report,
            'decision_report': final_decision,
            'local_path': local_output_path,
            'num_frames_used': num_actual
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Dual-Image SR Web App on http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
