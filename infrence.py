import os

# Silence OpenCV Qt font warning spam in headless/venv Linux setups.
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts.warning=false'
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts/truetype'

import cv2
import queue
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import carla

# --- 1. Settings & Paths ---
HOST = '127.0.0.1'
PORT = 2000

# Updated model paths
DEPTH_MODEL_PATH = "psmnet_clean_inference_gpu.pt"
SEG_MODEL_PATH = "deeplabv3_clean_inference_gpu.pt"

# Display size tuned so all three rows are visible on most laptop/desktop screens.
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

# CARLA server can take time to respond when loading a map.
CARLA_RPC_TIMEOUT_SEC = 60.0
MAP_LOAD_RETRIES = 3
MAP_LOAD_WAIT_SEC = 5.0

# Depth range for normalization (meters)
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 100.0

# --- 2. Hardware / Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--> Utilizing device:", device)

# ImageNet normalization (matching training data)
img_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("--> Loading TorchScript Models...")
try:
    # torch.jit.load requires zero external class definitions
    psmnet = torch.jit.load(DEPTH_MODEL_PATH, map_location=device).eval()
    deeplabv3 = torch.jit.load(SEG_MODEL_PATH, map_location=device).eval()
    print("--> Models successfully loaded on GPU.")
    
    # Verify models are actually on the target device
    if device.type == 'cuda':
        # For TorchScript, test inference to ensure device consistency
        test_tensor = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
        try:
            _ = psmnet(test_tensor, test_tensor)
            print(f"--> Device verification passed: models are on {device}")
        except Exception as e:
            print(f"[!] Models loaded but device test failed: {e}")
            print("[!] Falling back to CPU")
            device = torch.device("cpu")
            psmnet = torch.jit.load(DEPTH_MODEL_PATH, map_location=device).eval()
            deeplabv3 = torch.jit.load(SEG_MODEL_PATH, map_location=device).eval()
except Exception as e:
    print("[!] Error loading TorchScript models: ", e)
    import sys; sys.exit(1)

# --- 3. Helper Functions ---
CARLA_CLASSES = [
    (0,  "None",         (0, 0, 0)),
    (1,  "Buildings",    (70, 70, 70)),
    (2,  "Fences",       (190, 153, 153)),
    (3,  "Other",        (72, 0, 90)),
    (4,  "Pedestrians",  (220, 20, 60)),
    (5,  "Poles",        (153, 153, 153)),
    (6,  "RoadLines",    (157, 234, 50)),
    (7,  "Roads",        (128, 64, 128)),
    (8,  "Sidewalks",    (244, 35, 232)),
    (9,  "Vegetation",   (107, 142, 35)),
    (10, "Vehicles",     (0, 0, 255)),
    (11, "Walls",        (102, 102, 156)),
    (12, "TrafficSigns", (220, 220, 0)),
]

# Fast [H, W] class-id to [H, W, 3] RGB color mapping table.
SEG_COLOR_MAP = np.zeros((256, 3), dtype=np.uint8)
for class_id, _, rgb in CARLA_CLASSES:
    SEG_COLOR_MAP[class_id] = rgb

def preprocess_for_pytorch(img_bgr, target_device=None):
    """Convert OpenCV image to PyTorch tensor on the target device."""
    if target_device is None:
        target_device = device
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert to tensor (initially on CPU)
    tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float()
    # Scale to [0, 1]
    tensor = tensor / 255.0
    # Apply ImageNet normalization (matches training data)
    tensor = img_normalize(tensor)
    # Add batch dimension and move to target device
    tensor = tensor.unsqueeze(0).to(target_device)
    return tensor


def pad_to_multiple(tensor, multiple=32):
    """
    Pads a tensor [B, C, H, W] to the nearest multiple of 'multiple'.
    This prevents dimension mismatches in encoder-decoder models.
    Returns: (padded_tensor, pad_h, pad_w)
    """
    h, w = tensor.shape[2], tensor.shape[3]
    
    # Calculate how much padding is needed
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    
    # Pad format is (left, right, top, bottom)
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    return padded_tensor, pad_h, pad_w


def carla_depth_to_meters(depth_array):
    """
    Convert CARLA depth image to meters.
    CARLA encodes depth as: depth_meters = (R + G*256 + B*256^2) / (256^3 - 1)
    Returns float array in meters.
    """
    if isinstance(depth_array, torch.Tensor):
        depth_array = depth_array.cpu().numpy()
    
    # Ensure it's uint8 or convert
    if depth_array.dtype != np.uint8:
        depth_array = (depth_array * 255).astype(np.uint8)
    
    if len(depth_array.shape) == 3 and depth_array.shape[2] == 3:
        # RGB format: (R + G*256 + B*256^2) / (256^3 - 1)
        r = depth_array[:, :, 0].astype(np.float32)
        g = depth_array[:, :, 1].astype(np.float32)
        b = depth_array[:, :, 2].astype(np.float32)
        depth_m = (r + g * 256.0 + b * 256.0**2) / (256.0**3 - 1)
    else:
        # Single channel
        depth_m = depth_array.astype(np.float32) / 255.0 * DEPTH_MAX_M
    
    return depth_m


def colorize_depth(depth_data):
    """
    Colorize depth map using JET colormap.
    Handles both tensor and array inputs, converts to uint8 for visualization.
    """
    if isinstance(depth_data, torch.Tensor):
        depth_np = depth_data.squeeze().cpu().numpy()
    else:
        depth_np = depth_data
    
    # Normalize to 0-255 range
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_inv = 255 - depth_norm
    return cv2.applyColorMap(depth_inv, cv2.COLORMAP_JET)


def colorize_segmentation(seg_tensor):
    """Colorize segmentation output to RGB."""
    seg_np = seg_tensor.squeeze().cpu().numpy().astype(np.uint8)
    seg_rgb = SEG_COLOR_MAP[seg_np]
    return cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)


def parse_carla_image(carla_image, converter=None):
    """
    Parse CARLA image to numpy array.
    Handles conversion failures gracefully with fallback to raw data.
    """
    if converter is not None:
        try:
            converted = carla_image.convert(converter)
            if converted is not None:
                carla_image = converted
            else:
                print(f"[!] Converter {converter} failed, using raw data instead")
        except Exception as e:
            print(f"[!] Conversion error: {e}, using raw data instead")

    if carla_image.raw_data is None:
        raise ValueError("Image raw_data is None - sensor may not be ready")
    
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    return array[:, :, :3]


def compute_miou(pred_seg, gt_seg, num_classes=13):
    """
    Compute mean Intersection over Union (mIoU) between predicted and GT segmentation.
    Args:
        pred_seg: [H, W] predicted class indices
        gt_seg:   [H, W] ground truth class indices
        num_classes: total number of classes
    Returns:
        miou: float, mean IoU across all classes
    """
    pred_seg = pred_seg.astype(np.int64)
    gt_seg = gt_seg.astype(np.int64)
    
    iou_list = []
    for class_id in range(num_classes):
        pred_mask = (pred_seg == class_id)
        gt_mask = (gt_seg == class_id)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union > 0:
            iou = intersection / union
            iou_list.append(iou)
    
    miou = np.mean(iou_list) if iou_list else 0.0
    return miou


def compute_depth_metrics(pred_depth, gt_depth):
    """
    Compute depth estimation metrics.
    Args:
        pred_depth: [H, W] predicted depth in meters
        gt_depth:   [H, W] ground truth depth in meters
    Returns:
        rmse: Root Mean Squared Error
        delta_1_25: Percentage of pixels where max(pred/gt, gt/pred) < 1.25
    """
    # Flatten and remove invalid values
    pred_flat = pred_depth.flatten()
    gt_flat = gt_depth.flatten()
    
    valid_mask = (gt_flat > 0) & (gt_flat < DEPTH_MAX_M) & (pred_flat > 0) & (pred_flat < DEPTH_MAX_M)
    pred_valid = pred_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]
    
    if len(pred_valid) == 0:
        return 0.0, 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    
    # Delta 1.25 (accuracy)
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1_25 = (ratio < 1.25).sum() / len(ratio) * 100.0
    
    return rmse, delta_1_25


def draw_metrics_on_image(img, miou, rmse, delta_1_25):
    """
    Draw metrics text on image.
    Args:
        img: image array (BGR format)
        miou, rmse, delta_1_25: metric values
    Returns:
        img with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # Green
    thickness = 1
    
    metrics_text = [
        f"mIoU: {miou:.4f}",
        f"RMSE: {rmse:.4f}m",
        f"Delta1.25: {delta_1_25:.2f}%"
    ]
    
    y_offset = 30
    for i, text in enumerate(metrics_text):
        y = y_offset + (i * 25)
        cv2.putText(img, text, (10, y), font, font_scale, font_color, thickness)
    
    return img


# --- 4. Main CARLA 0.9.15 Loop ---
def main():
    actor_list = []
    try:
        print("--> Connecting to CARLA 0.9.15...")
        client = carla.Client(HOST, PORT)
        client.set_timeout(CARLA_RPC_TIMEOUT_SEC)

        world = None
        print(f"--> Loading map: Town01 (timeout={CARLA_RPC_TIMEOUT_SEC:.0f}s)")
        for attempt in range(1, MAP_LOAD_RETRIES + 1):
            try:
                world = client.load_world('Town01')
                break
            except RuntimeError as e:
                print(f"[!] Map load attempt {attempt}/{MAP_LOAD_RETRIES} failed: {e}")
                if attempt == MAP_LOAD_RETRIES:
                    raise RuntimeError(
                        "Failed to load Town01 after multiple attempts. "
                        "Ensure CARLA server is fully started and listening on 127.0.0.1:2000."
                    ) from e
                print(f"--> Waiting {MAP_LOAD_WAIT_SEC:.0f}s before retry...")
                time.sleep(MAP_LOAD_WAIT_SEC)

        # Give the simulator a brief moment to finalize world initialization.
        time.sleep(2.0)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # 1. Spawn Vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        # 2. Setup Stereo Cameras
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')

        transform_left = carla.Transform(carla.Location(x=2.0, y=-0.25, z=1.4))
        cam_left = world.spawn_actor(cam_bp, transform_left, attach_to=vehicle)
        actor_list.append(cam_left)

        transform_right = carla.Transform(carla.Location(x=2.0, y=0.25, z=1.4))
        cam_right = world.spawn_actor(cam_bp, transform_right, attach_to=vehicle)
        actor_list.append(cam_right)

        # 2b. Setup Ground Truth Sensors
        depth_gt_bp = blueprint_library.find('sensor.camera.depth')
        depth_gt_bp.set_attribute('image_size_x', '800')
        depth_gt_bp.set_attribute('image_size_y', '600')

        seg_gt_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_gt_bp.set_attribute('image_size_x', '800')
        seg_gt_bp.set_attribute('image_size_y', '600')

        depth_gt = world.spawn_actor(depth_gt_bp, transform_right, attach_to=vehicle)
        actor_list.append(depth_gt)
        seg_gt = world.spawn_actor(seg_gt_bp, transform_right, attach_to=vehicle)
        actor_list.append(seg_gt)

        # 3. Setup Queue
        image_queue = queue.Queue()
        cam_left.listen(lambda data: image_queue.put(('left', data)))
        cam_right.listen(lambda data: image_queue.put(('right', data)))
        depth_gt.listen(lambda data: image_queue.put(('depth_gt', data)))
        seg_gt.listen(lambda data: image_queue.put(('seg_gt', data)))

        print("--> Simulation Started. Engaging Live Feed...")
        
        while True:
            world.tick()
            frames = {}
            
            # Collect frames with timeout to avoid hanging
            start_time = time.time()
            timeout = 0.5  # seconds
            
            while len(frames) < 4 and (time.time() - start_time) < timeout:
                try:
                    cam_name, carla_img = image_queue.get(timeout=0.01)
                    if carla_img is None:
                        print(f"[!] Received None for {cam_name}, skipping")
                        continue
                    
                    if cam_name == 'left' or cam_name == 'right':
                        frames[cam_name] = parse_carla_image(carla_img)
                    elif cam_name == 'depth_gt':
                        frames[cam_name] = parse_carla_image(carla_img)
                    elif cam_name == 'seg_gt':
                        frames[cam_name] = parse_carla_image(carla_img)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[!] Error processing frame: {e}")
                    continue
            
            # Skip frame if we didn't get all 4
            if len(frames) < 4:
                print(f"[!] Only got {len(frames)}/4 frames, skipping this cycle")
                continue
            
            left_tensor = preprocess_for_pytorch(frames['left'], device)
            right_tensor = preprocess_for_pytorch(frames['right'], device)
            
            # Pad to multiple of 32 to prevent dimension mismatch (e.g., 76 vs 75)
            left_padded, pad_h, pad_w = pad_to_multiple(left_tensor, 32)
            right_padded, _, _ = pad_to_multiple(right_tensor, 32)
            
            # Verify tensors and models are on same device before inference
            try:
                with torch.no_grad():
                    # PSMNet Inference on padded images
                    depth_out_padded = psmnet(left_padded, right_padded)
                    if isinstance(depth_out_padded, tuple) or isinstance(depth_out_padded, list):
                        depth_out_padded = depth_out_padded[-1]
                    
                    # Crop output back to original CARLA resolution
                    if depth_out_padded.dim() == 4:
                        depth_out = depth_out_padded[:, :, :depth_out_padded.shape[2]-pad_h, :depth_out_padded.shape[3]-pad_w]
                    else:  # If shape is [B, H, W]
                        depth_out = depth_out_padded[:, :depth_out_padded.shape[1]-pad_h, :depth_out_padded.shape[2]-pad_w]
                    
                    # DeepLab Inference
                    seg_out = deeplabv3(right_padded)
                    if isinstance(seg_out, dict) and 'out' in seg_out:
                        seg_out = seg_out['out'].argmax(dim=1)
                    else:
                        seg_out = seg_out.argmax(dim=1)

                    # Crop segmentation to original (unpadded) size.
                    seg_out = seg_out[:, :seg_out.shape[1]-pad_h, :seg_out.shape[2]-pad_w]
            except RuntimeError as e:
                if "should be the same" in str(e) or "device" in str(e).lower():
                    print(f"[!] Device mismatch error: {e}")
                    print(f"[!] Left tensor device: {left_tensor.device}")
                    print(f"[!] Right tensor device: {right_tensor.device}")
                    print("[!] Skipping this frame and attempting recovery...")
                    continue
                else:
                    raise 
            
            # Convert ground truth depth to meters for metric calculation
            gt_depth_m = carla_depth_to_meters(frames['depth_gt'])
            
            # Extract predicted depth - keep as-is for visualization (raw model output)
            pred_depth_np = depth_out.squeeze().cpu().numpy()
            
            # For metrics: normalize predicted depth to comparable scale
            # Min-max normalize to 0-100m range for metric comparison
            pred_depth_normalized = np.interp(pred_depth_np, 
                                              (pred_depth_np.min(), pred_depth_np.max()),
                                              (DEPTH_MIN_M, DEPTH_MAX_M))
            pred_depth_m = pred_depth_normalized
            
            # Extract segmentation prediction
            seg_pred_np = seg_out.squeeze().cpu().numpy().astype(np.uint8)
            
            # Extract ground truth segmentation (convert from BGR to single channel class ID)
            # CARLA semantic segmentation encodes class in the R channel
            seg_gt_array = frames['seg_gt']
            seg_gt_np = seg_gt_array[:, :, 0]  # Use R channel (class ID)
            
            # Compute metrics
            miou = compute_miou(seg_pred_np, seg_gt_np, num_classes=13)
            rmse, delta_1_25 = compute_depth_metrics(pred_depth_m, gt_depth_m)
            
            # Visualizations
            vis_model_depth = cv2.resize(colorize_depth(pred_depth_np), (800, 600))
            vis_model_seg = cv2.resize(colorize_segmentation(seg_out), (800, 600))
            vis_gt_depth = cv2.resize(colorize_depth(gt_depth_m), (800, 600))
            vis_gt_seg = cv2.resize(colorize_segmentation(torch.from_numpy(seg_gt_np).unsqueeze(0)), (800, 600))
            
            # Ensure all visualizations are valid before stacking
            if any(v is None for v in [vis_model_depth, vis_model_seg, vis_gt_depth, vis_gt_seg]):
                print("[!] Some visualizations failed, skipping frame")
                continue
            
            # Add metrics overlay to visualizations
            vis_model_depth = draw_metrics_on_image(vis_model_depth.copy(), miou, rmse, delta_1_25)
            
            top_row = np.hstack((frames['left'], frames['right']))
            middle_row = np.hstack((vis_model_depth, vis_model_seg))
            bottom_row = np.hstack((vis_gt_depth, vis_gt_seg))
            grid_full = np.vstack((top_row, middle_row, bottom_row))
            grid = cv2.resize(grid_full, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
            
            cv2.imshow("CARLA v0.9.15 - TorchScript Pipeline", grid)
            
            # Print metrics to console for monitoring
            print(f"mIoU: {miou:.4f} | RMSE: {rmse:.4f}m | Delta1.25: {delta_1_25:.2f}%")
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    except KeyboardInterrupt:
        pass
    finally:
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        for actor in actor_list: actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()