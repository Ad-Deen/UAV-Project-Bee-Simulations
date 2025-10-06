import argparse
import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Camera Livestream')
    
    parser.add_argument('--camera-device', type=str, default='/dev/video0', help='Camera device path (e.g., /dev/ttyUSB0 or 0 for default camera)')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # Device setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize and load the model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Open camera stream
    cap = cv2.VideoCapture(args.camera_device)
    if not cap.isOpened():
        print(f"Error: Could not open camera at {args.camera_device}")
        exit()
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Visualization settings
    margin_width = 50
    cmap = matplotlib.colormaps['Spectral_r']
    
    print(f"Streaming from {args.camera_device}. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
# size (640 , 480)
        # Generate depth map
        depth = depth_anything.infer_image(raw_frame, args.input_size)
        print(depth.shape)
        np.save('my_depth.npy', depth)
        # Normalize depth map to [0, 255]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Visualize depth map
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Prepare output frame
        if args.pred_only:
            display_frame = depth
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            display_frame = cv2.hconcat([raw_frame, split_region, depth])
        
        # Display the frame
        cv2.imshow('Depth Anything V2 - Live', display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()