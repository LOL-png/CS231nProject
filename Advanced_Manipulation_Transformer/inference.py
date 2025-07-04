#!/usr/bin/env python3
"""
Advanced Manipulation Transformer Inference Script
Real-time inference with optimizations
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import time
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
import cv2

# Set DEX_YCB_DIR environment variable
os.environ['DEX_YCB_DIR'] = '/home/n231/231nProjectV2/dex-ycb-toolkit/data'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unified_model import UnifiedManipulationTransformer
from data.enhanced_dexycb import preprocess_image
from optimizations.flash_attention import replace_with_flash_attention
import logging

logger = logging.getLogger(__name__)

class InferencePipeline:
    """Optimized inference pipeline for real-time performance"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        use_fp16: bool = True,
        use_flash_attention: bool = True,
        benchmark: bool = False
    ):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.benchmark = benchmark
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Extract config from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.config = checkpoint.get('config', {})
        
        # Setup model
        self._setup_model(checkpoint_path, use_flash_attention)
        
        # Warmup if benchmarking
        if self.benchmark:
            self._warmup()
    
    def _setup_model(self, checkpoint_path: str, use_flash_attention: bool):
        """Load and setup model for inference"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Create model
        self.model = UnifiedManipulationTransformer(self.config.get('model', {}))
        
        # Apply optimizations
        if use_flash_attention:
            self.model = replace_with_flash_attention(self.model)
            logger.info("Enabled FlashAttention for inference")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle distributed training checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable mixed precision
        if self.use_fp16:
            self.model = self.model.half()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
            logger.info("Model compiled with torch.compile")
        
        logger.info("Model loaded successfully")
    
    def _warmup(self, num_runs: int = 10):
        """Warmup GPU for accurate benchmarking"""
        logger.info("Running warmup...")
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        if self.use_fp16:
            dummy_input = dummy_input.half()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model({'image': dummy_input})
        
        torch.cuda.synchronize()
        logger.info("Warmup complete")
    
    @torch.no_grad()
    def predict_single(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Dictionary with predictions:
            - hand_joints: 3D hand joint positions (21, 3)
            - hand_vertices: MANO vertices (778, 3)
            - object_poses: 6DoF object poses (N, 4, 4)
            - contact_points: Contact points (M, 3)
        """
        # Preprocess image
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        if self.use_fp16:
            image_tensor = image_tensor.half()
        
        # Run inference
        start_time = time.perf_counter()
        
        outputs = self.model({'image': image_tensor})
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Convert outputs to numpy
        results = {
            'inference_time_ms': inference_time
        }
        
        if 'hand_joints' in outputs:
            results['hand_joints'] = outputs['hand_joints'][0].cpu().numpy()
        
        if 'hand_vertices' in outputs:
            results['hand_vertices'] = outputs['hand_vertices'][0].cpu().numpy()
        
        if 'object_poses' in outputs:
            results['object_poses'] = outputs['object_poses'][0].cpu().numpy()
        
        if 'contact_points' in outputs:
            results['contact_points'] = outputs['contact_points'][0].cpu().numpy()
        
        return results
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on a batch of images
        
        Args:
            images: List of RGB images as numpy arrays
        
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            image_tensor = preprocess_image(image)
            image_tensors.append(image_tensor)
        
        # Stack into batch
        batch = torch.stack(image_tensors).to(self.device)
        if self.use_fp16:
            batch = batch.half()
        
        # Run inference
        start_time = time.perf_counter()
        
        outputs = self.model({'image': batch})
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        avg_time = inference_time / len(images)
        
        # Convert outputs to list of numpy arrays
        results = []
        for i in range(len(images)):
            result = {'inference_time_ms': avg_time}
            
            if 'hand_joints' in outputs:
                result['hand_joints'] = outputs['hand_joints'][i].cpu().numpy()
            
            if 'hand_vertices' in outputs:
                result['hand_vertices'] = outputs['hand_vertices'][i].cpu().numpy()
            
            if 'object_poses' in outputs:
                result['object_poses'] = outputs['object_poses'][i].cpu().numpy()
            
            if 'contact_points' in outputs:
                result['contact_points'] = outputs['contact_points'][i].cpu().numpy()
            
            results.append(result)
        
        return results
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = True,
        save_predictions: bool = True
    ) -> Dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            visualize: Whether to create visualization
            save_predictions: Whether to save predictions to file
        
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if needed
        if visualize and output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        frame_times = []
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            result = self.predict_single(frame_rgb)
            predictions.append(result)
            frame_times.append(result['inference_time_ms'])
            
            # Visualize if needed
            if visualize:
                vis_frame = self._visualize_frame(frame, result)
                if output_path:
                    out.write(vis_frame)
                else:
                    cv2.imshow('Predictions', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if visualize and output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Save predictions
        if save_predictions:
            pred_path = Path(output_path).stem + '_predictions.json' if output_path else 'predictions.json'
            self._save_predictions(predictions, pred_path)
        
        # Compute statistics
        stats = {
            'total_frames': len(predictions),
            'avg_inference_time_ms': np.mean(frame_times),
            'std_inference_time_ms': np.std(frame_times),
            'min_inference_time_ms': np.min(frame_times),
            'max_inference_time_ms': np.max(frame_times),
            'real_time_capable': np.mean(frame_times) < (1000 / fps)
        }
        
        logger.info(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        logger.info(f"Real-time capable: {stats['real_time_capable']}")
        
        return stats
    
    def _visualize_frame(self, frame: np.ndarray, predictions: Dict) -> np.ndarray:
        """Visualize predictions on frame"""
        vis_frame = frame.copy()
        
        # Draw hand joints
        if 'hand_joints' in predictions:
            joints_2d = self._project_to_2d(predictions['hand_joints'])
            for i, (x, y) in enumerate(joints_2d):
                cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Draw skeleton
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            
            for start, end in connections:
                pt1 = joints_2d[start].astype(int)
                pt2 = joints_2d[end].astype(int)
                cv2.line(vis_frame, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
        
        # Add inference time
        text = f"Inference: {predictions['inference_time_ms']:.1f} ms"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def _project_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Simple projection of 3D points to 2D (placeholder)"""
        # This is a simplified projection - in practice, use camera intrinsics
        # Assuming points are in camera coordinates
        focal_length = 500  # Placeholder
        center = np.array([112, 112])  # Image center
        
        points_2d = points_3d[:, :2] / (points_3d[:, 2:3] + 1e-8) * focal_length + center
        return points_2d
    
    def _save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_preds = []
        for pred in predictions:
            s_pred = {}
            for key, value in pred.items():
                if isinstance(value, np.ndarray):
                    s_pred[key] = value.tolist()
                else:
                    s_pred[key] = value
            serializable_preds.append(s_pred)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_preds, f)
        
        logger.info(f"Saved predictions to {output_path}")
    
    def benchmark(self, num_runs: int = 100, batch_sizes: List[int] = [1, 4, 8, 16]):
        """Run comprehensive benchmark"""
        logger.info("Running benchmark...")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create dummy batch
            dummy_batch = torch.randn(batch_size, 3, 224, 224, device=self.device)
            if self.use_fp16:
                dummy_batch = dummy_batch.half()
            
            # Time inference
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                
                with torch.no_grad():
                    _ = self.model({'image': dummy_batch})
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append((time.perf_counter() - start) * 1000)
            
            # Compute statistics
            times = np.array(times[10:])  # Skip first 10 for warmup
            results[f'batch_{batch_size}'] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'fps': 1000 / np.mean(times) * batch_size
            }
            
            logger.info(f"Batch size {batch_size}: {results[f'batch_{batch_size}']['mean_ms']:.2f} ms "
                       f"({results[f'batch_{batch_size}']['fps']:.1f} FPS)")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Advanced Manipulation Transformer Inference")
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 inference')
    parser.add_argument('--no-flash', action='store_true', help='Disable FlashAttention')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    
    args = parser.parse_args()
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        use_fp16=not args.no_fp16,
        use_flash_attention=not args.no_flash,
        benchmark=args.benchmark
    )
    
    if args.benchmark:
        # Run benchmark
        results = pipeline.benchmark()
        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"{key}: {value['mean_ms']:.2f} ms ({value['fps']:.1f} FPS)")
    
    elif args.input:
        # Process input
        input_path = Path(args.input)
        
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Process video
            stats = pipeline.process_video(
                str(input_path),
                output_path=args.output,
                visualize=True
            )
            print(f"\nProcessed {stats['total_frames']} frames")
            print(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
            
        else:
            # Process image
            image = cv2.imread(str(input_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            result = pipeline.predict_single(image_rgb)
            
            print(f"\nInference time: {result['inference_time_ms']:.2f} ms")
            print(f"Hand joints shape: {result.get('hand_joints', np.array([])).shape}")
            print(f"Object poses shape: {result.get('object_poses', np.array([])).shape}")
            
            # Visualize if output path given
            if args.output:
                vis_image = pipeline._visualize_frame(image, result)
                cv2.imwrite(args.output, vis_image)
                print(f"Saved visualization to {args.output}")
    
    else:
        print("Please provide either --input or --benchmark flag")


if __name__ == "__main__":
    main()