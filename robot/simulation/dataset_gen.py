"""
Simulation Dataset Generation Module
Generates training data in PyBullet/Gazebo simulation environment
Supports images, depth maps, segmentation masks, joint state recording
Pure Python implementation using PIL instead of cv2
"""

import time
import math
import json
import os
import random
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont


class RobotControllerBase:
    """Base class for robot controllers"""
    
    def __init__(self, robot_name: str = "robot"):
        self.robot_name = robot_name
        self._connected = False
    
    def connect(self) -> bool:
        """连接（基类默认实现：模拟连接）"""
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """断开连接（基类默认实现）"""
        self._connected = False
        return True
    
    def move_joint(self, joint_positions: List[float], velocity: float = 0.5, 
                   acceleration: float = 0.5) -> bool:
        """关节运动（基类默认实现：模拟运动）"""
        if not self._connected:
            return False
        return True
    
    def move_cartesian(self, pose: Tuple[float, float, float, float, float, float],
                       velocity: float = 0.2, acceleration: float = 0.2) -> bool:
        """笛卡尔空间运动（基类默认实现：模拟运动）"""
        if not self._connected:
            return False
        return True
    
    def get_joint_positions(self) -> List[float]:
        """获取关节位置（基类默认实现：返回零位）"""
        return [0.0] * 6
    
    def get_tcp_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取TCP位姿（基类默认实现：返回零位）"""
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def stop(self) -> bool:
        """停止运动（基类默认实现）"""
        return True
    
    def set_force_torque(self, force: Tuple[float, float, float], 
                         torque: Tuple[float, float, float]) -> bool:
        """设置力/力矩（基类默认实现：不支持）"""
        return False
    
    def force_control_enable(self, enable: bool) -> bool:
        """启用/禁用力控（基类默认实现：不支持）"""
        return False
    
    def get_force_torque(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """获取力/力矩（基类默认实现：返回零值）"""
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    
    def set_digital_out(self, pin: int, value: bool) -> bool:
        """设置数字输出（基类默认实现：记录日志）"""
        return False
    
    def get_digital_in(self, pin: int) -> bool:
        """获取数字输入（基类默认实现：返回False）"""
        return False
    
    def set_analog_out(self, pin: int, value: float) -> bool:
        """设置模拟输出（基类默认实现：不支持）"""
        return False
    
    def get_analog_in(self, pin: int) -> float:
        """获取模拟输入（基类默认实现：返回0.0）"""
        return 0.0


class SimDatasetGenerator:
    """Simulation Dataset Generator - Pure Python Implementation"""
    
    def __init__(self, robot: RobotControllerBase, sim_client: Any = None, 
                 output_dir: str = "./dataset"):
        """
        robot: Robot controller
        sim_client: Simulation client (PyBullet client id or Gazebo interface)
        output_dir: Output directory
        """
        self.robot = robot
        self.sim = sim_client
        self.output_dir = output_dir
        self._camera_intrinsics = None
        self._camera_extrinsics = None
        self._frame_counter = 0
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    def set_camera(self, intrinsics: Any, extrinsics: Any):
        """Set camera parameters"""
        self._camera_intrinsics = intrinsics
        self._camera_extrinsics = extrinsics
    
    def capture_frame(self, frame_id: int, robot_pose: Tuple, 
                      joint_positions: List[float]) -> Dict[str, Any]:
        """
        Capture a frame (image, depth, segmentation mask)
        Returns: {"image": PIL.Image, "depth": numpy array, "mask": PIL.Image}
        """
        # Actual rendering requires simulation
        # Here provides simulated implementation using PIL
        
        # Generate random image
        img_array = [[random.randint(0, 255) for _ in range(640)] for _ in range(480)]
        img = Image.new('RGB', (640, 480))
        pixels = img.load()
        for y in range(480):
            for x in range(640):
                pixels[x, y] = (img_array[y][x], 
                               (img_array[y][x] + 30) % 256, 
                               (img_array[y][x] + 60) % 256)
        
        # Draw text on image
        draw = ImageDraw.Draw(img)
        text = f"Frame {frame_id}"
        # Simple text drawing
        draw.text((10, 10), text, fill=(255, 255, 255))
        
        # Generate depth map (normalized)
        depth_array = [[random.random() for _ in range(640)] for _ in range(480)]
        
        # Generate mask
        mask = Image.new('L', (640, 480), 0)
        mask_draw = ImageDraw.Draw(mask)
        # Draw a simple rectangle as placeholder
        mask_draw.rectangle([200, 150, 440, 330], fill=128)
        
        return {
            "image": img, 
            "depth": depth_array, 
            "mask": mask,
            "pose": robot_pose,
            "joints": joint_positions
        }
    
    def record_frame(self, frame_id: int, extra_data: Dict = None) -> int:
        """Record a frame to disk"""
        pose = self.robot.get_tcp_pose()
        joints = self.robot.get_joint_positions()
        frame = self.capture_frame(frame_id, pose, joints)
        
        # Save image
        img_path = os.path.join(self.output_dir, "images", f"frame_{frame_id:06d}.png")
        frame["image"].save(img_path)
        
        # Save depth map (numpy format as .npy, or convert to image)
        import numpy as np
        depth_array = np.array(frame["depth"], dtype=np.float32)
        depth_path = os.path.join(self.output_dir, "depths", f"frame_{frame_id:06d}.npy")
        np.save(depth_path, depth_array)
        
        # Also save as 16-bit PNG for compatibility
        depth_img_path = os.path.join(self.output_dir, "depths", f"frame_{frame_id:06d}_16bit.png")
        depth_img = Image.fromarray((depth_array * 65535).astype(np.uint16), mode='I;16')
        depth_img.save(depth_img_path)
        
        # Save segmentation mask
        mask_path = os.path.join(self.output_dir, "masks", f"frame_{frame_id:06d}.png")
        frame["mask"].save(mask_path)
        
        # Save metadata
        metadata = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "robot_pose": list(pose),
            "joint_positions": joints,
            "extra": extra_data or {}
        }
        metadata_path = os.path.join(self.output_dir, f"frame_{frame_id:06d}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return frame_id
    
    def generate_sequence(self, num_frames: int = 100, random_motion: bool = True):
        """Generate continuous sequence data"""
        print(f"Generating {num_frames} frames...")
        for i in range(num_frames):
            if random_motion:
                # Randomly move robot (only in simulation)
                self._random_move()
            self.record_frame(i)
            if i % 10 == 0:
                print(f"Recorded {i}/{num_frames} frames")
        print("Dataset generation completed")
    
    def _random_move(self):
        """Randomly move robot in simulation"""
        # Get current pose, add random offset
        pose = self.robot.get_tcp_pose()
        new_pose = (
            pose[0] + random.uniform(-0.05, 0.05),
            pose[1] + random.uniform(-0.05, 0.05),
            pose[2] + random.uniform(-0.02, 0.02),
            pose[3] + random.uniform(-0.1, 0.1),
            pose[4] + random.uniform(-0.1, 0.1),
            pose[5] + random.uniform(-0.1, 0.1)
        )
        self.robot.move_cartesian(new_pose, velocity=0.05)
    
    def generate_robot_poses_dataset(self, num_poses: int = 1000) -> str:
        """
        Generate robot joint angles -> TCP pose dataset
        Used for training inverse kinematics or forward kinematics models
        """
        data = []
        for i in range(num_poses):
            # Random joint angles
            joints = [random.uniform(-math.pi, math.pi) for _ in range(6)]
            # Execute motion (requires simulation support)
            self.robot.move_joint(joints)
            time.sleep(0.02)
            pose = self.robot.get_tcp_pose()
            data.append({
                "joints": joints,
                "pose": list(pose)
            })
        
        output_file = os.path.join(self.output_dir, "robot_poses_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_file}")
        return output_file
    
    def generate_synthetic_images(self, num_images: int = 500, 
                                  with_variations: bool = True) -> str:
        """
        Generate synthetic image data (with lighting, background variations)
        Used for training object detection/segmentation models
        """
        images_dir = os.path.join(self.output_dir, "synthetic_images")
        os.makedirs(images_dir, exist_ok=True)
        
        for i in range(num_images):
            # Generate synthetic image with random shapes
            img = Image.new('RGB', (640, 480))
            draw = ImageDraw.Draw(img)
            
            if with_variations:
                # Random background color
                bg_color = (random.randint(0, 100), 
                           random.randint(0, 100), 
                           random.randint(0, 100))
                draw.rectangle([0, 0, 640, 480], fill=bg_color)
                
                # Randomly draw an object
                shape_type = random.choice(['circle', 'rectangle', 'ellipse'])
                center_x = random.randint(100, 540)
                center_y = random.randint(100, 380)
                color = (random.randint(0, 255), 
                        random.randint(0, 255), 
                        random.randint(0, 255))
                
                if shape_type == 'circle':
                    radius = random.randint(20, 80)
                    draw.ellipse([center_x - radius, center_y - radius,
                                 center_x + radius, center_y + radius], 
                                fill=color)
                elif shape_type == 'rectangle':
                    width = random.randint(40, 160)
                    height = random.randint(40, 160)
                    draw.rectangle([center_x - width//2, center_y - height//2,
                                   center_x + width//2, center_y + height//2],
                                  fill=color)
                else:  # ellipse
                    rx = random.randint(20, 80)
                    ry = random.randint(20, 80)
                    draw.ellipse([center_x - rx, center_y - ry,
                                 center_x + rx, center_y + ry],
                                fill=color)
            
            img.save(os.path.join(images_dir, f"synth_{i:06d}.png"))
        
        print(f"Generated {num_images} synthetic images")
        return images_dir
