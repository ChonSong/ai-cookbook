"""
Train Custom Card Detector

This script trains a YOLO model to detect poker cards from screenshots
of your specific poker application.

Requirements:
    - Annotated dataset (images + labels in YOLO format)
    - ultralytics or YOLOv5
    - GPU recommended for training

Usage:
    # Train with default settings
    python train_detector.py --data ./dataset

    # Train with custom parameters
    python train_detector.py --data ./dataset --epochs 100 --batch 16

    # Resume training
    python train_detector.py --resume ./runs/train/exp/weights/last.pt
"""

import argparse
from pathlib import Path
import yaml
import shutil


class YOLOTrainer:
    """Train YOLO model for card detection."""

    def __init__(self, model='yolov5s', device='cuda'):
        """
        Initialize YOLO trainer.

        Args:
            model (str): Base model to use (yolov5s, yolov5m, yolov8n, etc.)
            device (str): Training device ('cuda', 'cpu', or device ID)
        """
        self.model_name = model
        self.device = device

        # Try to load YOLO
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'{model}.pt')
            self.framework = 'ultralytics'
            print(f"✓ Loaded {model} from ultralytics")
        except ImportError:
            print("⚠ ultralytics not installed, trying YOLOv5...")
            try:
                import torch
                self.model = torch.hub.load('ultralytics/yolov5', model)
                self.framework = 'yolov5'
                print(f"✓ Loaded {model} from YOLOv5")
            except Exception as e:
                print(f"✗ Failed to load YOLO: {e}")
                print("Install with: pip install ultralytics")
                self.model = None
                self.framework = None

    def prepare_dataset(self, data_dir):
        """
        Prepare and validate dataset structure.

        Expected structure:
        data_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml

        Args:
            data_dir (Path): Dataset directory

        Returns:
            Path: Path to data.yaml config file
        """
        data_dir = Path(data_dir)

        # Check structure
        required_dirs = [
            data_dir / 'images' / 'train',
            data_dir / 'images' / 'val',
            data_dir / 'labels' / 'train',
            data_dir / 'labels' / 'val'
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"⚠ Creating missing directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

        # Check for images and labels
        train_images = list((data_dir / 'images' / 'train').glob('*.png')) + \
                      list((data_dir / 'images' / 'train').glob('*.jpg'))
        val_images = list((data_dir / 'images' / 'val').glob('*.png')) + \
                    list((data_dir / 'images' / 'val').glob('*.jpg'))

        print(f"✓ Found {len(train_images)} training images")
        print(f"✓ Found {len(val_images)} validation images")

        if len(train_images) == 0:
            print("✗ No training images found!")
            print("Please add annotated images to:", data_dir / 'images' / 'train')
            return None

        # Create or update data.yaml
        yaml_path = data_dir / 'data.yaml'
        
        # Define card classes
        card_classes = self._generate_card_classes()

        data_config = {
            'path': str(data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(card_classes),  # Number of classes
            'names': card_classes
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"✓ Dataset config saved: {yaml_path}")
        return yaml_path

    def _generate_card_classes(self):
        """
        Generate list of card classes.

        Returns:
            list: List of card class names
        """
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        suits = ['spades', 'hearts', 'diamonds', 'clubs']

        # Option 1: Each card as separate class (52 classes)
        # classes = [f"{rank}_{suit}" for rank in ranks for suit in suits]

        # Option 2: Generic "card" class (1 class - simpler for detection)
        classes = ['card']

        # Option 3: Classify by rank and suit separately (17 classes)
        # classes = ranks + suits

        return classes

    def train(self, data_yaml, epochs=100, batch_size=16, img_size=640,
              patience=50, save_dir='./runs/train', resume=None):
        """
        Train YOLO model.

        Args:
            data_yaml (Path): Path to data.yaml config
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            img_size (int): Image size for training
            patience (int): Early stopping patience
            save_dir (str): Directory to save results
            resume (str): Path to checkpoint to resume from

        Returns:
            dict: Training results
        """
        if self.model is None:
            print("✗ Model not loaded, cannot train")
            return None

        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {img_size}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")

        if self.framework == 'ultralytics':
            # Train with ultralytics
            results = self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                patience=patience,
                device=self.device,
                project=save_dir,
                name='exp',
                exist_ok=True,
                resume=resume
            )

        elif self.framework == 'yolov5':
            # Train with YOLOv5
            import subprocess
            cmd = [
                'python', '-m', 'yolov5.train',
                '--data', str(data_yaml),
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--img', str(img_size),
                '--patience', str(patience),
                '--device', self.device,
                '--project', save_dir,
                '--name', 'exp'
            ]

            if resume:
                cmd.extend(['--resume', resume])

            subprocess.run(cmd, check=True)
            results = None

        print("\n✓ Training completed!")
        return results

    def evaluate(self, data_yaml, weights):
        """
        Evaluate trained model on validation set.

        Args:
            data_yaml (Path): Path to data.yaml config
            weights (str): Path to trained weights

        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)

        if self.framework == 'ultralytics':
            from ultralytics import YOLO
            model = YOLO(weights)
            results = model.val(data=str(data_yaml))

            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")

            return results

        elif self.framework == 'yolov5':
            import subprocess
            cmd = [
                'python', '-m', 'yolov5.val',
                '--data', str(data_yaml),
                '--weights', weights,
                '--device', self.device
            ]

            subprocess.run(cmd, check=True)
            return None

    def export_model(self, weights, format='onnx'):
        """
        Export model to different format for deployment.

        Args:
            weights (str): Path to trained weights
            format (str): Export format ('onnx', 'torchscript', 'tflite', etc.)
        """
        print(f"\n→ Exporting model to {format}...")

        if self.framework == 'ultralytics':
            from ultralytics import YOLO
            model = YOLO(weights)
            model.export(format=format)
            print(f"✓ Model exported to {format}")

        elif self.framework == 'yolov5':
            import subprocess
            cmd = [
                'python', '-m', 'yolov5.export',
                '--weights', weights,
                '--include', format
            ]
            subprocess.run(cmd, check=True)


def create_sample_dataset_structure(output_dir):
    """
    Create sample dataset structure with README.

    Args:
        output_dir (Path): Output directory for dataset
    """
    output_dir = Path(output_dir)

    # Create directories
    dirs = [
        output_dir / 'images' / 'train',
        output_dir / 'images' / 'val',
        output_dir / 'labels' / 'train',
        output_dir / 'labels' / 'val'
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create README
    readme = output_dir / 'README.md'
    readme.write_text("""# Card Detection Dataset

## Structure

```
dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   └── val/            # Validation labels (YOLO format)
└── data.yaml           # Dataset configuration (auto-generated)
```

## Annotation Format

YOLO format (one .txt file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).

## Recommended Tools

1. **LabelImg**: https://github.com/heartexlabs/labelImg
2. **Roboflow**: https://roboflow.com/ (online, free tier available)
3. **CVAT**: https://cvat.org/ (self-hosted or cloud)

## Dataset Size Recommendations

- Minimum: 200-300 images (100 per class)
- Good: 500-1000 images
- Excellent: 1000+ images

## Tips

1. Capture images from actual poker app
2. Include various game states (pre-flop, flop, turn, river)
3. Vary lighting conditions
4. Include edge cases (overlapping cards, partially visible)
5. Split 80/20 for train/val

## Training

```bash
python 06_train_detector.py --data ./dataset --epochs 100
```
""")

    print(f"✓ Dataset structure created: {output_dir}")
    print(f"✓ README created: {readme}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train custom YOLO model for poker card detection"
    )
    parser.add_argument(
        "--data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model",
        default="yolov5s",
        help="Base model (yolov5s, yolov5m, yolov8n, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device (cuda, cpu, or device ID)"
    )
    parser.add_argument(
        "--resume",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--create-dataset",
        help="Create sample dataset structure at specified path"
    )
    parser.add_argument(
        "--evaluate",
        help="Evaluate model on validation set (provide weights path)"
    )

    args = parser.parse_args()

    # Create sample dataset structure
    if args.create_dataset:
        create_sample_dataset_structure(args.create_dataset)
        return

    # Validate inputs
    if not args.data and not args.evaluate:
        print("✗ Please specify --data for training or --create-dataset to setup")
        return

    # Initialize trainer
    trainer = YOLOTrainer(model=args.model, device=args.device)

    # Evaluation mode
    if args.evaluate:
        if not args.data:
            print("✗ Please specify --data for evaluation")
            return
        data_yaml = Path(args.data) / 'data.yaml'
        trainer.evaluate(data_yaml, args.evaluate)
        return

    # Prepare dataset
    data_yaml = trainer.prepare_dataset(args.data)
    if not data_yaml:
        print("✗ Dataset preparation failed")
        return

    # Train model
    trainer.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
