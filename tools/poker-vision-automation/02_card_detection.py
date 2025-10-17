"""
Card Detection Using Computer Vision

This script demonstrates two approaches for detecting poker cards:
1. YOLO (You Only Look Once) - Deep learning-based object detection
2. OpenCV Template Matching - Traditional computer vision approach

Requirements:
    - OpenCV (opencv-python)
    - PyTorch and YOLOv5/YOLOv8 (for YOLO approach)
    - Card templates (for template matching approach)

Usage:
    # Using YOLO
    python 02_card_detection.py --input image.png --method yolo

    # Using Template Matching
    python 02_card_detection.py --input image.png --method template

    # Process entire directory
    python 02_card_detection.py --input ./screenshots/ --output ./detected/
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json


class CardDetectorYOLO:
    """Card detection using YOLO (You Only Look Once) deep learning model."""

    def __init__(self, model_path="yolov5s.pt", confidence=0.5):
        """
        Initialize YOLO-based card detector.

        Args:
            model_path (str): Path to trained YOLO model
            confidence (float): Minimum confidence threshold (0-1)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

        # Try to load YOLO model
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded: {model_path}")
        except ImportError:
            print("⚠ ultralytics not installed. Using YOLOv5 alternative...")
            try:
                import torch
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                print(f"✓ YOLOv5 model loaded: {model_path}")
            except Exception as e:
                print(f"✗ Failed to load YOLO model: {e}")
                print("Please install: pip install ultralytics")

    def detect_cards(self, image):
        """
        Detect cards in an image using YOLO.

        Args:
            image (np.ndarray): Input image

        Returns:
            list: List of detected cards with bounding boxes and confidence
                  Format: [{"bbox": [x1, y1, x2, y2], "confidence": float, "class": str}]
        """
        if self.model is None:
            print("✗ Model not loaded. Cannot perform detection.")
            return []

        # Run inference
        results = self.model(image)

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get class name (if available)
                class_id = int(box.cls[0])
                class_name = result.names[class_id] if hasattr(result, 'names') else f"card_{class_id}"

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class": class_name
                })

        return detections


class CardDetectorTemplate:
    """Card detection using OpenCV template matching."""

    def __init__(self, template_dir="./templates", threshold=0.7):
        """
        Initialize template-based card detector.

        Args:
            template_dir (str): Directory containing card template images
            threshold (float): Matching threshold (0-1)
        """
        self.template_dir = Path(template_dir)
        self.threshold = threshold
        self.templates = self._load_templates()

    def _load_templates(self):
        """
        Load card templates from directory.

        Returns:
            dict: Dictionary of card templates {card_name: template_image}
        """
        templates = {}

        if not self.template_dir.exists():
            print(f"⚠ Template directory not found: {self.template_dir}")
            print("Creating example template structure...")
            self._create_template_structure()
            return templates

        # Load all template images
        for template_path in self.template_dir.glob("*.png"):
            card_name = template_path.stem
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is not None:
                templates[card_name] = template

        print(f"✓ Loaded {len(templates)} card templates")
        return templates

    def _create_template_structure(self):
        """Create example template directory structure."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        readme = self.template_dir / "README.txt"
        readme.write_text(
            "Place card template images here.\n"
            "Name format: rank_suit.png (e.g., ace_spades.png, 10_hearts.png)\n"
            "\n"
            "To create templates:\n"
            "1. Capture screenshots of individual cards from your poker app\n"
            "2. Crop each card to include only the card area\n"
            "3. Save with consistent naming\n"
            "4. Ensure all templates have similar size and quality\n"
        )

    def detect_cards(self, image):
        """
        Detect cards in an image using template matching.

        Args:
            image (np.ndarray): Input image

        Returns:
            list: List of detected cards with bounding boxes and confidence
        """
        if not self.templates:
            print("✗ No templates loaded. Cannot perform detection.")
            return []

        detections = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Match each template
        for card_name, template in self.templates.items():
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = gray_template.shape

            # Multi-scale template matching
            for scale in np.linspace(0.5, 1.5, 10):
                resized_template = cv2.resize(
                    gray_template,
                    (int(w * scale), int(h * scale))
                )

                # Skip if template is larger than image
                if (resized_template.shape[0] > gray_image.shape[0] or
                    resized_template.shape[1] > gray_image.shape[1]):
                    continue

                # Template matching
                result = cv2.matchTemplate(
                    gray_image,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED
                )

                # Find matches above threshold
                locations = np.where(result >= self.threshold)

                for pt in zip(*locations[::-1]):
                    x1, y1 = pt
                    x2 = x1 + int(w * scale)
                    y2 = y1 + int(h * scale)

                    # Get match confidence
                    confidence = result[y1, x1]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidence),
                        "class": card_name,
                        "scale": scale
                    })

        # Non-maximum suppression to remove overlapping detections
        detections = self._non_max_suppression(detections)

        return detections

    def _non_max_suppression(self, detections, overlap_threshold=0.3):
        """
        Remove overlapping detections using non-maximum suppression.

        Args:
            detections (list): List of detections
            overlap_threshold (float): IoU threshold for suppression

        Returns:
            list: Filtered detections
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                d for d in detections
                if self._compute_iou(best["bbox"], d["bbox"]) < overlap_threshold
            ]

        return keep

    def _compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]

        Returns:
            float: IoU value (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Compute union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0


def visualize_detections(image, detections, output_path=None):
    """
    Draw bounding boxes on image for detected cards.

    Args:
        image (np.ndarray): Input image
        detections (list): List of detections
        output_path (str): Optional path to save visualization

    Returns:
        np.ndarray: Image with bounding boxes drawn
    """
    result_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection.get("class", "card")

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"✓ Visualization saved: {output_path}")

    return result_image


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect poker cards using computer vision"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input image or directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./detected",
        help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["yolo", "template"],
        default="template",
        help="Detection method to use"
    )
    parser.add_argument(
        "--model",
        default="yolov5s.pt",
        help="Path to YOLO model (for YOLO method)"
    )
    parser.add_argument(
        "--templates",
        default="./templates",
        help="Directory containing card templates (for template method)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with bounding boxes"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    if args.method == "yolo":
        detector = CardDetectorYOLO(model_path=args.model, confidence=args.confidence)
    else:
        detector = CardDetectorTemplate(
            template_dir=args.templates,
            threshold=args.confidence
        )

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"✗ Failed to load image: {input_path}")
            return

        print(f"Processing: {input_path.name}")
        detections = detector.detect_cards(image)
        print(f"✓ Detected {len(detections)} cards")

        # Save results
        result_file = output_dir / f"{input_path.stem}_detections.json"
        with open(result_file, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"✓ Results saved: {result_file}")

        # Visualize if requested
        if args.visualize:
            vis_file = output_dir / f"{input_path.stem}_visualized.png"
            visualize_detections(image, detections, str(vis_file))

    elif input_path.is_dir():
        # Process directory
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        print(f"Processing {len(image_files)} images...")

        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            print(f"Processing: {image_file.name}")
            detections = detector.detect_cards(image)
            print(f"  Detected {len(detections)} cards")

            # Save results
            result_file = output_dir / f"{image_file.stem}_detections.json"
            with open(result_file, "w") as f:
                json.dump(detections, f, indent=2)

            # Visualize if requested
            if args.visualize:
                vis_file = output_dir / f"{image_file.stem}_visualized.png"
                visualize_detections(image, detections, str(vis_file))

        print(f"✓ Processed {len(image_files)} images")

    else:
        print(f"✗ Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
