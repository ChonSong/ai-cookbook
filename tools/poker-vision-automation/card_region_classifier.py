"""
Card Region Classifier

Classify detected cards into regions (hole cards, community cards, opponent cards)
based on their screen positions. Supports customizable region definitions.

Requirements:
    - OpenCV (opencv-python)

Usage:
    from card_region_classifier import CardRegionClassifier
    
    classifier = CardRegionClassifier(screen_resolution=(1080, 1920))
    region = classifier.classify_card(bbox)
"""

import cv2
import json
from pathlib import Path
from typing import Tuple, Dict, Optional


class CardRegionClassifier:
    """Classify cards into regions based on screen position."""
    
    def __init__(self, screen_resolution=(1080, 1920), config_file=None):
        """
        Initialize card region classifier.
        
        Args:
            screen_resolution (tuple): Screen resolution (height, width)
            config_file (str): Optional path to region config file
        """
        self.height, self.width = screen_resolution
        
        # Default region definitions (as percentages of screen)
        # These work for most poker apps with standard layouts
        self.regions = {
            'hole_cards': {
                'y_min': 0.65, 'y_max': 0.85,
                'x_min': 0.35, 'x_max': 0.65,
                'description': 'Your hole cards (bottom center)'
            },
            'community': {
                'y_min': 0.35, 'y_max': 0.55,
                'x_min': 0.20, 'x_max': 0.80,
                'description': 'Community cards (center)'
            },
            'opponent_top': {
                'y_min': 0.10, 'y_max': 0.30,
                'x_min': 0.35, 'x_max': 0.65,
                'description': 'Opponent cards (top center)'
            },
            'opponent_left': {
                'y_min': 0.35, 'y_max': 0.55,
                'x_min': 0.05, 'x_max': 0.25,
                'description': 'Opponent cards (left)'
            },
            'opponent_right': {
                'y_min': 0.35, 'y_max': 0.55,
                'x_min': 0.75, 'x_max': 0.95,
                'description': 'Opponent cards (right)'
            }
        }
        
        # Load custom regions if provided
        if config_file and Path(config_file).exists():
            self._load_regions(config_file)
    
    def _load_regions(self, config_file):
        """Load region definitions from JSON config."""
        try:
            with open(config_file) as f:
                custom_regions = json.load(f)
            self.regions.update(custom_regions)
            print(f"✓ Loaded region config from {config_file}")
        except Exception as e:
            print(f"⚠ Failed to load region config: {e}")
    
    def classify_card(self, bbox) -> str:
        """
        Determine which region a card belongs to.
        
        Args:
            bbox (tuple/list): Bounding box [x1, y1, x2, y2] or [x1, y1, width, height]
        
        Returns:
            str: Region name ('hole_cards', 'community', 'opponent_*', or 'unknown')
        """
        # Calculate center point of card
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Handle both formats: [x1, y1, x2, y2] and [x, y, width, height]
            if x2 < x1:  # Likely width/height format
                x_center = x1 + x2 / 2
                y_center = y1 + y2 / 2
            else:  # Coordinate format
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")
        
        # Normalize to 0-1 range
        x_norm = x_center / self.width
        y_norm = y_center / self.height
        
        # Check each region
        for region_name, bounds in self.regions.items():
            if (bounds['x_min'] <= x_norm <= bounds['x_max'] and 
                bounds['y_min'] <= y_norm <= bounds['y_max']):
                return region_name
        
        return 'unknown'
    
    def classify_cards(self, detections) -> Dict[str, list]:
        """
        Classify multiple card detections into regions.
        
        Args:
            detections (list): List of card detections with 'bbox' key
        
        Returns:
            dict: Cards grouped by region
        """
        classified = {
            'hole_cards': [],
            'community': [],
            'opponent': [],
            'unknown': []
        }
        
        for detection in detections:
            bbox = detection.get('bbox')
            if not bbox:
                continue
            
            region = self.classify_card(bbox)
            
            # Group opponent regions together
            if region.startswith('opponent'):
                classified['opponent'].append(detection)
            elif region in classified:
                classified[region].append(detection)
            else:
                classified['unknown'].append(detection)
        
        return classified
    
    def visualize_regions(self, image, output_path=None):
        """
        Draw region boundaries on an image for visualization.
        
        Args:
            image (np.ndarray): Image to draw on
            output_path (str): Optional path to save visualization
        
        Returns:
            np.ndarray: Image with region boundaries drawn
        """
        vis_image = image.copy()
        
        # Color map for different regions
        colors = {
            'hole_cards': (0, 255, 0),      # Green
            'community': (255, 255, 0),      # Yellow
            'opponent_top': (255, 0, 0),     # Red
            'opponent_left': (255, 0, 255),  # Magenta
            'opponent_right': (0, 255, 255)  # Cyan
        }
        
        # Draw each region
        for region_name, bounds in self.regions.items():
            # Convert normalized coordinates to pixels
            x1 = int(bounds['x_min'] * self.width)
            y1 = int(bounds['y_min'] * self.height)
            x2 = int(bounds['x_max'] * self.width)
            y2 = int(bounds['y_max'] * self.height)
            
            # Get color for this region
            color = colors.get(region_name, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = region_name.replace('_', ' ').title()
            cv2.putText(
                vis_image,
                label,
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Saved region visualization to {output_path}")
        
        return vis_image
    
    def calibrate_regions(self, screenshot_path, output_config='region_config.json'):
        """
        Interactive region calibration (placeholder for future implementation).
        
        Args:
            screenshot_path (str): Path to poker app screenshot
            output_config (str): Output path for calibrated regions
        """
        print("Interactive region calibration not yet implemented.")
        print("For now, manually edit region_config.json with your custom regions.")
        print("\nExample format:")
        print(json.dumps({
            'hole_cards': {
                'y_min': 0.65, 'y_max': 0.85,
                'x_min': 0.35, 'x_max': 0.65,
                'description': 'Your hole cards'
            }
        }, indent=2))
    
    def save_config(self, output_path='region_config.json'):
        """
        Save current region configuration to file.
        
        Args:
            output_path (str): Output path for config file
        """
        with open(output_path, 'w') as f:
            json.dump(self.regions, f, indent=2)
        print(f"✓ Region config saved to {output_path}")
    
    def get_region_info(self, region_name: str) -> Optional[Dict]:
        """
        Get information about a specific region.
        
        Args:
            region_name (str): Name of the region
        
        Returns:
            dict: Region information or None if not found
        """
        return self.regions.get(region_name)
    
    def print_regions(self):
        """Print information about all defined regions."""
        print("\n=== Card Region Definitions ===")
        for region_name, bounds in self.regions.items():
            print(f"\n{region_name}:")
            print(f"  X: {bounds['x_min']:.2f} - {bounds['x_max']:.2f} ({bounds['x_min']*100:.1f}% - {bounds['x_max']*100:.1f}%)")
            print(f"  Y: {bounds['y_min']:.2f} - {bounds['y_max']:.2f} ({bounds['y_min']*100:.1f}% - {bounds['y_max']*100:.1f}%)")
            if 'description' in bounds:
                print(f"  Description: {bounds['description']}")


def main():
    """Test and visualization utility."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Card Region Classifier - Visualize and test region definitions"
    )
    parser.add_argument(
        '--screenshot', '-s',
        help='Screenshot to visualize regions on'
    )
    parser.add_argument(
        '--resolution',
        default='1920x1080',
        help='Screen resolution (WIDTHxHEIGHT, e.g., 1920x1080)'
    )
    parser.add_argument(
        '--config', '-c',
        help='Region config file to load'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--save-config',
        help='Save current regions to config file'
    )
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print region definitions'
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (height, width)
    except:
        print(f"Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1920x1080)")
        return 1
    
    # Initialize classifier
    classifier = CardRegionClassifier(
        screen_resolution=resolution,
        config_file=args.config
    )
    
    # Print regions if requested
    if args.print:
        classifier.print_regions()
    
    # Save config if requested
    if args.save_config:
        classifier.save_config(args.save_config)
    
    # Visualize regions if screenshot provided
    if args.screenshot:
        image = cv2.imread(args.screenshot)
        if image is None:
            print(f"Failed to load image: {args.screenshot}")
            return 1
        
        output = args.output or 'region_visualization.png'
        classifier.visualize_regions(image, output)
        
        print(f"\n✓ Regions visualized on {args.screenshot}")
        print(f"✓ Saved to {output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
