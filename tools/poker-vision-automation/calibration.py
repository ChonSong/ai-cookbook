"""
Button Position Calibration Tool

Interactive tool to calibrate button positions for poker automation.
This allows customization for different poker apps with different layouts.

Requirements:
    - OpenCV (opencv-python)
    - A screenshot of your poker app showing all buttons

Usage:
    # Interactive calibration
    python calibration.py --screenshot screenshot.png

    # Load existing config and verify
    python calibration.py --screenshot screenshot.png --config button_config.json --verify
"""

import cv2
import json
import argparse
from pathlib import Path


class ButtonCalibrator:
    """Interactive tool to calibrate button positions for poker apps."""
    
    def __init__(self, screenshot_path, existing_config=None):
        """
        Initialize button calibrator.
        
        Args:
            screenshot_path (str): Path to poker app screenshot
            existing_config (str): Path to existing config to load
        """
        self.screenshot_path = Path(screenshot_path)
        if not self.screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
        
        self.image = cv2.imread(str(self.screenshot_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {screenshot_path}")
        
        self.display_image = self.image.copy()
        self.positions = {}
        self.current_button = None
        
        # Standard button names for poker apps
        self.button_names = ['fold', 'check', 'call', 'raise', 'bet', 'all_in']
        
        # Load existing config if provided
        if existing_config and Path(existing_config).exists():
            with open(existing_config) as f:
                self.positions = json.load(f)
            print(f"✓ Loaded existing configuration from {existing_config}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_button:
            self.positions[self.current_button] = [x, y]
            print(f"  {self.current_button}: ({x}, {y})")
            
            # Draw marker on image
            cv2.circle(self.display_image, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(
                self.display_image,
                self.current_button.upper(),
                (x + 15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow('Calibration', self.display_image)
    
    def calibrate(self, verify_only=False):
        """
        Run interactive calibration process.
        
        Args:
            verify_only (bool): If True, only display existing positions
        
        Returns:
            dict: Button positions {button_name: [x, y]}
        """
        window_name = 'Calibration'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        if verify_only:
            # Display existing positions
            self._draw_all_positions()
            print("\nVerifying existing button positions...")
            print("Press any key to close")
            cv2.imshow(window_name, self.display_image)
            cv2.waitKey(0)
        else:
            # Interactive calibration
            print("\n=== Button Position Calibration ===")
            print("Instructions:")
            print("  1. Click the CENTER of each button when prompted")
            print("  2. If you make a mistake, you can recalibrate later")
            print("  3. Press any key after each click to continue")
            print("  4. Press 's' to skip a button if not visible\n")
            
            for button in self.button_names:
                self.current_button = button
                print(f"\nClick the center of the '{button.upper()}' button")
                print("(Press 's' to skip if this button is not visible)")
                
                # Reset display
                self.display_image = self.image.copy()
                self._draw_all_positions()
                cv2.imshow(window_name, self.display_image)
                
                key = cv2.waitKey(0)
                if key == ord('s') or key == ord('S'):
                    print(f"  Skipped {button}")
                    continue
        
        cv2.destroyAllWindows()
        return self.positions
    
    def _draw_all_positions(self):
        """Draw all calibrated positions on the display image."""
        for button_name, pos in self.positions.items():
            if pos:
                x, y = pos
                cv2.circle(self.display_image, (x, y), 10, (0, 255, 0), 2)
                cv2.putText(
                    self.display_image,
                    button_name.upper(),
                    (x + 15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    
    def save(self, config_path='button_config.json'):
        """
        Save calibration to JSON file.
        
        Args:
            config_path (str): Output path for config file
        """
        config_path = Path(config_path)
        
        # Remove any skipped buttons
        positions_to_save = {k: v for k, v in self.positions.items() if v}
        
        with open(config_path, 'w') as f:
            json.dump(positions_to_save, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {config_path}")
        print(f"  Calibrated {len(positions_to_save)} buttons: {list(positions_to_save.keys())}")
    
    def validate(self):
        """
        Validate the calibrated positions.
        
        Returns:
            bool: True if validation passes
        """
        if not self.positions:
            print("✗ No button positions calibrated")
            return False
        
        # Check if at least fold and one action button are calibrated
        required_buttons = ['fold']
        action_buttons = ['call', 'check', 'raise', 'bet']
        
        has_fold = 'fold' in self.positions
        has_action = any(btn in self.positions for btn in action_buttons)
        
        if not has_fold:
            print("✗ Missing required 'fold' button position")
            return False
        
        if not has_action:
            print("✗ Missing at least one action button (call/check/raise/bet)")
            return False
        
        # Check all positions are within image bounds
        height, width = self.image.shape[:2]
        for button, pos in self.positions.items():
            x, y = pos
            if not (0 <= x < width and 0 <= y < height):
                print(f"✗ Button '{button}' position ({x}, {y}) is out of bounds")
                return False
        
        print(f"✓ Validation passed: {len(self.positions)} buttons calibrated")
        return True


def main():
    """Main entry point for calibration tool."""
    parser = argparse.ArgumentParser(
        description="Button Position Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run interactive calibration
    python calibration.py --screenshot poker_app.png
    
    # Verify existing calibration
    python calibration.py --screenshot poker_app.png --config button_config.json --verify
    
    # Calibrate with custom output file
    python calibration.py --screenshot poker_app.png --output my_config.json
        """
    )
    
    parser.add_argument(
        '--screenshot', '-s',
        required=True,
        help='Screenshot of poker app showing all buttons'
    )
    parser.add_argument(
        '--config', '-c',
        help='Existing config file to load (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        default='button_config.json',
        help='Output config file path (default: button_config.json)'
    )
    parser.add_argument(
        '--verify', '-v',
        action='store_true',
        help='Verify existing config instead of calibrating'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize calibrator
        calibrator = ButtonCalibrator(args.screenshot, args.config)
        
        # Run calibration or verification
        positions = calibrator.calibrate(verify_only=args.verify)
        
        # Save if not just verifying
        if not args.verify:
            if calibrator.validate():
                calibrator.save(args.output)
            else:
                print("\n⚠ Calibration did not pass validation")
                print("Please run calibration again to fix issues")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
