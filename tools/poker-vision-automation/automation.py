"""
Full Poker Automation System

This script integrates all components to create a complete poker automation system:
1. Screen capture from Android VM
2. Card detection
3. OCR recognition
4. Decision-making
5. Action execution via ADB

⚠️ WARNING: This is for educational purposes only. Using bots on real poker
platforms typically violates terms of service and may be illegal.

Usage:
    # Dry run (no actions executed)
    python automation.py --dry-run

    # Live automation with slow mode
    python automation.py --slow-mode --debug

    # Specific device
    python automation.py --device 192.168.1.100:5555
"""

import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Import our modules (in a real implementation, these would be proper imports)
# For demonstration, we'll show the integration pattern


class PokerAutomationSystem:
    """Complete poker automation system."""

    def __init__(self, device_id=None, debug=False, slow_mode=False, config_file='button_config.json'):
        """
        Initialize automation system.

        Args:
            device_id (str): Android device ID
            debug (bool): Enable debug output
            slow_mode (bool): Add delays between actions
            config_file (str): Path to button position config file
        """
        self.device_id = device_id
        self.debug = debug
        self.slow_mode = slow_mode

        # Component initialization
        print("Initializing Poker Automation System...")

        # Screen capture
        try:
            from screen_capture import AndroidScreenCapture
            self.screen_capture = AndroidScreenCapture(device_id)
            print("✓ Screen capture initialized")
        except ImportError as e:
            print(f"⚠ Screen capture module not available: {e}")
            self.screen_capture = None

        # Card detection
        try:
            from card_detection import CardDetectorTemplate
            self.card_detector = CardDetectorTemplate()
            print("✓ Card detector initialized")
        except ImportError as e:
            print(f"⚠ Card detection module not available: {e}")
            self.card_detector = None

        # OCR recognition
        try:
            from ocr_recognition import CardOCRRecognizer
            self.ocr_recognizer = CardOCRRecognizer()
            print("✓ OCR recognizer initialized")
        except ImportError as e:
            print(f"⚠ OCR module not available: {e}")
            self.ocr_recognizer = None

        # Poker strategy
        try:
            from poker_logic import PokerStrategy
            self.strategy = PokerStrategy()
            print("✓ Poker strategy initialized")
        except ImportError as e:
            print(f"⚠ Poker logic module not available: {e}")
            self.strategy = None

        # Game state
        self.game_state = {
            'hole_cards': [],
            'community_cards': [],
            'pot_size': 0,
            'current_bet': 0,
            'stack_size': 1000,
            'position': 'middle'
        }

        # Load button positions from config file
        self.button_positions = self._load_button_config(config_file)

    def _load_button_config(self, config_file):
        """
        Load button positions from JSON config file.
        
        Args:
            config_file (str): Path to button config file
            
        Returns:
            dict: Button positions {button_name: (x, y)}
        """
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                # Convert lists to tuples for consistency
                positions = {k: tuple(v) if isinstance(v, list) else v 
                           for k, v in config.items()}
                print(f"✓ Loaded button positions from {config_file}")
                if self.debug:
                    for button, pos in positions.items():
                        print(f"  {button}: {pos}")
                return positions
            except Exception as e:
                print(f"⚠ Failed to load button config: {e}")
        else:
            print(f"⚠ Button config not found: {config_file}")
            print("  Please run calibration.py first to calibrate button positions")
            print("  Using default positions (may not work correctly)")
        
        # Default fallback positions
        return {
            'fold': (540, 1500),
            'check': (540, 1400),
            'call': (540, 1400),
            'raise': (540, 1300),
            'bet': (540, 1300)
        }

    def capture_and_analyze(self):
        """
        Capture screen, detect cards, and analyze game state.

        Returns:
            dict: Analyzed game state
        """
        if self.debug:
            print("\n→ Capturing screen...")

        # Capture screenshot
        if self.screen_capture:
            screenshot_path = self.screen_capture.capture_screenshot()
            if not screenshot_path:
                return None

            # Load image
            import cv2
            image = cv2.imread(str(screenshot_path))
        else:
            print("✗ Screen capture not available")
            return None

        # Detect cards
        if self.debug:
            print("→ Detecting cards...")

        detections = []
        if self.card_detector:
            detections = self.card_detector.detect_cards(image)
            if self.debug:
                print(f"  Found {len(detections)} card regions")

        # Recognize cards
        if self.debug:
            print("→ Recognizing cards...")

        recognized_cards = []
        if self.ocr_recognizer and detections:
            recognized_cards = self.ocr_recognizer.recognize_from_detections(
                image, detections
            )
            if self.debug:
                for card in recognized_cards:
                    print(f"  {card['card']} (confidence: {card['ocr_confidence']:.1f}%)")

        # Update game state
        self._update_game_state(recognized_cards, image)

        return {
            'image': image,
            'detections': detections,
            'recognized_cards': recognized_cards,
            'game_state': self.game_state.copy()
        }

    def _update_game_state(self, recognized_cards, image):
        """
        Update internal game state from recognized cards.

        Args:
            recognized_cards (list): List of recognized cards
            image (np.ndarray): Screenshot image
        """
        # This is simplified - real implementation would need:
        # 1. Region-based card classification (hole cards vs community cards vs opponent cards)
        # 2. OCR for pot size, bet amounts, stack sizes
        # 3. Button detection and state recognition

        # For demonstration, we'll use heuristics based on card positions
        hole_cards = []
        community_cards = []

        for card in recognized_cards:
            if card['rank'] and card['suit']:
                # Simple heuristic: bottom cards are hole cards, center are community
                y_center = (card['bbox'][1] + card['bbox'][3]) / 2
                image_height = image.shape[0]

                if y_center > image_height * 0.7:
                    hole_cards.append(card['card'])
                elif image_height * 0.3 < y_center < image_height * 0.6:
                    community_cards.append(card['card'])

        self.game_state['hole_cards'] = hole_cards[:2]  # Max 2 hole cards
        self.game_state['community_cards'] = community_cards[:5]  # Max 5 community

    def make_decision(self):
        """
        Make poker decision based on current game state.

        Returns:
            dict: Decision with action and reasoning
        """
        if not self.strategy:
            print("✗ Strategy module not available")
            return None

        if len(self.game_state['hole_cards']) < 2:
            if self.debug:
                print("⚠ Not enough hole cards detected, cannot make decision")
            return None

        if self.debug:
            print("\n→ Making decision...")
            print(f"  Hole cards: {' '.join(self.game_state['hole_cards'])}")
            print(f"  Community: {' '.join(self.game_state['community_cards'])}")

        # Get recommendation
        recommendation = self.strategy.recommend_action(
            self.game_state['hole_cards'],
            self.game_state['community_cards'],
            self.game_state['pot_size'],
            self.game_state['current_bet'],
            self.game_state['stack_size'],
            self.game_state['position']
        )

        if self.debug:
            print(f"  → Action: {recommendation['action']}")
            print(f"  → Reasoning: {recommendation['reasoning']}")

        return recommendation

    def execute_action(self, action, dry_run=False):
        """
        Execute poker action via ADB.

        Args:
            action (dict): Action to execute
            dry_run (bool): If True, don't actually execute
        """
        action_type = action['action'].lower()

        if dry_run:
            print(f"\n[DRY RUN] Would execute: {action_type.upper()}")
            if action.get('amount'):
                print(f"[DRY RUN] Amount: {action['amount']}")
            return

        print(f"\n→ Executing action: {action_type.upper()}")

        # Get button position
        position = self.button_positions.get(action_type)
        if not position:
            print(f"✗ Unknown action: {action_type}")
            return

        x, y = position

        # Execute tap via ADB
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'tap', str(x), str(y)])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Tapped {action_type} button at ({x}, {y})")

            # For raise/bet actions, may need to enter amount
            if action_type in ['raise', 'bet'] and action.get('amount'):
                self._enter_bet_amount(action['amount'])

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to execute action: {e}")

        # Slow mode delay
        if self.slow_mode:
            time.sleep(2)

    def _enter_bet_amount(self, amount):
        """
        Enter bet amount via ADB.

        Args:
            amount (int): Bet amount to enter
        """
        # This is app-specific and would need customization
        # Common approach: tap on bet input field, then use ADB to input text

        # Example: tap bet amount field
        bet_input_position = (540, 1200)  # This needs calibration
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'tap', str(bet_input_position[0]), str(bet_input_position[1])])

        subprocess.run(cmd, capture_output=True)
        time.sleep(0.5)

        # Clear existing amount and enter new amount
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'text', str(amount)])

        subprocess.run(cmd, capture_output=True)
        print(f"  Entered bet amount: {amount}")

    def run_single_hand(self, dry_run=False):
        """
        Run automation for a single poker hand.

        Args:
            dry_run (bool): If True, don't execute actions
        """
        print("\n" + "="*60)
        print("PROCESSING HAND")
        print("="*60)

        # Capture and analyze
        analysis = self.capture_and_analyze()
        if not analysis:
            print("✗ Failed to capture and analyze screen")
            return

        # Make decision
        decision = self.make_decision()
        if not decision:
            print("⚠ Could not make decision")
            return

        # Execute action
        self.execute_action(decision, dry_run=dry_run)

        # Log result
        self._log_hand(analysis, decision)

    def run_continuous(self, dry_run=False, max_hands=None):
        """
        Run automation continuously.

        Args:
            dry_run (bool): If True, don't execute actions
            max_hands (int): Maximum number of hands to play
        """
        print("\n" + "="*60)
        print("STARTING CONTINUOUS AUTOMATION")
        print("="*60)
        print(f"Debug: {self.debug}")
        print(f"Slow Mode: {self.slow_mode}")
        print(f"Dry Run: {dry_run}")
        if max_hands:
            print(f"Max Hands: {max_hands}")
        print("\nPress Ctrl+C to stop")
        print("="*60)

        hand_count = 0

        try:
            while True:
                if max_hands and hand_count >= max_hands:
                    break

                # Wait between hands
                time.sleep(5 if self.slow_mode else 2)

                # Process hand
                self.run_single_hand(dry_run=dry_run)

                hand_count += 1
                print(f"\n✓ Completed hand {hand_count}")

        except KeyboardInterrupt:
            print("\n\n✓ Automation stopped by user")
            print(f"Total hands processed: {hand_count}")

    def _log_hand(self, analysis, decision):
        """
        Log hand details for review.

        Args:
            analysis (dict): Hand analysis
            decision (dict): Decision made
        """
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"hand_{timestamp}.json"

        log_data = {
            'timestamp': timestamp,
            'game_state': analysis['game_state'],
            'recognized_cards': [
                {k: v for k, v in card.items() if k != 'bbox'}
                for card in analysis['recognized_cards']
            ],
            'decision': decision
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        if self.debug:
            print(f"  Log saved: {log_file}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Full poker automation system (EDUCATIONAL USE ONLY)"
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Android device ID or IP:PORT"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without executing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--slow-mode",
        action="store_true",
        help="Add delays between actions"
    )
    parser.add_argument(
        "--single-hand",
        action="store_true",
        help="Process only one hand then exit"
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        help="Maximum number of hands to play"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="button_config.json",
        help="Path to button position config file (default: button_config.json)"
    )

    args = parser.parse_args()

    # Display warning
    print("\n" + "!"*60)
    print("WARNING: EDUCATIONAL USE ONLY")
    print("!"*60)
    print("This tool is for learning and research purposes.")
    print("Using bots on real poker platforms may:")
    print("  • Violate terms of service")
    print("  • Result in account bans")
    print("  • Be illegal in some jurisdictions")
    print("\nOnly use in controlled, private environments.")
    print("!"*60)

    if not args.dry_run:
        response = input("\nContinue with live automation? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting. Use --dry-run to test safely.")
            return

    # Initialize system
    automation = PokerAutomationSystem(
        device_id=args.device,
        debug=args.debug,
        slow_mode=args.slow_mode,
        config_file=args.config
    )

    # Run automation
    if args.single_hand:
        automation.run_single_hand(dry_run=args.dry_run)
    else:
        automation.run_continuous(
            dry_run=args.dry_run,
            max_hands=args.max_hands
        )


if __name__ == "__main__":
    main()
