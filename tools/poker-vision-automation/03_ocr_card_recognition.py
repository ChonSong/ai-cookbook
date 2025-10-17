"""
OCR Card Recognition Using Tesseract

This script extracts card rank and suit information from detected card regions
using Optical Character Recognition (OCR) with Tesseract.

Requirements:
    - pytesseract
    - Tesseract OCR installed on system
    - OpenCV for image preprocessing

Usage:
    # Recognize cards from image
    python 03_ocr_card_recognition.py --image screenshot.png

    # Process detected card regions
    python 03_ocr_card_recognition.py --image screenshot.png --detections detections.json

    # Process with specific preprocessing
    python 03_ocr_card_recognition.py --image screenshot.png --preprocess enhanced
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import argparse
import json
import re


class CardOCRRecognizer:
    """Extract card rank and suit using OCR."""

    # Card rank mapping
    RANK_PATTERNS = {
        'A': ['A', 'Ace'],
        '2': ['2', 'Two'],
        '3': ['3', 'Three'],
        '4': ['4', 'Four'],
        '5': ['5', 'Five'],
        '6': ['6', 'Six'],
        '7': ['7', 'Seven'],
        '8': ['8', 'Eight'],
        '9': ['9', 'Nine'],
        '10': ['10', 'Ten'],
        'J': ['J', 'Jack'],
        'Q': ['Q', 'Queen'],
        'K': ['K', 'King']
    }

    # Suit symbols
    SUIT_PATTERNS = {
        '♠': ['spades', 'spade', 's', '♠'],
        '♥': ['hearts', 'heart', 'h', '♥'],
        '♦': ['diamonds', 'diamond', 'd', '♦'],
        '♣': ['clubs', 'club', 'c', '♣']
    }

    def __init__(self, tesseract_config=None):
        """
        Initialize OCR recognizer.

        Args:
            tesseract_config (str): Custom Tesseract configuration
        """
        # Default Tesseract config optimized for cards
        if tesseract_config is None:
            tesseract_config = '--psm 6 --oem 3'
        
        self.tesseract_config = tesseract_config

        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            print("✓ Tesseract OCR found")
        except Exception as e:
            print(f"✗ Tesseract not found: {e}")
            print("Install from: https://github.com/tesseract-ocr/tesseract")

    def preprocess_card_image(self, image, method='standard'):
        """
        Preprocess card image for better OCR results.

        Args:
            image (np.ndarray): Input card image
            method (str): Preprocessing method ('standard', 'enhanced', 'adaptive')

        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == 'standard':
            # Simple thresholding
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'enhanced':
            # Advanced preprocessing
            # 1. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 2. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast = clahe.apply(denoised)
            
            # 3. Threshold
            _, processed = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'adaptive':
            # Adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

        else:
            processed = gray

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        return processed

    def extract_card_corner(self, image, corner='top-left', corner_size=0.3):
        """
        Extract corner region where rank and suit are typically located.

        Args:
            image (np.ndarray): Full card image
            corner (str): Which corner to extract
            corner_size (float): Fraction of card dimensions for corner

        Returns:
            np.ndarray: Corner region image
        """
        h, w = image.shape[:2]
        corner_h = int(h * corner_size)
        corner_w = int(w * corner_size)

        if corner == 'top-left':
            return image[0:corner_h, 0:corner_w]
        elif corner == 'top-right':
            return image[0:corner_h, w-corner_w:w]
        elif corner == 'bottom-left':
            return image[h-corner_h:h, 0:corner_w]
        elif corner == 'bottom-right':
            return image[h-corner_h:h, w-corner_w:w]
        else:
            return image

    def recognize_rank_suit(self, image, preprocess_method='enhanced'):
        """
        Recognize rank and suit from card image.

        Args:
            image (np.ndarray): Card image (full card or corner region)
            preprocess_method (str): Preprocessing method to use

        Returns:
            dict: Recognition result with rank, suit, and confidence
        """
        # Extract corner if full card provided
        if image.shape[0] > 100:  # Assume full card
            corner = self.extract_card_corner(image)
        else:
            corner = image

        # Preprocess
        processed = self.preprocess_card_image(corner, method=preprocess_method)

        # Run OCR
        text = pytesseract.image_to_string(
            processed,
            config=self.tesseract_config
        ).strip()

        # Get confidence data
        try:
            data = pytesseract.image_to_data(
                processed,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 0

        # Parse rank and suit
        rank = self._parse_rank(text)
        suit = self._parse_suit(text)

        return {
            'rank': rank,
            'suit': suit,
            'raw_text': text,
            'confidence': avg_confidence
        }

    def _parse_rank(self, text):
        """
        Parse card rank from OCR text.

        Args:
            text (str): OCR output text

        Returns:
            str: Recognized rank or None
        """
        text_upper = text.upper()

        # Check for exact matches
        for rank, patterns in self.RANK_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in text_upper:
                    return rank

        # Try to find any number
        numbers = re.findall(r'\d+', text)
        if numbers:
            num = int(numbers[0])
            if 2 <= num <= 10:
                return str(num)

        # Try to find face cards
        if 'A' in text_upper:
            return 'A'
        if 'J' in text_upper:
            return 'J'
        if 'Q' in text_upper:
            return 'Q'
        if 'K' in text_upper:
            return 'K'

        return None

    def _parse_suit(self, text):
        """
        Parse card suit from OCR text or symbol detection.

        Args:
            text (str): OCR output text

        Returns:
            str: Recognized suit symbol or None
        """
        text_lower = text.lower()

        # Check for suit names or symbols
        for suit, patterns in self.SUIT_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    return suit

        return None

    def recognize_from_detections(self, image, detections, preprocess_method='enhanced'):
        """
        Recognize ranks and suits for all detected cards.

        Args:
            image (np.ndarray): Full image
            detections (list): List of card detections with bounding boxes
            preprocess_method (str): Preprocessing method

        Returns:
            list: List of recognized cards with rank, suit, and bounding box
        """
        recognized_cards = []

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Extract card region
            card_region = image[y1:y2, x1:x2]

            # Recognize rank and suit
            result = self.recognize_rank_suit(card_region, preprocess_method)

            # Combine with detection info
            card_info = {
                'bbox': bbox,
                'rank': result['rank'],
                'suit': result['suit'],
                'card': self._format_card(result['rank'], result['suit']),
                'ocr_confidence': result['confidence'],
                'detection_confidence': detection.get('confidence', 0),
                'raw_text': result['raw_text']
            }

            recognized_cards.append(card_info)

        return recognized_cards

    def _format_card(self, rank, suit):
        """
        Format card as readable string.

        Args:
            rank (str): Card rank
            suit (str): Card suit symbol

        Returns:
            str: Formatted card string (e.g., "A♠", "10♥")
        """
        if rank and suit:
            return f"{rank}{suit}"
        elif rank:
            return rank
        elif suit:
            return suit
        else:
            return "Unknown"


def visualize_recognition(image, recognized_cards, output_path=None):
    """
    Visualize recognized cards on image.

    Args:
        image (np.ndarray): Input image
        recognized_cards (list): List of recognized cards
        output_path (str): Optional path to save visualization

    Returns:
        np.ndarray: Image with annotations
    """
    result_image = image.copy()

    for card in recognized_cards:
        x1, y1, x2, y2 = card['bbox']
        card_text = card['card']
        confidence = card['ocr_confidence']

        # Draw bounding box
        color = (0, 255, 0) if card['rank'] and card['suit'] else (0, 165, 255)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Draw card label
        label = f"{card_text} ({confidence:.1f}%)"
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
        description="Recognize poker card ranks and suits using OCR"
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Input image file"
    )
    parser.add_argument(
        "--detections",
        "-d",
        help="JSON file with card detections (from 02_card_detection.py)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--preprocess",
        choices=['standard', 'enhanced', 'adaptive'],
        default='enhanced',
        help="Image preprocessing method"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with recognized cards"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"✗ Failed to load image: {args.image}")
        return

    # Initialize recognizer
    recognizer = CardOCRRecognizer()

    # Process based on whether detections are provided
    if args.detections:
        # Load detections
        with open(args.detections, 'r') as f:
            detections = json.load(f)

        print(f"Processing {len(detections)} detected cards...")
        recognized_cards = recognizer.recognize_from_detections(
            image,
            detections,
            preprocess_method=args.preprocess
        )

    else:
        # Assume full image is a single card
        print("Processing image as single card...")
        result = recognizer.recognize_rank_suit(image, args.preprocess)
        recognized_cards = [{
            'bbox': [0, 0, image.shape[1], image.shape[0]],
            'rank': result['rank'],
            'suit': result['suit'],
            'card': recognizer._format_card(result['rank'], result['suit']),
            'ocr_confidence': result['confidence'],
            'raw_text': result['raw_text']
        }]

    # Display results
    print(f"\n✓ Recognized {len(recognized_cards)} cards:")
    for i, card in enumerate(recognized_cards, 1):
        print(f"  {i}. {card['card']} (Confidence: {card['ocr_confidence']:.1f}%)")

    # Save results
    input_path = Path(args.image)
    result_file = output_dir / f"{input_path.stem}_recognized.json"
    with open(result_file, "w") as f:
        json.dump(recognized_cards, f, indent=2)
    print(f"\n✓ Results saved: {result_file}")

    # Visualize if requested
    if args.visualize:
        vis_file = output_dir / f"{input_path.stem}_recognized.png"
        visualize_recognition(image, recognized_cards, str(vis_file))


if __name__ == "__main__":
    main()
