# Poker Vision Automation - Examples

This document provides practical examples and workflows for using the poker vision automation system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Workflows](#basic-workflows)
3. [Advanced Usage](#advanced-usage)
4. [Training Custom Models](#training-custom-models)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install ADB (if not already installed)
# Ubuntu/Debian:
sudo apt install adb

# macOS:
brew install android-platform-tools

# Verify ADB is working
adb version
```

### 2. Connect to Android Emulator

```bash
# Start your Android emulator (BlueStacks, NoxPlayer, etc.)

# List connected devices
adb devices

# Connect to emulator over network (if needed)
adb connect 127.0.0.1:5555

# Verify connection
adb shell echo "Connected!"
```

## Basic Workflows

### Workflow 1: Capture and Analyze Screenshots

**Step 1: Capture a single screenshot**

```bash
python 01_screen_capture.py --output ./screenshots/
```

**Step 2: Detect cards in the screenshot**

```bash
# Using template matching (no training required)
python 02_card_detection.py \
  --input ./screenshots/screenshot_*.png \
  --method template \
  --visualize

# Using YOLO (requires trained model)
python 02_card_detection.py \
  --input ./screenshots/screenshot_*.png \
  --method yolo \
  --model ./models/card_detector.pt \
  --visualize
```

**Step 3: Recognize card values**

```bash
python 03_ocr_card_recognition.py \
  --image ./screenshots/screenshot_*.png \
  --detections ./detected/screenshot_*_detections.json \
  --visualize
```

**Output**: Recognized cards with ranks and suits

### Workflow 2: Test Poker Logic

**Example 1: Evaluate hand strength**

```bash
# Strong hand
python 04_poker_logic.py \
  --hand "A♠ K♠" \
  --board "Q♠ J♠ 10♠"

# Output: Royal Flush, Rank: 10/10
```

**Example 2: Get action recommendation**

```bash
# Pre-flop decision
python 04_poker_logic.py \
  --hand "7♥ 2♦" \
  --pot 100 \
  --bet 50 \
  --stack 1000 \
  --position early \
  --strategy conservative

# Output: Fold (weak hand, too expensive)
```

**Example 3: Position-based strategy**

```bash
# Same hand, different positions
python 04_poker_logic.py \
  --hand "9♠ 9♥" \
  --pot 50 \
  --bet 10 \
  --position button \
  --strategy aggressive

# Output: Raise (pocket pair in late position)
```

### Workflow 3: Dry Run Automation

**Test full automation without executing actions**

```bash
python 05_full_automation.py \
  --dry-run \
  --debug \
  --single-hand

# This will:
# 1. Capture screenshot
# 2. Detect and recognize cards
# 3. Make decision
# 4. Show what action would be taken (without executing)
```

## Advanced Usage

### Continuous Capture and Analysis

**Capture screenshots at regular intervals**

```bash
# Capture every 2 seconds for 60 seconds
python 01_screen_capture.py \
  --continuous \
  --interval 2 \
  --duration 60 \
  --output ./game_session/

# Or capture a specific number
python 01_screen_capture.py \
  --continuous \
  --interval 1 \
  --max-captures 50 \
  --output ./game_session/
```

### Batch Processing

**Process multiple screenshots at once**

```bash
# Detect cards in all screenshots
python 02_card_detection.py \
  --input ./game_session/ \
  --output ./batch_results/ \
  --method template \
  --visualize

# Recognize all detected cards
for detection in ./batch_results/*_detections.json; do
  image="${detection/_detections.json/.png}"
  python 03_ocr_card_recognition.py \
    --image "$image" \
    --detections "$detection" \
    --visualize
done
```

### Custom Device Configuration

**Connect to specific device**

```bash
# List all devices
adb devices

# Use specific device
python 01_screen_capture.py \
  --device emulator-5554 \
  --output ./captures/

# Connect to network device
python 01_screen_capture.py \
  --connect 192.168.1.100 \
  --device 192.168.1.100:5555
```

## Training Custom Models

### Workflow: Train Card Detector

**Step 1: Create dataset structure**

```bash
python 06_train_detector.py \
  --create-dataset ./my_card_dataset
```

**Step 2: Collect and annotate data**

```bash
# Capture training images
python 01_screen_capture.py \
  --continuous \
  --interval 3 \
  --max-captures 500 \
  --output ./my_card_dataset/raw_images/

# Manually sort images into train/val splits
# Move 80% to my_card_dataset/images/train/
# Move 20% to my_card_dataset/images/val/

# Annotate using LabelImg or Roboflow
# Export annotations in YOLO format to labels/train/ and labels/val/
```

**Step 3: Train the model**

```bash
# Train with default settings
python 06_train_detector.py \
  --data ./my_card_dataset \
  --epochs 100 \
  --batch 16

# Train with GPU and larger batch size
python 06_train_detector.py \
  --data ./my_card_dataset \
  --model yolov8n \
  --epochs 200 \
  --batch 32 \
  --device cuda

# Resume training from checkpoint
python 06_train_detector.py \
  --data ./my_card_dataset \
  --resume ./runs/train/exp/weights/last.pt
```

**Step 4: Evaluate the trained model**

```bash
python 06_train_detector.py \
  --data ./my_card_dataset \
  --evaluate ./runs/train/exp/weights/best.pt
```

**Step 5: Use trained model for detection**

```bash
python 02_card_detection.py \
  --input ./test_images/ \
  --method yolo \
  --model ./runs/train/exp/weights/best.pt \
  --visualize
```

## Troubleshooting

### Issue: ADB not detecting device

```bash
# Kill and restart ADB server
adb kill-server
adb start-server

# Check USB debugging is enabled on device
# For emulators, ensure port is correct:
adb connect 127.0.0.1:5555  # Default for most emulators
adb connect 127.0.0.1:5556  # Alternative port
adb connect 127.0.0.1:62001 # BlueStacks
adb connect 127.0.0.1:62025 # NoxPlayer
```

### Issue: Card detection not working

**Solution 1: Adjust confidence threshold**

```bash
python 02_card_detection.py \
  --input ./test.png \
  --method template \
  --confidence 0.6  # Lower threshold for more detections
```

**Solution 2: Create card templates**

```bash
# Manually crop individual cards from screenshots
# Save them in ./templates/ with names like:
# ace_spades.png, king_hearts.png, etc.

# Then run detection
python 02_card_detection.py \
  --input ./test.png \
  --templates ./templates/ \
  --method template
```

### Issue: OCR misreading cards

**Solution: Try different preprocessing methods**

```bash
# Standard preprocessing
python 03_ocr_card_recognition.py \
  --image ./test.png \
  --preprocess standard

# Enhanced preprocessing (more aggressive)
python 03_ocr_card_recognition.py \
  --image ./test.png \
  --preprocess enhanced

# Adaptive thresholding
python 03_ocr_card_recognition.py \
  --image ./test.png \
  --preprocess adaptive
```

### Issue: Slow performance

**Optimization tips:**

```bash
# Use lower resolution
python 01_screen_capture.py \
  --output ./captures/ \
  # Then resize images before processing

# Use YOLO instead of template matching
python 02_card_detection.py \
  --method yolo \
  --model yolov5s.pt  # Smallest/fastest YOLO model

# Process on GPU
python 06_train_detector.py \
  --device cuda
```

## Integration Examples

### Example: Complete Hand Analysis Pipeline

```bash
#!/bin/bash
# analyze_hand.sh - Complete pipeline for analyzing a poker hand

# Capture screenshot
python 01_screen_capture.py --output ./temp/
SCREENSHOT=$(ls -t ./temp/screenshot_*.png | head -1)

# Detect cards
python 02_card_detection.py \
  --input "$SCREENSHOT" \
  --output ./temp/ \
  --method template \
  --visualize

# Recognize cards
DETECTIONS=$(ls -t ./temp/*_detections.json | head -1)
python 03_ocr_card_recognition.py \
  --image "$SCREENSHOT" \
  --detections "$DETECTIONS" \
  --output ./temp/ \
  --visualize

# Make decision (if cards recognized)
RECOGNIZED=$(ls -t ./temp/*_recognized.json | head -1)
# Parse JSON and call poker logic...

echo "Analysis complete! Check ./temp/ for results."
```

### Example: Monitoring Script

```python
# monitor_game.py - Monitor poker game and log decisions
import time
import subprocess
from pathlib import Path

def monitor_game(duration=3600):
    """Monitor game for specified duration."""
    start = time.time()
    hands_played = 0
    
    while time.time() - start < duration:
        # Capture screenshot
        subprocess.run([
            'python', '01_screen_capture.py',
            '--output', './monitoring/'
        ])
        
        # Detect and recognize cards
        # ... (implement card detection and recognition)
        
        # Log results
        hands_played += 1
        print(f"Hand {hands_played} analyzed")
        
        # Wait before next capture
        time.sleep(5)
    
    print(f"Monitoring complete. {hands_played} hands analyzed.")

if __name__ == '__main__':
    monitor_game(duration=3600)  # Monitor for 1 hour
```

## Safety and Best Practices

### Testing Checklist

- [ ] Test in offline mode first
- [ ] Use --dry-run flag initially
- [ ] Verify card detection accuracy on sample images
- [ ] Test OCR recognition separately
- [ ] Calibrate button positions for your specific app
- [ ] Monitor resource usage (CPU, memory)
- [ ] Implement error handling and logging
- [ ] Test with play money only
- [ ] Respect rate limits

### Production Considerations

1. **Accuracy**: Ensure 95%+ card recognition accuracy before live use
2. **Error Handling**: Implement robust error handling for all components
3. **Logging**: Log all decisions for review and debugging
4. **Monitoring**: Monitor system performance and accuracy
5. **Ethics**: Only use in permitted environments
6. **Legal**: Verify legality in your jurisdiction

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [ADB Documentation](https://developer.android.com/studio/command-line/adb)

## Getting Help

If you encounter issues:

1. Check the main README.md for troubleshooting tips
2. Verify all dependencies are installed correctly
3. Test each component separately
4. Review logs for error messages
5. Try with debug mode enabled: `--debug`

---

**Remember**: This tool is for educational purposes only. Always use responsibly and ethically.
