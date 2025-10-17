# Quick Start Guide

Get up and running with poker vision automation in 5 minutes!

## Prerequisites

- Python 3.8+
- Android emulator or VM running
- ADB installed

## Installation

### 1. Install Dependencies

```bash
cd tools/poker-vision-automation
pip install -r requirements.txt
```

### 2. Install System Dependencies

**Tesseract OCR:**
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

**ADB (Android Debug Bridge):**
- **Ubuntu/Debian**: `sudo apt install adb`
- **macOS**: `brew install android-platform-tools`
- **Windows**: Download [SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools)

## Quick Test

### Step 1: Verify ADB Connection

```bash
# Check if ADB is installed
adb version

# Connect to your emulator
adb connect 127.0.0.1:5555

# Verify connection
adb devices
```

You should see your device listed.

### Step 2: Capture a Screenshot

```bash
python screen_capture.py
```

✅ **Expected output**: Screenshot saved to `./output/screenshot_*.png`

### Step 3: Test Card Detection

```bash
python card_detection.py \
  --input ./output/screenshot_*.png \
  --method template \
  --visualize
```

✅ **Expected output**: Detection results in `./detected/` directory

### Step 4: Test OCR Recognition

```bash
python ocr_recognition.py \
  --image ./output/screenshot_*.png \
  --visualize
```

✅ **Expected output**: Recognized card information

### Step 5: Test Poker Logic

```bash
python poker_logic.py \
  --hand "A♠ K♠" \
  --board "Q♠ J♠ 10♠"
```

✅ **Expected output**: Hand evaluation (Royal Flush)

### Step 6: Calibrate Button Positions

Before running automation, calibrate button positions for your poker app:

```bash
# 1. Capture a screenshot with all buttons visible
python screen_capture.py

# 2. Run calibration tool
python calibration.py --screenshot output/screenshot_*.png

# 3. Click the center of each button when prompted
# - fold
# - call/check
# - raise/bet
# - all_in (if visible)

# This creates button_config.json
```

✅ **Expected output**: `button_config.json` created with button positions

### Step 7: Test Full Automation (Dry Run)

```bash
python automation.py \
  --dry-run \
  --debug \
  --single-hand
```

✅ **Expected output**: Complete analysis without executing actions

## Common Issues

### ❌ "Button config not found"

**Solution**: Run the calibration tool first

```bash
# Capture screenshot with all buttons visible
python screen_capture.py

# Run calibration
python calibration.py --screenshot output/screenshot_*.png
```

### ❌ "Cards in wrong regions"

**Solution**: Visualize and adjust region definitions

```bash
# Visualize default regions
python card_region_classifier.py --screenshot screenshot.png --output regions.png

# Check the visualization and adjust if needed
# Edit region_config.json if regions don't match your app layout
```

### ❌ "ADB not found"

**Solution**: Install ADB and add to PATH

```bash
# Ubuntu/Debian
sudo apt install adb

# Verify
which adb
```

### ❌ "No devices connected"

**Solution**: Connect to emulator

```bash
# Try different ports
adb connect 127.0.0.1:5555  # Standard
adb connect 127.0.0.1:62001 # BlueStacks
adb connect 127.0.0.1:62025 # NoxPlayer

# Check connection
adb devices
```

### ❌ "Tesseract not found"

**Solution**: Install Tesseract OCR

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Verify
tesseract --version
```

### ❌ "No cards detected"

**Solutions**:

1. **Check screenshot quality**: Ensure poker app is visible and cards are clear
2. **Create templates**: For template matching, you need card templates
3. **Train custom model**: For better accuracy, train a YOLO model with your app's cards

## Next Steps

### Configuration Options

Create a configuration file for persistent settings:

```bash
# Generate example config
python config.py --create-example

# Edit example_config.yaml with your preferences
nano example_config.yaml

# Use with automation
python automation.py --config example_config.yaml --dry-run
```

Example configuration adjustments:
- **Detection method**: Switch between YOLO and template matching
- **Strategy style**: Choose aggressive, balanced, or conservative play
- **Logging level**: Set DEBUG for detailed logs, INFO for normal operation
- **Screen resolution**: Match your emulator's resolution

### For Template Matching

1. Capture screenshots of individual cards from your poker app
2. Crop each card and save in `./templates/` directory
3. Name files like: `ace_spades.png`, `king_hearts.png`, etc.
4. Run detection again

### For YOLO Detection

1. Collect 200-500 screenshots from your poker app
2. Annotate cards using [LabelImg](https://github.com/heartexlabs/labelImg)
3. Train model: `python train_detector.py --data ./dataset --epochs 100`
4. Use trained model: `python card_detection.py --method yolo --model ./runs/train/exp/weights/best.pt`

## Safety Reminder

⚠️ **This tool is for educational purposes only!**

- ✅ Use in offline/private environments
- ✅ Test with play money
- ✅ Respect terms of service
- ❌ Do not use on real-money platforms without permission

## Getting Help

See detailed examples in [EXAMPLES.md](EXAMPLES.md) or full documentation in [README.md](README.md).

## Directory Structure

After running all scripts, you should have:

```
poker-vision-automation/
├── output/              # Screenshots
├── detected/            # Detection results
├── logs/                # Automation logs
├── templates/           # Card templates (create these)
├── models/             # Trained models (optional)
└── runs/               # Training runs (if training)
```

## Minimal Working Example

Here's the absolute minimum to get started:

```bash
# 1. Install
pip install opencv-python pytesseract numpy

# 2. Connect to emulator
adb connect 127.0.0.1:5555

# 3. Capture and view
python screen_capture.py
# Check ./output/ for screenshot

# 4. Test poker logic (no dependencies on actual game)
python poker_logic.py --hand "A♠ A♥" --board "A♦ K♠ Q♠"
```

## Support

For issues, questions, or contributions, please refer to the main repository documentation.

---

**Ready to dive deeper?** Check out [EXAMPLES.md](EXAMPLES.md) for comprehensive workflows!
