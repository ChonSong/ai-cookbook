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

### Step 6: Test Full Automation (Dry Run)

```bash
python automation.py \
  --dry-run \
  --debug \
  --single-hand
```

✅ **Expected output**: Complete analysis without executing actions

## Common Issues

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
