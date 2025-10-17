# Poker Vision Automation for Android VM

This guide demonstrates how to automate poker playing in an Android VM using computer vision, OCR, and automated decision-making. It combines object detection, optical character recognition, and Android automation to create an intelligent poker bot.

## ⚠️ Legal and Ethical Considerations

**Important:** This project is for educational and research purposes only. Using bots to play poker on real-money platforms typically violates terms of service and may be illegal. Only use this in:
- Local testing environments
- Private games with consent
- Educational demonstrations
- Research and development

Always respect platform terms of service and applicable laws.

## Overview

The system consists of four main components:

1. **Screen Capture**: Capture screenshots from Android VM using ADB
2. **Card Detection**: Detect poker cards using computer vision (OpenCV/YOLO)
3. **Card Recognition**: Extract card values and suits using OCR (Tesseract)
4. **Game Automation**: Make decisions and simulate touch events via ADB

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Android Emulator/VM                      │
│                      (Poker Application)                     │
└──────────────────┬────────────────────────┬─────────────────┘
                   │                        │
                   │ Screenshot             │ Touch Events
                   ↓                        ↑
┌──────────────────────────────────────────────────────────────┐
│                     Computer Vision System                    │
├──────────────────────────────────────────────────────────────┤
│  1. Screen Capture (ADB)                                     │
│  2. Card Detection (YOLO/OpenCV)                             │
│  3. Card Recognition (Tesseract OCR)                         │
│  4. Decision Logic (Poker AI)                                │
│  5. Action Execution (ADB Touch Events)                      │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.8+
- Android Debug Bridge (ADB)
- Android Emulator or VM (e.g., BlueStacks, NoxPlayer, Android Studio Emulator)

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Install Android SDK Platform Tools (includes ADB):
- **Linux/Mac**: `sudo apt install adb` or download from [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools)
- **Windows**: Download from [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools)

### Tesseract OCR Installation
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Quick Start

### 1. Connect to Android VM

```bash
# List connected devices
adb devices

# Connect to emulator (if not automatically detected)
adb connect 127.0.0.1:5555
```

### 2. Capture Screen

```bash
python screen_capture.py
```

### 3. Detect Cards

```bash
python card_detection.py
```

### 4. Run Full Automation

```bash
python automation.py
```

## Implementation Details

### 1. Screen Capture from Android VM

The `screen_capture.py` script captures screenshots from the Android emulator using ADB:

```python
# Capture screenshot and save to device
adb shell screencap -p /sdcard/screenshot.png

# Pull screenshot to local machine
adb pull /sdcard/screenshot.png ./output/
```

**Key Features:**
- Real-time screen capture
- Configurable capture rate
- Automatic image preprocessing
- Multiple device support

### 2. Card Detection Using Computer Vision

We support two approaches for card detection:

#### Option A: YOLO (You Only Look Once)
- **Pros**: Fast, accurate, real-time detection
- **Cons**: Requires training with labeled card images
- **Use Case**: Production systems with high accuracy needs

#### Option B: OpenCV Template Matching
- **Pros**: No training required, simple implementation
- **Cons**: Less robust to variations in card appearance
- **Use Case**: Quick prototypes and testing

The `card_detection.py` script demonstrates both approaches.

### 3. Card Recognition with OCR

The `ocr_recognition.py` script uses Tesseract OCR to:
- Extract rank (A, 2-10, J, Q, K)
- Identify suit (♠, ♥, ♦, ♣)
- Handle rotated or skewed cards
- Filter false positives

**OCR Optimization:**
- Preprocessing: grayscale conversion, thresholding, noise removal
- Region of interest (ROI) extraction
- Custom Tesseract configuration for card-specific fonts

### 4. Poker Decision Logic

The `poker_logic.py` implements basic poker strategy:
- Hand evaluation (straight, flush, full house, etc.)
- Pot odds calculation
- Position-based decisions
- Basic opponent modeling

**Advanced Features:**
- Integration with poker libraries (e.g., pokerkit, treys)
- Monte Carlo simulations
- Nash equilibrium strategies (for advanced users)

### 5. Android Automation via ADB

The `automation.py` orchestrates the entire system:

```python
# Simulate touch events
adb shell input tap <x> <y>

# Simulate swipe
adb shell input swipe <x1> <y1> <x2> <y2> <duration>

# Simulate text input
adb shell input text "bet_amount"
```

**Button Detection:**
- Fold, Call, Raise buttons located using computer vision
- Adaptive positioning based on screen resolution
- Fallback to predefined coordinates

## Training Your Own Card Detector

### Dataset Preparation

1. **Collect Screenshots**: Capture 500-1000 screenshots from your poker app
2. **Annotate Images**: Use tools like LabelImg or Roboflow
3. **Train YOLO Model**: Use `train_detector.py`

```bash
python train_detector.py --data ./dataset --epochs 100
```

### Annotation Tools
- [LabelImg](https://github.com/heartexlabs/labelImg) - Graphical annotation tool
- [Roboflow](https://roboflow.com/) - Online annotation and dataset management
- [CVAT](https://cvat.org/) - Computer Vision Annotation Tool

## Project Structure

```
poker-vision-automation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── screen_capture.py          # ADB screen capture
├── card_detection.py          # Computer vision card detection
├── ocr_recognition.py    # OCR for card values
├── poker_logic.py             # Decision-making logic
├── automation.py         # Complete automation system
├── train_detector.py          # Training script for custom detector
├── utils/
│   ├── adb_helper.py             # ADB utilities
│   ├── cv_utils.py               # Computer vision helpers
│   ├── ocr_utils.py              # OCR processing
│   └── poker_engine.py           # Poker game logic
├── models/
│   ├── yolov5s.pt                # Pre-trained YOLO model (optional)
│   └── card_detector.pt          # Custom trained model
├── templates/                     # Card templates for OpenCV matching
└── output/                        # Screenshots and results
```

## Example Workflows

### Testing with Static Images

```bash
# Test card detection on sample images
python card_detection.py --input ./test_images/ --output ./results/

# Test OCR extraction
python ocr_recognition.py --image ./test_images/hand.png
```

### Live Automation

```bash
# Run automation with debug mode
python automation.py --debug --slow-mode

# Run with specific strategy
python automation.py --strategy conservative
```

## Useful Resources

### Open-Source Projects
1. **[Poker Vision (MemDbg)](https://github.com/MemDbg/poker-vision)** - Complete poker vision system with OpenCV and OCR
2. **[Poker Hand Recognition](https://github.com/geaxgx/playing-card-detection)** - Playing card detection with deep learning
3. **[Poker Bot](https://github.com/dickreuter/Poker)** - Advanced poker bot with equity calculations

### Documentation and Tutorials
1. **[Roboflow Blog: Poker Vision](https://blog.roboflow.com/poker-vision/)** - Building poker computer vision systems
2. **[HackerNoon: Playing Poker with Computer Vision](https://hackernoon.com/playing-poker-with-computer-vision)** - Personal project insights
3. **[OpenCV Card Detection Tutorial](https://docs.opencv.org/4.x/dc/d16/tutorial_akaze_tracking.html)** - Feature detection and tracking

### Research Papers
1. **[arXiv: Self-Playing Poker Robot](https://arxiv.org/abs/2203.12755)** - Integration of robotics, computer vision, and algorithm design
2. **[YOLOv5 for Card Detection](https://github.com/ultralytics/yolov5)** - State-of-the-art object detection

### Android Automation
1. **[ADB Documentation](https://developer.android.com/studio/command-line/adb)** - Official Android Debug Bridge guide
2. **[Appium](http://appium.io/)** - Alternative to ADB for app automation
3. **[Scrcpy](https://github.com/Genymobile/scrcpy)** - Display and control Android devices

## Troubleshooting

### Common Issues

**ADB Connection Failed**
```bash
# Restart ADB server
adb kill-server
adb start-server

# Check device connection
adb devices
```

**Card Detection Not Working**
- Ensure good lighting and contrast
- Adjust detection confidence threshold
- Use template matching as fallback
- Retrain model with your specific card design

**OCR Misreading Cards**
- Preprocess images (grayscale, threshold, denoise)
- Use higher resolution screenshots
- Crop to card region before OCR
- Fine-tune Tesseract configuration

**Slow Performance**
- Use YOLO instead of template matching
- Reduce screenshot resolution
- Optimize detection regions (don't scan entire screen)
- Use GPU acceleration for YOLO inference

## Performance Optimization

### Speed Improvements
1. **Region of Interest (ROI)**: Only scan relevant screen areas
2. **Frame Skipping**: Don't process every frame
3. **GPU Acceleration**: Use CUDA for YOLO inference
4. **Parallel Processing**: Run detection and OCR concurrently

### Accuracy Improvements
1. **Model Fine-tuning**: Train on your specific poker app
2. **Ensemble Methods**: Combine multiple detection approaches
3. **Temporal Consistency**: Track cards across frames
4. **Confidence Thresholds**: Filter low-confidence detections

## Safety and Best Practices

### Development Guidelines
1. ✅ Test in offline/local environments
2. ✅ Use virtual currency or play money
3. ✅ Document all code for educational purposes
4. ✅ Respect platform terms of service
5. ❌ Do not use on real-money platforms without permission
6. ❌ Do not distribute for commercial use
7. ❌ Do not circumvent anti-bot measures

### Testing Safely
- Use Android emulator with isolated network
- Test with offline poker apps
- Create private test games
- Monitor resource usage
- Implement rate limiting

## Future Enhancements

- [ ] Deep reinforcement learning for optimal play
- [ ] Multi-table tournament support
- [ ] Advanced opponent modeling
- [ ] Integration with poker solvers (PioSOLVER, GTO+)
- [ ] Web-based dashboard for monitoring
- [ ] Support for different poker variants (Omaha, Stud, etc.)

## Contributing

Contributions are welcome! Please ensure all contributions:
1. Include clear documentation
2. Follow ethical guidelines
3. Add tests for new features
4. Update README with examples

## License

This project is for educational purposes only. Use responsibly and ethically.

## Acknowledgments

- YOLOv5 by Ultralytics
- OpenCV community
- Tesseract OCR project
- Android Open Source Project
- Poker community for strategy insights

## Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub
- Review the documentation links above
- Check existing projects and communities

---

**Remember**: This tool is designed for learning and research. Always play responsibly and respect the rules of the platforms you use.
