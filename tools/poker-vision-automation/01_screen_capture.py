"""
Screen Capture from Android VM using ADB

This script demonstrates how to capture screenshots from an Android emulator
or VM using Android Debug Bridge (ADB). The captured images serve as input
for card detection and OCR processing.

Requirements:
    - ADB installed and in PATH
    - Android emulator or VM running
    - Device connected via USB or network

Usage:
    python 01_screen_capture.py
    python 01_screen_capture.py --device 192.168.1.100:5555 --output ./captures/
"""

import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse


class AndroidScreenCapture:
    """Handles screen capture from Android devices via ADB."""

    def __init__(self, device_id=None, output_dir="./output"):
        """
        Initialize the screen capture system.

        Args:
            device_id (str): Specific device ID or IP:PORT for network connection
            output_dir (str): Directory to save captured screenshots
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify ADB is available
        self._check_adb()

    def _check_adb(self):
        """Check if ADB is installed and accessible."""
        try:
            result = subprocess.run(
                ["adb", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ ADB found: {result.stdout.split()[4]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ADB not found. Please install Android SDK Platform Tools:\n"
                "https://developer.android.com/studio/releases/platform-tools"
            )

    def get_devices(self):
        """
        Get list of connected Android devices.

        Returns:
            list: List of device IDs
        """
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse device list (skip header and empty lines)
        devices = []
        for line in result.stdout.split("\n")[1:]:
            if line.strip() and "device" in line:
                device_id = line.split()[0]
                devices.append(device_id)

        return devices

    def connect_device(self, ip_address, port=5555):
        """
        Connect to an Android device over network.

        Args:
            ip_address (str): IP address of the device/emulator
            port (int): ADB port (default: 5555)

        Returns:
            bool: True if connection successful
        """
        target = f"{ip_address}:{port}"
        try:
            result = subprocess.run(
                ["adb", "connect", target],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Connected to {target}")
            return "connected" in result.stdout.lower()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to connect to {target}: {e}")
            return False

    def capture_screenshot(self, filename=None):
        """
        Capture a single screenshot from the Android device.

        Args:
            filename (str): Optional filename for the screenshot

        Returns:
            Path: Path to the saved screenshot
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.png"

        # Build ADB command
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])

        # Capture screenshot to device storage
        device_path = "/sdcard/screenshot_temp.png"
        cmd.extend(["shell", "screencap", "-p", device_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True)

            # Pull screenshot from device to local machine
            local_path = self.output_dir / filename
            pull_cmd = ["adb"]
            if self.device_id:
                pull_cmd.extend(["-s", self.device_id])
            pull_cmd.extend(["pull", device_path, str(local_path)])

            subprocess.run(pull_cmd, check=True, capture_output=True)

            # Clean up device storage
            cleanup_cmd = ["adb"]
            if self.device_id:
                cleanup_cmd.extend(["-s", self.device_id])
            cleanup_cmd.extend(["shell", "rm", device_path])
            subprocess.run(cleanup_cmd, capture_output=True)

            print(f"✓ Screenshot saved: {local_path}")
            return local_path

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to capture screenshot: {e}")
            return None

    def capture_continuous(self, interval=1.0, duration=None, max_captures=None):
        """
        Capture screenshots continuously at specified intervals.

        Args:
            interval (float): Time between captures in seconds
            duration (float): Total duration to capture in seconds (None for infinite)
            max_captures (int): Maximum number of screenshots (None for unlimited)

        Returns:
            list: List of paths to captured screenshots
        """
        print(f"Starting continuous capture (interval: {interval}s)")
        if duration:
            print(f"Duration: {duration}s")
        if max_captures:
            print(f"Max captures: {max_captures}")

        screenshots = []
        start_time = time.time()
        capture_count = 0

        try:
            while True:
                # Check stopping conditions
                if duration and (time.time() - start_time) >= duration:
                    break
                if max_captures and capture_count >= max_captures:
                    break

                # Capture screenshot
                screenshot = self.capture_screenshot()
                if screenshot:
                    screenshots.append(screenshot)
                    capture_count += 1

                # Wait for next capture
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n✓ Capture stopped by user")

        print(f"✓ Captured {len(screenshots)} screenshots")
        return screenshots

    def get_screen_resolution(self):
        """
        Get the screen resolution of the Android device.

        Returns:
            tuple: (width, height) in pixels
        """
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(["shell", "wm", "size"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse output like "Physical size: 1080x1920"
            size_str = result.stdout.strip().split(":")[-1].strip()
            width, height = map(int, size_str.split("x"))
            return width, height
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"✗ Failed to get screen resolution: {e}")
            return None


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Capture screenshots from Android VM using ADB"
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device ID or IP:PORT (e.g., 192.168.1.100:5555)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Capture interval in seconds (for continuous mode)"
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Enable continuous capture mode"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration for continuous capture in seconds"
    )
    parser.add_argument(
        "--max-captures",
        "-n",
        type=int,
        help="Maximum number of screenshots to capture"
    )
    parser.add_argument(
        "--connect",
        help="Connect to device at IP address (e.g., 192.168.1.100)"
    )

    args = parser.parse_args()

    # Initialize screen capture
    capture = AndroidScreenCapture(device_id=args.device, output_dir=args.output)

    # Connect to device if specified
    if args.connect:
        if not capture.connect_device(args.connect):
            return

    # List connected devices
    devices = capture.get_devices()
    if not devices:
        print("✗ No devices connected. Please connect an Android device or emulator.")
        print("\nTo connect to an emulator over network:")
        print("  adb connect 127.0.0.1:5555")
        return

    print(f"✓ Found {len(devices)} device(s): {', '.join(devices)}")

    # Get screen resolution
    resolution = capture.get_screen_resolution()
    if resolution:
        print(f"✓ Screen resolution: {resolution[0]}x{resolution[1]}")

    # Capture mode
    if args.continuous:
        capture.capture_continuous(
            interval=args.interval,
            duration=args.duration,
            max_captures=args.max_captures
        )
    else:
        # Single capture
        capture.capture_screenshot()


if __name__ == "__main__":
    main()
