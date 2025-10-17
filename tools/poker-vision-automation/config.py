"""
Configuration Management System

Centralized configuration for poker automation system using dataclasses and YAML.
Provides type-safe configuration with sensible defaults.

Requirements:
    - pyyaml (for YAML config files)

Usage:
    from config import AutomationConfig
    
    # Load from YAML
    config = AutomationConfig.from_yaml('config.yaml')
    
    # Use defaults
    config = AutomationConfig()
    
    # Save to YAML
    config.to_yaml('config.yaml')
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠ PyYAML not available. Install with: pip install pyyaml")


@dataclass
class DetectionConfig:
    """Card detection configuration."""
    method: str = 'template'  # 'yolo' or 'template'
    confidence_threshold: float = 0.5
    model_path: str = 'yolov5s.pt'
    template_dir: str = './templates'
    use_cache: bool = True


@dataclass
class OCRConfig:
    """OCR configuration."""
    preprocess_method: str = 'enhanced'  # 'basic', 'enhanced', 'adaptive'
    tesseract_config: str = '--psm 6 --oem 3'
    corner_size: float = 0.3
    min_confidence: float = 50.0


@dataclass
class StrategyConfig:
    """Poker strategy configuration."""
    style: str = 'balanced'  # 'aggressive', 'balanced', 'conservative'
    position_aware: bool = True
    min_hand_strength: float = 0.4
    bluff_frequency: float = 0.1
    pot_odds_threshold: float = 2.0


@dataclass
class RegionConfig:
    """Screen region configuration."""
    config_file: Optional[str] = None
    auto_calibrate: bool = False
    screen_resolution: tuple = (1920, 1080)


@dataclass
class ButtonConfig:
    """Button position configuration."""
    config_file: str = 'button_config.json'
    calibration_required: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = './logs'
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class AutomationConfig:
    """Main automation configuration."""
    device_id: Optional[str] = None
    debug: bool = False
    slow_mode: bool = False
    capture_interval: float = 1.0
    
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    buttons: ButtonConfig = field(default_factory=ButtonConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'AutomationConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path (str): Path to YAML config file
            
        Returns:
            AutomationConfig: Loaded configuration
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        config_dict = {}
        
        # Top-level configs
        for key in ['device_id', 'debug', 'slow_mode', 'capture_interval']:
            if key in data:
                config_dict[key] = data[key]
        
        # Nested configs
        if 'detection' in data:
            config_dict['detection'] = DetectionConfig(**data['detection'])
        if 'ocr' in data:
            config_dict['ocr'] = OCRConfig(**data['ocr'])
        if 'strategy' in data:
            config_dict['strategy'] = StrategyConfig(**data['strategy'])
        if 'regions' in data:
            # Handle tuple conversion for screen_resolution
            if 'screen_resolution' in data['regions'] and isinstance(data['regions']['screen_resolution'], list):
                data['regions']['screen_resolution'] = tuple(data['regions']['screen_resolution'])
            config_dict['regions'] = RegionConfig(**data['regions'])
        if 'buttons' in data:
            config_dict['buttons'] = ButtonConfig(**data['buttons'])
        if 'logging' in data:
            config_dict['logging'] = LoggingConfig(**data['logging'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'AutomationConfig':
        """
        Load configuration from JSON file.
        
        Args:
            path (str): Path to JSON config file
            
        Returns:
            AutomationConfig: Loaded configuration
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path) as f:
            data = json.load(f)
        
        # Similar parsing as from_yaml
        config_dict = {}
        
        for key in ['device_id', 'debug', 'slow_mode', 'capture_interval']:
            if key in data:
                config_dict[key] = data[key]
        
        if 'detection' in data:
            config_dict['detection'] = DetectionConfig(**data['detection'])
        if 'ocr' in data:
            config_dict['ocr'] = OCRConfig(**data['ocr'])
        if 'strategy' in data:
            config_dict['strategy'] = StrategyConfig(**data['strategy'])
        if 'regions' in data:
            if 'screen_resolution' in data['regions'] and isinstance(data['regions']['screen_resolution'], list):
                data['regions']['screen_resolution'] = tuple(data['regions']['screen_resolution'])
            config_dict['regions'] = RegionConfig(**data['regions'])
        if 'buttons' in data:
            config_dict['buttons'] = ButtonConfig(**data['buttons'])
        if 'logging' in data:
            config_dict['logging'] = LoggingConfig(**data['logging'])
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path (str): Output path for YAML file
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to save YAML configs. Install with: pip install pyyaml")
        
        # Convert to dict
        config_dict = self.to_dict()
        
        # Write to file
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Configuration saved to {path}")
    
    def to_json(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path (str): Output path for JSON file
        """
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {path}")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return asdict(self)
    
    def print_config(self):
        """Print current configuration in a readable format."""
        print("\n=== Automation Configuration ===")
        print(f"\nDevice ID: {self.device_id or 'Auto-detect'}")
        print(f"Debug Mode: {self.debug}")
        print(f"Slow Mode: {self.slow_mode}")
        print(f"Capture Interval: {self.capture_interval}s")
        
        print("\n--- Detection Settings ---")
        print(f"Method: {self.detection.method}")
        print(f"Confidence: {self.detection.confidence_threshold}")
        if self.detection.method == 'yolo':
            print(f"Model: {self.detection.model_path}")
        else:
            print(f"Template Dir: {self.detection.template_dir}")
        
        print("\n--- OCR Settings ---")
        print(f"Preprocess: {self.ocr.preprocess_method}")
        print(f"Min Confidence: {self.ocr.min_confidence}%")
        
        print("\n--- Strategy Settings ---")
        print(f"Style: {self.strategy.style}")
        print(f"Position Aware: {self.strategy.position_aware}")
        print(f"Min Hand Strength: {self.strategy.min_hand_strength}")
        
        print("\n--- Region Settings ---")
        print(f"Config File: {self.regions.config_file or 'Default'}")
        print(f"Resolution: {self.regions.screen_resolution[0]}x{self.regions.screen_resolution[1]}")
        
        print("\n--- Button Settings ---")
        print(f"Config File: {self.buttons.config_file}")
        
        print("\n--- Logging Settings ---")
        print(f"Log Directory: {self.logging.log_dir}")
        print(f"Log Level: {self.logging.log_level}")
        print()


def create_example_config(output_path='example_config.yaml'):
    """
    Create an example configuration file with documented options.
    
    Args:
        output_path (str): Path for example config file
    """
    example_yaml = """# Poker Automation System Configuration
# This is an example configuration with all available options

# Device settings
device_id: null  # Auto-detect, or specify like "192.168.1.100:5555"
debug: true
slow_mode: true
capture_interval: 2.0  # Seconds between captures

# Card detection settings
detection:
  method: template  # 'yolo' or 'template'
  confidence_threshold: 0.5
  model_path: yolov5s.pt  # For YOLO method
  template_dir: ./templates  # For template method
  use_cache: true

# OCR settings
ocr:
  preprocess_method: enhanced  # 'basic', 'enhanced', 'adaptive'
  tesseract_config: '--psm 6 --oem 3'
  corner_size: 0.3
  min_confidence: 50.0

# Poker strategy settings
strategy:
  style: balanced  # 'aggressive', 'balanced', 'conservative'
  position_aware: true
  min_hand_strength: 0.4
  bluff_frequency: 0.1
  pot_odds_threshold: 2.0

# Screen region settings
regions:
  config_file: null  # Path to custom region config, or null for defaults
  auto_calibrate: false
  screen_resolution: [1920, 1080]  # [width, height]

# Button position settings
buttons:
  config_file: button_config.json
  calibration_required: true

# Logging settings
logging:
  log_dir: ./logs
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
  max_log_size: 10485760  # 10MB in bytes
  backup_count: 5
  console_output: true
"""
    
    with open(output_path, 'w') as f:
        f.write(example_yaml)
    
    print(f"✓ Example configuration created: {output_path}")
    print("  Edit this file and use with: --config {output_path}")


def main():
    """CLI utility for config management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Configuration Management Utility"
    )
    parser.add_argument(
        '--create-example',
        action='store_true',
        help='Create example configuration file'
    )
    parser.add_argument(
        '--validate',
        help='Validate a configuration file'
    )
    parser.add_argument(
        '--print',
        dest='print_config',
        help='Load and print a configuration file'
    )
    parser.add_argument(
        '--convert',
        help='Convert config file to different format'
    )
    parser.add_argument(
        '--output',
        help='Output file for conversion'
    )
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config()
        return 0
    
    if args.validate:
        try:
            if args.validate.endswith('.yaml') or args.validate.endswith('.yml'):
                config = AutomationConfig.from_yaml(args.validate)
            else:
                config = AutomationConfig.from_json(args.validate)
            print(f"✓ Configuration is valid: {args.validate}")
            return 0
        except Exception as e:
            print(f"✗ Configuration is invalid: {e}")
            return 1
    
    if args.print_config:
        try:
            if args.print_config.endswith('.yaml') or args.print_config.endswith('.yml'):
                config = AutomationConfig.from_yaml(args.print_config)
            else:
                config = AutomationConfig.from_json(args.print_config)
            config.print_config()
            return 0
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            return 1
    
    if args.convert:
        if not args.output:
            print("✗ --output is required for conversion")
            return 1
        
        try:
            # Load config
            if args.convert.endswith('.yaml') or args.convert.endswith('.yml'):
                config = AutomationConfig.from_yaml(args.convert)
            else:
                config = AutomationConfig.from_json(args.convert)
            
            # Save in new format
            if args.output.endswith('.yaml') or args.output.endswith('.yml'):
                config.to_yaml(args.output)
            else:
                config.to_json(args.output)
            
            print(f"✓ Converted {args.convert} to {args.output}")
            return 0
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            return 1
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main())
