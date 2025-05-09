# Hand Volume Controller

Control your system volume with hand gestures using your webcam.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python GestureControl.py
```

## Controls

- **Right Hand**: Adjust volume by changing distance between thumb and index finger
- **Left Hand**: Hold palm open for 2 seconds to lock/unlock volume control
- Press 'q' to exit 

## How It Works 

- **Hand Detection**: Uses MediaPipe Hands model to detect and track 21 3D landmarks on each hand at 30fps
- **Gesture Recognition**: 
  - Left palm open detection: Evaluates finger extension by comparing y-coordinates of PIP and DIP joints
  - Volume control: Calculates Euclidean distance between thumb (landmark 4) and index finger (landmark 8)
- **Volume Adjustment**: Maps finger distance range [15px-200px] to volume range [0%-100%] using linear interpolation
- **Lock Mechanism**: Implements a state machine with temporal threshold of 2 seconds for palm gesture
- **System Integration**: Uses macOS `osascript` subprocess calls to control system volume via AppleScript 