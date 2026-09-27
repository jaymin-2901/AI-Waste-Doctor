"""
Image Preprocessing module for AI Waste Doctor.
Handles cropping, BGR-to-RGB conversion, resizing, float normalization, and batch expansion.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def preprocess_frame(frame, target_size=(224, 224), normalization_mode="-1_to_1", crop_box=None):
    """
    Preprocess an OpenCV BGR video frame for TensorFlow model inference.
    
    Parameters:
    -----------
    frame : np.ndarray
        Full BGR image frame from OpenCV.
    target_size : tuple (width, height)
        Target dimensions expected by the Keras model (default: (224, 224)).
    normalization_mode : str
        "0-1" -> pixel / 255.0
        "-1_to_1" -> (pixel / 127.5) - 1.0 (Teachable Machine standard)
    crop_box : tuple (x, y, w, h) or None
        Coordinates of scan zone to crop. If None, uses the full frame.

    Returns:
    --------
    np.ndarray : Batch-expanded float32 array of shape (1, height, width, 3)
    np.ndarray : Cropped RGB image (height, width, 3) for preview if needed
    """
    if frame is None:
        raise ValueError("Input frame is None")

    h, w = frame.shape[:2]

    # 1. Crop scan zone if provided
    if crop_box is not None:
        x, y, box_w, box_h = crop_box
        # Ensure box coordinates are within frame bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        box_w = max(1, min(box_w, w - x))
        box_h = max(1, min(box_h, h - y))
        crop = frame[y:y + box_h, x:x + box_w]
    else:
        crop = frame

    # 2. Convert BGR to RGB
    if cv2 is not None:
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        # Fallback if cv2 is not imported (assuming RGB or simple array slicing)
        rgb_img = crop[..., ::-1]

    # 3. Resize to target size (width, height)
    target_w, target_h = target_size
    if cv2 is not None:
        resized_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Simple placeholder resize if cv2 unavailable
        resized_img = rgb_img

    # 4. Convert to float32
    float_img = resized_img.astype(np.float32)

    # 5. Normalize pixel values
    if normalization_mode == "-1_to_1":
        # Teachable Machine format: scale [0, 255] to [-1, 1]
        normalized_img = (float_img / 127.5) - 1.0
    else:
        # Standard scale [0, 255] to [0, 1]
        normalized_img = float_img / 255.0

    # 6. Add batch dimension (1, H, W, C)
    batch_img = np.expand_dims(normalized_img, axis=0)

    return batch_img, rgb_img
