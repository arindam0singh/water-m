"""
Simulate screen-recording style distortions against a video. Ill do anything but DSA!!!!!!!!!!
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from utils import fourcc_for_path


def jpeg_roundtrip(frame: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to a frame in memory."""
    quality = int(np.clip(quality, 1, 100))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return decoded


def distort_frame(
    frame: np.ndarray,
    blur_kernel: int = 3,
    jpeg_quality: int = 65,
    resize_scale: float = 0.85,
    brightness: float = 1.05,
    noise_std: float = 2.0,
) -> np.ndarray:
    
    attacked = frame.copy()
    height, width = attacked.shape[:2]

    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        attacked = cv2.GaussianBlur(attacked, (blur_kernel, blur_kernel), 0)

    attacked = jpeg_roundtrip(attacked, jpeg_quality)

    if resize_scale > 0 and resize_scale != 1.0:
        small_size = (
            max(8, int(width * resize_scale)),
            max(8, int(height * resize_scale)),
        )
        attacked = cv2.resize(attacked, small_size, interpolation=cv2.INTER_AREA)
        attacked = cv2.resize(attacked, (width, height), interpolation=cv2.INTER_LINEAR)

    attacked = np.clip(attacked.astype(np.float32) * brightness, 0, 255)

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, attacked.shape).astype(np.float32)
        attacked = np.clip(attacked + noise, 0, 255)

    return attacked.astype(np.uint8)


def attack_video(
    input_path: str,
    output_path: str,
    blur_kernel: int = 3,
    jpeg_quality: int = 65,
    resize_scale: float = 0.85,
    brightness: float = 1.05,
    noise_std: float = 2.0,
    max_frames: int | None = None,
) -> None:
    """Create an attacked copy of a video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Input video contains no readable frames")

    is_color = first_frame.ndim == 3
    writer = cv2.VideoWriter(
        output_path,
        fourcc_for_path(output_path),
        fps,
        (width, height),
        isColor=is_color,
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not open output video writer: {output_path}")

    frame_index = 0
    frame = first_frame
    while True:
        attacked = distort_frame(
            frame,
            blur_kernel=blur_kernel,
            jpeg_quality=jpeg_quality,
            resize_scale=resize_scale,
            brightness=brightness,
            noise_std=noise_std,
        )
        writer.write(attacked)

        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    writer.release()
    print(f"Attacked {frame_index} frame(s).")
    print(f"Output written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate screen-recording attacks.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output attacked video path")
    parser.add_argument("--blur-kernel", type=int, default=3, help="Gaussian kernel size")
    parser.add_argument("--jpeg-quality", type=int, default=65, help="JPEG quality 1-100")
    parser.add_argument("--resize-scale", type=float, default=0.85, help="Downscale factor")
    parser.add_argument("--brightness", type=float, default=1.05, help="Brightness multiplier")
    parser.add_argument("--noise-std", type=float, default=2.0, help="Gaussian noise std dev")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attack_video(
        input_path=args.input,
        output_path=args.output,
        blur_kernel=args.blur_kernel,
        jpeg_quality=args.jpeg_quality,
        resize_scale=args.resize_scale,
        brightness=args.brightness,
        noise_std=args.noise_std,
        max_frames=args.max_frames,
    )
