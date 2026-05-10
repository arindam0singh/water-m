"""
Embed a deterministic DCT spread-spectrum watermark into a video.

Example:
    python embed.py input.mp4 watermarked.mp4 --public-key "alice-key" --salt "secret"
"""

from __future__ import annotations

import argparse

import cv2

from utils import (
    DEFAULT_BITS,
    DEFAULT_SALT,
    derive_watermark_bits,
    embed_luma_channel,
    fourcc_for_path,
    merge_luma,
    seed_from_key,
    split_luma,
)


def embed_video(
    input_path: str,
    output_path: str,
    public_key: str,
    salt: str = DEFAULT_SALT,
    bit_length: int = DEFAULT_BITS,
    strength: float = 12.0,
    max_frames: int | None = None,
) -> None:
    """Embed the same deterministic watermark across video frames."""
    watermark_bits = derive_watermark_bits(public_key, salt, bit_length)
    seed = seed_from_key(public_key, salt)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first_ok, first_frame = cap.read()
    if not first_ok:
        cap.release()
        raise ValueError("Input video contains no readable frames")

    is_color = first_frame.ndim == 3
    writer_size = (width, height)
    writer = cv2.VideoWriter(
        output_path,
        fourcc_for_path(output_path),
        fps,
        writer_size,
        isColor=is_color,
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not open output video writer: {output_path}")

    frame_index = 0
    frame = first_frame
    while True:
        luma, chroma, was_color = split_luma(frame)
        marked_luma = embed_luma_channel(luma, watermark_bits, seed, strength)
        writer.write(merge_luma(marked_luma, chroma, was_color))

        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    writer.release()
    print(f"Embedded {bit_length} watermark bits into {frame_index} frame(s).")
    print(f"Output written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a DCT video watermark.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output watermarked video path")
    parser.add_argument("--public-key", required=True, help="User public key string")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="Secret salt")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, help="Watermark length")
    parser.add_argument(
        "--strength",
        type=float,
        default=12.0,
        help="Embedding strength; try 8-20 for this PoC",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick experiments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed_video(
        input_path=args.input,
        output_path=args.output,
        public_key=args.public_key,
        salt=args.salt,
        bit_length=args.bits,
        strength=args.strength,
        max_frames=args.max_frames,
    )
