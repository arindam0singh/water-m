"""
Blind extraction and correlation-based detection for the DCT watermark.

Example:
    python extract.py watermarked.mp4 --public-key "alice-key" --salt "secret"
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from utils import (
    DEFAULT_BITS,
    DEFAULT_SALT,
    DetectionResult,
    derive_watermark_bits,
    extract_votes_from_luma,
    load_luma_for_detection,
    seed_from_key,
)


def extract_watermark(
    input_path: str,
    public_key: str,
    salt: str = DEFAULT_SALT,
    bit_length: int = DEFAULT_BITS,
    max_frames: int | None = None,
) -> DetectionResult:
    """
    Blindly recover bits and compare them with the expected public-key watermark.

    The original video is not used. Robustness comes from accumulating many
    noisy votes across blocks and frames.
    """
    expected_bits = derive_watermark_bits(public_key, salt, bit_length)
    seed = seed_from_key(public_key, salt)
    total_votes = np.zeros(bit_length, dtype=np.float64)
    total_counts = np.zeros(bit_length, dtype=np.int64)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        luma = load_luma_for_detection(frame)
        votes, counts = extract_votes_from_luma(luma, expected_bits, seed)
        total_votes += votes
        total_counts += counts

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    if frame_count == 0:
        raise ValueError("No readable frames found")

    recovered_bits = (total_votes >= 0).astype(np.uint8)
    bit_matches = int(np.sum(recovered_bits == expected_bits))
    bit_accuracy = bit_matches / float(bit_length)

    expected_symbols = np.where(expected_bits == 1, 1.0, -1.0)
    vote_symbols = total_votes / np.maximum(total_counts, 1)
    denom = np.linalg.norm(vote_symbols) * np.linalg.norm(expected_symbols)
    correlation = float(np.dot(vote_symbols, expected_symbols) / denom) if denom else 0.0

    return DetectionResult(
        recovered_bits=recovered_bits,
        expected_bits=expected_bits,
        bit_matches=bit_matches,
        bit_accuracy=bit_accuracy,
        correlation=correlation,
        votes_per_bit=total_votes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a DCT video watermark.")
    parser.add_argument("input", help="Input watermarked or attacked video path")
    parser.add_argument("--public-key", required=True, help="User public key string")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="Secret salt")
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, help="Watermark length")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick experiments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = extract_watermark(
        input_path=args.input,
        public_key=args.public_key,
        salt=args.salt,
        bit_length=args.bits,
        max_frames=args.max_frames,
    )

    print(f"Detected: {result.detected}")
    print(f"Bit accuracy: {result.bit_accuracy:.3f} ({result.bit_matches}/{args.bits})")
    print(f"Correlation: {result.correlation:.3f}")
    print("Recovered bits:")
    print("".join(str(int(bit)) for bit in result.recovered_bits))
