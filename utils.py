"""
Shared utilities for DCT-based forensic video watermarking.

This module contains the deterministic watermark derivation, block-DCT helpers,
and the coefficient-pair embedding/extraction primitives used by embed.py and
extract.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_SALT = "demo-secret-salt-change-me"
DEFAULT_BITS = 256

# Mid-frequency 8x8 DCT coefficient pairs. DC/very low frequencies are avoided
# because they are visually sensitive; high frequencies are avoided because
# compression tends to erase them.
MID_FREQ_PAIRS = (
    ((2, 3), (3, 2)),
    ((2, 4), (4, 2)),
    ((3, 4), (4, 3)),
    ((1, 4), (4, 1)),
    ((2, 5), (5, 2)),
    ((3, 5), (5, 3)),
)


@dataclass(frozen=True)
class DetectionResult:
    """Result returned by blind extraction."""

    recovered_bits: np.ndarray
    expected_bits: np.ndarray
    bit_matches: int
    bit_accuracy: float
    correlation: float
    votes_per_bit: np.ndarray

    @property
    def detected(self) -> bool:
        """Simple PoC detection decision."""
        return self.bit_accuracy >= 0.70 and self.correlation > 0.20


def derive_watermark_bits(
    public_key: str,
    salt: str = DEFAULT_SALT,
    bit_length: int = DEFAULT_BITS,
) -> np.ndarray:
    """
    Derive a deterministic binary watermark from a public key and secret salt.

    SHA-256 produces 256 bits. If more bits are requested, additional digest
    blocks are generated with a counter prefix.
    """
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")

    bit_chunks: list[np.ndarray] = []
    counter = 0
    while sum(len(chunk) for chunk in bit_chunks) < bit_length:
        payload = f"{counter}|{salt}|{public_key}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        bit_chunks.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        counter += 1

    return np.concatenate(bit_chunks)[:bit_length].astype(np.uint8)


def seed_from_key(public_key: str, salt: str = DEFAULT_SALT) -> int:
    """Return a stable 32-bit seed for deterministic coefficient selection."""
    digest = hashlib.sha256(f"seed|{salt}|{public_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def block_plan(
    num_blocks: int,
    watermark_bits: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the deterministic spread-spectrum schedule.

    Each 8x8 block carries one watermark bit. The bit index, coefficient pair,
    and chip sign are pseudo-random but fully reproducible from the public key
    and salt. Repeating bits over many blocks/frames gives robust majority votes.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if watermark_bits.size == 0:
        raise ValueError("watermark_bits cannot be empty")

    rng = np.random.default_rng(seed)
    bit_indices = np.arange(num_blocks, dtype=np.int64) % watermark_bits.size
    rng.shuffle(bit_indices)

    pair_indices = rng.integers(0, len(MID_FREQ_PAIRS), size=num_blocks)
    chips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=num_blocks)
    return bit_indices, pair_indices, chips


def iter_8x8_blocks(channel: np.ndarray):
    """Yield top-left coordinates and 8x8 views over a 2D image channel."""
    usable_h = channel.shape[0] - (channel.shape[0] % 8)
    usable_w = channel.shape[1] - (channel.shape[1] % 8)
    for y in range(0, usable_h, 8):
        for x in range(0, usable_w, 8):
            yield y, x, channel[y : y + 8, x : x + 8]


def block_count(frame_shape: tuple[int, int]) -> int:
    """Return the number of complete 8x8 blocks in a frame."""
    height, width = frame_shape[:2]
    return (height // 8) * (width // 8)


def split_luma(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """
    Extract a luma/grayscale channel for watermarking.

    Color videos are converted to YCrCb and only Y is modified so the output
    remains visually close to the input. Grayscale videos are processed directly.
    """
    if frame.ndim == 2:
        return frame.astype(np.float32), None, False

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return y.astype(np.float32), cv2.merge((cr, cb)), True


def merge_luma(luma: np.ndarray, chroma: np.ndarray | None, was_color: bool) -> np.ndarray:
    """Merge the modified luma channel back into a displayable frame."""
    clipped = np.clip(luma, 0, 255).astype(np.uint8)
    if not was_color:
        return clipped

    if chroma is None:
        raise ValueError("chroma must be provided for color frames")
    cr, cb = cv2.split(chroma)
    ycrcb = cv2.merge((clipped, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def embed_luma_channel(
    luma: np.ndarray,
    watermark_bits: np.ndarray,
    seed: int,
    strength: float,
) -> np.ndarray:
    """
    Embed watermark bits into the luma channel with 8x8 block DCT.

    A bit is encoded by forcing the signed difference between a mid-frequency
    coefficient pair to agree with the target bit and pseudo-random chip sign.
    """
    watermarked = luma.copy()
    num_blocks = block_count(luma.shape)
    bit_indices, pair_indices, chips = block_plan(num_blocks, watermark_bits, seed)

    for block_number, (y, x, block) in enumerate(iter_8x8_blocks(watermarked)):
        bit = watermark_bits[bit_indices[block_number]]
        target = 1.0 if bit == 1 else -1.0
        target *= chips[block_number]

        dct_block = cv2.dct(block.astype(np.float32) - 128.0)
        (a_y, a_x), (b_y, b_x) = MID_FREQ_PAIRS[pair_indices[block_number]]

        coeff_a = float(dct_block[a_y, a_x])
        coeff_b = float(dct_block[b_y, b_x])
        diff = coeff_a - coeff_b

        # Adjust the pair just enough to make the desired signed difference
        # visible to the detector while keeping pixel changes small.
        desired_diff = target * strength
        if target * diff < strength:
            adjustment = (desired_diff - diff) / 2.0
            dct_block[a_y, a_x] += adjustment
            dct_block[b_y, b_x] -= adjustment

        restored = cv2.idct(dct_block) + 128.0
        watermarked[y : y + 8, x : x + 8] = restored

    return watermarked


def extract_votes_from_luma(
    luma: np.ndarray,
    expected_bits: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blindly extract correlation votes from one luma channel.

    For each block, the detector checks whether the same deterministic
    coefficient-pair difference is positive or negative after de-spreading by
    the chip sign.
    """
    num_blocks = block_count(luma.shape)
    bit_indices, pair_indices, chips = block_plan(num_blocks, expected_bits, seed)

    votes = np.zeros(expected_bits.size, dtype=np.float64)
    counts = np.zeros(expected_bits.size, dtype=np.int64)

    for block_number, (_, _, block) in enumerate(iter_8x8_blocks(luma.astype(np.float32))):
        dct_block = cv2.dct(block - 128.0)
        (a_y, a_x), (b_y, b_x) = MID_FREQ_PAIRS[pair_indices[block_number]]
        diff = float(dct_block[a_y, a_x] - dct_block[b_y, b_x])

        bit_index = bit_indices[block_number]
        votes[bit_index] += diff * chips[block_number]
        counts[bit_index] += 1

    return votes, counts


def load_luma_for_detection(frame: np.ndarray) -> np.ndarray:
    """Convert an input frame to the luma/grayscale channel used for extraction."""
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def fourcc_for_path(path: str) -> int:
    """Pick a broadly compatible codec from the output extension."""
    lower = path.lower()
    if lower.endswith(".avi"):
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")
