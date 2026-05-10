# DCT Video Watermarking PoC

This proof of concept embeds a deterministic invisible watermark into video
luma frames using 8x8 DCT blocks and spread-spectrum style repetition.

## Algorithm

1. Derive watermark bits from `SHA-256(counter | salt | public_key)`.
2. For each frame, work on the grayscale/luma channel.
3. Split the luma channel into 8x8 blocks and run DCT on each block.
4. For each block, deterministically choose a watermark bit, a mid-frequency
   coefficient pair, and a pseudo-random chip sign from the public key seed.
5. Encode the bit by slightly forcing the signed difference between the chosen
   coefficient pair.
6. During blind extraction, repeat the same deterministic schedule and recover
   each bit by correlation/majority voting across all blocks and frames.

The original video is not required for extraction.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Embed:

```bash
python embed.py input.mp4 watermarked.mp4 \
  --public-key "user-public-key-string" \
  --salt "shared-secret-salt" \
  --strength 12
```

Simulate screen-recording distortions:

```bash
python attacks.py watermarked.mp4 attacked.mp4 \
  --blur-kernel 3 \
  --jpeg-quality 60 \
  --resize-scale 0.85 \
  --brightness 1.05 \
  --noise-std 2.0
```

Extract and compare against the expected public-key watermark:

```bash
python extract.py attacked.mp4 \
  --public-key "user-public-key-string" \
  --salt "shared-secret-salt"
```

## Notes

- Increase `--strength` for harsher attacks, but very high values may become
  visible.
- Use more frames for better robustness because the detector accumulates votes.
- This is a teaching PoC, not a production forensic watermarking system.
