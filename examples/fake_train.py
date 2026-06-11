"""Simulated training run for demoing/testing tlog without a GPU.

    python examples/fake_train.py --steps 2000 --sleep 0.02

Then in another pane:  tlog watch
"""

from __future__ import annotations

import argparse
import math
import random
import time

import tlog


def make_image(step: int, w: int = 96, h: int = 96, seed: int = 0):
    """A colorful gradient that sharpens as training 'converges' (no numpy)."""
    rng = random.Random(seed * 100003 + step)
    noise = max(0.02, 0.6 * math.exp(-step / 800))
    pixels = bytearray()
    for y in range(h):
        for x in range(w):
            r = x / w
            g = y / h
            b = 0.5 + 0.5 * math.sin(step / 200 + (x + y) / 24)
            for v in (r, g, b):
                v += rng.uniform(-noise, noise)
                pixels.append(min(255, max(0, int(v * 255))))
    return bytes(pixels), w, h


def log_fake_images(key: str, step: int, n: int, quality: float):
    """Log images using raw (bytes, w, h, channels) tuples — no numpy needed."""
    images = []
    for i in range(n):
        pixels, w, h = make_image(step, seed=i + int(quality * 10))
        images.append((pixels, w, h, 3))
    tlog.log_images(key, images, step=step, caption=f"recon @ step {step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="demo")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-freq", type=int, default=10)
    ap.add_argument("--eval-freq", type=int, default=200)
    ap.add_argument("--image-freq", type=int, default=500)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tlog.init(project=args.project, name=args.name, config=vars(args))

    warmup = 100
    t_prev = time.time()
    for step in range(1, args.steps + 1):
        time.sleep(args.sleep)

        lr = args.lr * min(1.0, step / warmup) * (
            0.5 * (1 + math.cos(math.pi * step / args.steps))
        )
        base = 2.2 * math.exp(-step / 600) + 0.35
        noise = rng.gauss(0, 0.04 + 0.1 * math.exp(-step / 300))
        loss = max(0.01, base + noise)
        charb = loss * 0.6 + rng.gauss(0, 0.01)
        ssim_l = loss * 0.25 + rng.gauss(0, 0.005)
        dino = loss * 0.15 + rng.gauss(0, 0.005)

        if step % args.log_freq == 0:
            now = time.time()
            dt = (now - t_prev) / args.log_freq
            t_prev = now
            tlog.log(
                {
                    "loss/total": loss,
                    "loss/charb": charb,
                    "loss/ssim": ssim_l,
                    "loss/dino": dino,
                    "training/lr": lr,
                    "timing/step_time": dt,
                    "timing/samples_per_sec": args.batch_size / dt,
                    "timing/mfu_percent": 38 + rng.gauss(0, 1.5),
                    "memory/allocated_gb": 61.2 + rng.gauss(0, 0.3),
                },
                step=step,
            )
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e}")

        if step % args.eval_freq == 0:
            tlog.log(
                {
                    "eval/ssim": min(0.99, 0.6 + 0.4 * (1 - math.exp(-step / 700)) + rng.gauss(0, 0.005)),
                    "eval/psnr": 18 + 9 * (1 - math.exp(-step / 700)) + rng.gauss(0, 0.1),
                    "eval/fid": max(2.0, 80 * math.exp(-step / 500) + rng.gauss(0, 0.5)),
                },
                step=step,
            )
            print(f"step {step:5d} | eval done")

        if step % args.image_freq == 0:
            log_fake_images("eval/recon", step, n=3, quality=step / args.steps)
            print(f"step {step:5d} | logged recon images")

    tlog.finish()
    print("done")


if __name__ == "__main__":
    main()
