# baseline vs high-lr

Comparing the two demo VAE runs: same model and data, `lr` raised 3.3x in
**high-lr**. Verdict: the higher learning rate is *noisier in training but
lands in the same place* — slightly better FID, slightly worse SSIM. Not a
meaningful win at this scale.

## Training loss

The high-lr run pays for its speed with visible noise; smoothed (EMA 0.9)
the curves nearly converge by step 1500.

```tlog chart
key: loss/total
smooth: 0.9
```

## Eval quality

FID is the one metric where high-lr finishes ahead. PSNR/SSIM favor the
baseline by a hair.

```tlog chart
key: eval/fid
logy: true
```

```tlog table
columns: config.lr, eval/fid min, eval/ssim max, eval/psnr last, loss/total last
```

## Reconstructions

Side by side at the logged checkpoints — by step 1500 both are visually
indistinguishable.

```tlog images
key: eval/recon
last: 2
```

---

Generated from `runs/demo` with `tlog report examples/report.md demo`.
