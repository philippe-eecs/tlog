# Review: does the higher LR help?

*A sample agent-written review doc. Render + comment with:*
`tlog review examples/review.md demo`
*— the charts and reconstructions draw in your terminal, then your editor opens
with a comment slot under each heading. The agent then reads your notes with*
`tlog comments --doc review.md --json`.

## Summary

I compared **baseline** and **high-lr** (same model/data, `lr` raised 3.3x).
My read: the higher LR trains noisier but lands in the same place — marginally
better FID, marginally worse SSIM. I don't think it's a real win at this scale,
but I'd like your call before I kill the sweep.

## Training loss

Smoothed (EMA 0.9), the curves nearly converge by step 1500; high-lr is just
noisier early.

```tlog chart
key: loss/total
smooth: 0.9
```

## Eval quality

FID is the only metric where high-lr finishes ahead; PSNR/SSIM favor baseline
by a hair.

```tlog chart
key: eval/fid
logy: true
```

```tlog table
columns: config.lr, eval/fid min, eval/ssim max, loss/total last
```

## Reconstructions

By the last checkpoint they're visually indistinguishable to me — curious if you
see a difference.

```tlog images
key: eval/recon
last: 2
```

## Decision

Proposed: keep baseline's LR, drop high-lr. Tell me if you want a third point
between the two before I commit.
