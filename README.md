# tlog

A lightweight, local-first experiment logger for neural network training.
wandb-shaped API, **zero dependencies** in your training environment, and three
clean ways to look at your runs from a SLURM cluster with nothing but a
terminal:

| viewer | command | when |
|---|---|---|
| **terminal dashboard** | `tlog watch` | live charts in a tmux pane — the default |
| **live web dashboard** | `tlog serve` | wandb-like browser UI through an SSH/VS Code port-forward |
| **self-contained HTML** | `tlog export -o report.html` | one file with charts + images; preview in VS Code, scp it, share it |

Everything is plain append-only JSONL in a run directory: grep-able,
rsync-able, crash-safe, no daemon, no cloud, no account.

```
 ● demo/baseline (da064b) · step 1500 · finished
  loss  eval  training  timing  memory  console

 loss/charb                              0.3158    loss/dino                              0.07182
  1.552 ┤⡧⣼                                        0.3882 ┤⡧⣼
        │⠇⢹⢿⣠⢀                                            │⡇⢹⣶⣀⣀
        │   ⠹⢹⠢⣧⣄⣀                                        │   ⠛⢹⠢⡦⣆⢀
        │      ⠁⠋⠋⠳⢶⢤⡀                                    │       ⠋⠉⠢⣴⣀⣀
        │           ⠘⠙⠦⠦⢴⢄⣀⡀                              │           ⠘⠙⢢⡧⢦⣀⢀
        │                 ⠉⠙⠛⠓⠶⠤⣤⣠⣠⣀⡀                     │                 ⠉⠙⠋⠳⠶⢤⣤⣴⣠⡄⡀
 0.2771 ┤                         ⠉⠙⠉⠛⠋⠓⠲⠚⠴⠖⠤⠤⠦⡦  0.06998 ┤                         ⠁⠙⠉⠋⠋⠑⠳⠒⠲⠶⠤⡶⣦⣦
        10                                 1490           10                                 1490

 loss/ssim                                0.131    loss/total                              0.5004
 0.6492 ┤⡧⣼                                         2.605 ┤⡧⣼
        │⠇⢹⢶⣀⢀                                            │⠇⢹⢶⣀
        │   ⠹⠹⠦⣶⣄⣀                                        │   ⠹⢹⠢⣦⣄⣀
        │       ⠛⠋⠣⣴⣠⡀                                    │       ⠋⠋⠲⣴⢤⣀
        │          ⠈⠘⠙⠲⠦⢤⣀⢀                               │           ⠘⠙⠦⠦⢤⣀⣀
        │                 ⠉⠙⠋⠓⢴⠤⢤⣤⣠⡀⡀                     │                 ⠈⠙⠋⠲⠴⠤⣤⣠⢠⡀⡀
 0.1174 ┤                         ⠁⠉⠉⠛⠊⠛⠲⠖⠴⠒⠤⡴⠦⣤   0.4672 ┤                         ⠉⠉⠉⠛⠊⠛⠲⠖⠴⠖⠤⠤⠦⣦
        10                                 1490           10                                 1490

 ←/→ pages · ↑/↓ scroll · 1-9 cols (auto) · s smooth (0) · l log (off) · q quit
```

*An actual `tlog watch` frame — braille-canvas charts in a plain tmux pane.*

## Install

```bash
pip install tlog-ml          # distribution is tlog-ml; you still `import tlog`
# or for development:
git clone https://github.com/philippe-eecs/tlog && cd tlog
pip install -e ".[dev]"
```

The core has **zero dependencies** — nothing to conflict with your torch/jax
pins. PIL is used opportunistically if present (image encoding, report
downscaling); otherwise a pure-stdlib PNG encoder takes over.

## Quickstart

```python
import tlog

run = tlog.init(project="vitok", name="vae-L16", config=vars(args))

for step in range(steps):
    ...
    if step % log_freq == 0:
        tlog.log({"loss/total": loss, "training/lr": lr,
                  "timing/mfu_percent": mfu}, step=step)
    if step % eval_freq == 0:
        tlog.log({f"eval/{k}": v for k, v in eval_stats.items()}, step=step)
        tlog.log_images("eval/recon", [orig, recon], step=step)  # torch/np/PIL

tlog.finish()
```

Then, in another tmux pane:

```bash
tlog                # == tlog watch: live dashboard of the latest run
tlog ls             # table of runs: step, last loss, slurm job, status
tlog tail           # live captured console output of the latest run
tlog serve          # web UI on :8585 (VS Code auto-forwards the port)
tlog export run-a run-b -o compare.html    # side-by-side report
```

Key namespaces (`loss/`, `eval/`, `timing/`, ...) become chart groups / TUI
pages automatically.

## What gets captured

`tlog.init()` records, without being asked:

- **SLURM**: job id, job name, partition, nodelist, array task id, and the
  actual `sbatch` script that launched the job (saved as `launch.sh`)
- **git**: commit, branch, dirty flag, and a `diff.patch` of uncommitted changes
- **environment**: argv, entrypoint, hostname, user, python/torch/CUDA
  versions, GPU models, world size
- **system metrics** (background thread, 10s interval): GPU util/mem/temp/power
  per device via nvidia-smi, CPU%, RAM — shown as their own chart groups
- **console**: stdout/stderr teed to `console.log` (tqdm-safe; viewers resolve
  `\r` overwrites)

## How it works

tlog is two decoupled halves that only meet at the filesystem: a **write
path** that lives inside your training process, and a **read path** (the
viewers) that runs anywhere that can see the same disk. There is no daemon,
no database, no socket between them — a run *is* a directory:

```
runs/<project>/<name>__<timestamp>__<id>/
├── meta.json          # identity + environment snapshot + restart history
├── config.json        # your hyperparameters (vars(args))
├── metrics.jsonl      # one JSON object per log() call, append-only
├── system.jsonl       # sampled GPU/CPU/RAM
├── console.log        # teed stdout/stderr
├── launch.sh          # captured sbatch script (under SLURM)
├── diff.patch         # uncommitted git changes
└── media/             # PNGs + index.jsonl mapping them to (key, step)
```

### The write path never blocks training

`log()` serializes one JSON line and appends it. Lines are written whole and
flushed, so a crash loses at most the line in flight and can never corrupt
history; `fsync` runs on a 30s timer to bound hard-failure data loss without
paying sync cost per step. Everything slow happens off the hot path: git
diff / nvidia-smi / `scontrol` captures run in a background thread after
init, system sampling and the liveness heartbeat are daemon threads, and
framework versions are read from `sys.modules` instead of importing anything.

### Preemption-safe by construction

SLURM requeues a preempted job with the same job id and bumps
`SLURM_RESTART_COUNT`. `init(resume="auto")` (the default) detects that,
finds the run directory it created before the preemption, and keeps
appending — recording a restart event in `meta.json`. Restarting from an
older checkpoint re-logs some steps; instead of rewriting files (dangerous),
**readers keep the last value logged per (metric, step)**, so charts come out
continuous and the storage stays strictly append-only. Explicit resume:
`tlog.init(id="a1b2c3", resume="must")`.

### The read path is one engine with three faces

`store.py` discovers runs, tails JSONL incrementally (remembering byte
offsets, parsing only complete new lines), applies keep-last dedup, and
downsamples with **min/max/mean buckets** — a one-step loss spike survives
being squeezed into a 200-px chart instead of being averaged away. Debiased
EMA smoothing (same formula as wandb) sits on top. The three viewers are just
renderers over this engine:

- **TUI**: each terminal cell is a 2×4 braille dot grid, so a tmux pane
  becomes a pixel canvas; charts are drawn with Bresenham lines and repainted
  on the alternate screen buffer. Pure ANSI — no curses, works over any SSH.
- **Web**: a stdlib `ThreadingHTTPServer` with JSON endpoints; the browser
  polls every 3s and refetches only runs whose files changed (mtime-keyed).
- **Export**: the *same* frontend with data, images (base64), and uPlot
  inlined into one HTML file. One codebase, a mode flag, two surfaces.

### Liveness without IPC

A daemon thread touches `heartbeat` every 15s. Viewers call a run *running*
if the heartbeat is fresh, *finished* if `finish()` marked it, and *dead* if
neither — which is how a SIGKILLed job shows up correctly with no process
ever being asked.

## Distributed training

`tlog.init()` is a no-op on non-zero ranks (it checks the `RANK` env var set
by torchrun/SLURM), so you can call it unguarded — or keep your existing
`if rank == 0:` guard; both are fine.

## Migrating from wandb

```diff
-import wandb
+import tlog

-wandb.init(project=args.project, name=args.name, config=vars(args))
+tlog.init(project=args.project, name=args.name, config=vars(args))

-wandb.log(avg, step=step)
+tlog.log(avg, step=step)

-wandb.finish()
+tlog.finish()
```

Runs land in `./runs` by default; set `TLOG_DIR=/scratch/$USER/runs` (or pass
`dir=`) to keep them on scratch.

## The viewers in detail

**`tlog watch [run]`** — braille line charts with min/max bands, one page per
metric group plus a console page; the grid auto-sizes to the pane and scrolls
when a group has more charts than fit. Keys: `←/→` pages · `↑/↓` (or `j/k`)
scroll charts / console history · `1`–`9` force column count, `0` auto (or
`--cols N`) · `s` smoothing (EMA 0 → 0.6 → 0.9 → 0.99) · `l` log scale ·
`q` quit.

**`tlog serve [root]`** — open `http://localhost:8585` through VS Code Remote
(auto port-forward) or `ssh -L 8585:localhost:8585 cluster`. Multi-run
overlay charts with synced cursors, smoothing slider, log scale, a media tab
laid out **runs-as-columns × steps-as-rows** for side-by-side recon/eval
comparison, a config tab that highlights differing hyperparameters, and live
console.

**`tlog export <runs...> -o report.html`** — the same UI frozen into a single
file (images downscaled to ≤512px by default; `--max-image-px 0` keeps
originals). No server, no internet — works in VS Code's HTML preview.

## Demo without a GPU

```bash
python examples/fake_train.py --steps 2000 &
tlog watch
```

## Prior art

[trackio](https://github.com/gradio-app/trackio), [aim](https://github.com/aimhubio/aim),
TensorBoard, and MLflow all live in adjacent space. tlog's niche is the
combination: a zero-dependency stdlib-only core safe to drop into any
training env, files you can grep as the source of truth, SLURM-native
metadata + preemption semantics, a terminal dashboard designed for a tmux
pane on a GPU cluster, and single-file HTML reports — in ~2,700 lines of
Python you can read in an afternoon.

## Tests

```bash
python -m pytest tests/
```

## License

MIT
