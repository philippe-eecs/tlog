# tlog

A local-first experiment logger **and a terminal review buddy** for working with
a coding agent on a cluster. wandb-shaped logging, plus a way for an agent to
show you full-res images, plots, and markdown **in the terminal** — and for you
to leave comments in `$EDITOR` that the agent reads back. No browser, no cloud,
no leaving the terminal.

The headline is smart dispatch — **`tlog <thing>` just works**, where `<thing>`
is a run, a project, a group, a saved set, or a file:

| you run | tlog does |
|---|---|
| `tlog` | live dashboard of the latest run |
| `tlog baseline high-lr` | overlay runs / a project / a group / a saved set, live |
| `tlog plot.png` · `tlog notes.md` | render the image / markdown **in the terminal** (full-res on Ghostty/kitty) |
| `tlog review analysis.md` | render a doc, then open `$EDITOR` to comment on it |
| `tlog comments --json` | the agent reads your open comments and revises |
| `tlog serve` / `tlog report` | browser dashboard / self-contained HTML report |

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

 ←/→ pages · ↑/↓ scroll · 1-9 cols (auto) · s smooth (0) · l log (off) · c comment · q quit
```

*An actual `tlog watch` frame — braille-canvas charts in a plain tmux pane.*

## Install

```bash
pip install tlog-ml          # distribution is tlog-ml; you still `import tlog`
pip install "tlog-ml[video]" # adds `tlog show clip.mp4` (bundled ffmpeg)

# for development:
git clone https://github.com/philippe-eecs/tlog && cd tlog
pip install -e ".[dev]"
```

tlog has a single runtime dependency, **Pillow**, used for robust full-res
image decode/encode in the terminal (it pip-installs cleanly on any cluster —
no system libraries). torch / numpy are still only touched if your training
code already imported them. Video support is the one optional extra and bundles
its own ffmpeg via `imageio-ffmpeg`, so even `tlog show clip.mp4` needs no
system binary.

### Full-res images on a cluster (Ghostty, kitty, iTerm2 — even through tmux)

tlog draws true-pixel images with the kitty or iTerm2 graphics protocol when it
detects a capable terminal, and falls back to 24-bit half-blocks (`▀`) anywhere
else, so something always renders — even SSH into a bare terminal. Two notes for
the common cluster setup:

- **Ghostty/kitty through tmux**: tmux normally eats graphics escapes. Turn on
  passthrough once — `tmux set -g allow-passthrough on` — and tlog wraps its
  escapes so full-res images come through. Add `export TLOG_TERM=ghostty` (or
  `kitty`/`iterm2`) to your shell rc so tlog still knows the outer terminal can
  show pixels (tmux hides the usual signals). Without these you get half-blocks
  and a one-line hint explaining how to upgrade.
- Force a backend any time with `--images kitty|iterm2|halfblock|off`.

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
tlog                          # live dashboard of the latest run
tlog baseline high-lr         # overlay runs / a project / a group / a saved set
tlog ls                       # table of runs: group, step, last loss, status
tlog show eval.png notes.md   # render images / markdown in the terminal
tlog show baseline --console  # captured console output of a run
tlog serve                    # web UI on :8585 (VS Code auto-forwards the port)
tlog export a b -o compare.html    # one self-contained interactive HTML file
tlog report spec.md a b            # markdown narrative with live blocks -> HTML
```

Key namespaces (`loss/`, `eval/`, `timing/`, ...) become chart groups / TUI
pages automatically.

### Grouping & linking runs to compare

```python
tlog.init(project="vitok", name="lr-3e-4", group="lr-sweep")   # tag at init
```

```bash
tlog lr-sweep                       # overlay every run in the group, live
tlog link big-models vae-L vae-H    # save an ad-hoc set (an agent can build these)
tlog sets                           # list saved sets
tlog big-models                     # view the set; `r`/`v` cycle focus / hide runs
```

## The terminal review loop (you ↔ your coding agent)

A run is just files, so a coding agent on the same cluster can already read your
metrics and images directly. The new half is letting it **show you things and
collect your feedback without anyone opening a browser**:

```bash
# the agent renders an analysis (prose + live charts/images) in your terminal,
# then drops you into $EDITOR to comment section-by-section:
tlog review analysis.md

# you type feedback under the section headings, save, quit. The agent reads it:
tlog comments --doc analysis.md --json     # open comments, machine-readable
tlog resolve <id> -m "fixed"               # mark one done after addressing it
```

Comments aren't only for docs. Anything can be a target, and they all land in
one append-only store (`<runs>/.tlog/comments.jsonl`):

```bash
tlog comment run:demo/baseline@1000:eval/recon -m "mode collapse here"
tlog comment file:model.py:42 -m "this init looks wrong"
```

…and inside `tlog watch`, press **`c`** to comment on the focused run/chart/image
right there — a `✎N` badge marks runs with open comments. One loop: the agent
shows (`tlog show`/`review`), you comment (`$EDITOR`), the agent reads
(`tlog comments --json`) and revises.

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

By default a rerun **re-attaches by name**: `init(project="p", name="exp")`
twice points at the same run directory and keeps appending, instead of
littering `runs/` with parallel copies. SLURM requeues (same job id, bumped
`SLURM_RESTART_COUNT`) re-attach the same way and record a restart event in
`meta.json`. Restarting from an older checkpoint re-logs some steps; instead of
rewriting files (dangerous), **readers keep the last value logged per (metric,
step)**, so a requeue continues forward and a from-scratch rerun overwrites the
overlapping steps — storage stays strictly append-only either way. Want a fresh
parallel run? `new=True` (or a new name). Clean slate in the same dir?
`reset=True`. Explicit id resume still works: `tlog.init(id="a1b2c3",
resume="must")`.

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

**`tlog watch [runs...]`** — braille line charts with min/max bands, one page
per metric group plus media and console pages; the grid auto-sizes to the
pane and scrolls when a group has more charts than fit.

- **Compare runs**: `tlog watch baseline high-lr` (or name a project dir to
  take all its runs) overlays every metric wandb-style, one color per run,
  with a legend line. The `r` key cycles which run the console page shows.
- **Media page**: logged images render *in the terminal* — true pixel images on
  Ghostty/kitty/iTerm2/WezTerm (incl. through tmux with `allow-passthrough`, see
  Install), half-block thumbnails everywhere else. Runs are columns, steps are
  rows, like the web media tab. (`--images off` hides the page.)

Keys: `←/→` pages · `↑/↓` (or `j/k`) scroll charts / media steps / console
history · `m` cycle media key · `r` cycle focused run · `v` hide/show the focused
run · `c` comment on the focused run/view · `1`–`9` force column count, `0` auto
(or `--cols N`) · `s` smoothing (EMA 0 → 0.6 → 0.9 → 0.99) · `l` log scale ·
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

**`tlog report spec.md [runs...]`** — custom pages: write plain markdown and
drop in ```` ```tlog ```` blocks where you want live elements, then render to
one self-contained HTML file (`--open` pops a browser). Prose narrates;
blocks pull from the runs:

````markdown
## Eval quality

FID is the one metric where high-lr finishes ahead.

```tlog chart
key: eval/fid
smooth: 0.9        # optional EMA; raw stays as a faint line
logy: true
runs: baseline, high-lr   # optional — defaults to the runs on the CLI
```

```tlog table
columns: config.lr, eval/fid min, eval/ssim max, loss/total last
```

```tlog images
key: eval/recon
last: 2            # or steps: 500, 1500
```
````

Three block types: `chart` (multi-run SVG overlay), `table` (one row per
run; columns are metric keys with an optional `min`/`max`/`last` aggregator,
or `config.*` values), and `images` (runs-as-columns × steps-as-rows grid).
Because a run is just files and the spec is just markdown, reports are easy
for both humans and coding agents to compose — ask an agent to inspect your
runs and it can write the analysis *and* the page that shows the evidence
(see `examples/report.md`).

## Let a coding agent review your runs

Because a run is nothing but files on disk — `metrics.jsonl`, `config.json`,
PNGs under `media/` — a coding agent (Claude Code, etc.) on the same cluster can
inspect a run with no API key, no server, and no browser. The review loop then
happens entirely in the terminal:

> "Compare `baseline` and `high-lr`. Look at the loss curves and the eval recon
> images, then write up whether the higher LR helped."

1. The agent reads the JSONL and PNGs, writes `analysis.md` (prose plus
   `chart`/`table`/`images` blocks), and runs **`tlog review analysis.md`** — the
   charts and side-by-side reconstructions render right in your pane (full-res on
   Ghostty), then `$EDITOR` opens with a comment slot per section.
2. You type reactions under the sections that matter and save. The agent runs
   **`tlog comments --doc analysis.md --json`**, addresses each point, marks them
   `tlog resolve`d, and revises the doc.
3. Repeat until you're happy — then `tlog report analysis.md a b -o out.html` if
   you also want a shareable file for your laptop.

The same loop works on runs directly (`tlog baseline`, press `c`) and on source
(`tlog comment file:model.py:42 -m "…"`). The agent never needed anything but the
run directory and your terminal. See `examples/review.md` for a sample doc and
`examples/report.md` for the HTML-report form.

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
