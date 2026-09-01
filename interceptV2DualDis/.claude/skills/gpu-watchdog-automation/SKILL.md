---
name: gpu-watchdog-automation
description: Use when working with, debugging, or extending the unattended GPU-sharing/auto-training automation for this project (/home/robocup/IsaakB/intercept_gpu_watchdog.sh, cron */30 * * * *). Covers the full priority state machine, the state files it reads/writes, the additive-resume iteration quirk it has to account for, and known edge cases -- including several confirmed live (contention chains with zero idle gaps, the mislabeled-final-checkpoint bug interacting with its target check).
---

# Intercept GPU Watchdog Automation

## What it is

A single unattended bash script, `/home/robocup/IsaakB/intercept_gpu_watchdog.sh`,
scheduled via cron (`*/30 * * * *`) on this machine (robocup). It is **pure
shell** -- no Claude/LLM involvement at runtime. Claude only writes/edits the
script and reads its logs when asked; cron invokes it directly and it runs
to completion on its own every 30 minutes, forever, with zero AI cost.

It is the sole automation for this project. It supersedes two earlier,
narrower scripts that are **no longer scheduled** (kept on disk only for
reference, do not re-add their cron entries without a reason):
- `intercept_autotrain.sh` (used to fire once daily at 23:59, only checked
  for new commits)
- `intercept_nightly_resume.sh` (used to fire once daily at 00:00, only
  resumed the tracked checkpoint lineage)

Both behaviors were folded into one script with a single priority order,
per user request (2026-08-31), specifically so a new commit can **override**
an in-progress resume rather than waiting for it to finish first.

## The priority state machine

Re-evaluated completely fresh every single tick (no persistent process,
no memory between invocations except the state files below):

1. **We're running AND someone else's job also appears on the GPU
   (contention)** -> pause. Push the latest checkpoint (see "the persistent
   local-edits problem" below), stop our process. Takes priority over
   *everything* else, including a pending new commit.
2. **Someone else's job only, we're not running** -> wait. Do nothing.
3. **A new training-relevant commit exists** (compared against
   `last_trained_commit.txt`, same path filter as the old autotrain
   script: `mdp/`, `tasks/`, `robots/`, `motions/data/`, `rsl_rl_multi/`,
   `scripts/train.py`) -> override. If we were mid-resume on the old
   lineage, push+stop it first -- **its remaining iterations toward 20k
   are abandoned, not finished first.** Then pull and launch fresh on the
   new commit.
4. **No new commit, already running, no contention** -> nothing to do.
5. **No new commit, GPU idle** -> resume the tracked lineage
   (`resume_run_dir.txt`) from its latest checkpoint, aimed at absolute
   iteration 20000. Throttled to ~hourly *actual* attempts via
   `last_idle_attempt_epoch.txt`, even though the cron itself fires every
   30 min (checking for contention/new-commits more often is cheap and
   desired; repeatedly re-launching the same resume is not).
6. **No new commit, GPU idle, tracked lineage already >=20000 or none
   exists** -> nothing to do.

PID classification: every PID `nvidia-smi --query-compute-apps` reports is
checked via `ps -p <pid> -o cmd=` against the pattern `sgk_train.*MultiDisc`.
Match = ours, no match = someone else's.

## State files (all under `/home/robocup/IsaakB/intercept_autotrain_logs/`)

| File | Meaning |
|---|---|
| `resume_run_dir.txt` | Absolute path to the run directory the watchdog should resume from when idle. Updated on every successful launch (fresh or resumed) once the new run is *confirmed* running. |
| `last_trained_commit.txt` | The last commit SHA the watchdog (or a manual launch, if kept in sync -- see below) has confirmed training on. Compared against `origin/v2-blue-ball-waypoint` each tick. |
| `last_idle_attempt_epoch.txt` | Unix timestamp of the last actual idle-resume attempt, purely to throttle state-5 to ~hourly. |
| `<timestamp>_watchdog.log` | Full stdout/stderr of one tick, one file per invocation. |
| `train_<run-name>.log` | Full training stdout for a watchdog-launched run, same as any manually launched run's log. |

**Manual launches must keep these in sync.** Every time a human (or Claude,
interactively) launches or resumes a run *outside* the watchdog, update both
`last_trained_commit.txt` (to `git rev-parse HEAD`) and `resume_run_dir.txt`
(to the new run's directory) immediately after confirming it started.
Otherwise the watchdog either launches a redundant duplicate next tick (if
the commit marker is stale) or loses track of the real current lineage (if
the resume pointer is stale). See `interceptV2DualDis/CLAUDE.md`'s Training
Run Monitoring section for the exact commands.

## The additive-resume iteration quirk

`--agent.resume True --agent.max-iterations N` does **not** mean "run until
absolute iteration N." The runner computes
`tot_iter = current_learning_iteration + num_learning_iterations`
(`him_amp_on_policy_runner.py`) -- `N` is *added* to wherever the loaded
checkpoint already is. Confirmed the same additive pattern exists in at
least one other project on this machine too
(`BoosterLab/scripts/AMP/amp/amp_on_policy_runner.py:568-569`), so don't
assume a project without this exact code is safe from it -- verify per
project if it matters.

The watchdog's `resume_tracked_lineage()` handles this correctly: it reads
the latest checkpoint's real iteration number from its filename, then
computes `OFFSET = TARGET_ITERATIONS - LATEST_ITER` and passes `OFFSET` (not
`TARGET_ITERATIONS`) as `--agent.max-iterations`, so the *result* lands on
absolute 20000 regardless of where the checkpoint started from.

## Known edge cases

- **Zero-gap contention chains (confirmed live, 2026-08-31 into 09-01):**
  another project on this machine ran ~8 back-to-back jobs through a full
  night with literally no idle moment between any of them -- one finishes,
  the next launches within the same 30-minute window. The watchdog cannot
  catch a gap that never exists; it correctly logged "someone else's job
  only, waiting" every single tick with no false starts. This is expected
  behavior, not a bug -- there was nothing to catch.
- **The mislabeled-final-checkpoint bug (see `docs/BugFixes.md`) turns out
  harmless here, for the wrong reason.** A run that completes its full
  20000-iteration schedule sometimes saves its final checkpoint under an
  inflated filename (e.g. `model_39750.pt` when the real trained iteration
  count is 20000 -- a save-point counter mutation bug, documented
  elsewhere). `resume_tracked_lineage()`'s check `LATEST_ITER >=
  TARGET_ITERATIONS` still correctly concludes "nothing to resume" in this
  case (39750 >= 20000 is true), just via an inflated number rather than
  the real one. The practical outcome (stop trying to resume) is right
  either way, but don't trust this file's number for anything beyond that
  boolean check.
- **Cold-start requires manual seeding.** With no `resume_run_dir.txt` and
  no `last_trained_commit.txt`, the very first tick seeds the commit marker
  to current HEAD (so no commit looks "new") and finds no state file to
  resume from -- it does *nothing* on its own. A human has to manually
  point `resume_run_dir.txt` at an existing run+checkpoint, or roll back
  `last_trained_commit.txt` to something behind HEAD, to bootstrap either
  the resume path or the new-commit path.
- **The persistent local-edits problem.** This checkout has several
  files with real, never-committed local edits that have persisted across
  the entire project history (`../.claude/settings.local.json`,
  `../CLAUDE.md`, `../commands.txt`, `goalkeeper_multidisc_amp_cfg.py`).
  Every git operation the watchdog does (push, rebase, pull) stashes these
  four paths first and restores them after, mirroring the exact manual
  dance used throughout this project's history. `commands.txt` is the only
  one that reliably conflicts on stash-pop (since the watchdog itself
  overwrites it with the launch command) -- resolved automatically with
  "ours." **A real, once-nearly-happened risk:** `git reset --hard` (used
  once, manually, to pin an old commit for an experiment) discarded these
  edits outright since it wasn't preceded by a stash -- recovered only
  because the dropped stash object hadn't been garbage-collected yet
  (`git fsck --unreachable`). The watchdog's own git operations were
  designed specifically to avoid ever repeating that mistake -- never add a
  `git reset --hard` to this script without stashing first.
- **No overlap protection.** If a single tick takes long enough to still be
  running when the next `*/30` fires (the confirm-wait loop alone can take
  up to 5 minutes, plus warp/kernel compile time on a fresh launch), two
  watchdog invocations could run concurrently. Not yet observed live, no
  lock file exists to prevent it -- worth adding (e.g. `flock`) if it ever
  actually happens.
- **PID misclassification risk.** The "ours" check is a hardcoded string
  match (`sgk_train.*MultiDisc`). If this project's task ID or invocation
  ever changes (e.g. switching to the single-discriminator task, or a
  differently-named entry point), the watchdog would misclassify its own
  process as "someone else's" -- it would then think the GPU is occupied by
  a stranger and just wait forever, never recognizing its own job. Update
  the grep pattern in `intercept_gpu_watchdog.sh` if the launch command
  ever changes.
- **Real git conflicts beyond `commands.txt` are not handled.** The
  stash/rebase dance assumes the only real conflict is on `commands.txt`.
  A genuine content conflict on any other stashed file (or a rebase
  conflict against actual training code, which shouldn't happen since the
  watchdog never edits training code itself) would leave the repo mid-
  operation with no automatic recovery -- would need a human to intervene
  and clean up before the next tick can do anything useful.
