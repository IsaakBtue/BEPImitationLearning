---
name: gpu-watchdog-automation
description: Use when working with, debugging, or extending the unattended GPU-sharing/auto-training automation for this project (/home/robocup/IsaakB/intercept_gpu_watchdog.sh, cron */10 * * * *). Covers the full priority state machine, the state files it reads/writes, the additive-resume iteration quirk it has to account for, and known edge cases -- including several confirmed live (contention chains with zero idle gaps, the mislabeled-final-checkpoint bug interacting with its target check).
---

# Intercept GPU Watchdog Automation

## What it is

A single unattended bash script, `/home/robocup/IsaakB/intercept_gpu_watchdog.sh`,
scheduled via cron (`*/10 * * * *`, tightened from `*/30` -> `*/15` -> `*/10`
on 2026-09-07 -- a tick itself is just `nvidia-smi`/`ps` checks and an
occasional `git fetch`, no GPU/compute cost, and contention detection IS the
tick, so a tighter cron directly means noticing another job sooner) on this
machine (robocup). It is **pure shell** -- no Claude/LLM involvement at
runtime. Claude only writes/edits the script and reads its logs when asked;
cron invokes it directly and it runs to completion on its own every 10
minutes, forever, with zero AI cost.

**Temporarily disabled overnight 2026-09-07 -> 2026-09-08** for a deliberate
GPU-sharing session with a teammate (both training on the GPU at once,
watchdog would otherwise pause on seeing the teammate's PIDs as contention).
A one-shot cron entry (`0 15 8 9 * intercept_watchdog_reenable.sh`) restores
the `*/10` schedule at 15:00 on 2026-09-08 and removes itself.

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
   desired; repeatedly re-launching the same resume is not). **Skipped
   entirely if `no_autoresume.flag` is present** (see below) -- a new
   commit still overrides and clears the flag; this path alone does not.
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
| `no_autoresume.flag` | **NEW 2026-09-09 (user request).** Presence blocks ONLY priority-5 idle-resume. Set by Claude whenever the user says "stop training" in chat without asking for a relaunch -- the user does not want the watchdog quietly taking the GPU back later on its own. Cleared automatically by priority 3 (a new commit always overrides, per explicit user request -- "only when there is a new commit it can go automatically"), and should also be cleared by Claude any time the user explicitly asks to resume/continue a stopped run from chat. Empty/content-less -- only its existence matters. |
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

**Manual stop from chat, no relaunch requested: also touch `no_autoresume.flag`.**
`touch $LOG_DIR/no_autoresume.flag` right after stopping the process (and
pushing its checkpoint, as always). This is a distinct case from a
stop-then-immediately-relaunch (e.g. "stop and resume from X") -- in that
case don't bother setting the flag at all, since it would just be cleared
again a moment later. If the user comes back later and asks to resume, `rm
-f $LOG_DIR/no_autoresume.flag` as part of that action.

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
- **No overlap protection -- confirmed live twice, 2026-09-09.** Not a
  watchdog-vs-watchdog overlap (that specific scenario still unobserved),
  but the closely related case: a human/Claude manually launches or resumes
  a run at almost the exact moment a `*/10` tick lands on "GPU idle, no new
  commit" (priority 5) or "new commit, GPU idle" (priority 3) and reaches
  the same conclusion independently -- both launch within the same second,
  producing two live training processes racing for the GPU. Happened once
  on a fresh launch (`6144_overshootleakfix` vs. the watchdog's own
  `6144_watchdog`, same tick) and once on a resume (`6144_resumefrom1500`
  vs. the watchdog's own `6144_watchdogresume`, same tick, right after a
  manual stop left the GPU briefly idle). Both times: no OOM/crash observed
  from the brief dual-process window, resolved by killing the
  watchdog-launched duplicate and keeping the manually-launched one. No
  lock file exists to prevent this -- `flock` around the whole script would
  close it, but the user judged the actual odds of it happening "subliminal"
  (2026-09-09) and asked not to bother. If it starts happening more often,
  revisit with a lock.
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
