# Handoff: Multi-Discriminator AMP implementation (interceptV2DualDis)

Written because the session's rate limit is about to cut off. Read this file
first in the next session to resume exactly where this left off.

## Where everything lives

- **Worktree:** `/home/ibouwmeest/BEPImitationLearning/.claude/worktrees/multi-disc-amp`
  (this file's directory). Branch: `worktree-multi-disc-amp`. Created via the
  `EnterWorktree` tool with `name: "multi-disc-amp"` — to resume, use
  `EnterWorktree` with `path: "/home/ibouwmeest/BEPImitationLearning/.claude/worktrees/multi-disc-amp"`.
- **Working project:** `interceptV2DualDis/` inside the worktree — a standalone
  copy of `SimpleGoalKeeper` (do NOT touch `SimpleGoalKeeper/` or the shared
  `beyondAMP/` clone under either project — hard constraint from the plan).
- **Python env:** `interceptV2DualDis/.venv` already built (`uv sync --python 3.11`
  — MUST be 3.11, not whatever `uv sync` defaults to, or `mjlab` import breaks).
  Activate with `source .venv/bin/activate` from inside `interceptV2DualDis/`
  before any `python3`/`pytest` command — each new shell needs this, it does
  not persist across separate Bash tool calls.
- **Plan being executed:** `interceptV2DualDis/docs/superpowers/plans/2026-07-02-multi-discriminator-amp.md`
  (8 tasks; read this for full task text/interfaces if anything below is unclear).
- **Design spec behind the plan:** `interceptV2DualDis/docs/superpowers/specs/2026-07-02-multi-discriminator-amp-design.md`
- **Progress ledger:** `.superpowers/sdd/progress.md` (in the worktree root) —
  append-only log of completed tasks, kept in sync with this file.
- **Task briefs/reports so far:** `.superpowers/sdd/task-{1,2,3,4,5,6}-brief.md`
  and matching `-report.md` files, plus `.superpowers/sdd/review-*.diff` review
  packages — all still on disk, reusable if you need to re-check what a
  reviewer actually saw.

## What this is

The user asked (in the parent session, not this worktree) for a 4-region,
multi-discriminator AMP system ported from `Humanoid-Goalkeeper`'s original
`HIMPPO`/`ActorCritic` design (6 discriminators there; simplified to 4 here:
`left_near`, `left_far`, `right_near`, `right_far`, split by side and by the
existing 0.35m-0.65m distance-tier convention at a 0.5m near/far cutoff).
Full HIM-style integration was requested explicitly — history-stacked
observations, ball/region estimator heads on the actor, 4 independent
discriminators + 4 independent motion buffers (near regions get the
single-step motion clip, far regions get the double-step clip only,
triple-step motion is excluded from AMP entirely in this track).

We are executing the plan via **subagent-driven-development**: fresh
implementer subagent per task (model chosen by task complexity — haiku for
mechanical/self-contained tasks, sonnet for multi-file integration, opus for
the highest-risk correctness review), task reviewer subagent after each
(spec compliance + code quality verdicts), fix-and-re-review loop for any
Critical/Important finding, ledger updated after each clean review.

**User's explicit standing instruction, not yet acted on:** after all 8 tasks
land and the final whole-branch review passes, "spin up a lot of tests you
can think of to fully test if we done the implementation correctly" — i.e.
go beyond the plan's own prescribed tests and build an extensive additional
verification pass before considering this done. This is tracked as the
"Final" task (see Task List state below) — do not skip it.

## Exact state right now

**Tasks 1-6 are implemented and their code is committed and tests pass.**
**Tasks 1-5 are fully reviewed and Approved.** **Task 6 has NOT been reviewed
yet — a real bug was found and fixed in it before review could happen; the
fix is committed but the task-reviewer subagent has never been dispatched
for it.** This is the very next action.

Commit log (newest first) on `worktree-multi-disc-amp`, all inside `interceptV2DualDis/`:

```
797f3ee fix(multidisc): replace redundant actor_history group with a real single-step actor_current group   <- Task 6 fix, UNREVIEWED
3a9197a feat(multidisc): add goalkeeper_multidisc_env_cfg with history group and region events               <- Task 6 original, superseded by 797f3ee's fix
cf35613 fix(multidisc): call amp_normalizer.update() per region, matching single-disc AMPPPO                 <- Task 4+5, re-reviewed Approved
9e4235a feat(multidisc): wire per-region policy-side AMP replay buffers into MultiDiscAMPPPO                 <- Task 5 half of merged 4+5
e327f83 feat(multidisc): add MultiDiscAMPPPO with region-routed expert loss and auxiliary est/region losses  <- Task 4 half of merged 4+5
3fcdb3c feat(multidisc): add HimActorCritic with history+ball+region estimator heads                         <- Task 3, reviewed Approved
f167c5d feat(multidisc): add ball/region ground-truth terms for critic obs                                   <- Task 2, reviewed Approved
d66db6d feat(multidisc): add static 4-region assignment and region-conditioned ball spawn                    <- Task 1, reviewed Approved
5a77bb6 fix(interceptV2DualDis): add missing motion-data gitignore whitelist + track NPZ/PKL files            <- prerequisite fix, not in plan
```

Full test suite as of `797f3ee`: **45 passed**, 0 failed, only pre-existing
`torch.jit.script` deprecation warnings (unrelated, harmless). Verified
directly, not just trusted from a report — ran
`cd interceptV2DualDis && source .venv/bin/activate && python3 -m pytest tests/ -q`
myself right before writing this file.

### The bug that was found and fixed in Task 6 (read this before touching Task 8)

Task 6's original implementation (`3a9197a`) built a new `"actor_history"`
observation group by reusing `cfg.observations["actor"].terms` (the SAME
`ObservationTermCfg` objects the base, unmodified `goalkeeper_env_cfg()`
already uses). Two things made this wrong:

1. The base `goalkeeper_env_cfg()` **already** sets
   `cfg.observations["actor"].history_length = 10` — so "actor" was already
   the 10-step history-stacked group before Task 6 ever ran. The plan
   (written before this was discovered) wrongly assumed the base actor group
   was single-step, and designed a separate history group to fix that
   non-existent gap.
2. `mjlab.managers.observation_manager.ObservationManager` mutates
   `term_cfg.history_length` **in place** when a group-level override is set
   (`observation_manager.py` ~line 438-439). Since "actor_history" shared the
   exact same term objects as "actor", it could never have differed from
   "actor" anyway — and worse, there was then no group anywhere providing a
   genuine single-step observation, which `HimActorCritic.act(obs_current,
   obs_history)` (Task 3) needs as its first argument.

**Fix (`797f3ee`):** removed "actor_history"; added `"actor_current"` built
from `dataclasses.replace(term_cfg, history_length=0)` clones of "actor"'s
terms — decoupled from the mutated originals. Result:
- `cfg.observations["actor"]` — unchanged, still `history_length=10`, this
  IS the history source. **Task 8's runner must read `obs_history` from the
  "actor" group**, not a group called "actor_history" (that name no longer
  exists — the plan text still says "actor_history" in a few places; treat
  "actor" as its replacement whenever you see that in the plan).
- `cfg.observations["actor_current"]` — new, `history_length=0` on every
  term, this is the `obs_current` source for `HimActorCritic`.

**This changes Task 8 from what the plan currently says.** The plan's Task 8
draft code has `_get_actor_history_obs` reading a `"actor_history"` group key
and treats `env.get_observations()` (which reads the `"actor"` group) as if
it were single-step for `obs_current` — neither is true anymore. When you
implement Task 8, the dispatch prompt MUST explicitly correct this:
- `obs_history` ← `env.unwrapped.observation_manager.compute()["actor"]`
  (already 10-step-flattened, shape `(N, num_one_step_obs * 10)`)
- `obs_current` ← `env.unwrapped.observation_manager.compute()["actor_current"]`
  (single-step, shape `(N, num_one_step_obs)`)
- `num_one_step_obs` should be computed from the `"actor_current"` group's
  dim, NOT from `env.num_obs` (which is `AMPEnvWrapper`'s notion of the
  `"actor"` group's dim — already `*10` given the above, so using it
  directly for `num_one_step_obs` would be off by a factor of 10).

Do not silently trust the plan's Task 8 section for this specific point —
it predates this discovery. Everything else in the plan's Task 8 section
(discriminator construction, motion dataset construction, region-id/ball-gt
critic-obs indexing, save/load, logging) is still believed accurate, but
re-verify against actual current file contents as the plan itself already
instructs, since Task 6/6-fix already showed the plan's assumptions can be
stale in specific spots.

## Immediate next action

1. **Dispatch a task reviewer for Task 6** (covering both `3a9197a` and its
   fix `797f3ee` as one diff, same pattern as the Task 4+5 merge — generate
   with `scripts/review-package cf35613 797f3ee` from
   `/home/ibouwmeest/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.0/skills/subagent-driven-development/`,
   run from the worktree root). Use the Task 6 brief at
   `.superpowers/sdd/task-6-brief.md` and report at
   `.superpowers/sdd/task-6-report.md` (the fix agent appended a section to
   this report before being interrupted — check it's actually there; if the
   interrupted agent didn't get to that step, note in the reviewer dispatch
   that the report may be incomplete for the fix half and the reviewer should
   rely on the diff, which is authoritative).
   - Model: sonnet is fine (this is a real but bounded correctness fix,
     already independently verified by me — the controller — via a full
     `pytest tests/ -q` run showing 45/45 passing before writing this file).
   - Global constraints to hand the reviewer: same as Task 6's original
     dispatch (see `.superpowers/sdd/task-6-brief.md` and the plan's Global
     Constraints section) PLUS explicitly: verify `"actor_history"` no
     longer exists as a key in `cfg.observations`, verify `"actor_current"`
     has `history_length=0` on every term via fresh (non-shared) object
     construction, verify `"actor"` still has `history_length=10` unchanged.
2. If Approved (or after any fix+re-review loop): mark Task 6 complete in
   both the TaskList tool and `.superpowers/sdd/progress.md`, append a line
   like `Task 6: complete (commits cf35613..797f3ee, review clean, Approved)`.
3. Proceed to Task 7 (`scripts/task-brief PLAN 7`), then Task 8 (with the
   corrected `obs_history`/`obs_current` group names spelled out explicitly
   in that dispatch per the section above — do not let the implementer
   subagent discover the plan's stale assumption the hard way).
4. After Task 8's review is clean, dispatch the final whole-branch code
   reviewer (`superpowers:requesting-code-review`'s `code-reviewer.md`
   template, most capable model available, package via
   `scripts/review-package <merge-base> HEAD` where merge-base is
   `git merge-base master HEAD` — the worktree branched from `master` at
   commit `7f335b2`, so that should be the merge-base, but confirm with the
   actual `git merge-base` command rather than assuming).
5. **Then, per the user's explicit request**, build and run an extensive
   additional test pass beyond what the plan itself specifies — think about
   what "fully test if we done the implementation correctly" means for an
   RL training pipeline: beyond the existing unit tests (which cover shapes,
   routing isolation, config wiring), consider things like: an actual short
   multi-region training run (mirrors Task 8 Step 4's manual smoke test,
   which itself was never run yet — do that too, it's still open) checking
   all 4 discriminators get nonzero gradient updates over several iterations
   and losses are finite/non-NaN; a check that region-assignment truly
   partitions envs 1:1 with no overlap/gaps at realistic `num_envs` (e.g.
   6144, not just the small test values used in unit tests); a check that
   `LeftTripleStep`/`RightTripleStep` motion files are never loaded into any
   of the 4 region buffers even by accident (e.g. assert on the actual
   `MotionDataset.motion_files` list at runtime, not just the config dict);
   an end-to-end `HimActorCritic` forward pass wired to real env observation
   shapes (not just synthetic tensors) to catch any dimension mismatch the
   unit tests' synthetic dims might have masked. Use your judgment for the
   rest — the user wants thoroughness here, not just plan-compliance.
6. Only after that: use `superpowers:finishing-a-development-branch` to
   decide how to land this (merge to master, PR, etc.) — do not merge
   without that skill's guidance and without the user's sign-off, since
   merging is exactly the kind of action that needs explicit confirmation.

## Task List tool state (for cross-checking after resume)

If the TaskList tool's state is still intact next session, it should show:
1. Task 1 — completed
2. Task 2 — completed
3. Task 3 — completed
4. Task 4+5 (merged) — completed
5. Task 6 — in_progress (needs review dispatch, see above)
6. Task 7 — pending
7. Task 8 — pending
8. Final (whole-branch review + user's extra test pass) — pending

If the tool's state was lost, recreate from this list — the ledger and git
log are the source of truth regardless of what the TaskList tool shows.

## Things worth remembering that aren't obvious from the plan file alone

- **Model selection pattern used so far:** haiku for Tasks 1/2/3 (mechanical,
  complete code in the brief, self-contained or near-self-contained); sonnet
  for Task 4+5 (real integration judgment, multi-file, forking beyondAMP
  classes) and Task 6 (multi-file config wiring); opus for the Task 4+5
  *review* specifically (highest correctness stakes — cross-region gradient
  isolation) both times (initial + re-review). Reviewers otherwise sonnet.
  Keep this pattern for Tasks 7/8 unless a task turns out simpler/harder
  than expected.
- **User decided (mid-plan):** merge plan Tasks 4 and 5 into one implementer
  dispatch, since Task 4's own text intentionally ships a placeholder that
  Task 5 immediately fixes, and a standalone reviewer would flag it as
  Critical — merging avoided ever exposing that placeholder to review in
  isolation. If a similar staged-placeholder pattern appears in Tasks 7/8
  (it shouldn't, based on the plan text as written), apply the same
  merge-before-review approach rather than asking again — this preference is
  now established.
- **Reviewers should always get real signature verification, not brief-trust:**
  every review dispatch so far has explicitly told the reviewer to check
  forked/reused upstream (`beyondAMP`) class signatures against the ACTUAL
  current source, not just the plan's inline code sketches — this caught
  nothing wrong in `beyondAMP`'s classes themselves, but did surface the
  Task 6 actor/actor_current bug (found by the controller reading the
  implementer's own passing-mention "concern" and verifying it, not by a
  reviewer — worth continuing to read implementer concerns skeptically-but-
  seriously rather than dismissing them as pre-hedged uncertainty).
- **`interceptV2DualDis`'s motion NPZ/PKL files were originally NOT tracked
  in git at all** (missing `.gitignore` whitelist entry, silently excluded by
  a blanket `*.npz`/`*.pkl` rule) — this was fixed as a prerequisite before
  Task 1 (`5a77bb6`), on the worktree branch, not on `master`. If a future
  fresh clone/worktree of `master` is used for anything before this branch
  merges, the same missing-motion-data problem will reappear on `master`
  until this branch (or an equivalent fix) lands there.
- **Do not re-run the smoke test from a stale mental model:** Task 8's plan
  text includes a manual (non-automated) smoke-test step
  (`uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 8
  --agent.max-iterations 2 --agent.save-interval 1`, with
  `MUJOCO_GL=egl WANDB_MODE=offline` env vars, same pattern already proven to
  work for the plain `interceptV2DualDis` copy earlier this session) — this
  has never actually been run yet since Task 8 hasn't been implemented. Do
  not skip it; it's the first point where the whole pipeline (env partition
  → history/estimator heads → 4 discriminators → runner) gets exercised
  together rather than in isolated unit tests.
