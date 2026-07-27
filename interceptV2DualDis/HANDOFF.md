# Handoff Notes

Running log of what to watch for after recent reward changes, for whoever
(human or AI) picks up training monitoring next. See `docs/BugFixes.md` for
full rationale/evidence on each item — this file is just the "what to watch"
distillation.

## 2026-07-27 -- feet_slippage/foot_clearance/post-save-orientation/success batch

Two parallel change sets landed together today (merge commit after
`b14ad6d` + `def6226`), both touching post-save leading/trailing-limb
behavior via different, complementary mechanisms (see `docs/BugFixes.md`'s
"Merge reconciliation" entry for why they don't conflict). **None of this
has been validated against a live training run yet** — everything below is
pytest + smoke-test verified only (real env, zero-action steps, no learning).
The next step for any of these is watching real training curves.

### What changed
- `feet_slippage`: excludes genuine foot-ball contact (both feet) from
  counting as ground slippage. Known limitation: the `ball_contact` sensor
  can still miss brief/glancing touches, so this reduces, not eliminates,
  false positives.
- `foot_clearance`: linear-ramp-then-flat-cap replaced with a symmetric bump
  peaking at 10cm — now penalizes lifting a foot *higher* than 10cm too,
  which it never did before.
- `postlegdofpos`/`postupperdofpos`/`postwaistdofpos`: retargeted from the
  crouched `default_joint_pos` to a straight-leg/45-deg-arms/centered-waist
  stance (`_POST_SAVE_STANCE_MAP`). `postwaistdofpos` weight 1.0→3.0.
- New `trailing_foot_forward_continuous` (+3.0, always active): trailing
  foot rewarded for pointing along the robot's forward axis.
- New `postleadfootorientation` (+2.0, `behind`-gated): leading foot
  rewarded for rotating back to forward after the save.
- `success`: doubling retiered off `softstop`/`cleanstop` (1.0x/2.0x/3.0x)
  instead of `stopball` — **the metric's scale changed**, pre- and
  post-this-fix `Episode_Reward/success` values are not directly comparable.

### What to watch once a training run has real data

- **`feet_slippage`**: should no longer crash hard (e.g. 2.65→0.13-style
  drops, see `docs/BugFixes.md` 2026-07-23 entry) exactly at genuine save
  contact moments. Some residual dips from missed sensor frames are
  expected — the question is whether the *floor* it settles at during
  active saves rises compared to pre-fix runs, not whether it's perfectly
  smooth.
- **`foot_clearance`**: expect this value to be *lower* than before for any
  policy that was lifting feet well past 10cm (that's the fix working as
  intended, not a regression) — check whether mean foot-lift height (not
  just the reward number) actually converges toward ~10cm over training,
  not just whether the reward number itself looks different from history.
- **`success`**: don't compare raw magnitude against pre-2026-07-27 runs
  (different scale/semantics now). Instead watch whether it reaches its
  3.0x tier at a non-trivial rate as training progresses — if it stays
  pinned near 1.0x-2.0x for a long time, that says `cleanstop` (the hardest
  save-quality event) essentially never fires, which is useful signal on
  its own about save quality, separate from whether `success` itself
  "looks converged."
- **`trailing_foot_forward_continuous` / `postleadfootorientation`**: brand
  new, no historical baseline to compare against. Watch that both trend
  upward from ~0 (their untrained starting point) rather than staying flat
  or going negative — negative/flat over many thousand iterations would
  mean the new incentive isn't being learned, worth a follow-up look.
- **`postlegdofpos` / `postupperdofpos` / `postwaistdofpos`**: the retarget
  (crouch → straight-leg stance) should make these noticeably easier to
  hold than before. Compare against the pre-fix stuck floors reported in
  `docs/BugFixes.md`'s "unify post-save recovery" entry
  (`postlegdofpos≈0.07`, `postwaistdofpos≈0.31`, both from the
  `blue_v2_nearstickreach_2026-07-26` run) — if they're still stuck near
  those same floors after a comparable number of iterations post-fix, the
  stance retarget didn't actually solve the "policy won't hold this pose"
  problem and needs a fresh look, not just a bigger weight.
- **General**: multiple reward *shapes* changed (not just weights) in this
  batch — if resuming an existing checkpoint rather than starting fresh,
  be alert for the kind of value-function discontinuity this project has
  hit before when the effective reward function shifts under a
  policy trained against the old one (see the 2026-07-05 multi-disc
  `schedule="adaptive"` resume-collapse entries in `docs/BugFixes.md` for
  what that failure mode looks like in practice, though the cause there was
  different).
