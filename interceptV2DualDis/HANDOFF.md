# Handoff Notes

Running log of what to watch for after recent reward changes, for whoever
(human or AI) picks up training monitoring next. See `docs/BugFixes.md` for
full rationale/evidence on each item — this file is just the "what to watch"
distillation.

## 2026-08-06 (later same day) -- far-region arm-swing batch: postshoulderdofpos, penalize_arm_above_shoulder always-on, far bound 1.1->1.0, arm-column damping, + a 3rd wiring-bug fix

Five changes landed together, all stemming from a `model_12500.pt`
(`6144_shoulderscopewiring_2026-08-06`) force-region probe that found the
shoulder had zero pose-recovery pull anywhere in the episode. See
`docs/BugFixes.md`'s three 2026-08-06 entries for full rationale/evidence.
**None of this has been validated against a live training run yet.**

### What changed
- `penalize_arm_above_shoulder`: no longer gated to the "steady" (>20 steps
  post-save) window -- now fires for the whole post-save (`behind`) window.
- New `postshoulderdofpos` (weight 5.0, `kernel_scale=0.15`,
  `during_scale=0.3`): separate reward/kernel for the 4 shoulder joints,
  deliberately not merged into `postupperdofpos` to avoid re-mixing
  shoulder's larger error into elbow's kernel.
- Far outer region bound narrowed 1.1m -> 1.0m (`regions.py`, `events.py`,
  `goalkeeper_env_cfg.py` train+play, all 3 synced locations).
- 6 far-region AMP reference clips (`new_doublestepleft/right_{booster_t1,
  short,wide}.npz`) had their 8 arm `joint_pos` columns smoothed (9-tap box
  filter) -- mean arm speed down 31-55%, peak down 17-35%. True originals at
  `motions/data_pre_armdamp_backup/` (local scratch, git-ignored, not
  committed -- recoverable via git history if deleted).
- **Separately found and fixed:** `postupperdofpos`'s earlier same-day
  "shoulder-scope wiring fix" never actually worked -- mjlab's
  `RewardManager` only resolves `SceneEntityCfg` objects passed explicitly
  via `params`, never a function's own default argument. `postupperdofpos`
  was silently running on ALL 21 joints (not 4) this whole time despite that
  fix's own verification claiming otherwise. Both `postupperdofpos` and the
  new `postshoulderdofpos` now pass `asset_cfg` explicitly. Checked all 33
  scoped-default rewards in `rewards.py` for the same bug class -- this was
  the only real instance.

### What to watch once a training run has real data
- **`postshoulderdofpos`**: brand new, no baseline. Watch it trend upward
  from its untrained starting point. If it stays near 0 or the arms still
  look wild specifically DURING the dive (not just post-save), the
  conservative `during_scale=0.3` may be too weak -- consider raising it
  gradually (mirrors `postupperdofpos`'s own 0.3->0.5->0.8->1.0 history, but
  don't assume that same endpoint transfers -- shoulder's excursion is
  larger).
- **`postupperdofpos`**: now genuinely elbow-only for the first time (prior
  runs, including everything up through `model_12500.pt`, were secretly
  whole-body). Expect this run's `postupperdofpos` values to look
  DIFFERENT from every prior checkpoint's history in `docs/BugFixes.md` --
  that's the fix working, not a new regression. Compare against the
  `kernel_scale=0.15` sensitivity math in that entry, not against old
  checkpoint numbers directly.
- **`penalize_arm_above_shoulder`**: watch for the exact failure mode its
  original 20-step gate was designed to prevent -- if `footreach`/
  `Episode_Termination/ball_exit`/`Train/mean_episode_length` regress
  compared to pre-this-batch runs, the always-on gate may be fighting
  legitimate dive/immediate-post-save balance motion, and should get a
  narrower window back (not necessarily the old 20-step one).
- **Far outer bound 1.0m**: `far_travel_curriculum`'s cu=0 seed gap
  self-adapts (proportional formula) -- no separate check needed beyond
  confirming `env._far_outer` actually reaches 1.0 at full curriculum
  (`ball_difficulty` saturated).
- **Arm-column damping**: watch (via `sgk_play --force-region left_far`)
  whether the AMP-driven pre-save arm motion looks less jerky specifically
  during the approach/dive -- this only touches the reference clips, so any
  improvement should show up as smoother tracking toward a still-genuine
  double-step arm swing, not a frozen/flat one. If arm-swing quality gets
  WORSE, note the 2026-07-15/16 history of this exact region's AMP-arm
  masking flipping back and forth -- this is a new, different lever (motion
  smoothness, not masking), but worth comparing against that history if it
  doesn't help.
- **General**: this is the third same-session change to
  `postupperdofpos`'s wiring specifically (shoulder-scope default 2026-08-03,
  wiring-not-applied fix + params-passthrough fix both 2026-08-06) --
  worth extra scrutiny on this term's actual behavior in the next run
  before trusting any further tuning built on top of it.

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

## 2026-07-28 -- shin/knee wrong-foot blind spot fix (pulled), + new `postheadingorientation`/`arm_dof_vel`

- **`penalize_wrong_foot_ball_contact`** (pulled from `f7856a2`): now also
  catches wrong-side SHIN/KNEE contact, not just foot geoms — this was a
  real detection blind spot (17 genuine events found completely invisible
  to the old sensor via real-checkpoint replay). Same -100 weight, no new
  tuning. Watch whether this term's logged magnitude actually *increases*
  post-fix (it should — it's now catching contact it was blind to before)
  and whether the underlying behavior (catching/resting with the trailing
  leg) declines over training now that it's a visible penalty.
- **New `postheadingorientation`** (+2.5, `behind`-gated): whole-body yaw
  heading recovery — no historical baseline. Watch that it trends upward
  from ~0 like the other new post-save terms did on 2026-07-27, and
  separately watch (via play/viewer) whether the reported left/right hip
  drift after a save actually decreases.
- **New `arm_dof_vel`** (-5e-3, always active, arm joints only): first
  empirical weight guess, real risk it's too strong and fights legitimate
  in-dive counterbalance motion rather than only damping post-save idle
  swinging. Watch `foot_proximity`/`softstop`/`single_foot_save` for any
  regression (would suggest the arms needed that motion for balance during
  the save itself) alongside whether visible arm swinging actually
  decreases.

## 2026-07-29 -- reverted arm_torque_limits/arm_action_rate_l2/arm_action_acc_l2 (pulled 2026-07-28), postupperdofpos during_scale 0.3->0.5

The pulled arm-penalty batch (`e76ac55`) regressed footreach/ball_exit/
episode-length and caused `postupperdofpos` itself to collapse ~10x
(confirmed via matched-iteration comparison, `docs/BugFixes.md`) — the 3
movement/effort penalties were fighting legitimate dive counterbalance
motion. Reverted all 3; kept `arm_dof_vel` (small, G1-grounded, minor
contributor) and raised `postupperdofpos`'s pre-save strength instead,
since that's the mechanism that actually targets arm *pose* rather than
arm *movement*.

**What to watch on the next run:**
- `footreach`/`Episode_Termination/ball_exit`/`Train/mean_episode_length`
  should recover toward (or exceed) the pre-2026-07-28 baseline
  (`6144_headingarmfix_2026-07-28`) at matched iterations — if they don't,
  the regression wasn't (only) about the 3 reverted terms.
- `postupperdofpos` should no longer collapse the way it did — watch it
  stay comparable to `postlegdofpos`/`postwaistdofpos` rather than falling
  far below them again.
- The actual stated goal (arm not ending up behind the body post-save) is
  a play/viewer observation, not a wandb metric — worth a live check once
  there's a checkpoint worth watching.

## 2026-07-29 (same day, follow-up) -- postupperdofpos kernel_scale 1.0->0.15 (far-region gradient was fully saturated)

Comparing `armrevert`'s `model_19000` against itself with `--force-region`
pinned to `left_near` vs `left_far` showed the arm-recovery gradient was
essentially dead for far regions specifically — post-save arm error was
~50x larger there (0.155 near vs 7.81 far) and the old `exp(-1.0*err)`
kernel had already collapsed to ~0.0004 at that magnitude, meaning
`postupperdofpos` was providing zero real signal for far-region recovery
regardless of `during_scale`. Lowered `kernel_scale` to 0.15 so the reward
doesn't vanish at the error magnitudes far dives actually produce
(`exp(-0.15*7.81)=0.31`) while barely changing near-region's near-ceiling
value (`exp(-0.15*0.155)=0.977`).

**What to watch on the next run:**
- Re-run the same `--force-region left_far` vs `left_near` probe against a
  later checkpoint (the methodology is in `docs/BugFixes.md` — reset,
  step with the trained policy, read `env._post_save_stance_target` vs
  actual arm joint positions, split by `_softstop_flag`) to confirm
  far-region post-save `postupperdofpos` is now actually nonzero and
  climbing, not just that the standalone kernel math checks out.
- Watch (via `sgk_play --force-region left_far`) whether the arm actually
  stops looking "really weird" post-save on far shots specifically —
  that's the real test, not any wandb aggregate (which averages all 4
  regions together and would dilute a far-only improvement).
- `postlegdofpos`/`postwaistdofpos` were NOT touched and use the same
  `exp(-k*err)` shape — worth the same near/far probe if far-region leg or
  waist recovery also looks off, since they likely share this failure mode.

## 2026-08-01 -- far-region AMP clips got +8deg forward torso tilt (pulled); postupperdofpos during_scale 0.8->1.0 (no more pre-/post-save distinction)

The pulled clip edit is display/data-quality only — root orientation in
these NPZ files isn't read by `reset_from_motion_data` or the AMP
observation (joint_pos/joint_vel only), so it should have **zero direct
effect** on trained-policy behavior. If a later run's fall-back tendency
actually improves, that's not from this pull; look elsewhere (reward
shaping, RSI, curriculum).

`postupperdofpos`'s `during_scale` is now 1.0 — full strength the entire
episode, pre- and post-save alike, no discount during the approach/dive.
This is the fourth retune of this parameter (0.3→0.5→0.8→1.0). **The
specific risk to watch**: `during_scale` existed below 1.0 in the first
place because pulling the arms toward a static target pose *during* an
active dive can fight the counterbalance motion a real save needs — the
exact mechanism that got `arm_torque_limits`/`arm_action_rate_l2`/
`arm_action_acc_l2` reverted on 2026-07-29. Watch `footreach`/`ball_exit`/
`Train/mean_episode_length` on the next run for the same regression
signature seen back then; if they drop, `during_scale=1.0` is too strong
and needs to come back down.
