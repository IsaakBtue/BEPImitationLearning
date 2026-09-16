# SimpleGoalKeeper — CLAUDE.md

## Response Style

- Keep answers short.
- Plain English, simple language — no jargon-heavy or overly technical phrasing unless the user used the term first.
- Use bullet points, not prose.
- Just tell the user what to do — skip explanation/rationale unless asked.
- **Hard cap: a handful of short bullets per response, not a report.** No multi-paragraph investigations, no restating evidence already given earlier in the conversation, no hedging/caveat paragraphs. If more detail is truly needed, give the one-line conclusion first and wait to be asked before expanding. (2026-08-13 user request: "your claude outputs of texts are too long.")
- **Reinforced 2026-08-23** ("also answer shorter in your outputs") — still too long even during research/investigation turns. Applies to research/diagnosis responses too, not just implementation ones.
- **Reinforced 2026-09-04** ("write in plain english and write short answers because these are too long and too complicated") — third time this has needed saying. Explain mechanisms (e.g. "what is catchstep") in one or two plain sentences, not a docstring-style breakdown. Don't dump full before/after tables or multi-paragraph investigation writeups unless explicitly asked for a deep dive — short summary first, offer to expand.
- **Always give the plot link (2026-09-16, "always give me the new plot"):** any time a code change affects a value/shape an already-published artifact plot shows (e.g. a reward-shaping constant the plot visualizes), update and republish that plot and put its URL in the response — don't just say "updated" or "republished" without restating the link, even when the URL is unchanged from before.

## Phase 1 Scope

**Phase 1 focuses exclusively on foot-based goalkeeping.** The robot must intercept incoming balls using its feet only. There are no hand rewards, no arm-specific observations, and no hand-related AMP body names.

Hand rewards and arm observations are explicitly out of scope and should not be added until Phase 2 is started with a new design review.

## Project Purpose

Standalone, simplified goalkeeper training environment for the Booster T1 humanoid using:
- **mjlab** (MuJoCo-Warp RL framework)
- **beyondAMP** (simplified AMP integration — no custom 6-discriminator runner)
- **21-DOF headless T1** (head joints removed from action/observation space)

## Project Origin

`SimpleGoalKeeper` is a **foot-only** experimental track, distinct from `Humanoid-Goalkeeper` (the original paper) and `Imitationlearningbooster` (the T1 hand-catching port). Both of those use hands/arms; here the robot may only use its feet. AMP motion priors encourage natural bipedal motion while the task rewards focus entirely on foot-ball contact.

## Design Rule: Always Verify Against Humanoid-Goalkeeper First

**Before adding, changing, or removing ANY reward term, spawn parameter, observation, termination condition, curriculum stage, or training hyperparameter**, you MUST:

1. **Read the upstream G1 code** in `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/` — specifically `base/legged_robot.py` (all reward functions, reset logic, behind/deflection conditions) and `g1/g1_29_config.py` (all weights, ranges, thresholds).
2. **Confirm the G1 equivalent exists** and quote the exact line/function. If it does not exist in G1, that is a red flag — explain why it's needed.
3. **Explicitly justify every divergence** — state WHY the upstream G1 value is wrong for this setup (MuJoCo vs PhysX, feet vs hands, mjlab API difference, etc.).
4. **Document it** in the "Divergences from G1 Upstream" table below.

This rule exists because G1 is the only proven working reference. Every undocumented divergence is a potential source of a local optimum or training failure. "It seems reasonable" is not a justification — G1 must be the baseline.

## Training / Play Parity Rule

**Training and play ball spawn parameters must always match.** Whenever `dist_range`, `speed_range`, `y_start_range`, `y_end_range`, or `spawn_z` are changed in the training `reset_ball` block (`goalkeeper_env_cfg.py` around line 515), the play block (around line 589) must be updated to the same values in the same commit. A policy evaluated on a different distribution than it was trained on gives misleading results.

## Visualization Honesty Rule

**Never let `play.py` (or any other viewer/diagnostic tool) render something that could be mistaken for what the model actually trains on, without clearly labeling the difference.** `play.py`'s ghost overlay correctly replays a reference clip's *real, full* root+joint data (`GhostMotionCommand`'s own docstring: "world space, independent of the robot") — that is accurate for judging what a clip looks like, but AMP's discriminator only ever observes `joint_pos`/`joint_vel` (see `goalkeeper_multidisc_amp_cfg.py`'s `_MULTIDISC_AMP_OBS_TERMS`) and never root orientation. A root-orientation edit (e.g. a torso-lean adjustment) can look completely convincing on screen while being 100% invisible to AMP — confirmed directly (2026-08-01): `joint_pos` byte-identical, yet Trunk world-pitch measured `0.00°` once root was forced level (exactly what AMP effectively "sees"), vs. `+8.05°` in the real playback.

- If a tool/flag exists specifically to preview or debug a training-relevant signal (an AMP reference clip, an observation term, a reward target), it must either (a) render *only* the actual observed quantity, or (b) offer an explicit, clearly-labeled mode that does — never silently blend "looks right to a human" with "is what the network sees."
- **Pick a physically-grounded anchor when stripping root information, not an arbitrary one.** The first attempt at an "AMP-eye view" forced root orientation to identity — mathematically valid (it does cancel out root, leaving a pure function of joint angles), but it silently broke this project's own feet-always-flat invariant (every frame, both feet, `tilt_xy ~ 0`), since the recorded joint angles were tuned assuming the real root, not an arbitrary level one. The render looked "unbalanced"/leaning for reasons that had nothing to do with the actual edit being checked. Fixed by re-anchoring to whichever foot is currently planted instead (same root-cancelling math, any body in the chain works) — the foot is already known-flat/grounded, so the rendering stays physically sensible. Expect a visible jump at each stance-foot switch; that's inherent to a foot-anchored view, not a bug.
- Use `sgk_play ... --amp-eye-view True --motion-file <clip>` (`scripts/amp_eye_view.py`) to see any reference clip exactly as AMP does (re-anchored to the planted foot each frame, `joint_pos`/`joint_vel` untouched) before concluding a reference-motion edit will affect training.
- When editing AMP reference data, verify under **both** lenses: the real viewer (does it look right to a human) and the AMP-eye view (does the change actually reach the discriminator). An edit that only changes the first is a visualization/diagnostic-only change and must be documented as such (see the 2026-08-01 root-tilt entry in `docs/BugFixes.md` for the pattern).

## Git Commit & Push Rule

**Every push must include ALL modified files** (`git add -A`) unless the user explicitly says otherwise. Never leave uncommitted working-tree changes behind when pushing.

**Before pushing, check for anything still pending confirmation.** When the user says "git push," scan the conversation for reward-code changes that were listed for approval (per the Change Approval Workflow above) but never got an explicit yes/no. Push only what was actually confirmed — don't silently apply an unconfirmed change just because a push was requested, and don't silently drop it either. List the still-pending items back to the user in short bullets so they can confirm, deny, or leave them for later. (2026-09-16 incident: a "git push" was correctly scoped to only the confirmed changes, but nothing flagged the still-pending ones — a different Claude session later reported a value as unchanged and it took a manual check to confirm it really was still pending, not lost in the push.)

## Documentation Update Timing

**Only update `docs/BugFixes.md` and other `.md` files (including this file's own divergence-table rows) as part of actually pushing** — not after every individual code edit during a session. Write the doc entries as a batch alongside the commit that includes them, not one entry per fix as the session goes. **Only push when the user explicitly tells you to push** — never proactively, even after a batch of validated, tested changes. (2026-09-11 user request: "only update the bugfixes and all other .md only when you also take the action to push changes, and only push changes when the user tells you to" — this reverses this session's own prior pattern of documenting immediately after every fix.)

## Change Approval Workflow

**Before making ANY code changes**, you MUST:

1. **List all changes** you plan to make in a clear, bullet-point format
2. **Show this list to the user** and wait for explicit approval
3. **DO NOT apply changes until approved**
4. **Only after approval**: apply changes and document them

This prevents accidental modifications, keeps the user informed of scope, and ensures changes match the actual request.

**UPDATED 2026-09-16 (user request):** "when i ask something directly to change something you go ahead do it, but when its more like a research question or what is better to do kinda question then you wait for confirm but stuff like this just do it because i asked you to." This narrows steps 1-3 above: they apply to **decisions Claude would be making** (an open-ended "what should we do about X" / "what's better" question, or anything where the specific value/approach isn't already fully pinned down by the user's own words) — not to a fully-specified direct instruction. The test: does the message already name the exact parameter/file AND the exact new value/formula ("make X 0.17", "put orange at 0.14", "use the tanh")? If yes, apply it immediately, no list-and-wait round-trip. If the user is asking what to do, or the instruction leaves the actual value/approach for Claude to pick, list it and wait as below.

**A request phrased as a direct instruction does not waive this** (2026-08-07 incident, prior to the 2026-09-16 update above): the user asked to "add the penalty again with the knee contact because it converged learning to touch the chin" — a specific, direct-sounding instruction — and Claude applied it straight to `rewards.py`/`goalkeeper_env_cfg.py` without first listing the change and waiting for confirmation. The user had to explicitly stop and ask for a revert. Kept here as the reasoning behind why this ever needed saying — the 2026-09-16 update above is a deliberate, explicit relaxation of it for the fully-specified case, not a reversal of the underlying caution: an ambiguous or Claude-decided value still needs confirmation.

- **Reward/training-affecting code** (anything in `rewards.py`, reward weights/params in `goalkeeper_env_cfg.py`, observation/termination/curriculum logic, or anything else that changes what a future training run optimizes for) — apply immediately when the user's own message already fully specifies the exact parameter and exact new value/formula; list the specific function/file/parameter and the exact before→after change and wait for explicit confirmation only when the value/approach isn't already fully pinned down (a research/"what's better" question, or an instruction that leaves a real choice to Claude). These changes are expensive to discover were wrong (only visible after hours of training) and target a currently-running or about-to-run process, not just the repo state — so still document the exact change made (see Documentation Update Timing) even when applied without a pre-approval round-trip.
- **Viewer/diagnostic-only code** (P-panel promotions, plot additions, debug print statements, `sgk_play`-only visualizations) — no training effect, safe to apply directly without a pre-approval round-trip, exactly as this file already treats WandB/logging-only changes.

If a request is genuinely ambiguous about which category it falls in, ask.

**Mandatory skill load for reward-shaping code:** before writing, editing, or reviewing anything in `rewards.py`, any reward weight/param in `goalkeeper_env_cfg.py`, or any reward-affecting observation/termination/curriculum logic, load `.claude/skills/reward-shaping-scene-entity-cfg/SKILL.md` via the Skill tool. It covers two distinct pitfall classes, both with real incident histories in this project: `SceneEntityCfg`/`asset_cfg` resolution gotchas (Part 1), and decaying/shrinking-target reward design (Part 2 — division-by-target instability, unsigned-distance/directional-condition mismatches, overlapping geometric constants, shared-aggregate state contamination across bodies, unaccounted geometry baselines, duplicate reward mechanisms). Part 3 of that skill makes verification itself mandatory for any shape-based reward change: graph the intended curve/shape before implementing, then build (or reuse) a static teleport probe script (`scripts/probe_leading_foot_lift.py` is the reference pattern) AND a live animated `--agent` for `sgk_play` (`scripted_blue_approach` is the reference pattern) so the change can actually be watched, not just reasoned about. A passing test suite and clean `ast.parse` are necessary, not sufficient, for this class of change — do not report a reward-shaping change as complete without having done all three verification steps.


## Divergences from G1 Upstream

*(Full chronological history for every row below lives in `git log`/`git show` on this file and in `docs/BugFixes.md`. This table states current values only.)*

| Parameter | G1 value | SimpleGoalKeeper value | Justification |
|-----------|----------|------------------------|---------------|
| `bad_orientation` termination formula | `norm(projected_gravity[:, :2]) > 0.8` (`legged_robot.py:257`), sourced from `"pelvis"` | **G1-exact:** `norm(projected_gravity_b[:, :2]) > 0.8`, sourced from `root_link_quat_w` | Was wired to mjlab's built-in arccos-based formula (a different physical quantity); ported G1's literal XY-norm formula instead. |
| Robot reset (RSI) — legacy | Fixed standing pose | Old tier-pool NPZ mechanism, dead for training since 2026-07-01, kept only for the `sgk_play_rsi` diagnostic script | See the "RSI mechanism" row below for actual current training-time behavior. |
| Ball spawn height | 0.15–0.4 m (chest-compatible) | 0.05–0.35 m (foot-to-shin) | Ball spawn was too high, forcing upper-body contact — lowered for feet-only goalkeeping. |
| `ang_vel_xy` weight | -0.1 (roll+pitch only) | -0.1 (same) | Restored after drifting to -0.5 with no G1-based justification. |
| `ang_vel_z` (yaw) | Not penalized (free) | -0.5 | G1 yaws to extend hand reach; feet have ~10cm reach radius, so yawing only enables spinning exploits. |
| `_ball_is_behind` threshold | `delta_vx > 2.0` (`legged_robot.py:1377`) | `delta_vx > 1.0` | MuJoCo soft contact produces smaller impulses than PhysX — 2.0 m/s never fired on slow-ball saves. |
| `ball_positive_vx` | Not in G1 | Removed | Caused ball-chasing after save (continuous reward never deactivates). |
| Ball spawn frame | Robot-local | World (global) frame | Keeps `stopball`/`stayonline`/`ball_exit_termination` (all world-X) aligned with the ball trajectory. |
| Ball spawn angular velocity | N/A | Pure-rolling ω at spawn | Zero-ω spawn caused a sliding phase that lost enough speed to false-positive `stopball` with no robot contact. |
| Ball/save contact restitution | Per-shape material property, randomized every reset | `randomize_foot_ball_restitution` (dampratio ∈ U(0.35,1.0) on foot geoms, train only) | MuJoCo has no literal restitution scalar and the foot geoms' `priority=1` makes the ball's own solref irrelevant to contact — randomizing the foot achieves G1's actual per-episode bounciness effect. |
| RSI foot floor penetration | N/A | +0.030 m root Z offset in `MotionResetManager` | NPZ foot capsule geometry sits 0.03m below the body-frame floor reference; uncorrected this caused a violent reset "pump-up." |
| RSI motion selection | Random frame from all 6 motions | Distance-conditioned: ball crossing Y → (side × step-count) pool | Random selection gave the policy no correlated hint about which motion matches the spawned ball. |
| Ball spawn `y_start_range` | ±0.8 m (ILB) | ±0.3 m, sampled directly at every difficulty (not curriculum-lerped) | ±0.5m lateral spawn + wide target created unreachable diagonal shots; G1 samples spawn Y from a fixed range never touched by curriculum. |
| Ball spawn `y_end_range` (goal target) | Motion-dependent, wide, genuinely curriculum-scaled by G1's `command_ranges` | ±1.0 m with ±0.15 m dead zone at d=0, curriculum-scaled | Confirmed via direct G1 code re-derivation that this dimension's curriculum coupling IS a real G1 match. |
| `stopball`/`softstop` `in_front` threshold | N/A | `ball_x_local > -0.3` | Deflection accumulates gradually; a strict `>0.0` check missed saves completing just past the goal line. |
| RSI in play mode | RSI active | RSI active in play mode too | Play mode was accidentally popping the RSI reset event; removed the pop so play matches the training distribution. |
| `feet_slippage` ball-contact gate | N/A | Suppressed when ball within 0.5m of either foot | The ground-contact sensor can't distinguish ball contact from ground contact; without gating this fought the foot sweeping into the ball. |
| `footreach`/`foot_proximity` phase2 target | Live ball when close, frozen when far | Frozen crossing_y when far (≥0.5m), live ball Y/Z when close — matches G1's own switch exactly | Restored after an earlier fix had frozen the target for the whole episode, breaking phase2 coupling to the real ball. |
| `footreach` `vel_sigma` multiplier | 3.0 (max 10×) | 3.0 (max 10×), decays linearly to neutral over the last 0.30m before the green target | G1 catches with hands (full speed at contact is fine); this project strikes a free ball with a foot, where full speed means kicking it away instead of a controlled save. |
| Two-stage blue/green waypoint mechanism | N/A (direct reach only) | v2 mechanism: `_get_reach_target_y`, `blue_ball_landed`, `blue_overshoot_penalty`, `blue_stick_landing` — midpoint targeting until the assigned foot genuinely lands there, then switches to the true crossing point | Extensively retuned across many sessions (landing radius, distance-from-start/green floors, settle-count leakiness). Current landing_radius default 0.12m (curriculum-eased). See `docs/BugFixes.md` for the full tuning history. |
| Trailing-foot positional targeting ("orange ball") | N/A | `_get_orange_reach_target_y`, `orange_foot_proximity`, `orange_ball_landed`, `orange_overshoot_penalty`, `orange_stick_landing` — trailing-foot mirror of blue, target derived relative to blue's own position with a guaranteed minimum gap | Trailing foot had no positional target of its own; `trailing_foot_forward_continuous` is orientation-only. |
| Trailing-foot second waypoint ("red ball") | N/A | `_get_red_reach_target_y` etc. — active only once both blue and orange are genuinely landed, anchored near green with a capped offset | Closes the gap where orange never graduated to a live-ball target once landed. |
| Trailing-foot urgency reward (`trailing_foot_reach`) | N/A | Sigmoid-reach × `vel_sigma` urgency for the trailing foot's orange→red sequence | Mirrors `footreach`'s urgency mechanism but without the ball-position-specific phase logic (orange/red are fixed waypoints, not live-ball targets). |
| Sequence-promptness reward (`sequence_promptness`) | N/A | One-shot bonus for completing the full blue→orange→red→save relay with margin, paid once at the save moment as an average of completed-stage margins | No G1 equivalent — G1 has no multi-stage waypoint sequence to be prompt about. |
| `stopball` `delta_vel_threshold` | 2.0 m/s | 0.6 m/s | G1's 2.0 m/s is calibrated for high-speed hand catches (3-6 m/s); SGK balls max ~2.0 m/s, so 2.0 would require a near-perfect stop. |
| Ball spawn timing (speed) | `t_flight = 0.4 + 0.6×rand()`, hardcoded, no curriculum | `t_flight_range` sampled directly (hard end) with a `domain_rand_curriculum`-driven lerp from an easier range at low difficulty (`_EASY_T_FLIGHT_RANGE`, currently (0.75, 1.45)) | Deliberate divergence beyond G1 (an easy on-ramp), not a parity fix — see `docs/BugFixes.md`, 2026-09-10/11. |
| Ball-spawn curriculum coupling | Only the target/catch window (Y/Z) is curriculum-scaled; spawn distance, spawn Y, and `t_flight` are hardcoded full-range constants | Only `y_end_range` (and region-specific `far_travel_curriculum`) is curriculum-scaled; `dist_range`/`y_start_range`/`t_flight_range` sampled directly from their full range | Re-derived from G1 source line-by-line; confirmed G1 never curriculum-scales spawn distance/reaction-time but DOES scale the target window. |
| `stopball` curriculum | `weight = stop_init × (1 + 0.5 × cu)` | base 20 → max 50 at cu=3 | Previously had no curriculum at all; ported G1's mechanism. |
| Episode length (training) | ILB: 3s | SGK: 3s | 6s wasted compute — `footreach` zeros post-deflection, diluting `stopball`'s per-step signal over the unused remainder. |
| RSI split | 100% RSI (previous SGK) | **0% RSI (100% default-standing-pose branch)** | Superseded twice: first the `continue_keep` donor-copy branch was region-scoped (a deliberate, documented divergence from G1's own unscoped behavior); then a true-motion-sampling RSI attempt was tried and reverted; `rsi_fraction` is now 0.0, matching HUSKY's own approach (no RSI, fixed start every episode) rather than DeepMimic's RSI-favorable evidence. See `docs/BugFixes.md`, 2026-09-13. |
| `penalize_sharpcontact` threshold | 1000 N | 1800 N | Progressive widening across many training runs after normal aggressive stepping repeatedly false-positived the penalty. |
| `ball_difficulty` curriculum `ep_len_divisor` | 48 | 47 | Matches every other reward curriculum's divisor (were desynchronized in WandB otherwise). |
| WandB logging | N/A | Batch per iteration (single `wandb.log` call) | Per-metric calls each auto-incremented WandB's step counter, scattering one iteration's metrics across multiple x-axis positions. |
| `airborne_at_save` reward | N/A | Removed | Secondary quality metric judged to add training noise without clear early-stage benefit. |
| RSI mechanism (`reset_from_motion_data`) | `continue_keep` branch (80%): unscoped `torch.randint` donor copy, no clamp; else branch (20%): scale/offset default pose, clip to soft joint limits | Literal G1 port for both branches (after fixing 4 found divergences: missing clamp, missing else-branch randomization, wrong 50/50 split, hard- vs soft-limit clip) | The donor-copy branch is now ADDITIONALLY region-scoped (deliberate SGK divergence, RSI's own purpose is defeated by a wrong-region donor) — but currently dead code, since `rsi_fraction=0.0` (see "RSI split" row above). |
| AMP discriminator motion sampling | N/A | Double/triple-step files boosted 4× vs. uniform-by-frame-count | These motions were underrepresented (~36% of frames) relative to their training importance. |
| AMP expert-transition playback speed | Randomized position `motion_ids + ratio`, `ratio=(fps/env_fps)*U(0.25,1.25)`, floor/ceil-interpolated | Now G1-exact (was fixed adjacent-frame `t,t+1`, zero speed diversity) | Gave the discriminator zero velocity-magnitude diversity to learn from. |
| AMP normalizer transition coverage | Updates using the full concatenated `[state, next_state]` vector every step | Now G1-exact in effect (was state-half only) | The running stats used to normalize every `next_state` sample were computed only from `state` samples. |
| `action_rate_l2`/`action_acc_l2` weight mapping | `_reward_smoothness` (2nd order), weight -0.1 | `action_acc_l2` (2nd order, G1's true match) = -0.1; `action_rate_l2` (1st order, no G1 equivalent) = -0.05 | An earlier audit had assigned G1's weight to the wrong (1st-order) term — swapped. |
| `feet_slippage` kernel steepness | `exp(-10*contactvel)` | `exp(-10*contactvel)` (was `exp(-50*contactvel)`) | A porting error since this function's first commit; docstring already claimed -10, code didn't match. |
| `feet_slippage` "in contact" source | Any ground contact (feet never touch the ball in G1) | Now excludes genuine foot-ball contact via the `ball_contact` sensor | SGK's feet ARE the ball-contact effector — "any contact" wrongly counted a genuine save impact as slippage. |
| `dof_vel_limits` threshold | Per-joint URDF limits × 0.9 soft margin | Now per-joint × 0.9, sourced from T1 motor specs (was a flat 10 rad/s cap) | T1's MJCF defines no joint velocity limit at all; the flat cap silently never fired for T1's slower joints. |
| Save-event reward group peak magnitude | `_reward_stopball` alone, curriculum-scaled 100→250 | 6-term group (`stopball`+`softstop`+`single_foot_save`+`cleanstop`+`inner_face_orientation_save`+`foot_inner_face_continuous`) combined peak = 250.0 | Scaled the group's base weights down so the combined peak matches G1's ceiling exactly. |
| `success` reward | `(success_flag+1)*(dist<0.15)`, weight 5→12.5 | Ported: uses `env._sb_flag`, foot-to-crossing_point distance, same 0.15 threshold and weight | Was missing entirely from SGK's reward table before this port. |
| `success` reward — v2 landing gate | N/A | Also gated by `landing_ok` (same gate `stopball`/`softstop` use) | Base term could fire just from beelining to the final target, bypassing the two-stage waypoint mechanism entirely. |
| `success` reward — flag source | Set inside `_reward_stopball` | Retiered off `softstop`/`cleanstop` instead (1.0x/2.0x/3.0x ladder) | Deliberate divergence beyond G1 — ties the doubling to actual outcome quality (a clean stop), not the loosest deflection event. |
| Trailing-foot orientation shaping | N/A (no hand-orientation reward of any kind in G1) | `trailing_foot_forward_continuous` (always active) + `postleadfootorientation` (behind-gated, mutually exclusive with the leading-foot term) | The trailing foot previously had zero orientation shaping anywhere in the episode. |
| Leading-foot save-orientation target angle (`_FOOT_TARGET_ANGLE_DEG`) | N/A | **70° off forward** (current) | Extensively retuned (0.01→80 across many live-checkpoint replays balancing Hip_Yaw physical reach limits against whole-body yaw-spin reaction torque). See `docs/BugFixes.md` for the full history. |
| Leading-foot PRE-LANDING target angle (`_PRE_LANDING_TARGET_ANGLE_DEG`) | N/A (no pre/post-landing phase distinction exists in G1) | **15°** (current) | Before blue lands, the foot targets this shallower pre-rotation instead of the full block-posture angle; gated so narrow crossings (no blue waypoint) target the full angle immediately. History: 0°→20°→45°→20°→15°. |
| `postleadfootorientation` ground-contact gate | N/A | Airborne-only (zero while grounded) | An unconditional version pulled an already-planted foot to keep rotating, causing floor slippage. |
| Post-save foot airtime | N/A | `postsave_foot_airtime` — flat, time-boxed (~0.6s) bonus for staying airborne right after the save | Gives pressure to delay landing so there's hangtime to rotate during, without reopening the "post-save hopping" failure mode `foot_clearance` guards against. |
| Near/far region boundary + far outer bound | N/A (region system has no G1 equivalent) | 0.6m / 1.0m | Narrowed from earlier wider values after live evidence; kept in sync across `regions.py`/`rewards.py`/`events.py`. |
| Near-region `y_end` inner floor + sampling shape | N/A | Floor ~0 (was 0.15m), sampling skewed toward the outer/harder end (`u**0.5`, was uniform) | Center-of-goal near-region spawns had little to learn from; skewed toward the more informative outer end. |
| `penalize_wrong_foot_ball_contact` sensor coverage | N/A | Asymmetric: WRONG side = whole leg (sole/shin/knee), CORRECT side = knee only, chin/head via proximity check | Extensively iterated (7+ same-day fixes) after live evidence that the raw contact sensor under-detected genuine wrong-side touches. |
| Whole-body post-save yaw heading | N/A (static hand-catch task) | `postheadingorientation` — root forward-axis alignment with its own save-moment heading | `postorientation` is yaw-invariant and `ang_vel_z` only penalizes rate, not final heading — neither corrects a settled post-save yaw drift. |
| Arm swinging (counterbalance) damping | Flat, non-curriculum, whole-body smoothness terms only | `arm_dof_vel` — `joint_vel_l2` scoped to the 8 arm joints | Concentrates the existing `dof_vel` mechanism onto the arms for undiluted gradient. |
| ~~Arm torque-limit excess~~ / ~~Arm action-rate/acceleration~~ | REVERTED | Added then reverted the next day | Over-suppressed legitimate in-dive counterbalance motion (regressed footreach/ball_exit/episode-length, collapsed `postupperdofpos`). The user's actual goal (arm not ending up behind the body) was a pose question, addressed instead via `postupperdofpos`'s `during_scale`. |
| Actor ball-observation visibility gate | Gated every step by flying/vanish logic, including during training; critic ungated (warmup-only) | **CORRECTED 2026-09-14** (was stale/self-contradicting this file's own line 319): actor's registered `ball_pos_b` term (`goalkeeper_env_cfg.py:818`, `func=ball_pos_xy_b`) passes `always_visible=False`, so it IS gated by the full flying/random_vanish mask (`_compute_ball_visibility`) — same mechanism `ball_pos_b`/`ball_vel_b` use — PLUS two additional zeroing conditions on top: `hide_behind_torso=True` (zero once ball is behind torso front edge) and `hide_after_steps=75`. The function's own docstring ("G1's warmup blackout, random vanish...intentionally NOT ported") is stale — corrected in `observations.py`, see below. Critic's `ball_pos_b` uses `warmup_only=True` (initial_vanish gate only, not the full flying/random_vanish mask) — not the `always_visible=True` this row previously implied; not independently re-verified for `ball_vel_b`. | The "Post-save release gate v2" design intent (visible through approach/save, zero after) was never actually live in the registered actor term — `always_visible=False` means the flying/vanish gate always applies on top. Left as-is (not changed to `always_visible=True`) since that's an intentional-vs-accidental question outside this session's scope — flagging only that the doc was wrong, not proposing a behavior change. |
| `cleanstop` payout shape | N/A | Continuous scale of end ball speed (best at 0.2 m/s, worst at 1.0 m/s), gated off genuine sole contact | Binary threshold at one cliff wasted gradient across the whole achievable speed range. |
| Effector type | Hands | Feet only | Phase 1 scope. |
| Multi-disc `obs_current` sourcing | Sliced from the history tensor's newest frame | Fetched from a separate `"actor_current"` observation group with independently-sampled noise/delay | Assessed by two independent reviews as functionally acceptable for a learned MLP (same noise distribution, decorrelated not biased) — documented per this project's divergence rule rather than fixed, given the fix's complexity vs. benefit. |
| Multi-disc PPO `schedule` | `"adaptive"` | `"adaptive"` (matches G1 — confirmed 2026-09-14 G1 also runs `schedule="adaptive"` with its own `region_estimator` in the same optimizer group and trains fine) | This row previously said `"fixed"` after a 2026-07 investigation, but the code/checkpoints have run `"adaptive"` since 2026-07-06 — that row was stale, corrected here. The `region_estimator`+`adaptive` combo is not uniquely risky to SGK; ruled out as a root cause for the double-step training stall. |
| Actor/critic observation scaling | Fixed per-term `obs_scales` | Now scaled to match (`ang_vel=0.25`, `joint_vel=0.05`, critic `lin_vel=2.0`/`ball_vel=0.2`) | SGK previously set no scale on any observation term. |
| `empirical_normalization` runner flag | Declared but dead in G1 too | `False` | Was pure noise — the runner never even read this key. Wiring a real normalizer would be a genuine feature addition, not implemented. |
| Per-observation delay | None (G1 only delays actions) | Removed (was a per-term 0-2 tick delay on every actor observation) | A second, compounding source of temporal staleness with no G1 analog; SGK's real action-delay equivalent is untouched. |
| Multi-disc actor/critic hidden dims | `[512, 256, 256]` | `[512, 256, 256]` (was `[512, 256, 128]`) | Git-history audit found no rationale anywhere for the 128 value; reverted to the layer sizes that actually trained the reference G1 policy. |
| PPO policy/value smoothness regularizer | Ported (mixing-based smoothness penalty on policy mean and value) | Ported (was missing — deferred due to a storage-class limitation, since resolved without touching the shared storage class) | Closes a real feature gap, not a parity nuance. |
| `num_steps_per_env` (PPO batch size) | 100 (batch 614,400/update) | 24 (tried 100→OOM→64→reverted to 24) | A live saturation investigation found no measurable improvement from a larger per-discriminator batch — reverted to regain throughput. |
| R1 gradient penalty coefficient | Effective 0.5 | **Effective 0.5 — matches G1** (reverted 2026-09-14, `lambda_=100→5` in `multi_disc_amp_ppo.py`) | Was raised to effective 10 on 2026-07-21 (deliberate divergence, see git history) after a G1-baseline comparison suggested G1's own 0.5 was "healthy" while SGK's was collapsing. Live wandb data across ~28k subsequent iterations showed the raise never fixed the collapse (`mean_discri_logits` still pinned -50 to -85 throughout) — reverted to G1-exact as part of a broader discriminator-health investigation; see `docs/BugFixes.md`, 2026-09-14. |
| `random_vanish` window upper bound | Flat `randint(0,30)`, same for every env, no flight-duration scaling | **Deliberate divergence beyond G1, added 2026-09-14** — per-env, derived from that env's own `_t_flight` (`ceil(t_flight/dt)`, clamped above `vanish_floor`) instead of a flat 30-step cap | G1's `t_flight` is fixed at 0.4-1.0s so a flat 30-step (0.6s) window never much exceeds a flight; SGK's `t_flight_range` extends well beyond that (up to ~1.45s), so the old flat cap could force the ball to vanish before a long/far double-step episode's second-step decision point, every time, not just by chance. |
| AMP discriminator observation content | `dof_pos` only, 58 dims | `joint_pos` + `joint_vel`, 84 dims — deliberate divergence, not a parity fix | G1 never needs velocity because every G1 region is a single atomic motion; this project's far regions need a genuine multi-step gait `joint_pos` alone can't distinguish from a passive weight-shift. |
| Post-success leg pose recovery | No G1 equivalent (G1's legs mostly just stand/brace) | `postlegdofpos` — mirrors `postupperdofpos`'s shape, targets the 12 leg joints | SGK saves with its feet, so the LEGS (not arms) are the limb thrown into an extreme configuration during a dive. |
| AMP discriminator policy-sample generator | Deterministic 5× reuse per transition, own-region-share minibatches | Was a genuine bug (avg ~20x reuse with replacement, cross-region-total sizing instead of own-share) — now fixed to match G1's cadence exactly | Unintended sampling-scheme drift found during an AMP-saturation investigation, not a deliberate divergence. |

## Frame Convention

Ball spawning uses **world (global) frame** (`reset_ball_rolling` in `mdp/events.py`).
Ball spawns at `(env_origin_x + x_start, env_origin_y + y_start, floor_z + spawn_z)` in
world frame, aimed at `(env_origin_x - 0.3, env_origin_y + y_end)` — 0.3 m behind the
goal line so the ball retains forward momentum at interception.

Ball always approaches in world **-X** direction. All reward terms that gate on ball direction
also use world-X (stopball: `root_link_lin_vel_w[:, 0]`; stayonline: world-X position;
ball_exit_termination: world-X). Observations (`ball_pos_b`, `ball_vel_b`) are in robot
body frame — consistent because `ang_vel_z=-2.0` keeps robot facing world +X.

Key frame notes:
- `noretreat`: body-frame X velocity (correct even when robot yaws during a dive)
- `stopball`/`softstop`: world-X velocity — consistent with world-frame ball spawn
- `footreach`/`crossing_y`: world Y alignment (ball_y_w − robot_y_w)

## Key Files

| File | Purpose |
|------|---------|
| `src/simple_goalkeeper/robots/t1_constants.py` | T1 actuator configs, action scale, home keyframe |
| `src/simple_goalkeeper/robots/xmls/` | T1 headless XML + ball XML + STL assets |
| `src/simple_goalkeeper/mdp/observations.py` | ball_pos_b, ball_vel_b (visibility system), foot positions |
| `src/simple_goalkeeper/mdp/events.py` | reset_ball_local_frame, tick_catchstep |
| `src/simple_goalkeeper/mdp/rewards.py` | 5 goalkeeper reward terms (feet-only) |
| `src/simple_goalkeeper/tasks/goalkeeper_env_cfg.py` | Full env config |
| `src/simple_goalkeeper/tasks/goalkeeper_amp_cfg.py` | AMPRunnerCfg |
| `src/simple_goalkeeper/tasks/__init__.py` | Task registration |
| `src/simple_goalkeeper/motions/data/` | NPZ motion files (converted from PKL) |
| `src/simple_goalkeeper/scripts/train.py` | Training entry point |
| `src/simple_goalkeeper/scripts/play.py` | Play/evaluation entry point |
| `src/simple_goalkeeper/scripts/pkl_to_npz.py` | PKL→NPZ motion converter |
| `src/simple_goalkeeper/scripts/mirror_motion.py` | Left<->right NPZ motion mirror tool (`sgk_mirror`, self-inverse). See `docs/superpowers/specs/2026-07-26-motion-mirror-tool-design.md`. |
| `.claude/skills/debugging-mujoco-contact-sensors/` | **Claude skill.** Read before debugging any contact-sensor/geom-based reward (`ball_contact`, `leg_ball_contact`, `head_ball_contact`, etc.) that fires on the wrong body part, too rarely, or inconsistently with the viewer. Came out of the 2026-07-30 `penalize_wrong_foot_ball_contact` debugging loop (see `docs/BugFixes.md`) — codifies checking `primary_names` directly, computing real surface-gap distances instead of trusting `found`, and verifying against a real trained checkpoint rather than zero-action. Includes a reusable probe script template. |
| `.claude/skills/editing-amp-motion-data/` | **Claude skill.** Read before creating, viewing, or hand-correcting any reference motion NPZ clip (`motions/data/*.npz`). Came out of the 2026-07-31 far-region swing-leg straightening fix (see `docs/BugFixes.md`) — codifies the Kinematic Chain Rule (editing hip/knee changes the foot's world orientation even though the ankle's own joint values are untouched), the per-frame Newton-solve flat-foot fix, the stance/swing detection + FK-refit + re-level editing pipeline, and which viewer (`sgk_play ...-WithOverlay`, not bare `sgk_view`) actually shows the real scene. Includes a reusable damp+flatten script template. |
| `.claude/skills/reward-shaping-scene-entity-cfg/` | **Claude skill — mandatory load before touching ANY reward code (see Change Approval Workflow above).** Part 1 (original scope): came out of the 2026-08-06 `postupperdofpos` saga (three same-day wiring bugs, see `docs/BugFixes.md`) — codifies why an explicit `params["asset_cfg"]` always overrides a function's own default, why NOT passing one does NOT mean the default gets resolved, and why `body_ids`/`joint_ids` don't preserve declared name order. Includes a reusable `verify_reward_scoping.py` script. Part 2 (added 2026-09-08, from `leading_foot_lift`'s 6-pass shrinking-target saga): decaying/shrinking-target reward design pitfalls — division-by-target instability, unsigned-vs-directional distance metrics, overlapping geometric constants, shared-aggregate state contamination across bodies, unaccounted geometry baselines, duplicate reward mechanisms. Part 3: makes verification mandatory for any shape-based reward change — graph the design first, then build/reuse a static teleport probe (`scripts/probe_leading_foot_lift.py`) AND a live animated `sgk_play` agent (`scripted_blue_approach`). |

## beyondAMP Location

Cloned at `./beyondAMP/`. The four packages are installed as editable:
- `beyondAMP/source/beyondAMP` → `beyondAMP` package
- `beyondAMP/source/rsl_rl_amp` → `rsl-rl-amp` package
- `beyondAMP/source/amp_tasks` → `amp-tasks` package
- `beyondAMP/source/amp_tasks_mjlab` → `amp-tasks-mjlab` package

## Motion Files

NPZ format, 21-DOF headless T1 joint order. Expected arrays:
- `fps`: sampling rate
- `joint_pos` (T, 21): joint positions (absolute, matching T1 default pose reference)
- `joint_vel` (T, 21): joint velocities via finite differences
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`: body kinematics from FK

Convert PKL → NPZ:
```bash
uv run sgk_convert --input-dir /path/to/Motions --output-dir src/simple_goalkeeper/motions/data
```

Mirror an NPZ clip left<->right (self-inverse transform; negates world Y, mirrors
root quaternion, swaps+signs the 21 joints via `t1_headless.xml`'s Pitch/Roll/Yaw
axis convention, recomputes all body kinematics via FK):
```bash
uv run sgk_mirror --input-file src/simple_goalkeeper/motions/data/<left_clip>.npz \
                   --output-file <output>.npz
```


## Reward Design

Phase 1 reward structure (ported from proven Imitationlearningbooster pattern).

*(Full chronological history for every row below lives in `git log`/`git show` on this file and in `docs/BugFixes.md`. This table states current values only.)*

| Term | Weight | Purpose |
|------|--------|---------|
| `stopball` | +5.21→13.02 (curriculum) | One-time bonus when ball is deflected (delta_vx > 0.6 m/s). Primary signal. |
| `softstop` | +36.46→91.15 (curriculum) | One-time bonus on full velocity reversal. |
| `success` | +5.0→12.5 (curriculum) | Continuing, foot-to-crossing-point proximity (`dist < 0.15 m`), gated by the two-stage landing gate. Multiplier retiered 1.0x/2.0x/3.0x off `softstop`/`cleanstop`. |
| `footreach` | +10→20 (curriculum) | Phase1: lateral alignment. Phase2: sigmoid reach × vel_sigma (1–10×). Two-stage blue/green targeting with a blue decel-zone and a green-overshoot penalty mirroring the blue one. |
| `near_stick_reach` | +8→16 (curriculum) | Near-region analog of `blue_stick_landing` — dense "close AND slow" reward preventing rapid oscillation-farming near the near-region target. |
| `blue_ball_landed` | +10→20 (curriculum) | One-shot bonus when the assigned foot genuinely lands at the blue midpoint on a wide crossing. |
| `blue_overshoot_penalty` | -30→-60 (curriculum) | Penalty for the assigned foot advancing past blue before landing there. |
| `blue_stick_landing` | +8→16 (curriculum) | Dense "close AND slow" reward near blue, peaking at a genuine plant. |
| `blue_trunk_drive` | +5→10 (curriculum) | Trunk (whole-body) lateral velocity toward the current two-stage target — toward blue (decaying near it) while approaching, toward green (undecayed) once landed. Fills the pre-close-range locomotion gap `footreach`'s own vel_sigma leaves open. An acceleration sibling (`blue_trunk_drive_acc`) was tried and removed — rewarding raw acceleration let oscillation farm reward. |
| `blue_green_transition_track` | +5.0 | HUSKY-inspired (arXiv 2602.03205) trajectory-guided reward: eases the leading foot's target Y+height from its blue-landing position toward green over a window sized off the ball's live remaining time-to-arrival (`window_frac_of_remaining=0.56`). Height target is an up-then-down arc reusing `leading_foot_lift`'s own kernel. Shares implementation (`_husky_transition_track`) with `start_blue_transition_track` below. |
| `start_blue_transition_track` | +5.0 | Sibling instance of the same shared mechanism, covering the FIRST leg (start→blue). Window sized off the ball's SAMPLED `t_flight` (not live remaining time, since this leg triggers essentially at episode start) with `window_frac_of_remaining=0.45`. Deactivates the instant blue genuinely lands, handing off cleanly to `blue_green_transition_track`. |
| `orange_foot_proximity` | +2.5 | Trailing-foot mirror of `foot_proximity` — dense pull toward the orange target, wide crossings only. |
| `orange_ball_landed` | base 5.0, peak 12.5 | One-shot bonus when the trailing foot genuinely lands at orange. |
| `orange_overshoot_penalty` | base -30.0, peak -75.0 | Penalty for the trailing foot advancing past orange while unlanded. |
| `orange_stick_landing` | base 4.0, peak 10.0 | Dense "close AND slow" bonus near orange. |
| `red_foot_proximity` | +2.5 | Dense pull toward red, active only once both blue and orange are genuinely landed. |
| `red_ball_landed` | base 5.0, peak 12.5 | One-shot bonus at red. |
| `red_overshoot_penalty` | base -30.0, peak -75.0 | Penalty for advancing past red while unlanded. |
| `red_stick_landing` | base 4.0, peak 10.0 | Dense "close AND slow" bonus near red. |
| `trailing_foot_reach` | +10.0 | General urgency reward for the trailing foot's orange→red sequence, same sigmoid×vel_sigma mechanism as `footreach` without the ball-position-specific phase logic. |
| `sequence_promptness` | +3.0 | One-shot bonus rewarding the full blue→orange→red→save relay for margin, paid once at the save moment. |
| `leading_foot_lift` (was `foot_clearance`) | +2.0 | Smooth bump peaking when the LEADING foot is 10cm above the floor, scoped to the leading foot only (trailing foot has its own `trailing_foot_lift`). Fades toward a near-zero target as the foot approaches blue's landing radius (exponential decay, `k=1.0`), restoring once genuinely landed. Corrects for a ~0.03m foot-resting-height baseline offset. |
| `trailing_foot_lift` | +2.0 | Mirrors `leading_foot_lift`'s exact design (same baseline correction, same shrinking-target-near-waypoint decay) for the trailing foot's start→orange→red journey. Target height currently 0.03m. |
| `stayonline` | -2.0 | Penalty for drifting away from goal line (X displacement). |
| `noretreat` | -2.0 | Penalty for retreating backward. |
| `feetorientation` | +15.0, sigma=100.0 | Flat feet (gravity aligned with foot Z). Weight/sigma both raised substantially from early defaults after live measurement showed a visibly bad tilt still scoring most of max reward. |
| ~~`foot_ang_vel_xy`~~ | REMOVED | Penalized foot roll+pitch angular velocity — superseded by `ankle_pitch_vel`/`ankle_roll_vel`, which read each ankle joint's own local velocity directly instead of a whole-body world-frame read contaminated by hip/knee cross-talk. |
| ~~`foot_ang_vel_z`~~ | REMOVED | Penalized foot yaw angular velocity (Hip_Yaw reaction-torque spin) — deleted as part of a full revert isolating this reward group back to a known-good checkpoint's exact training-time state. |
| `ankle_pitch_vel` | -1.0 | `joint_vel_l2` scoped to Ankle_Pitch, always active, no gate — root-caused via a live joint-level probe as the dominant driver of pre-save foot-pitch spikes. |
| ~~`ankle_pitch_pos`~~ / ~~`ankle_roll_pos`~~ | REMOVED | Pulled the ankle toward its own default JOINT angle, which can actively fight genuine landing flatness mid-dive — `feetorientation`'s weight was raised instead to measure the real world-frame outcome. |
| `ankle_roll_vel` | -1.0 | Same mechanism as `ankle_pitch_vel`, scoped to Ankle_Roll. |
| `postorientation` | +3.0 | Posture recovery targeting a 15° forward lean (not dead-vertical) — always active, since AMP has no root-orientation observation of its own. Render-verified the lean direction matches the toe-pointing direction. |
| `postangvel` | +3.0 | Low XY angular velocity reward, post-save only. |
| `postlinvel` | +1.0 | Low forward velocity reward, post-save only. |
| `postupperdofpos` | +5.0 | exp(-err) elbow/wrist recovery to a `_POST_SAVE_STANCE_MAP` target (matches Booster's own official T1 default pose), post-save. Scope narrowed to match G1's own elbow+wrist-only scope (shoulder was never in G1's version — its inclusion here was saturating the kernel on far-region dives). `kernel_scale=0.15`, `during_scale=1.0` (both tuned to avoid saturating on large far-region pre-save error). |
| `penalize_arm_above_shoulder` | -2.0 | Supplementary to `postupperdofpos` — targets the specific "hand above its own shoulder" failure geometrically, active for the whole post-save window. |
| `postshoulderdofpos` | +5.0 | Separate kernel from `postupperdofpos` for the 4 shoulder joints specifically (mixing shoulder's much larger error magnitude into elbow's kernel was the original saturation bug). |
| `angular_momentum_penalty` | -0.02 | Penalizes whole-body angular momentum (native MuJoCo `subtreeangmom` sensor) to encourage natural arm swing — small/gentle, complements `postupperdofpos` rather than replacing it. |
| `postwaistdofpos` | +3.0 | exp(-err) waist recovery to default, post-save only. |
| `postlegdofpos` | +1.0 | exp(-err) leg (hip/knee/ankle) recovery to `_POST_SAVE_STANCE_MAP`'s straight-leg stance, post-save only. No G1 equivalent by design. |
| `penalize_baseheight` | -100.0 | Penalty when root/Trunk height drops below 0.59m. Replaced a shank-based gate that false-positived on legitimate deep lunges. |
| `penalize_sharpcontact` | -100.0 | Binary penalty when mean foot contact force > 1800 N. |
| `penalize_self_collision` | -50.0 | Binary penalty on any Trunk-subtree self-collision. |
| `penalize_wrong_foot_ball_contact` | -100.0 | Binary per-step penalty for wrong-side ball contact. Final shape (after extensive same-day iteration): WRONG side = whole leg (sole/shin/knee), LEADING side = knee only, chin/head via a distance-based proximity check. |
| `single_foot_save` | +34.72→69.44 (curriculum, ×2 at cu≥3) | One-time bonus: stopball+softstop both fire within a short window on the correct foot. Gated on sole-only contact (not shin/knee). |
| `cleanstop` | +17.36→34.72 (curriculum, ×2 at cu≥3) | One-time bonus: ball nearly dead after a correct-foot, sole-contact deflection. Continuous payout scale of end ball speed. |
| `contact_yield_velocity` | +5.0 | One-shot bonus at the exact contact instant, rewarding the leading foot's velocity yielding backward normal to its orientation target (momentum absorption). Symmetric clamp so a non-yielding contact doesn't read as "no contact." |
| `inner_face_orientation_save` | +17.36→34.72 (curriculum, ×2 at cu≥3) | One-time bonus: correct foot turned to the block-posture target angle (within `tolerance_deg=5.0`) at the save moment. |
| `postleadfootplantspeed` | +3.0 | One-shot bonus for the leading foot's ground-impact speed landing near 0.1 m/s (a soft, not slammed, touchdown). |
| `foot_inner_face_continuous` | +3.47→6.94 (curriculum, ×2 at cu≥3) | Continuous reward for rotating the assigned foot's inner face toward the ball. Target gated on `env._blue_landed_genuine`: `_PRE_LANDING_TARGET_ANGLE_DEG` (15°, straight-forward-adjacent) before landing, `_FOOT_TARGET_ANGLE_DEG` (70°) after — with the pre-landing state applying for the whole episode on narrow crossings (no blue waypoint to wait for). Deactivates the instant `softstop` fires, handing off to the one-shot `inner_face_orientation_save`. |
| `trailing_foot_forward_continuous` | +3.0 | Continuous reward for the trailing foot's alignment with the robot's own forward axis, always active. |
| `postleadfootorientation` | +6.0, `window_steps=30` (~0.6s) | Continuous reward for the leading foot rotating back toward forward post-save, airborne-only, capped to a fixed post-save window (bounds how long the Hip_Yaw reaction-torque yaw drift is incentivized). |
| `postsave_foot_airtime` | +2.0, `window_steps=30` (~0.6s) | Bonus for the assigned foot staying airborne right after the save, within the same fixed window `postleadfootorientation` uses, linear ramp rewarding sustained hangtime. |
| `postheadingorientation` | +2.5 | Continuous reward for the root's yaw heading returning to ITS OWN heading captured at the save moment (not a fixed world direction), post-save window only. |
| `feet_slippage` | +3.0 | exp(-10*contactvel) — rewards stable foot contact, excludes genuine ball contact from counting as slippage. |
| `dof_pos_limits` | -3.0 | Joint limit violation penalty. |
| `dof_vel_limits` | -2.0 | Penalty for per-joint velocity above real T1 motor limits × 0.9. |
| `torque_limits` | -3.0 (flat, not curriculum) | Per-joint torque limit violation. |
| ~~`arm_torque_limits`~~ | REVERTED | See Divergences table. |
| `ang_vel_xy` | -0.1 | Penalise rolling/pitching. |
| `ang_vel_z` | -0.5 | Penalise yaw rotation. |
| `deviation_waist_joint` | -0.001 | Waist joint regularisation, always active. |
| `torques` | -1e-5 | Normalized torque L2. |
| `action_rate_l2` | -0.05 | Action smoothness (1st order). |
| `action_acc_l2` | -0.1 | Action jerk penalty (2nd order — G1's true structural match). |
| ~~`arm_action_rate_l2` / `arm_action_acc_l2`~~ | REVERTED | See Divergences table. |
| `dof_vel` | -5e-4 | Joint velocity regularisation. |
| `arm_dof_vel` | -5e-3 | Same mechanism as `dof_vel`, scoped to the 8 arm joints for concentrated gradient. |
| `dof_acc` | -2.5e-7 | Joint acceleration penalty. |

**Terminations:** `time_out`, `bad_orientation` (G1-exact formula, ~53° tilt), `base_height` (<0.4m), `ball_exit` (behind goal -0.5m), `sharpforce` (>2600N mean foot force).

**`_ball_is_behind` semantics:** `(ball_x < 0) | (delta_vx > 1.0)` — matches ILB exactly. Fires the moment stopball fires, deactivating `footreach` and activating post-save recovery rewards immediately.

**Removed (created stand-still or wrong local optimum):**
- `ball_positive_vx`: caused ball-chasing after save
- `successland`: became a ball-chasing reward with feet-only goalkeeping
- `ball_vx_reduction`: peaked when ball stopped naturally — rewarded doing nothing
- `foot_to_ball` (std=0.15): zero gradient at typical spawn distance
- `posture`: AMP handles motion naturalness; this incentivised standing still

**Ball visibility:** actor's `ball_pos_b` uses `always_visible=False` in both training and play, matching G1's warmup/flying/random-vanish gating during training itself. Critic's `ball_pos_b`/`ball_vel_b` remain `always_visible=True`, matching G1's ungated privileged obs.


**FIX 2026-07-14:** this section previously documented only the plain
single-discriminator task. A training run was launched off this doc on
2026-07-13 (`green_doubletriple25x_2026-07-13`) and silently used the wrong
architecture for ~17 hours before being caught by comparing checkpoint
`state_dict` keys against a known-multi-disc checkpoint. The real target
architecture is the multi-discriminator task (region-conditioned AMP,
history encoder, ball/region estimators) — use `-MultiDisc`.

```bash
# Convert motions (once):
uv run sgk_convert --input-dir /home/isaak/BEPImitationlearning/Motions --output-dir src/simple_goalkeeper/motions/data

# Train:
uv run sgk_train Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --num-envs 4096

# Play (zero policy sanity check):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --agent zero --num-envs 1

# Play (trained checkpoint):
uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc --checkpoint-file logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_500.pt
```

**FIX 2026-09-09 (user request, "still not showing up in wandb"):** this
machine's default `wandb` login (`~/.netrc`) resolves to a teammate's
account (`l-a-j-alewijns`), not the project owner's. Any manual `sgk_train`
launch that doesn't override this silently syncs to the teammate's wandb
project instead — training itself works fine (confirmed via
`debug-internal.log`: continuous `200 OK` filestream requests), it's just
invisible to the owner, who gets a 404/no-team-access error on the printed
URL. The watchdog script (`intercept_gpu_watchdog.sh`) already does this
correctly via `export WANDB_API_KEY="$(cat "$HOME/IsaakB/wandbapilink")"`
before every launch it makes — **any manual launch/resume must do the same**
or it silently lands on the wrong account. Verify by checking the printed
`Currently logged in as: ...` line matches `i-p-b-bouwmeester`, not
`l-a-j-alewijns`.

## Exporting a Checkpoint to ONNX for Deployment

**Observation scaling (`base_ang_vel*0.25`, `joint_vel*0.05`, etc.) is already automatic during training** — it's baked unconditionally into `goalkeeper_env_cfg()`, the single config function both `train.py` and `play.py` call, since `e930b425da` (2026-07-22). Nothing about training needs to change for this, and no run needs to be restarted on account of it. The part that is **not** automatic is producing a deployable artifact that carries the same scaling — that's a manual step, run it every time a checkpoint is handed off for deployment:

```bash
uv run sgk_export <checkpoint.pt> [--output path.onnx] [--device cpu]
```

This always derives the per-term scale live from the checkpoint's own env config and bakes it into the ONNX graph as its first op — the exported graph accepts **raw, unscaled** sensor values, so a deploy consumer never needs to independently track or hardcode the training-side scale convention. Do not hand-write an export or a deploy-side observation-scaling step instead — a prior hand-export (`model_18500`) skipped this and shipped a checkpoint that received `joint_vel` ~20x its trained-on magnitude, most visible during fast motions like a swing-leg lift. Applying the exact scale a policy was trained on at inference time is standard practice for any deployed learned-control policy (the RL equivalent of feature normalization in classical ML) — it introduces no sim-to-real gap by itself; the only real risk is inconsistency between train-time and deploy-time scaling, which baking the scale into the export eliminates by construction. Full mechanism, HIM-architecture handling, and how to verify an export without the training stack: `.claude/skills/exporting-him-checkpoints-to-onnx/SKILL.md`.

## Training Run Monitoring

**After launching any training run, ask the user: "Do you want scheduled monitoring?"** Don't set it up unasked, and don't re-explain this workflow each time — just do it once they say yes.

If yes, set up a session cron job with this standing behavior:

- **Every 2 hours:** health check. Confirm the worker process is still alive. Read the latest TensorBoard scalars and check for divergence — `Episode/Episode_Metrics/mean_action_acc` should stay O(1), not blow up toward O(1e6)+; `Loss/*` terms (surrogate, AMP, AMP_grad, est_ball, est_region, etc.) should stay small and finite, never NaN/inf. Report the reward terms actually relevant to whatever's being iterated on. If a live diagnostic script exists for the specific behavior being trained (e.g. a success-rate or landing-rate probe), run it once the run has enough iterations for the result to be meaningful, and report the number.
- **At iteration 5000:** a deeper checkpoint, not just another routine ping. Look at the full trend (not just the latest value) for the key diagnostic and the loss curves, and form a judgment: is this run fundamentally stuck, or genuinely progressing? If it looks stuck:
  1. **Stop it** (kill the process).
  2. Decide which subsystem is the more likely culprit — if it looks like the motion-prior/AMP mechanism itself is still off, dispatch a fresh, independently-framed subagent (skeptical, verify-from-scratch, no access to prior conclusions) to re-compare against the frozen G1 reference for anything still missed. If it looks like the task-specific reward mechanism is the problem, think through concrete tweaks and **implement one directly** — don't just report and wait.
  3. Follow the standard fix cycle: run tests, live smoke test, document in `docs/BugFixes.md` and `HANDOFF.md`, commit the fix, push the final checkpoint of the stopped run, launch the new run, confirm a clean start, and update the monitoring cron to track it.
  4. If the run looks like it's genuinely progressing, don't stop it — just report the trend and keep going with routine bihourly checks.

**Always push the latest checkpoint before stopping or restarting any run**, whether for a scheduled-monitoring fix or an ad hoc one — never let a checkpoint go un-pushed when a run is about to be killed.

**Unified GPU watchdog:** `/home/robocup/IsaakB/intercept_gpu_watchdog.sh` (cron, `*/30 * * * *`) is the single automation for this project — it supersedes both the old 23:59 fresh-launch-on-commit script (`intercept_autotrain.sh`) and the old fixed-00:00 resume script (`intercept_nightly_resume.sh`); neither is scheduled anymore, both kept only for reference. Every 30 minutes it classifies GPU state (`nvidia-smi` PIDs cross-checked against each process's own cmdline — ours = matches `sgk_train.*MultiDisc`) and applies one priority order:

1. **We're running AND someone else's job also appears (contention):** pauses first, above everything else, including a pending new commit — pushes the latest checkpoint (stashing/restoring the persistent unrelated local edits around it, same as any manual push) and stops our process, yielding the GPU. The resume pointer (`resume_run_dir.txt`) is left pointing at that same run dir, so the next tick picks it back up automatically once free.
2. **Someone else's job only, we're not running:** waits — can't launch into an occupied GPU even if a new commit is pending.
3. **A new training-relevant commit exists** (compared against `last_trained_commit.txt`, same training-relevant path filter as the old autotrain script): overrides whatever's running — if we were mid-resume on the old lineage, pushes+stops it first (the old lineage's remaining iterations toward 20k are abandoned, not finished first) — then pulls and launches **fresh** on the new commit, updating both `last_trained_commit.txt` and `resume_run_dir.txt` to the new run.
4. **No new commit, already running, no contention:** nothing to do.
5. **No new commit, GPU idle:** resumes the run directory recorded in `resume_run_dir.txt` from its latest checkpoint, aimed at absolute iteration 20000 (computes the correct additive `--agent.max-iterations` offset itself — resuming is additive to the loaded checkpoint's iteration, not absolute). Throttled to ~hourly actual attempts via `last_idle_attempt_epoch.txt`, even though the cron itself fires every 30 min.
6. **No new commit, GPU idle, tracked lineage already ≥20000 or none exists:** nothing to do.

**Every time a training run is launched or resumed manually (not by the script itself), update both marker files** so the watchdog doesn't redo the same work or lose track of the current lineage:
```bash
git rev-parse HEAD > /home/robocup/IsaakB/intercept_autotrain_logs/last_trained_commit.txt
echo "/full/path/to/the/run/directory" > /home/robocup/IsaakB/intercept_autotrain_logs/resume_run_dir.txt
```
Do this every time, not just when asked — including whenever the user says something like "stop [training] and resume later" (or otherwise clearly signals they want to pick this run back up, not abandon it). If the user later just says "resume" (or "continue"), read `resume_run_dir.txt` to find the run dir + its latest checkpoint and launch with `--agent.resume True --agent.load-run <dir> --agent.load-checkpoint <file>` rather than asking which checkpoint they mean.
This way both a later "resume" request from the user and the nightly script automatically pick up the same checkpoint without re-deriving which one. If the user later just says "resume" (or "continue"), read this file to find the run dir + its latest checkpoint and launch with `--agent.resume True --agent.load-run <dir> --agent.load-checkpoint <file>` rather than asking which checkpoint they mean.

## Reading TensorBoard / WandB Episode Reward Metrics

mjlab's `reward_manager` logs `Episode_Reward/X` with **two scaling factors** baked in:

```
Episode_Reward/X = (Σ over episode of [reward_fn(obs) × weight × dt]) / max_episode_length_s
```

Where `dt = 0.02 s` and `max_episode_length_s = 3.0 s` (150 steps).

**The logged value is NOT a raw per-episode sum.** It is divided by `max_episode_length_s`, so it represents "reward per second" rather than "reward per episode". This is done in `reward_manager.py`:
```python
value = value * term_cfg.weight * scale          # scale = dt = 0.02
self._episode_sums[name] += value
# on episode end:
extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
```

### Converting logged values to meaningful metrics

**One-shot rewards** (fire once per episode: `softstop`, `stopball`, `single_foot_save`):
```
event_rate = logged_value × max_episode_length_s / (weight × dt)
           = logged_value × 3.0 / (weight × 0.02)
```
Example: softstop logged = 1.09, weight = 210 → rate = 1.09 × 3.0 / (210 × 0.02) = **77.9%** of episodes

**Per-step rewards** (e.g., `ang_vel_xy`, `feetorientation`): logged value ≈ average per-step value × weight (already divided by max_episode_length_s cancels the dt accumulation for continuous rewards).

**Binary termination-linked rewards** (e.g., `penalize_sharpcontact` at 1800 N): same formula as one-shot, since they also fire on at most one step per episode.

This scaling is the source of the "softstop ≈ 1.09 looks tiny" confusion — 1.09 is actually a 78% save rate. Always apply the formula before interpreting one-shot metrics.

## Standalone Constraint

**No runtime imports from `Imitationlearningbooster`, `BoosterT1mjlab`, or `HandWavingMotion`.**
All needed assets and constants are copied into this folder.
