# Double-Step Research & Root-Cause Report

**Date:** 2026-07-18
**Scope:** `interceptV2DualDis` (region-conditioned, multi-discriminator AMP fork of
`SimpleGoalKeeper`, Booster T1, foot-only goalkeeping). Research + audit only — no training
code, reward weights, or configs are changed by this document. Any of the recommendations below
still need to go through this project's Change Approval Workflow before being applied.

## Why this document exists

Three weeks and 40+ documented, evidence-backed fixes (`docs/BugFixes.md`, 2026-06-30 through
2026-07-16, latest commit `e62b468`) still have not produced a stable, sustained genuine
double/triple-step behavior on far ball crossings. The user asked for a fresh, full
investigation: research papers, GitHub repos, whether a different motion dataset is needed, a
full code audit, and a comparison against the frozen `Humanoid-Goalkeeper` (G1) upstream — ending
in a concrete, prioritized list of things to change or try.

## Sources read this session

- `interceptV2DualDis/CLAUDE.md` (full divergence-from-G1 table), `CONTEXT.md`, and the complete
  `docs/BugFixes.md` (631 lines, every entry).
- `docs/superpowers/specs/2026-07-02-multi-discriminator-amp-design.md` and current
  `src/simple_goalkeeper/mdp/regions.py`.
- Motion inventory: `motions/raw/*.pkl` vs. `motions/data/*.npz`.
- `Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py` (`end_regions`,
  `jump_scale`, `_reward_eereach`, `_reward_airfeetorientation`, `_reward_successland`, lines
  916-960 and 1361-1466) and `Humanoid-Goalkeeper/README.md`.
- The project's own reference paper, **arXiv:2510.18002** (*Humanoid Goalkeeper: Learning from
  Position-Conditioned Task-Motion Constraints*), via full-text fetch — confirmed via the
  README's own arXiv badge, not guessed.
- Live web search to independently verify every arXiv ID cited (both the ones already referenced
  internally in `BugFixes.md` and new ones pulled in this session), plus one public code repo.

## Key findings

### 1. The core gap: G1 has no analogue for "reach a far target via multiple sequential steps"

This is the single biggest finding and reframes everything else on this list.

G1's 6 regions (`legged_robot.py:916-924`) are **not** "near/far x left/right" the way this
project's 4 regions are. The paper describes them as three **height tiers per side** (low/mid/up),
and the code assigns per-region reward behavior accordingly:

- `leftstep`/`rightstep` (low tier): flat, small `vel_sigma` — no jump, no speed boost. This is
  the **easiest** region — the ball is basically already at the foot.
- `lefthand`/`righthand` (mid tier): hand-reach boost keyed to hand velocity.
- `leftjump`/`rightjump` (up/far tier): escalating `jump_scale = 3.0 + 3.0*curriculumupdate`
  **plus** two dedicated safety rewards that only activate for those regions —
  `_reward_airfeetorientation` and `_reward_successland` (`legged_robot.py:1427-1466`), managing
  foot orientation while airborne and landing quality, because G1's far region is a **literal
  dive** (`root_states[:,2] > 1.0`, torso leaves the ground).

Every one of G1's 6 regions is solved by a single atomic motion (a small weight shift, an arm
reach, or one dive) executed from the current stance. G1 never needs the robot to relocate its
base across multiple discrete footsteps to reach a target — reach is extended via the arm or via
one jump, never by walking there.

`BugFixes.md`'s 2026-07-15 entry already brushed against this ("No G1-equivalent fix exists for a
grounded task" when porting `jump_scale` to far regions) but never stated the structural
conclusion: **the double/triple-step behavior this project needs has no proven recipe to port
from the reference implementation at all.** Every other mechanism here (RSI, AMP parity,
curriculum, region-estimator, ball spawn geometry) could be, and was, verified line-for-line
against G1. This one specifically cannot be — because in the paper's own foot-only "step" tier is
the *trivial* case, not the hard one. Foot-only goalkeeping inverted which region is hard, and
nothing in the reference method was ever designed to solve the hard version.

**Implication:** the last three weeks of otherwise excellent, rigorous bug-hunting — RSI physics,
AMP magnitude/init/replay-buffer parity, curriculum ratchet bugs, region-assignment sampling
bugs, wrong-foot/wrong-sensor exploits, the settle-window multi-call bug — were all necessary
hygiene. None of them could have supplied the one thing G1 never had to supply either: a working
recipe for *compositional, multi-step* skill acquisition. That's a different kind of problem than
an implementation bug, and needs a different kind of fix (see recommendations B/C below).

### 2. The reference motion for the target skill is synthetic, not captured, and its pacing was reverse-engineered from the reward budget

`LeftDoubleStep`/`RightDoubleStep`/`LeftTripleStep`/`RightTripleStep` (and even the baseline
`LeftStep`/`Rightstep`) have **no raw PKL** — unlike every other clip (`1-1` through `8-1`,
`*Safe*`), which is real motion capture. The "own" suffix and total absence of a capture source
indicates these were hand-authored or procedurally built, not captured from a human or robot
performing a genuine double/triple-step save.

Their playback speed has been tuned twice (2x, then 2.5x — `docs/BugFixes.md` 2026-07-12) purely
to match the ball's flight-time budget (0.58-1.01s), not because 2.5x was independently verified
as the clip's natural, physically-plausible pace. A flat playback-speed multiplier scales joint
*velocities* by the same factor without re-solving dynamics — at 2.5x, the reference the AMP
discriminator is rewarding may already be a borderline-unnatural gait (the same entry reports
peak joint velocities of 3.1-5.5 rad/s post-retime — well under the 10 rad/s hard limit, but
getting into unusual territory for a stepping motion, and never independently validated).

Net effect: the one piece of ground truth for "what does a correct double-step actually look
like," which both the AMP discriminator and RSI seeding depend on, is the least-validated asset
in the whole pipeline. This deserves to be treated as a first-class hypothesis, not a footnote.

### 3. AMP's own literature documents exactly this collapse-to-simplest-style failure mode

The "Multi-AMP" paper (**arXiv:2203.14912**, Vollenweider/Rudin et al., ETH RSL/ANYmal — very
plausibly the design inspiration for this project's per-region-discriminator architecture) states
directly: when multiple priors could plausibly cover the same task, "the policy might either go
for the more straightforward style... or find a hybrid motion similar to both clips." That is a
documented, *expected* AMP failure mode, not a bug — and it matches the repeatedly observed
symptom here ("one continuous fast lunge instead of a paced multi-step approach") almost exactly.
The paper's own mitigations — curriculum staging, disturbance injection during the critical
phase, joint-velocity-based early termination for the harder skill — are a fuller toolkit than
what's currently deployed, which leans almost entirely on reward-term conjunctions (landing
radius + contact + speed + settle-window).

### 4. The reward mechanism is a hand-built approximation of a staged/hierarchical curriculum, implemented as flat single-timestep conjunctive gates — and that specific shape produced most of the debugging pain

The blue/green waypoint system (`_get_reach_target_y`, `blue_ball_landed`,
`blue_overshoot_penalty`, `blue_stick_landing`, the settle-window/speed/radius landing detector)
is, in spirit, exactly the right idea — force a genuine intermediate stop before the final target,
i.e. a 2-phase curriculum over the trajectory. But it's implemented as a sparse, multi-condition
conjunctive event computed inside a shared per-step helper called from 7 different reward
terms — precisely the class of mechanism that produced the multi-call settle-count bug, the
RSI-free-credit latch bug, the wide-crossing sign/magnitude sampling bug, and the "genuine rate
looked like 90%+ but was actually 1.4%" measurement chain documented across nine separate
2026-07-07 through 2026-07-09 `BugFixes.md` entries.

Two research passes already dispatched from within this project (07-09 entry) independently
converged on **arXiv:2605.30896** ("Zero Collapse": policy-gradient methods in discontinuous
reward landscapes rapidly collapse to near-zero reward and stay there, especially for
actor-critic methods) and **arXiv:2103.06846** (rare-event scarcity degrades policy-gradient
learning as a reward-bearing event's frequency falls) as the best-fit explanation for the observed
43%→35%→2% self-reinforcing collapse. Both are real, independently re-verified papers (confirmed
via live search this session), and both diagnoses are consistent with findings #3 and #4: a
sparse, conjunctive, all-or-nothing landing event is close to worst-case for policy-gradient
stability.

### 5. What's genuinely solid and should not be re-litigated

The AMP-vs-G1 parity work (discriminator width, grad-penalty scaling, weight init, reward
smoothing, replay-buffer clearing), the RSI physics fixes, the curriculum ratchet/oscillation fix
(now correctly bidirectional and EMA-smoothed, matching G1), and the region-estimator LR-collapse
fix (own optimizer param group) are all independently well-evidenced and match G1 exactly
wherever G1 has an equivalent mechanism. Treat these as a stable foundation, not something to
revisit without new evidence.

## Papers and repos (verified this session, not guessed)

| Source | Why it matters | Suggested action |
|---|---|---|
| **arXiv:2510.18002** — *Humanoid Goalkeeper: Learning from Position-Conditioned Task-Motion Constraints* (this project's own reference paper) | Confirms finding #1: 6 regions = 3 height-tiers × 2 sides; `leftstep`/`rightstep` is the *easy* tier; no multi-step composition anywhere in the reference method. Also runs curriculum-free difficulty sweeps (`Speed-Easy/Hard`, `Range-Easy/Hard`) worth mirroring as an eval protocol. | Do a full manual read of Fig. 4 (motion curation pipeline) and the reward-equations section directly from the PDF (`https://arxiv.org/pdf/2510.18002`) for anything an automated pass might miss. |
| **arXiv:2005.04323** — *ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills* (Xie et al., SCA 2020, Best Paper) | Directly about teaching a biped precise, discrete, sequential foot placement via **curriculum** — 4 curriculum strategies beat a no-curriculum baseline, which fails outright. Closest literature match to "make a biped commit to discrete multi-step travel instead of one lunge." An earlier internal research pass (07-09) rejected it as "not directly applicable" because it curricula over terrain, not reward gating — but the transferable lesson is the *curriculum-over-required-travel-distance* pattern itself (recommendation B), which this project doesn't currently do explicitly. | Clone `https://github.com/belinghy/SteppingStone` (public, verified). Read `https://arxiv.org/abs/2005.04323`. |
| **arXiv:2203.14912** — *Advanced Skills through Multiple Adversarial Motion Priors in RL* (Vollenweider, Rudin et al., ETH RSL/ANYmal) | Very likely the direct architectural ancestor of this project's per-region-discriminator design. Documents the exact collapse-to-simpler-style failure mode being observed (finding #3); its mitigations (curriculum, disturbance injection, joint-velocity-based termination) go beyond what's implemented today. No public code found. | Read `https://arxiv.org/pdf/2203.14912`. |
| **arXiv:2509.21810** — *Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors (CAMP)* | Alternative to N separate discriminators: **one** discriminator conditioned on a skill/style embedding plus a skill-conditioned reward. Worth comparing against the current 4-discriminator design, which the original multi-disc design spec already flags as batch-starved (each region trains on only `num_envs/4` samples). | Read `https://arxiv.org/pdf/2509.21810`. |
| **arXiv:2411.01000** — *Enhancing Model-Based Step Adaptation for Push Recovery through RL of Step Timing and Region* | Confirms the field's actual working solution to "learn multi-step footwork" is **hierarchical**: RL learns only the high-level decision (footstep region + timing); a QP/DCM planner handles feasible foot placement and balance; a low-level controller executes it. No prior work found trains a flat monolithic PPO+AMP policy to output raw joint torques for a multi-step maneuver end-to-end. Directly informs recommendation C. | Read `https://arxiv.org/pdf/2411.01000`. |
| **arXiv:2605.30896** (Zero Collapse) / **arXiv:2103.06846** (rare-event scarcity) | Already cited internally (07-09 entry) as the best-fit explanation for the 43%→2% collapse; independently re-verified as real this session. Worth reading in full before deciding whether to keep patching the sparse-event reward shape or replace it (recommendation D). | Read `https://arxiv.org/pdf/2605.30896`; search for `arXiv:2103.06846`. |
| AMASS / LAFAN1 motion datasets | If recommendation A is pursued: LAFAN1 (used by several AMP/ASE-style humanoid papers) contains real human sidestep/direction-change locomotion; AMASS is broader but less curated for sport-specific shuffle-steps. Either needs retargeting to the T1's 21-DOF headless rig via a `pkl_to_npz.py`-style pipeline. | Only pursue if recommendation A is chosen — check current access/license terms before downloading. |

## Prioritized list of changes/experiments to try

Ordered by expected leverage, not ease of implementation. **B and C are the structural bets that
address finding #1 directly; everything after that is refinement of the existing approach.**

**A. Treat the reference motion as suspect, not settled.** Either (a) source or capture a real
double/triple-step reference — mocap of an actual human/robot lateral shuffle-step, or a properly
re-solved/retargeted clip from LAFAN1-style data — instead of a synthetic clip played back at an
arbitrarily chosen 2.5x, or (b) if staying with the synthetic clip, validate its physical
plausibility directly (peak accelerations, a ZMP/COM stability check via open-loop MuJoCo replay)
before trusting AMP to shape a policy toward it. Neither has been done yet — the pace was tuned to
match the reward budget, not validated against the motion itself.

**B. Add an explicit curriculum over *required travel distance*, not just ball speed/difficulty.**
ALLSTEPS' central lesson: start the far-region target within single-step reach and *widen it over
training* — mirror the existing `ball_difficulty` curriculum pattern, but apply it to
`_REGION_Y_END_RANGE`'s far bound (currently a fixed `(0.5, 1.3)` from iteration 0). This gives
the policy a continuous path from "succeeds at the easy end of far" to "succeeds at the hard end,"
instead of requiring it to discover the full 1.3m multi-step behavior from a cold start against a
fixed wide target — which the Multi-AMP paper's account of AMP's collapse-to-simplest-style
failure mode predicts will fail exactly as observed.

**C. Prototype a hierarchical decomposition instead of pushing further on the flat monolithic
policy.** Train a standalone "step to point P and stop" primitive first, on a much simpler task
(robot starts standing, P is a random point within ~0.3-0.4m, reward = arrive + settle, no ball,
no AMP or a single clean step clip only). This isolates exactly the capability — genuine
stop-at-a-point — that has been the single hardest-to-achieve piece across nine `BugFixes.md`
entries (the entire blue-landing-gate saga). Once that primitive reliably works, compose it:
either (i) feed the full task a *sequence* of 2 waypoints for far balls instead of the current
single blue/green switch, or (ii) keep the task flat but reuse the validated primitive's reward
shape as the far-region shaping term instead of the current from-scratch conjunctive gate. This
follows the field's actual solved pattern (arXiv:2411.01000) instead of continuing to harden a
from-scratch flat-policy attempt at a problem the literature solves hierarchically.

**D. Replace the sparse, multi-condition "landing" event with a shape that can't zero-collapse.**
Per Zero Collapse (arXiv:2605.30896), a discontinuous, conjunctive, one-shot reward is close to
worst-case for policy-gradient stability once the policy drifts outside the success basin. The
`blue_stick_landing` basin-widening (07-09) was a step in the right direction, but the underlying
event (`_BLUE_SETTLE_STEPS` consecutive-step contact+radius+speed conjunction) is still sparse and
multi-conditional. Consider a dense, always-differentiable shaping signal for the whole
approach-and-stop phase (e.g., a potential-based term keyed to "distance-to-waypoint decreasing,
then speed-at-waypoint decreasing" as one smooth function of time) instead of a binary event gate,
so gradient never fully vanishes off the exact success manifold.

**E. Reassess whether 4 independent per-region discriminators are pulling their weight**, given
each trains on only `num_envs/4` samples (already flagged as an open risk in the original design
spec). CAMP (arXiv:2509.21810) suggests a single skill-conditioned discriminator may avoid that
batch-starvation without losing region-specific style separation.

**F. Continue validating the in-flight same-step-gate fix (`e62b468`,
`green_samestepgate_armregion_2026-07-16`), but treat it as maintenance, not the main lever.** The
historical pattern across every prior escalation (iterations 5000, 6500, 8000+) is that fixing the
latest measurement/gating bug produces a temporary-looking improvement that doesn't hold —
consistent with finding #1: no amount of correct measurement fixes the underlying "no proven
recipe" problem. Keep the bihourly monitoring per the standing policy, but budget the next block
of deep work toward B/C rather than another round of gate-tightening.

**G. Full audit conclusion.** No further undocumented divergences from G1 were found this session
beyond what `CLAUDE.md`'s divergence table and finding #1 already cover — the AMP/RSI/curriculum
machinery is in good, G1-matching shape. The gap is not a hidden bug; it's that this specific
behavior has no G1 precedent to audit against in the first place.

## Next steps

Implementing any of A-E is a separate, explicitly-approved follow-up per this project's Change
Approval Workflow — this document is research/audit only.
