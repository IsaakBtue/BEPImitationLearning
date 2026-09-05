---
name: scoping-claude-login-to-tmux-session
description: Use when a plain terminal outside a specific tmux session should NOT be pre-authenticated as Claude Code, but a designated tmux session (e.g. a long-running training session) should stay logged in. Covers how to scope CLAUDE_CONFIG_DIR to one named tmux session via .bashrc, and why a plain `bash -c` test of the scoping function will falsely look broken.
---

# Scoping Claude Code Login to One tmux Session

## Problem

Claude Code stores login state in `~/.claude/.credentials.json` (default) or
wherever `CLAUDE_CONFIG_DIR` points. By default every shell on the machine
shares the same store, so opening `claude` in any terminal is pre-authenticated
with the same account. To keep a long-running/trusted session (e.g. a
`tmux` session left attached for days during training) as the only
pre-authenticated context, and require login everywhere else, you need a
**separate config directory** that only gets selected inside that one named
tmux session.

## Recipe

1. Create an isolated config dir, e.g. `~/.claude_training/`, and seed it
   with a valid `.credentials.json` (copy from `~/.claude/.credentials.json`,
   mode `600`).
2. In `~/.bashrc`, add a function that checks the current tmux session name
   and exports/unsets `CLAUDE_CONFIG_DIR` accordingly, then wire it into
   `PROMPT_COMMAND` (runs on every prompt, so it re-evaluates if you attach
   to a different session from the same shell):

   ```bash
   _claude_config_scope() {
       if [ -n "$TMUX" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = "trainingIsaak" ]; then
           export CLAUDE_CONFIG_DIR="$HOME/.claude_training"
       else
           unset CLAUDE_CONFIG_DIR
       fi
   }
   PROMPT_COMMAND="_claude_config_scope${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
   ```

3. Clear the default store (`~/.claude/.credentials.json`) so a plain
   terminal actually prompts for login instead of silently working. Do this
   LAST, and not while any live `claude` process still depends on that
   store's access token being valid/refreshable mid-session.

## Verification Gotcha: `bash -c` Looks Broken But Isn't

`.bashrc` starts with an early-return guard for non-interactive shells
(`[[ $- != *i* ]] && return`, near the top of the default Ubuntu `.bashrc`).
`bash -c '...'` is non-interactive, so sourcing `.bashrc` that way silently
no-ops the whole file — `_claude_config_scope` never gets defined, and a
naive test looks like the scoping "isn't working" when actually it was never
loaded. This produced a real false negative during setup: `bash -c 'source
~/.bashrc; ...'` reported `CLAUDE_CONFIG_DIR=` (empty) even inside the
intended session.

**Correct verification:**
- `bash -i -c '...' < /dev/null` (forces interactive mode) — will emit a
  harmless `cannot set terminal process group` / `no job control` warning to
  stderr; that's expected, not a failure.
- Or just check the real running shell directly: `echo $CLAUDE_CONFIG_DIR`
  typed interactively inside vs. outside the target tmux session.
- To confirm the tmux-name check itself: `tmux display-message -p '#S'` run
  from inside the target session should print exactly the session name used
  in the `.bashrc` comparison (case-sensitive, exact match).

## Gotcha: Step 3 Doesn't Stay Fixed — Re-Verify Periodically

Clearing `~/.claude/.credentials.json` (recipe step 3) is a one-time file
deletion, not an enforced invariant. Anything that runs `claude`/`/login` in
a shell where the scoping function evaluated to "not the target session" —
another tmux session, a plain terminal, even a stray non-interactive
process — will silently repopulate the default store with a fresh, valid
token. The scoping logic itself (steps 1–2) staying correctly wired is not
evidence that step 3 is still in effect; they're independent facts.

**Confirmed 2026-08-12:** despite the scoping function being correctly
active inside `trainingIsaak` (`CLAUDE_CONFIG_DIR` set as expected), the
default `~/.claude/.credentials.json` was found holding a valid,
non-expired token (written 09:43:54, ~10 minutes before that session's own
`~/.claude_training` token was refreshed) — i.e. `claude`/`/login` had been
run outside the target session at some point, and nothing caught it.

**To re-verify step 3 is actually holding**, don't just check whether the
file exists — check whether it's live:

```bash
python3 -c "
import json, time
d = json.load(open('$HOME/.claude/.credentials.json'))['claudeAiOauth']
now = time.time()*1000
print('default store logged in:', d['expiresAt'] > now)
"
```

If this prints `True` from outside `trainingIsaak`, the isolation has been
breached and step 3 needs to be redone (`rm ~/.claude/.credentials.json`,
run from outside the target tmux session, only once no live `claude`
process there still needs that store's token).

## Why `PROMPT_COMMAND`, Not a One-Time `.bashrc` Line

A plain `export CLAUDE_CONFIG_DIR=...` guarded by an `if` only evaluates once,
at shell start. If the same shell later attaches to a different tmux session
(`tmux switch-client`) or the tmux session is renamed, a one-time check goes
stale. Binding the check to `PROMPT_COMMAND` re-runs it before every prompt,
so it tracks the *current* session name live.

## Quick Reference

| Question | Answer |
|---|---|
| Where does the scoping logic live? | `~/.bashrc`, `_claude_config_scope` function on `PROMPT_COMMAND` |
| What decides which store is used? | `tmux display-message -p '#S'` compared against the target session name |
| Why did `bash -c` testing show it broken? | Non-interactive shells skip `.bashrc` entirely via its own early-return guard — use `bash -i -c '...' < /dev/null` instead |
| Is it safe to clear `~/.claude/.credentials.json` immediately? | No — only after no live `claude` process still needs that store's token (check for pending token refresh / active session first) |
| Does this survive switching tmux sessions in the same shell? | Yes, because it's on `PROMPT_COMMAND`, not a one-time startup check |
| Does clearing the default store once mean it stays cleared? | No — any `claude`/`/login` run outside the target session repopulates it. Re-check `expiresAt` periodically, not just file presence (confirmed breached 2026-08-12) |
