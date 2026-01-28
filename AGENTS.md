# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Release Notes - v0.3.2

### Breaking Changes
- **Dice ordering aligned**: `dice[1]` is now HIGH die, `dice[2]` is now LOW die
  - Action encoding: `loc1` uses high die, `loc2` uses low die
  - Observation slots: slot 0 (ch 13-18) = high die, slot 1 (ch 19-24) = low die
  - This aligns the entire system: observation → action → execution
- **Removed backwards-compatible constructor**: Use `clone(g)` for copying game state
- **Added `obs_type` field to BackgammonGame**: All constructors now accept `obs_type` keyword

### New Features
- **Flat observations**: Same features as 3D but without spatial broadcasting
  - `observe_minimal_flat()` → 330 values (vs 30×1×26=780 in 3D)
  - `observe_full_flat()` → 362 values (vs 62×1×26=1612 in 3D)
  - `observe_biased_flat()` → 422 values (vs 122×1×26=3172 in 3D)
- **Hybrid observations**: Board spatial (12×26) + globals flat (NamedTuple)
  - `observe_minimal_hybrid()` → (board=12×26, globals=18)
  - `observe_full_hybrid()` → (board=12×26, globals=50)
  - `observe_biased_hybrid()` → (board=12×26, globals=110)
  - Ideal for conv1d on board, then concatenate globals before dense layers
- **Configurable observation type per game**:
  - `initial_state(obs_type=:minimal_flat)` - set at creation
  - `set_obs_type!(g, :full)` - change observation type
  - `observe(g)` - dispatch to correct observation function
  - `obs_dims(g)` or `obs_dims(:minimal)` - get dimensions
- **Nine observation types**: `:minimal`, `:full`, `:biased`, `:minimal_flat`, `:full_flat`, `:biased_flat`, `:minimal_hybrid`, `:full_hybrid`, `:biased_hybrid`

### Bug Fixes
- **Fixed doubles second action move count**: Previously hard-coded to bin 2 (2 moves), now correctly computes 0/1/2 playable moves when player may be blocked by a prime
- **Removed PyCall from test target**: `Pkg.test()` no longer requires PyCall (gnubg tests are run separately)

### Cleanup
- **Removed `vector_observation` alias**: Use `observe_full()` directly (benchmark.jl updated)
- **Removed dead code**: `get_legal_source_locs` allocating function (unused after is_action_valid refactor)
- **Fixed README doubles_only wording**: Clarified that API returns all 21 indices with 6 having non-zero probability

### Tests Added
- Tests for `clone()` function
- Tests for doubles second action move count
- Tests for flat observations
- Tests for hybrid observations
- Tests for `observe()`, `obs_dims()`, `set_obs_type!()`

### gnubg Validation (2026-01-27)
Full validation against gnubg reference implementation:

| Test | Games | Positions | Mismatches |
|------|-------|-----------|------------|
| Final states (hybrid) | 500 | 44,303 | 0 |
| Legal actions (direct) | 100 | 9,004 | 0 |

All legal actions and final board states match gnubg exactly.

---

## Release Notes - v0.3.0

### Breaking Changes
- **Removed deprecated API**: `observe_fast()`, `observe_fast!()`, `OBS_SIZE_FAST`, `OBS_SIZE_FULL` are removed
  - Use `observe_minimal()` / `observe_minimal!()` instead
  - Use `OBS_CHANNELS_MINIMAL` / `OBS_CHANNELS_FULL` for size constants
- **PyCall moved to [extras]**: Core module no longer requires Python installation
  - PyCall is only loaded during tests (for gnubg validation)
  - Users who don't need gnubg integration get a lighter install

### New Features
- **`clone(g)`**: Safe deep copy of game state with fresh buffers (recommended for MCTS)
- **`legal_actions` caching**: Results cached until state changes (~2x faster for repeated calls)
- **Accurate move count for non-doubles**: Observation now correctly encodes 0/1/2 playable dice

### Performance Improvements
- `is_action_valid()` now uses internal buffers (zero allocations)
- `legal_actions()` cached in game object, invalidated on state changes

### AlphaZero.jl Compatibility (v0.3.2)

**Breaking change**: The `BackgammonGame` struct now has an `obs_type` field. Update your `GI.current_state` implementation:

```julia
# REQUIRED: Use clone() instead of direct construction
function GI.current_state(g::BackgammonGame)
    return clone(g)
end
```

### Migration Guide

```julia
# Old (deprecated)
obs = observe_fast(g)           # → observe_minimal(g)
observe_fast!(buf, g)           # → observe_minimal!(reshape(buf, 30, 1, 26), g)
size = OBS_SIZE_FAST            # → OBS_CHANNELS_MINIMAL

# New - copying game state
# Old (fragile - depends on field layout)
copy = BackgammonGame(g.p0, g.p1, g.dice, ...)
# New (recommended)
copy = clone(g)

# New - observation API (v0.3.2)
g = initial_state(obs_type=:minimal_flat)  # 330-element vector
obs = observe(g)                            # Dispatches based on obs_type
dims = obs_dims(g)                          # Returns 330

# Available obs_type values:
# :minimal (30×1×26), :full (62×1×26), :biased (122×1×26)
# :minimal_flat (330), :full_flat (362), :biased_flat (422)
# :minimal_hybrid, :full_hybrid, :biased_hybrid (NamedTuple with board + globals)
```

---

## Build & Test Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests directly
julia --project test/runtests.jl

# Run benchmark (30 second duration)
julia --project benchmark.jl

# Start Julia REPL with project
julia --project
```

## Architecture

BackgammonNet.jl is a high-performance Backgammon engine for AlphaZero.jl integration. The key design decisions:

### Bitboard Representation
The game state uses two `UInt128` bitboards (`p0`, `p1`) with 4-bit nibbles per position:
- Indices 1-24: Board points
- Index 0: P1 off (borne off)
- Index 25: P0 off
- Index 26: P0 bar
- Index 27: P1 bar

Bit manipulation uses nibble operations: `get_count`, `incr_count`, `decr_count` (shifting by `idx << 2`).

### Canonical vs Physical Coordinates
- P0 moves from point 1→24→off (increasing)
- P1 moves from point 24→1→off (decreasing)
- `g[i]` accessor returns canonical view: positive = current player's checkers, negative = opponent's
- P1's canonical index `i` maps to physical index `25-i`

### IMPORTANT: Backgammon Point Numbers vs Physical Indices
This is a common source of confusion. Standard backgammon uses point numbers 1-6 for the home board, where:
- **6-point** = furthest from bearing off (highest point number)
- **1-point** = closest to bearing off (lowest point number)

The mapping to physical indices is **inverted** for P0:

| P0 Physical Index | Backgammon Point | Distance to Off |
|-------------------|------------------|-----------------|
| 19                | 6-point          | 6               |
| 20                | 5-point          | 5               |
| 21                | 4-point          | 4               |
| 22                | 3-point          | 3               |
| 23                | 2-point          | 2               |
| 24                | 1-point          | 1               |

For P1 (home board physical 1-6):

| P1 Physical Index | Backgammon Point | Distance to Off |
|-------------------|------------------|-----------------|
| 6                 | 6-point          | 6               |
| 5                 | 5-point          | 5               |
| 4                 | 4-point          | 4               |
| 3                 | 3-point          | 3               |
| 2                 | 2-point          | 2               |
| 1                 | 1-point          | 1               |

**Over-Bear Rule:** When bearing off with a die larger than needed, you can ONLY over-bear from the HIGHEST occupied point (furthest from off). This means:
- P0: Check for checkers at LOWER physical indices (19 to src-1) before allowing over-bear
- P1: Check for checkers at HIGHER physical indices (src+1 to 6) before allowing over-bear

"Higher" in backgammon = further from off = LOWER physical index for P0, HIGHER physical index for P1.

### Action Encoding
Actions are encoded as `(loc1 * 26) + loc2 + 1` where:
- `loc1`: source for HIGH die (dice[1]) - 0=bar, 1-24=points, 25=pass
- `loc2`: source for LOW die (dice[2])

**Dice ordering is consistent throughout:**
- `DICE_OUTCOMES` stores tuples as (high, low)
- `g.dice[1]` = high die, `g.dice[2]` = low die
- Observation slot 0 (ch 13-18) = high die, slot 1 (ch 19-24) = low die
- Action loc1 uses high die, loc2 uses low die

For doubles, two actions are needed per turn (`remaining_actions = 2`).

### Two-Mode API
1. **Deterministic (`step!`)**: Auto-rolls dice after each action. Used for RL training.
2. **Explicit (`apply_action!`/`apply_chance!`)**: Manual dice control. Used for MCTS.

A game starts at a chance node (dice = [0,0]). Call `sample_chance!` or use `step!` to roll.

### Key Files
- `src/game.jl`: `BackgammonGame` struct, move application, termination/scoring, bitboard helpers
- `src/actions.jl`: `legal_actions`, action encoding/decoding, move generation with forced-move rules
- `src/observation.jl`: 3-tier observation system: `observe_minimal` (30ch), `observe_full` (62ch), `observe_biased` (122ch)

### Game Rules Enforced
- Bar entry priority (must enter from bar before moving other pieces)
- Bearing off validation with over-bear rules (uses precomputed bitmasks)
- Forced maximum dice usage
- Higher die preference when only one die can be used
- Gammon/Backgammon scoring multipliers (reward: 1/2/3)

### Valid Shortcuts in Logic Validation

**Doubles don't need higher-die or multi-action enforcement:**
- For doubles (e.g., rolling 3-3), all 4 moves use the **same die value**
- The "higher die" rule only applies when dice have different values (e.g., 5-3)
- No special validation needed because all moves are equivalent
- The code gates higher-die logic with `if d1 != d2` in `src/actions.jl`

**Why this is correct:**
- Non-doubles (e.g., 5-3): If only one die can be used, must use the higher (5)
- Doubles (e.g., 3-3): All moves are "3", so there's no higher/lower distinction
- The max-dice rule (use as many dice as possible) is still enforced for doubles

**Compile-time validation control:**
- `ENABLE_SANITY_CHECKS` constant controls redundant validation in `is_action_valid`
- Set to `true` during development for correctness verification
- Set to `false` for large-scale training once `legal_actions` is thoroughly tested
- The duplicate validation between `legal_actions` and `is_action_valid` is intentional

**Known behavior - history on error:**
- In `apply_action!`, history is updated before moves are validated
- When `ENABLE_SANITY_CHECKS=true` (default), invalid actions are rejected before history update
- When `ENABLE_SANITY_CHECKS=false`, an invalid action may be added to history before error is thrown
- This is acceptable since the error propagates and the game state is unusable anyway
- Callers catching errors should treat the game as corrupted

### Observation System (3-Tier)

Three observation tiers with increasing feature complexity (shape: `C × 1 × 26`):

| Tier | Channels | Content |
|------|----------|---------|
| `observe_minimal` | 30 | Raw board (threshold 1-6+) + dice (2 slots) + move count (4 bins) + off counts |
| `observe_full` | 62 | + arithmetic features (dice_sum, dice_delta, pips, contact, stragglers, remaining) |
| `observe_biased` | 122 | + strategic features (primes, anchors, blots, builders) |

**Spatial layout (1-indexed, Julia convention):**
- Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
- Indices 2-25: Points 1-24 in canonical order (entry → home)
- Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)

This symmetric layout enables 1D CNN kernels to naturally capture both bar→entry point relationships: my bar adjacent to points 1-6, and opponent's bar adjacent to points 19-24.

**Hierarchy property:** Each tier extends the previous (`full[1:30] == minimal`, `biased[1:62] == full`).

**In-place versions:** `observe_minimal!`, `observe_full!`, `observe_biased!` for high-throughput scenarios (MCTS, batch eval) to avoid GC pressure.

**Design rationale for Width 26:**
- Entry logic: MyBar (Index 1) adjacent to Point 1 (Index 2) allows kernels to learn "can I enter?" patterns
- Defense logic: OppBar (Index 26) adjacent to Point 24 (Index 25) allows kernels to detect home board threats
- Off counts stay as global scalars (channels 29-30) since they don't have spatial adjacency relationships

### Dice Encoding Design (Channels 13-28)

The dice encoding uses 2 one-hot slots (12 channels) + 4-bin move count one-hot (4 channels):

| Channels | Content |
|----------|---------|
| 13-18 | Dice slot 0: high die (one-hot, values 1-6) |
| 19-24 | Dice slot 1: low die (one-hot, values 1-6) |
| 25-28 | Move count one-hot (bins 1, 2, 3, 4 moves) |

**Design philosophy:**
- **Always show rolled values**: Dice slots always show what was rolled (e.g., 5-3 or 4-4), regardless of `remaining_actions`
- **Action mask handles legality**: Network learns current move legality from the legal action mask, not from dice encoding
- **4-bin move count**: Explicitly encodes how many moves this turn (1, 2, 3, or 4)

**Move count encoding semantics:**
- **Doubles**: Compute exact playable moves (1-4) and activate corresponding bin
  - "Strong doubles" (4 moves) vs "weak/blocked doubles" (1-3 moves) are now distinguishable
- **Non-doubles**: Compute exact playable moves (0, 1, or 2) and activate corresponding bin
  - Uses cached `legal_actions` result (see MCTS Integration section below)
- **Chance node** (no dice): All bins zero
- **Completely blocked** (0 moves): All bins zero

**Why 4-bin one-hot over single doubles flag?**
- More explicit representation: network knows exact move count (1, 2, 3, or 4)
- Distinguishes "strong doubles" from "blocked doubles" more explicitly
- Also distinguishes "both dice playable" from "only 1 die playable" for non-doubles
- Only 3 extra channels (16 total vs 13 for old encoding)
- Performance: uses cached `legal_actions` result (no redundant computation)

**Why compute "playable dice" precisely?**
- **Doubles:** "Blocked doubles" occur when the player is primed/trapped and can't use all 4 dice
  - Computing actual playability requires simulating moves via `_compute_playable_dice_doubles`
- **Non-doubles:** Blocking situations where only 1 die (or 0) can be played
  - `_compute_playable_dice_non_doubles` checks legal actions for PASS slots
- Both functions use cached `legal_actions` result via MCTS integration (see below)

### Test Utilities (`test/runtests.jl`)

The `make_test_game` helper uses **perspective-relative** board indices:

```
Board indices 1-24: Points (canonical coordinates)
  - Positive values = my checkers
  - Negative values = opponent checkers

Board indices 25-28 (bars and off):
  - Index 25: My bar (positive count)
  - Index 26: Opponent bar (negative count)
  - Index 27: My off (positive count)
  - Index 28: Opponent off (negative count)
```

When `current_player=0`: "my" = P0, "opponent" = P1
When `current_player=1`: "my" = P1, "opponent" = P0

This abstracts away the physical bitboard layout for easier test writing.

### Performance Tradeoffs in legal_actions

**Why `unique!(actions)` is needed:**
- For non-doubles, paths A (d1 then d2) and B (d2 then d1) can generate the same action
- Example: Moving same checker with both dice may have identical encoded actions
- `unique!` deduplicates in-place to avoid returning duplicate actions

**Why `decode_action` is called in filter! loops:**
- This is called for every action during filtering, which has some overhead
- Alternative: Track max_usage per-action during generation (more complex code)
- Current approach prioritizes correctness and maintainability over micro-optimization
- For RL training, actions are typically sampled from policy, not enumerated

### MCTS Integration & Legal Actions Caching

When used with AlphaZero.jl MCTS, the call sequence during simulation is:

1. **`current_state`** (observation): Called at every node visit to get state for tree lookup
2. **`is_chance_node`**: Check node type
3. **Decision node path:**
   - `available_actions` → calls `legal_actions(g)` on every visit (HOTSPOT)
   - Action selected, then `play!` → calls `apply_action!(g, action)`
4. **Chance node path:**
   - `chance_outcomes(g)` called
   - Game cloned, `apply_chance!(game, outcome)` applied
5. **`vectorize_state`** (neural network input): Called only once per node during expansion

**Performance design decisions:**

**Legal actions caching:**
- `legal_actions()` is cached in `g._actions_buffer` with `g._actions_cached` flag
- Cache is invalidated in `apply_action!`, `apply_chance!`, `switch_turn!`, and `reset!`
- Subsequent calls return cached buffer until state changes
- This is critical since `available_actions` is called on every MCTS visit

**Why cache in game object, not state:**
- The State object (returned by `current_state`) is used as a Dictionary key in MCTS trees
- If cached actions were in State, hashing/equality would become slow and memory would explode
- Cache in mutable Game object, invalidate on state changes

**Observation using cached actions:**
- `observe_minimal/full/biased` can call `legal_actions(g)` internally
- Since `available_actions` is called BEFORE `vectorize_state` in MCTS, the cache is already warm
- No duplicate computation: observation functions hit the cached result

**Clone behavior:**
- Cloning mainly happens at Chance nodes where decision-move cache is empty
- This avoids copying large action lists during stochastic branching
- Use `clone(g)` for safe deep copies with fresh buffers

**AlphaZero.jl Compatibility:**
- AlphaZero directly constructs `BackgammonGame` in `GI.current_state`
- A backwards-compatible constructor preserves the old 12-argument form
- Recommended: Migrate to `clone(g)` for cleaner, future-proof copying
- Internal fields (`_actions_buffer`, `_actions_cached`, etc.) may change between versions

---

## gnubg Validation

### Validation Status: PASSES (0 mismatches)

The Julia implementation has been validated against gnubg by comparing **final board states** after each turn. All unique final positions computed by Julia match exactly what gnubg computes.

**Validated commit:** `c48ef94` (2026-01-27)

**Latest validation run:**
- Final states (hybrid): **500 games, 44,303 positions, 0 mismatches** (~63 pos/sec, ~0.7 games/sec)
- Legal actions (direct): **100 games, 9,004 positions, 0 mismatches**
- Reward validation: **5,000 games, 0 mismatches** (~3,734 games/sec)

### Why Two gnubg Interfaces?

We have two separate interfaces for gnubg, each serving a different purpose:

| Interface | Method | Speed | Used For |
|-----------|--------|-------|----------|
| `GnubgInterface.jl` | PyCall | ~85k evals/sec | Position evaluation, playing games |
| `gnubg_bridge.jl` | CLI | ~63 pos/sec | Validation (enumerating ALL legal moves) |

**Why not just use PyCall for everything?**

1. **PyCall `gnubg.probabilities()`** - Works correctly, returns position evaluation (win/gammon/backgammon probabilities). This is what `GnubgInterface.jl` uses.

2. **PyCall `gnubg.moves()`** - Returns move sequences but has historically shown issues with move legality in some versions. More importantly, it returns move *sequences* (e.g., "13/7") not unique final *positions*, making it unsuitable for validation.

3. **CLI `gnubg hint`** - Returns all legal moves correctly. We use this to validate Julia's `legal_actions()` by comparing final board states.

**Bottom line:** PyCall is fast and works for evaluation. CLI is slow but necessary for comprehensive validation because it's the only reliable way to enumerate all legal moves from gnubg.

### Validation Scripts

| Script | Speed | Description |
|--------|-------|-------------|
| `test/gnubg_hybrid.jl` | ~63 pos/sec, ~0.7 games/sec | Parallel CLI validation (compares final board states) |
| `test/validate_rewards.jl` | ~3,700 games/sec | Fast Julia-only reward validation |

```bash
# Run move validation (uses gnubg CLI, 4 threads)
julia --project -t4 test/gnubg_hybrid.jl 500

# Run reward validation (fast, Julia-only)
julia --project test/validate_rewards.jl 5000
```

### Reward Validation Details

Rewards are validated based on backgammon scoring rules:
- **Normal win (±1)**: Loser has borne off at least one checker
- **Gammon (±2)**: Loser has not borne off any checkers
- **Backgammon (±3)**: Gammon + loser has checker on bar or in winner's home board

**Observed reward distribution (5000 random games):**
| Reward | Type | Count | Percentage |
|--------|------|-------|------------|
| +1 | P0 normal | 941 | 18.8% |
| +2 | P0 gammon | 894 | 17.9% |
| +3 | P0 backgammon | 673 | 13.5% |
| -1 | P1 normal | 917 | 18.3% |
| -2 | P1 gammon | 905 | 18.1% |
| -3 | P1 backgammon | 670 | 13.4% |

This shows ~37% normal wins, ~36% gammons, and ~27% backgammons, which is reasonable for random play.

### Key Implementation Files
- `test/gnubg_bridge.jl` - Core gnubg CLI interface (board conversion, move parsing)
- `test/gnubg_hybrid.jl` - Parallel validation using CLI (compares Julia vs gnubg final states)
- `test/validate_rewards.jl` - Fast Julia-only reward validation
- `test/GnubgInterface.jl` - PyCall-based interface for position evaluation and game playing

### Technical Notes

**Why Final State Comparison (not Move Sequences)**:

Julia and gnubg represent move sequences differently:
- gnubg uses compact notation: "13/7" (one checker moves 13→8→7 using both dice)
- Julia encodes separate sources: (loc1, loc2) where loc1 uses die1, loc2 uses die2

This means gnubg "13/7" vs Julia's "(8→7, 13→8)" are different notations for moves that
reach the SAME final position. The final state comparison correctly validates that all
reachable positions match, which is what matters for game correctness.

**Board Format (for gnubg Python module)**:
```
board[0] = OPPONENT's checkers
board[1] = ON-ROLL player's checkers
```

**Coordinate mappings (0-indexed gnubg_idx)**:
- P0 on roll: `julia_idx = 24 - gnubg_idx`
- P1 on roll: `julia_idx = gnubg_idx + 1`

---

## GnubgInterface Module (test/GnubgInterface.jl)

A clean, high-level interface for integrating gnubg neural network evaluation into Julia code.

### Quick Start

```julia
include("test/GnubgInterface.jl")
using .GnubgInterface

# Evaluate a position
g = BackgammonNet.initial_state()
BackgammonNet.sample_chance!(g, MersenneTwister(42))
equity = evaluate(g)                    # cubeless money equity
probs = evaluate_probs(g)               # (win, win_g, win_bg, lose_g, lose_bg)

# Get best move
action, equity = best_move(g)

# Play games between agents
result = play_game(GnubgAgent(), RandomAgent(seed=1); seed=1, verbose=true)
result = play_match(GnubgAgent(ply=1), RandomAgent(), 100; verbose=true)
```

### Ply Settings (Search Depth)

Control gnubg's search depth for accuracy vs speed tradeoff:

```julia
# Global default
set_default_ply!(1)                     # affects all subsequent calls
get_default_ply()                       # check current default

# Per-call override
evaluate(g; ply=2)                      # 2-ply for this call only
best_move(g; ply=0)                     # neural net only

# Per-agent setting
gnubg_strong = GnubgAgent(ply=2)        # always uses 2-ply
gnubg_fast = GnubgAgent(ply=0)          # always uses 0-ply
```

### Weights & Biases (wandb) Integration

Monitor match progress with wandb logging:

```julia
# Initialize wandb
init_wandb("backgammon-eval", name="gnubg-vs-random", config=Dict("ply"=>2))

# Play match (automatically logs metrics)
result = play_match(GnubgAgent(ply=2), RandomAgent(), 1000; verbose=true)

# Finish logging
finish_wandb()
```

Logged metrics: `games`, `agent0_wins`, `agent1_wins`, `agent0_win_rate`, `agent0_points`, `agent1_points`, `avg_moves`

### Agent Types

| Agent | Description |
|-------|-------------|
| `GnubgAgent(ply=0)` | Uses gnubg neural net evaluation |
| `RandomAgent(seed=42)` | Plays random legal moves |
| `CustomAgent(move_fn)` | Custom function `move_fn(game) -> action` |

### Dice Control

**Julia controls the dice**, not gnubg. In `play_game`, dice are rolled via `BackgammonNet.sample_chance!(g, rng)` using a seeded `MersenneTwister`. gnubg only evaluates positions - it never sees or influences dice rolls.

### Performance Benchmarks

Tested on typical hardware (single-threaded Julia):

| Operation | Ply 0 | Ply 1 | Ply 2 |
|-----------|-------|-------|-------|
| Raw evaluation | ~84,000/sec | ~72,000/sec | ~35/sec |
| Best move selection | ~7,500/sec | ~5,200/sec | ~0.3/sec |
| Games (gnubg vs random) | 29/sec | 22/sec | 0.04/sec |

**Baselines:**
| Operation | Speed |
|-----------|-------|
| Pure Julia (Random vs Random) | 54 games/sec |
| Pure Julia random games (no PyCall) | ~3,500 games/sec |

**Latest benchmark (2026-01-27):**
```
PyCall gnubg.probabilities(): ~84,000 evals/sec
gnubg ply-0 vs Random: 29 games/sec
gnubg ply-1 vs Random: 22 games/sec
gnubg ply-2 vs Random: 0.04 games/sec (~25 sec/game)
```

**Key observations:**
- Ply 0 → Ply 1: ~25% slower (1-ply adds some overhead)
- Ply 1 → Ply 2: ~550x slower (exponential minimax search)
- Pure Julia is ~2x faster than ply-0 (no PyCall overhead)
- For training, ply-0 recommended - ply-2 is far too slow
- gnubg may use internal parallelism for ply-2, but Julia side is single-threaded

### PyCall Configuration

**IMPORTANT:** gnubg must be installed in the Python that PyCall uses.

Check current configuration:
```julia
using PyCall
println(PyCall.python)  # Shows which Python PyCall uses
```

If gnubg is in a different Python (e.g., pyenv), reconfigure PyCall:
```bash
# Set PYTHON to the correct interpreter and rebuild
julia --project -e '
ENV["PYTHON"] = "/path/to/python/with/gnubg"
using Pkg
Pkg.build("PyCall")
'
```

Common issue: PyCall defaults to Julia's Conda Python, but gnubg may be installed in system Python or pyenv. The error message will say `ModuleNotFoundError: No module named 'gnubg'`.

### Lessons Learned

**1. gnubg `best_move()` and `moves()` are buggy:**
- They return illegal moves (don't respect blocking)
- Only `gnubg.probabilities(board, ply)` works correctly
- Solution: Julia generates legal moves + gnubg evaluates positions

**2. Board format is perspective-relative:**
```
board[0] = OPPONENT's checkers (25 elements)
board[1] = ON-ROLL player's checkers (25 elements)
```
Both arrays: index 0 = bar, indices 1-24 = points from on-roll player's perspective.

**3. Ply parameter to gnubg.probabilities():**
- `ply=0`: Neural net evaluation only (fast)
- `ply=1`: 1-ply lookahead (slightly slower, more accurate)
- `ply=2`: 2-ply lookahead (very slow, even more accurate)

**4. Equity calculation from probabilities:**
```julia
equity = (win - lose) + (win_g - lose_g) + 2*(win_bg - lose_bg)
# where lose = 1 - win
```

**5. GnubgAgent win rate vs RandomAgent:**
- Ply 0: ~75% win rate
- Ply 1: ~60% win rate (small sample, expected to be higher with more games)
- gnubg is a strong player even at ply 0
