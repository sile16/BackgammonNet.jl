# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `loc1`: source for die 1 (0=bar, 1-24=points, 25=pass)
- `loc2`: source for die 2

For doubles, two actions are needed per turn (`remaining_actions = 2`).

### Two-Mode API
1. **Deterministic (`step!`)**: Auto-rolls dice after each action. Used for RL training.
2. **Explicit (`apply_action!`/`apply_chance!`)**: Manual dice control. Used for MCTS.

A game starts at a chance node (dice = [0,0]). Call `sample_chance!` or use `step!` to roll.

### Key Files
- `src/game.jl`: `BackgammonGame` struct, move application, termination/scoring, bitboard helpers
- `src/actions.jl`: `legal_actions`, action encoding/decoding, move generation with forced-move rules
- `src/observation.jl`: 3-tier observation system: `observe_minimal` (38ch), `observe_full` (70ch), `observe_biased` (130ch)

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
| `observe_minimal` | 38 | Raw board (threshold 1-6+) + dice (one-hot 4×6) + off counts |
| `observe_full` | 70 | + arithmetic features (dice_sum, dice_delta, pips, contact, stragglers, remaining) |
| `observe_biased` | 130 | + strategic features (primes, anchors, blots, builders) |

**Spatial layout (1-indexed, Julia convention):**
- Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
- Indices 2-25: Points 1-24 in canonical order (entry → home)
- Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)

This symmetric layout enables 1D CNN kernels to naturally capture both bar→entry point relationships: my bar adjacent to points 1-6, and opponent's bar adjacent to points 19-24.

**Hierarchy property:** Each tier extends the previous (`full[1:38] == minimal`, `biased[1:70] == full`).

**In-place versions:** `observe_minimal!`, `observe_full!`, `observe_biased!` for high-throughput scenarios (MCTS, batch eval) to avoid GC pressure.

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

---

## gnubg Validation

### Validation Status: PASSES (0 mismatches)

The Julia implementation has been validated against gnubg by comparing **final board states** after each turn. All unique final positions computed by Julia match exactly what gnubg computes.

**Validated commit:** `a5e7db3` (2026-01-16)

**Latest validation run:**
- Move validation: **500 games, 46,174 positions, 0 mismatches** (~78 pos/sec)
- Reward validation: **5,000 games, 0 mismatches** (~3,734 games/sec)

### Why Two gnubg Interfaces?

We have two separate interfaces for gnubg, each serving a different purpose:

| Interface | Method | Speed | Used For |
|-----------|--------|-------|----------|
| `GnubgInterface.jl` | PyCall | ~85k evals/sec | Position evaluation, playing games |
| `gnubg_bridge.jl` | CLI | ~78 pos/sec | Validation (enumerating ALL legal moves) |

**Why not just use PyCall for everything?**

1. **PyCall `gnubg.probabilities()`** - Works correctly, returns position evaluation (win/gammon/backgammon probabilities). This is what `GnubgInterface.jl` uses.

2. **PyCall `gnubg.moves()`** - Returns move sequences but has historically shown issues with move legality in some versions. More importantly, it returns move *sequences* (e.g., "13/7") not unique final *positions*, making it unsuitable for validation.

3. **CLI `gnubg hint`** - Returns all legal moves correctly. We use this to validate Julia's `legal_actions()` by comparing final board states.

**Bottom line:** PyCall is fast and works for evaluation. CLI is slow but necessary for comprehensive validation because it's the only reliable way to enumerate all legal moves from gnubg.

### Validation Scripts

| Script | Speed | Description |
|--------|-------|-------------|
| `test/gnubg_hybrid.jl` | ~0.8 games/sec | Parallel CLI validation (compares final board states) |
| `test/validate_rewards.jl` | ~3,700 games/sec | Fast Julia-only reward validation |

```bash
# Run move validation
julia --project -t 4 test/gnubg_hybrid.jl 500

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

Tested on typical hardware:

| Operation | Ply 0 | Ply 1 | Ply 2 |
|-----------|-------|-------|-------|
| Raw evaluation | 85,706/sec | 72,480/sec | 35/sec |
| Best move selection | 7,561/sec | 5,193/sec | ~0.3/sec |
| Games (gnubg vs random) | 46/sec | 1.5/sec | very slow |

**Baselines:**
| Operation | Speed |
|-----------|-------|
| Random vs Random (with equity logging) | 422 games/sec |
| Pure Julia random games (no gnubg) | 3,547 games/sec |

**Key observations:**
- Ply 0 and Ply 1 have similar raw eval speed (~72-86k/sec)
- Ply 2 is ~2000x slower (gnubg does deep minimax search)
- `play_game` calls `evaluate()` for equity logging on every move, adding overhead
- Main bottleneck at ply 0: ~11 legal moves per position × evaluation cost

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
