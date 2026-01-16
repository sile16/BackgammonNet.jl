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
- `src/observation.jl`: `vector_observation` (86-dim) and `observe_fast` (34-dim) for neural network input

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

### Observation Code Duplication

The observation functions appear duplicated but serve different purposes:
- `observe_fast` / `observe_full`: Allocating versions for convenience
- `observe_fast!` / `observe_full!`: In-place versions for high-throughput scenarios

This is an intentional performance pattern. For MCTS or batch evaluation, use the in-place versions
with pre-allocated buffers to avoid GC pressure.

### Performance Tradeoffs in legal_actions

**Why `unique!(actions)` is needed (line 250):**
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

**Latest validation run (5000 games):**
- Move validation: **0 mismatches** (460,945 positions checked vs gnubg)
- Reward validation: **0 mismatches** (normal/gammon/backgammon scoring)

### Validation Scripts

| Script | Speed | Status | Description |
|--------|-------|--------|-------------|
| `test/gnubg_hybrid.jl` | ~0.9 games/sec | **PASSES** | Parallel workers + small batches (recommended) |
| `test/gnubg_parallel.jl` | ~0.5 games/sec | **PASSES** | Parallel gnubg processes |
| `test/validate_full.jl` | ~0.1 games/sec | **PASSES** | Single-threaded CLI validation |
| `test/validate_rewards.jl` | ~3500 games/sec | **PASSES** | Fast reward-only validation (Julia-only) |
| `test/validate_games.jl` | ~0.09 games/sec | **PASSES** | Full game + reward validation |

**Recommended for validation**: `gnubg_hybrid.jl` for move validation + `validate_rewards.jl` for reward validation

```bash
# Run move validation (recommended)
julia --project -t 4 test/gnubg_hybrid.jl 100

# Run reward validation (fast, Julia-only)
julia --project test/validate_rewards.jl 5000

# Full validation (slow but thorough)
julia --project test/validate_full.jl 1000
```

### Reward Validation Details

Rewards are validated based on backgammon scoring rules:
- **Normal win (±1)**: Loser has borne off at least one checker
- **Gammon (±2)**: Loser has not borne off any checkers
- **Backgammon (±3)**: Gammon + loser has checker on bar or in winner's home board

**Observed reward distribution (5000 random games):**
| Reward | Type | Count | Percentage |
|--------|------|-------|------------|
| +1 | P0 normal | 931 | 18.6% |
| +2 | P0 gammon | 898 | 18.0% |
| +3 | P0 backgammon | 679 | 13.6% |
| -1 | P1 normal | 927 | 18.5% |
| -2 | P1 gammon | 901 | 18.0% |
| -3 | P1 backgammon | 664 | 13.3% |

This shows ~37% normal wins, ~36% gammons, and ~27% backgammons, which is reasonable for random play.

### Key Implementation Files
- `test/gnubg_bridge.jl` - Core gnubg CLI interface (board conversion, move parsing)
- `test/gnubg_hybrid.jl` - Fast parallel validation (recommended)
- `test/gnubg_parallel.jl` - Parallel single-query validation
- `test/validate_full.jl` - Original single-threaded validation

### Technical Notes

**PyCall Interface (gnubg.moves())**: The Python gnubg module's `moves()` function returns move sequences, NOT unique final positions. This makes it unsuitable for direct validation. The CLI `hint` command is used instead.

**Board Format (for gnubg Python module)**:
```
board[0] = OPPONENT's checkers
board[1] = ON-ROLL player's checkers
```

**Coordinate mappings (0-indexed gnubg_idx)**:
- P0 on roll: `julia_idx = 24 - gnubg_idx`
- P1 on roll: `julia_idx = gnubg_idx + 1`

**Why Final State Comparison (not Move Sequences)**:

Julia and gnubg represent move sequences differently:
- gnubg uses compact notation: "13/7" (one checker moves 13→8→7 using both dice)
- Julia encodes separate sources: (loc1, loc2) where loc1 uses die1, loc2 uses die2

This means gnubg "13/7" vs Julia's "(8→7, 13→8)" are different notations for moves that
reach the SAME final position. The final state comparison correctly validates that all
reachable positions match, which is what matters for game correctness.

`test/validate_moves.jl` compares move sequences directly but shows representation differences
(not logic bugs). Use final state validation for correctness checking.

---

## gnubg Fast Evaluation Interface (test/gnubg_pycall.jl)

### Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| `gnubg.probabilities()` | ~131k/sec | Raw neural net evaluation |
| `evaluate_position_gnubg()` | ~48k/sec | With board conversion |
| `get_best_move_hybrid()` | ~1.4k/sec | Julia moves + gnubg eval |
| `play_game_hybrid()` | ~1.5 games/sec | Full games vs gnubg |

### Key Functions

```julia
include("test/gnubg_pycall.jl")

evaluate_position_gnubg(g)    # Fast position evaluation (~48k/sec)
get_best_move_hybrid(g)       # Best move using Julia moves + gnubg eval
play_game_hybrid(julia_player; seed, verbose)  # Play game vs gnubg
play_game_vs_gnubg_cli(julia_player; seed, verbose)  # CLI-based (slower, for comparison)
```

### Lessons Learned

**1. gnubg Python module `best_move()` and `moves()` are BUGGY:**
- They don't respect blocking rules and return illegal moves
- Only `gnubg.probabilities()` works correctly for neural net evaluation
- Use hybrid approach: Julia generates legal moves + gnubg evaluates positions

**2. Direct ccall to gnubg C library doesn't work:**
- gnubg-nn-pypi is a Python extension module (.cpython-*.so)
- Requires full Python runtime, sys.path, and import machinery
- Attempting direct ccall causes segfaults in `BearoffInit`
- PyCall is fast enough (~131k raw evals/sec)

**3. Board format for gnubg Python module:**
```
board[0] = OPPONENT's checkers (25 elements: bar + 24 points)
board[1] = ON-ROLL player's checkers
```

**4. Position ID validation:**
- Known gnubg starting position ID: `4HPwATDgc/ABMA`
- Python module produces different ID due to board format bug
- Use final board state comparison, not position IDs

**5. Move notation differences:**
- gnubg: compact notation like "13/7" (one checker moves 13→8→7)
- Julia: separate sources (loc1, loc2) where each uses one die
- Same final positions, different representations

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
