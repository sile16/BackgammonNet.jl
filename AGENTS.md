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
