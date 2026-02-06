# BackgammonNet.jl

High-performance Backgammon implementation in Julia, designed for AlphaZero.jl.

## Features
- Optimized bitboard representation (UInt128).
- Unified action space (680 actions): checker moves (1-676) + cube actions (677-680).
- Chance action indices 1-21 for dice outcomes (in doubles-only mode, 6 have non-zero probability).
- Strictly enforced Backgammon rules (forcing moves, max dice usage).
- Cube decisions (double/take/pass) with correct ownership tracking.
- Match play with Crawford rule, post-Crawford detection, and Jacoby rule.
- Two modes of operation: Deterministic Step (auto-chance) and Explicit Phase (manual-chance).

## Usage

### 1. Deterministic Mode (RL Standard)
Use `step!` to automatically handle dice rolls. The environment will always return a state where it is a player's turn to move (unless terminated).

```julia
using BackgammonNet
using Random

g = initial_state()
# initial_state starts at a chance node (waiting for roll). 
# We must sample it first if we want to start from a player state, 
# or just use step! which handles it.
sample_chance!(g) 

while !game_terminated(g)
    actions = legal_actions(g)
    
    # Pick an action (e.g., random)
    action = rand(actions)
    
    # step! applies the action AND automatically resolves subsequent chance nodes
    step!(g, action)
    
    println("Current Player: $(current_player(g))")
    println("Board: $(g)")
end
```

### 2. Explicit Phase Mode (Advanced / MCTS)
Control exactly when and how stochastic events (dice rolls) occur.

```julia
using BackgammonNet

g = initial_state()

# check if we need to roll dice
if is_chance_node(g)
    # Get all possible outcomes and probabilities
    outcomes = chance_outcomes(g) # Vector of (outcome_idx, prob)
    
    # Apply a specific outcome (e.g., Roll 3-4)
    # 3-4 is index 13 (see DICE_OUTCOMES)
    apply_chance!(g, 13)
else
    # It's a player's turn
    actions = legal_actions(g)
    action = actions[1]
    
    # Apply the move deterministically
    apply_action!(g, action)
end
```

## API Reference

### Core
- `initial_state(; first_player=nothing, short_game=false, doubles_only=false)`: Returns a new game (starts at chance node).
- `reset!(g; first_player=nothing, short_game=false, doubles_only=false)`: Resets game to initial state without reallocating.
- `clone(g)`: Creates a deep copy with fresh internal buffers (recommended for MCTS).
- `legal_actions(g)`: Returns valid action indices (cached until state changes). Checker play: 1-676. Cube decision: [677, 678]. Cube response: [679, 680]. Chance: 1-21.
- `game_terminated(g)`: Bool.
- `winner(g)`: Returns winning player ID (0 or 1) or `nothing` if not terminated.
- `g.reward`: Player 0's reward. Without cube: ±1/±2/±3. With cube: ±(multiplier × cube_value). Positive if P0 wins.

### Action Encoding (Unified Action Space)
**Checker moves (1-676):** Two source locations for the two dice:
- Location 0 = bar, 1-24 = board points, 25 = pass
- `encode_action(loc1, loc2) = loc1*26 + loc2 + 1`
- `decode_action(action)` returns `(loc1, loc2)`
- For non-doubles: loc1 uses die1, loc2 uses die2
- For doubles: both locations use the same die value

**Cube actions (677-680):**
- `ACTION_CUBE_NO_DOUBLE` (677), `ACTION_CUBE_DOUBLE` (678)
- `ACTION_CUBE_TAKE` (679), `ACTION_CUBE_PASS` (680)
- `MAX_ACTIONS = 680`

### Stepping
- `step!(g, action, rng)`: High-level step. Applies action, then `sample_chance!` until deterministic.
- `apply_action!(g, action)`: Applies player move. Errors if called on chance node.
- `apply_chance!(g, outcome_idx)`: Sets dice. Errors if called on player node.
- `sample_chance!(g, rng)`: continuously applies random chance outcomes until state is deterministic.
- `is_chance_node(g)`: Returns true if waiting for dice roll.
- `chance_outcomes(g)`: Returns `[(outcome_idx, prob), ...]`. In `doubles_only` mode, keeps 21 entries with 0 probability for non-doubles.

### Observation (3-Tier System)
Three observation tiers with increasing feature complexity (shape: `C × 1 × 26`):

| Function | Channels | Description |
|----------|----------|-------------|
| `observe_minimal(g)` | 42 | Raw board (threshold encoded) + dice (2 slots) + move count (4 bins) + off counts + cube/match state (12ch) |
| `observe_full(g)` | 74 | + arithmetic features (dice_sum, dice_delta, pips, contact, etc.) |
| `observe_biased(g)` | 134 | + strategic features (primes, anchors, blots, builders) |

Also available: flat (342/374/434 values), hybrid (board 12×26 + globals 30/62/122).

**Spatial Layout (1-indexed, Julia convention):**
- Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
- Indices 2-25: Points 1-24 in canonical order (entry → home)
- Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)

Each tier builds on the previous: `full[1:42] == minimal`, `biased[1:74] == full`.

In-place versions available: `observe_minimal!`, `observe_full!`, `observe_biased!`.

### Cube & Match Play
```julia
# Enable cube decisions
g = initial_state()
g.cube_enabled = true

# Set up match play (7-point match, score 4-2, Crawford game)
init_match_game!(g, my_score=4, opp_score=2, match_length=7, is_crawford=true)

# Context observation for policy conditioning (12 features)
ctx = context_observation(g)
```

### Initialization Options
- `first_player`: Set to `0` or `1` to choose starting player, or `nothing` for random.
- `short_game`: When `true`, uses a symmetric board position with 113 pips per player (45.7% shorter than standard 208 pips) for faster games.
- `doubles_only`: When `true`, all dice rolls are doubles (1-1 through 6-6 with uniform probability).

## Performance

### Caching & Thread Safety
- `legal_actions(g)` results are cached in the game object until state changes
- Cache is automatically invalidated by `apply_action!`, `apply_chance!`, `step!`, `reset!`
- **Not thread-safe**: Don't call `legal_actions` or `is_action_valid` concurrently on the same game object
- For MCTS: Use `clone(g)` to create independent game copies

### Benchmarks
Benchmarked on Apple M1 MacBook (single core) without sanity checks:

| Metric | Value |
|--------|-------|
| Games/sec | ~4,668 |
| Actions/sec | ~491,000 |
| Avg. actions/game | ~112 |

With sanity checks
   Mode        │ Games/sec │ Actions/sec │ Overhead │
  ├───────────────────┼───────────┼─────────────┼──────────┤
  │ Sanity Checks ON  │ 3,452     │ 362,744     │ -26%     │

Run the benchmark yourself:
```bash
julia --project benchmark.jl
```

## Structure
- `src/game.jl`: Core structs, step logic, state management, cube/match state, `GamePhase` enum.
- `src/actions.jl`: Move generation, validation, encoding, cube action constants (677-680), `may_double`.
- `src/observation.jl`: 3-tier observation system (minimal/full/biased), cube/match channels (31-42), `context_observation`.
- `tools/validate_cube_match.jl`: Comprehensive cube/match validation (637 checks).
- `requirements.txt`: Python dependencies for gnubg integration and wandb logging.
