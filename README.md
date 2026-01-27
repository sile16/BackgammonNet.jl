# BackgammonNet.jl

High-performance Backgammon implementation in Julia, designed for AlphaZero.jl.

## Features
- Optimized bitboard representation (UInt128).
- Player action indices 1-676, encoding two locations (0-25 each): `action = loc1*26 + loc2 + 1`.
- Chance action indices 1-21 for dice outcomes (or 6 specific indices in doubles-only mode).
- Strictly enforced Backgammon rules (forcing moves, max dice usage).
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
- `legal_actions(g)`: Returns valid action indices. At player nodes: 1-676 (encoding two locations). At chance nodes: 1-21 (dice outcomes). In `doubles_only` mode, use `chance_outcomes(g)` to see which have non-zero probability.
- `game_terminated(g)`: Bool.
- `winner(g)`: Returns winning player ID (0 or 1) or `nothing` if not terminated.
- `g.reward`: Player 0's reward (Single: ±1, Gammon: ±2, Backgammon: ±3). Positive if P0 wins.

### Action Encoding (Player Nodes)
Player actions are integers 1-676, encoding two source locations for the two dice:
- Location 0 = bar, 1-24 = board points, 25 = pass
- `encode_action(loc1, loc2) = loc1*26 + loc2 + 1`
- `decode_action(action)` returns `(loc1, loc2)`
- For non-doubles: loc1 uses die1, loc2 uses die2
- For doubles: both locations use the same die value

### Stepping
- `step!(g, action, rng)`: High-level step. Applies action, then `sample_chance!` until deterministic.
- `apply_action!(g, action)`: Applies player move. Errors if called on chance node.
- `apply_chance!(g, outcome_idx)`: Sets dice. Errors if called on player node.
- `sample_chance!(g, rng)`: continuously applies random chance outcomes until state is deterministic.
- `is_chance_node(g)`: Returns true if waiting for dice roll.
- `chance_outcomes(g)`: Returns `[(outcome_idx, prob), ...]`. In `doubles_only` mode, keeps 21 entries with 0 probability for non-doubles.

### Observation (3-Tier System)
Three observation tiers with increasing feature complexity (shape: `C × 1 × 25`):

| Function | Channels | Description |
|----------|----------|-------------|
| `observe_minimal(g)` | 38 | Raw board (threshold encoded) + dice (one-hot, high-to-low) + off counts |
| `observe_full(g)` | 70 | + arithmetic features (dice_sum, dice_delta, pips, contact, etc.) |
| `observe_biased(g)` | 130 | + strategic features (primes, anchors, blots, builders) |

Each tier builds on the previous: `full[1:38] == minimal`, `biased[1:70] == full`.

In-place versions available: `observe_minimal!`, `observe_full!`, `observe_biased!`.

### Initialization Options
- `first_player`: Set to `0` or `1` to choose starting player, or `nothing` for random.
- `short_game`: When `true`, uses a modified board position with pieces closer to bearing off (faster games for training).
- `doubles_only`: When `true`, all dice rolls are doubles (1-1 through 6-6 with uniform probability).

## Performance

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
- `src/game.jl`: Core structs, step logic, and state management.
- `src/actions.jl`: Move generation, validation, and encoding.
- `src/observation.jl`: 3-tier observation system (minimal/full/biased).
- `requirements.txt`: Python dependencies for gnubg integration and wandb logging.
