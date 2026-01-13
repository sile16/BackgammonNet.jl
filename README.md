# BackgammonNet.jl

High-performance Backgammon implementation in Julia, designed for AlphaZero.jl.

## Features
- Optimized bitboard representation (UInt128).
- Canonical Action Space (0-25) and Observation Space.
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
- `legal_actions(g)`: Returns valid action indices.
- `game_terminated(g)`: Bool.
- `winner(g)`: Returns winning player ID (0 or 1) or `nothing`.
- `reward(g)`: Access `.reward` field for Player 0's perspective (Single: 1, Gammon: 2, Backgammon: 3).

### Stepping
- `step!(g, action, rng)`: High-level step. Applies action, then `sample_chance!` until deterministic.
- `apply_action!(g, action)`: Applies player move. Errors if called on chance node.
- `apply_chance!(g, outcome_idx)`: Sets dice. Errors if called on player node.
- `sample_chance!(g, rng)`: continuously applies random chance outcomes until state is deterministic.
- `is_chance_node(g)`: Returns true if waiting for dice roll.
- `chance_outcomes(g)`: Returns `[(outcome_idx, prob), ...]`.

### Observation
- `vector_observation(g)`: AlphaZero-compatible feature vector.

### Initialization Options
- `first_player`: Set to `0` or `1` to choose starting player, or `nothing` for random.
- `short_game`: When `true`, uses a modified board position with pieces closer to bearing off (faster games for training).
- `doubles_only`: When `true`, all dice rolls are doubles (1-1 through 6-6 with uniform probability).

## Performance

Benchmarked on Apple M1 MacBook (single core):

| Metric | Value |
|--------|-------|
| Games/sec | ~7,100 |
| Actions/sec | ~800,000 |
| Observations/sec | ~800,000 |
| Avg. actions/game | ~112 |

Run the benchmark yourself:
```bash
julia --project benchmark.jl
```

## Structure
- `src/game.jl`: Core structs, step logic, and state management.
- `src/actions.jl`: Move generation, validation, and encoding.
- `src/observation.jl`: Vectorization and canonical views.
