# BackgammonNet.jl

High-performance Backgammon implementation in Julia, designed for AlphaZero.jl.

## Features
- Optimized `SVector` board representation.
- Full `pgx`-compatible action space and logic.
- Fast and Full observation encoders .
- Strictly enforced Backgammon rules (forcing moves, max dice usage).


## Usage

```julia
using BackgammonNet

game = initial_state()
obs = observe(game) # 86-element vector

actions = legal_actions(game) # Indices 1-676

play!(game, actions[1])
```

## Structure
- `src/game.jl`: Core structs and game flow.
- `src/actions.jl`: Move generation and validation.
- `src/observation.jl`: Vectorization.