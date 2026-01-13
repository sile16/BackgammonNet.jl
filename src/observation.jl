# --- Observation Constants ---

# Canonical board indices
const BAR_IDX = 25
const OPP_BAR_IDX = 26

# Feature vector sizes
const OBS_SIZE_FAST = 34        # Compact observation size
const OBS_SIZE_FULL = 86        # Full observation size

# Normalization constants
const CHECKER_NORM = 15.0f0     # Max checkers per player (for board normalization)
const DICE_NORM = 4.0f0         # Max dice count for doubles (remaining_actions * 2)
const SINGLE_DIE_VALUE = 0.25f0 # 1/4 for non-doubles (one die out of possible 4)
const PIP_NORM = 375.0f0        # Max pip difference for normalization (~25 * 15)

# Feature section offsets
const DICE_OFFSET = 28          # Dice features start after board (indices 29-34)
const BLOT_OFFSET = 38          # Blot detection starts at index 39
const BLOCK_OFFSET = 62         # Block detection starts at index 63

# Home board boundaries (canonical indices)
const MY_HOME_START = 19        # My home board: 19-24 (canonical)
const OPP_HOME_END = 6          # Opponent home board: 1-6 (canonical)
const BAR_PIP_VALUE = 25        # Pip value for checkers on bar

# Heuristic feature indices
const IDX_RACE = 35             # Race indicator
const IDX_CAN_BEAR_MY = 36      # Can current player bear off
const IDX_CAN_BEAR_OPP = 37     # Can opponent bear off
const IDX_PIP_DIFF = 38         # Normalized pip count difference

"""
    _encode_dice!(obs::AbstractVector{Float32}, d1, d2, remaining_actions)

Internal helper to encode dice values into observation vector.
Writes to indices DICE_OFFSET+1 through DICE_OFFSET+6.

For doubles: encodes remaining dice count (remaining_actions * 2) / 4 at die position.
For non-doubles: encodes 0.25 at each die position.
"""
@inline function _encode_dice!(obs::AbstractVector{Float32}, d1, d2, remaining_actions)
    if remaining_actions > 0 && d1 > 0 && d2 > 0
        if d1 == d2
            # Doubles: remaining_actions: 2 -> 4 dice, 1 -> 2 dice
            count = Float32(Int(remaining_actions) * 2) / DICE_NORM
            @inbounds obs[DICE_OFFSET + d1] = count
        else
            # Non-doubles: each die contributes 1/4 of max
            @inbounds obs[DICE_OFFSET + d1] = SINGLE_DIE_VALUE
            @inbounds obs[DICE_OFFSET + d2] = SINGLE_DIE_VALUE
        end
    end
    return nothing
end

"""
    observe_fast(g::BackgammonGame) -> Vector{Float32}

Generate a compact 34-element feature vector for the current game state.

# Feature Layout (34 elements)
- Indices 1-28: Board positions normalized by 15 (my checkers positive, opponent negative)
  - 1-24: Points (canonical view from current player's perspective)
  - 25: My bar
  - 26: Opponent's bar
  - 27: My borne off
  - 28: Opponent's borne off
- Indices 29-34: Dice encoding (one-hot style, normalized by 4 for doubles count)
  - Non-doubles: 0.25 at each die position
  - Doubles: (remaining_actions * 2) / 4 at the die position

This is a minimal observation suitable for simple policies or when memory is constrained.
For richer features including blot/block detection and heuristics, use `observe_full`.
"""
function observe_fast(g::BackgammonGame)
    obs = zeros(Float32, OBS_SIZE_FAST)

    # Board positions normalized by max checkers
    @inbounds for i in 1:DICE_OFFSET
        obs[i] = Float32(g[i]) / CHECKER_NORM
    end

    # Dice encoding
    _encode_dice!(obs, g.dice[1], g.dice[2], g.remaining_actions)

    return obs
end

"""
    observe_full(g::BackgammonGame) -> Vector{Float32}

Generate a comprehensive 86-element feature vector for the current game state.

This is the default observation used by `vector_observation` and `observe`.

# Feature Layout (86 elements)
- Indices 1-28: Board positions normalized by 15 (same as `observe_fast`)
- Indices 29-34: Dice encoding (same as `observe_fast`)
- Index 35: Race indicator (1.0 if no contact possible, 0.0 otherwise)
- Index 36: Can bear off (current player) - 1.0 if all checkers in home board
- Index 37: Can bear off (opponent) - 1.0 if all opponent checkers in their home
- Index 38: Pip count difference normalized by 375 ((my_pip - opp_pip) / 375)
- Indices 39-62: Blot detection (24 points)
  - 1.0 if my blot (single checker), -1.0 if opponent blot, 0.0 otherwise
- Indices 63-86: Block detection (24 points)
  - 1.0 if my block (2+ checkers), -1.0 if opponent block, 0.0 otherwise

# Performance Note
This function allocates a new vector on each call. For high-throughput scenarios,
consider using `observe_full!` (in-place version) with a pre-allocated buffer.

See also: `observe_fast`, `observe_full!`
"""
function observe_full(g::BackgammonGame)
    obs = zeros(Float32, OBS_SIZE_FULL)

    # Extract all 28 board values ONCE into stack-allocated array
    vals = MVector{28, Int8}(undef)
    @inbounds for i in 1:DICE_OFFSET
        vals[i] = g[i]
        obs[i] = Float32(vals[i]) / CHECKER_NORM
    end

    # Dice encoding
    _encode_dice!(obs, g.dice[1], g.dice[2], g.remaining_actions)

    # Heuristics - use cached vals instead of g[i]
    min_my = BAR_PIP_VALUE
    max_opp = 0
    my_pip = 0
    opp_pip = 0
    can_bear_my = (vals[BAR_IDX] == 0)
    can_bear_opp = (vals[OPP_BAR_IDX] == 0)

    @inbounds for i in 1:NUM_POINTS
        val = vals[i]
        if val > 0
            if i < min_my; min_my = i; end
            my_pip += val * (BAR_PIP_VALUE - i)
            # Check bearing off for my checkers (need all in home 19-24)
            if can_bear_my && i < MY_HOME_START
                can_bear_my = false
            end
        elseif val < 0
            if i > max_opp; max_opp = i; end
            opp_pip += (-val) * i
            # Check bearing off for opp checkers (need all in home 1-6)
            if can_bear_opp && i > OPP_HOME_END
                can_bear_opp = false
            end
        end

        # Blot/Block detection
        idx_blot = BLOT_OFFSET + i
        idx_block = BLOCK_OFFSET + i
        if val == 1
            obs[idx_blot] = 1.0f0
        elseif val == -1
            obs[idx_blot] = -1.0f0
        end
        if val >= 2
            obs[idx_block] = 1.0f0
        elseif val <= -2
            obs[idx_block] = -1.0f0
        end
    end

    @inbounds begin
        my_pip += vals[BAR_IDX] * BAR_PIP_VALUE
        opp_pip += (-vals[OPP_BAR_IDX]) * BAR_PIP_VALUE

        is_race = (vals[BAR_IDX] == 0 && vals[OPP_BAR_IDX] == 0 && min_my > max_opp)
        obs[IDX_RACE] = Float32(is_race)
        obs[IDX_CAN_BEAR_MY] = Float32(can_bear_my)
        obs[IDX_CAN_BEAR_OPP] = Float32(can_bear_opp)
        obs[IDX_PIP_DIFF] = (my_pip - opp_pip) / PIP_NORM
    end

    return obs
end

"""
    observe_fast!(obs::AbstractVector{Float32}, g::BackgammonGame) -> AbstractVector{Float32}

In-place version of `observe_fast`. Fills the first $(OBS_SIZE_FAST) elements of `obs`.

For high-throughput scenarios, pre-allocate a buffer and reuse it:

```julia
obs = Vector{Float32}(undef, 34)
for game in games
    observe_fast!(obs, game)
    # use obs...
end
```

See also: `observe_fast`, `observe_full!`
"""
function observe_fast!(obs::AbstractVector{Float32}, g::BackgammonGame)
    # Zero out the buffer
    @inbounds for i in 1:OBS_SIZE_FAST
        obs[i] = 0.0f0
    end

    # Board positions normalized by max checkers
    @inbounds for i in 1:DICE_OFFSET
        obs[i] = Float32(g[i]) / CHECKER_NORM
    end

    # Dice encoding
    _encode_dice!(obs, g.dice[1], g.dice[2], g.remaining_actions)

    return obs
end

"""
    observe_full!(obs::AbstractVector{Float32}, g::BackgammonGame) -> AbstractVector{Float32}

In-place version of `observe_full`. Fills the first $(OBS_SIZE_FULL) elements of `obs`.

For high-throughput scenarios, pre-allocate a buffer and reuse it:

```julia
obs = Vector{Float32}(undef, 86)
for game in games
    observe_full!(obs, game)
    # use obs...
end
```

See also: `observe_full`, `observe_fast!`
"""
function observe_full!(obs::AbstractVector{Float32}, g::BackgammonGame)
    # Zero out the buffer
    @inbounds for i in 1:OBS_SIZE_FULL
        obs[i] = 0.0f0
    end

    # Extract all 28 board values ONCE into stack-allocated array
    vals = MVector{28, Int8}(undef)
    @inbounds for i in 1:DICE_OFFSET
        vals[i] = g[i]
        obs[i] = Float32(vals[i]) / CHECKER_NORM
    end

    # Dice encoding
    _encode_dice!(obs, g.dice[1], g.dice[2], g.remaining_actions)

    # Heuristics
    min_my = BAR_PIP_VALUE
    max_opp = 0
    my_pip = 0
    opp_pip = 0
    can_bear_my = (vals[BAR_IDX] == 0)
    can_bear_opp = (vals[OPP_BAR_IDX] == 0)

    @inbounds for i in 1:NUM_POINTS
        val = vals[i]
        if val > 0
            if i < min_my; min_my = i; end
            my_pip += val * (BAR_PIP_VALUE - i)
            if can_bear_my && i < MY_HOME_START
                can_bear_my = false
            end
        elseif val < 0
            if i > max_opp; max_opp = i; end
            opp_pip += (-val) * i
            if can_bear_opp && i > OPP_HOME_END
                can_bear_opp = false
            end
        end

        # Blot/Block detection
        idx_blot = BLOT_OFFSET + i
        idx_block = BLOCK_OFFSET + i
        if val == 1
            obs[idx_blot] = 1.0f0
        elseif val == -1
            obs[idx_blot] = -1.0f0
        end
        if val >= 2
            obs[idx_block] = 1.0f0
        elseif val <= -2
            obs[idx_block] = -1.0f0
        end
    end

    @inbounds begin
        my_pip += vals[BAR_IDX] * BAR_PIP_VALUE
        opp_pip += (-vals[OPP_BAR_IDX]) * BAR_PIP_VALUE

        is_race = (vals[BAR_IDX] == 0 && vals[OPP_BAR_IDX] == 0 && min_my > max_opp)
        obs[IDX_RACE] = Float32(is_race)
        obs[IDX_CAN_BEAR_MY] = Float32(can_bear_my)
        obs[IDX_CAN_BEAR_OPP] = Float32(can_bear_opp)
        obs[IDX_PIP_DIFF] = (my_pip - opp_pip) / PIP_NORM
    end

    return obs
end

const vector_observation = observe_full