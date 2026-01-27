# ============================================================================
# Observation Module - Three-Tier Neural Network Input Encoding
# ============================================================================
#
# Provides three observation types that build on each other:
#   - Minimal (38 channels): Raw board + dice only
#   - Full (70 channels): + arithmetic features (no strategic bias)
#   - Biased (130 channels): + hand-crafted strategic features
#
# Shape: (C, 1, 25) where width = 1 bar + 24 board points
#
# Spatial Layout (for CNN topology):
#   Index 1: Bar (adjacent to entry points for 1D convolutions)
#   Indices 2-25: Points 1-24 in canonical order (entry → home)
#
# Design Philosophy:
#   - Minimal: Network must learn everything from raw state
#   - Full: Pre-compute arithmetic to save network capacity
#   - Biased: Include TD-Gammon/gnubg strategic features
#
# ============================================================================

# --- Observation Dimensions ---
const OBS_WIDTH = 25  # Bar at index 1, points 1-24 at indices 2-25

# Channel counts for each tier
const OBS_CHANNELS_MINIMAL = 38
const OBS_CHANNELS_FULL = 70      # 38 minimal + 32 full features (39-70)
const OBS_CHANNELS_BIASED = 130   # 70 full + 60 biased features (71-130)

# --- Normalization Constants ---
const OFF_NORM = 15.0f0           # Max checkers per player
const PIP_NORM = 167.0f0          # Starting pip count (natural scale)
const DICE_SUM_NORM = 12.0f0      # Max dice sum (6+6)
const DICE_DELTA_NORM = 5.0f0     # Max dice delta |6-1|
const OVERFLOW_NORM = 10.0f0      # For 6+ threshold encoding: (n-5)/10

# --- Board Layout Constants ---
const MY_HOME_START = 19          # My home board: points 19-24 (canonical)
const OPP_HOME_END = 6            # Opponent home board: points 1-6 (canonical)
const BAR_PIP_VALUE = 25          # Pip value for checkers on bar

# ============================================================================
# Channel Layout Documentation (1-indexed)
# ============================================================================
#
# MINIMAL (38 channels, 1-38):
#   1-6:   My checker thresholds (>=1, >=2, >=3, >=4, >=5, 6+)
#   7-12:  Opponent checker thresholds
#   13-18: Dice slot 0 one-hot (values 1-6)
#   19-24: Dice slot 1 one-hot
#   25-30: Dice slot 2 one-hot
#   31-36: Dice slot 3 one-hot
#   37:    My off count (/15)
#   38:    Opponent off count (/15)
#
# FULL adds 32 channels (39-70):
#   39:    dice_sum (/12)
#   40:    dice_delta |d1-d2| (/5)
#   41:    Contact indicator (1=contact, 0=race)
#   42:    My pip count (/167)
#   43:    Opponent pip count (/167)
#   44:    Pip difference (my-opp, /167, clipped [-1,1])
#   45:    Can bear off (me)
#   46:    Can bear off (opponent)
#   47-52: My stragglers (outside home) threshold
#   53-58: Opponent stragglers threshold
#   59-64: My remaining (15-off) threshold
#   65-70: Opponent remaining threshold
#
# BIASED adds 60 channels (71-130):
#   71-76:   My prime length threshold
#   77-82:   Opponent prime length threshold
#   83-88:   My home board blocks threshold
#   89-94:   Opponent home board blocks threshold
#   95-100:  My anchors (in opponent's home) threshold
#   101-106: Opponent anchors threshold
#   107-112: My blot count threshold
#   113-118: Opponent blot count threshold
#   119-124: My builder count threshold
#   125-130: Opponent builder count threshold
#
# ============================================================================

# --- Minimal Observation Helpers ---

"""
    _get_checker_counts(g::BackgammonGame, pt::Int) -> (my::Int, opp::Int)

Get checker counts at a board point (1-24) from current player's perspective.
Returns (my_count, opp_count) where both are non-negative.
"""
@inline function _get_checker_counts(g::BackgammonGame, pt::Int)
    val = g[pt]  # Uses canonical accessor from game.jl
    if val > 0
        return (Int(val), 0)
    elseif val < 0
        return (0, Int(-val))
    else
        return (0, 0)
    end
end

"""
    _get_bar_counts(g::BackgammonGame) -> (my::Int, opp::Int)

Get bar checker counts for both players.
"""
@inline function _get_bar_counts(g::BackgammonGame)
    my_bar = Int(g[25])      # My bar (positive)
    opp_bar = Int(-g[26])    # Opp bar (stored negative, convert to positive)
    return (my_bar, opp_bar)
end

"""
    _get_off_counts(g::BackgammonGame) -> (my::Int, opp::Int)

Get borne-off checker counts for both players.
"""
@inline function _get_off_counts(g::BackgammonGame)
    my_off = Int(g[27])      # My off (positive)
    opp_off = Int(-g[28])    # Opp off (stored negative, convert to positive)
    return (my_off, opp_off)
end

"""
    _encode_threshold_6!(obs, ch_base, w, count)

Encode count using 1,2,3,4,5,6+ threshold scheme at a single spatial position.
Channel 5 uses overflow encoding: (count-5)/10 for count >= 6.
"""
@inline function _encode_threshold_6!(obs::AbstractArray{Float32,3}, ch_base::Int, w::Int, count::Int)
    @inbounds begin
        obs[ch_base + 1, 1, w] = Float32(count >= 1)
        obs[ch_base + 2, 1, w] = Float32(count >= 2)
        obs[ch_base + 3, 1, w] = Float32(count >= 3)
        obs[ch_base + 4, 1, w] = Float32(count >= 4)
        obs[ch_base + 5, 1, w] = Float32(count >= 5)
        obs[ch_base + 6, 1, w] = count >= 6 ? Float32(count - 5) / OVERFLOW_NORM : 0.0f0
    end
    return nothing
end

"""
    _encode_board!(obs, g)

Encode board state using threshold encoding (channels 1-12).
Per-point encoding with 6 channels per player.

Spatial layout: [Bar, Point1, Point2, ..., Point24]
- Index 1: Bar (for CNN topology - adjacent to entry points 1-6)
- Indices 2-25: Points 1-24 in canonical order
"""
function _encode_board!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    # Bar at spatial index 1 (for CNN topology - adjacent to entry points)
    my_bar, opp_bar = _get_bar_counts(g)
    _encode_threshold_6!(obs, 0, 1, my_bar)
    _encode_threshold_6!(obs, 6, 1, opp_bar)

    # Points 1-24 at spatial indices 2-25
    @inbounds for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)
        _encode_threshold_6!(obs, 0, pt + 1, my_count)
        _encode_threshold_6!(obs, 6, pt + 1, opp_count)
    end

    return nothing
end

"""
    _get_dice_slots(g::BackgammonGame) -> NTuple{4, Int}

Get 4 dice slots ordered high to low. Empty/used slots are 0.
For doubles, all 4 slots have the same value (reduced as dice are used).
For non-doubles, slots 0-1 have die values, slots 2-3 are 0.
"""
@inline function _get_dice_slots(g::BackgammonGame)
    d1, d2 = g.dice[1], g.dice[2]
    remaining = g.remaining_actions

    if d1 == 0 || d2 == 0 || remaining == 0
        return (0, 0, 0, 0)
    end

    if d1 == d2
        # Doubles: remaining_actions = 2 means 4 dice, 1 means 2 dice
        # But we show all 4 initially, then clear as used
        # remaining=2 -> 4 dice, remaining=1 -> 2 dice
        n_dice = Int(remaining) * 2
        return (
            n_dice >= 1 ? Int(d1) : 0,
            n_dice >= 2 ? Int(d1) : 0,
            n_dice >= 3 ? Int(d1) : 0,
            n_dice >= 4 ? Int(d1) : 0
        )
    else
        # Non-doubles: 2 dice, order high to low
        high = max(Int(d1), Int(d2))
        low = min(Int(d1), Int(d2))
        if remaining == 2
            return (high, low, 0, 0)
        else  # remaining == 1, one die used
            # We don't know which was used, show both (network learns from context)
            return (high, low, 0, 0)
        end
    end
end

"""
    _get_original_dice_sum(g::BackgammonGame) -> Int

Get sum of original dice roll (before any moves).
"""
@inline function _get_original_dice_sum(g::BackgammonGame)
    d1, d2 = g.dice[1], g.dice[2]
    if d1 == 0 || d2 == 0
        return 0
    end
    return Int(d1) + Int(d2)
end

"""
    _encode_dice_onehot!(obs, g)

Encode dice using one-hot encoding for 4 slots (channels 13-36).
Slots are ordered high-to-low (largest die value first).
Each slot gets 6 channels for values 1-6. All zeros if slot is empty.
"""
function _encode_dice_onehot!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    slots = _get_dice_slots(g)

    @inbounds for slot in 0:3
        die_val = slots[slot + 1]
        base_ch = 12 + slot * 6  # 12, 18, 24, 30

        if die_val > 0
            # One-hot encode: channel (base + die_val - 1) = 1
            ch = base_ch + die_val  # 1-indexed channel
            for w in 1:OBS_WIDTH
                obs[ch, 1, w] = 1.0f0
            end
        end
        # If die_val == 0, all 6 channels stay 0
    end

    return nothing
end

"""
    _encode_off!(obs, g)

Encode borne-off counts (channels 37-38).
"""
function _encode_off!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    my_off, opp_off = _get_off_counts(g)

    @inbounds for w in 1:OBS_WIDTH
        obs[37, 1, w] = Float32(my_off) / OFF_NORM
        obs[38, 1, w] = Float32(opp_off) / OFF_NORM
    end

    return nothing
end

# --- Full Observation Helpers ---

"""
    _broadcast_scalar!(obs, ch, value)

Broadcast a scalar value across all spatial positions.
"""
@inline function _broadcast_scalar!(obs::AbstractArray{Float32,3}, ch::Int, value::Float32)
    @inbounds for w in 1:OBS_WIDTH
        obs[ch, 1, w] = value
    end
    return nothing
end

"""
    _encode_threshold_broadcast!(obs, ch_base, count)

Encode count using threshold and broadcast across all spatial positions.
"""
function _encode_threshold_broadcast!(obs::AbstractArray{Float32,3}, ch_base::Int, count::Int)
    @inbounds for w in 1:OBS_WIDTH
        obs[ch_base + 1, 1, w] = Float32(count >= 1)
        obs[ch_base + 2, 1, w] = Float32(count >= 2)
        obs[ch_base + 3, 1, w] = Float32(count >= 3)
        obs[ch_base + 4, 1, w] = Float32(count >= 4)
        obs[ch_base + 5, 1, w] = Float32(count >= 5)
        obs[ch_base + 6, 1, w] = count >= 6 ? Float32(count - 5) / OVERFLOW_NORM : 0.0f0
    end
    return nothing
end

"""
    _encode_threshold_capped!(obs, ch_base, count, max_val)

Encode count with a maximum value (no overflow, pure binary).
Used for features with known max like prime length (6).
"""
function _encode_threshold_capped!(obs::AbstractArray{Float32,3}, ch_base::Int, count::Int, max_val::Int=6)
    @inbounds for w in 1:OBS_WIDTH
        for t in 1:max_val
            obs[ch_base + t, 1, w] = Float32(count >= t)
        end
    end
    return nothing
end

"""
    _compute_full_features(g::BackgammonGame)

Compute all features needed for full observation.
Returns named tuple with pip counts, contact, can_bear_off, stragglers.
"""
function _compute_full_features(g::BackgammonGame)
    my_pips = 0
    opp_pips = 0
    my_stragglers = 0    # Checkers outside home (points 1-18 for me)
    opp_stragglers = 0   # Checkers outside home (points 7-24 for opponent)
    can_bear_my = true
    can_bear_opp = true
    min_my = 25
    max_opp = 0

    for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)

        if my_count > 0
            my_pips += my_count * (BAR_PIP_VALUE - pt)
            min_my = min(min_my, pt)
            if pt < MY_HOME_START  # Not in home (points 19-24)
                can_bear_my = false
                my_stragglers += my_count
            end
        end

        if opp_count > 0
            opp_pips += opp_count * pt
            max_opp = max(max_opp, pt)
            if pt > OPP_HOME_END  # Not in their home (points 1-6)
                can_bear_opp = false
                opp_stragglers += opp_count
            end
        end
    end

    my_bar, opp_bar = _get_bar_counts(g)
    my_off, opp_off = _get_off_counts(g)

    # Add bar to pips and stragglers
    my_pips += my_bar * BAR_PIP_VALUE
    opp_pips += opp_bar * BAR_PIP_VALUE
    my_stragglers += my_bar
    opp_stragglers += opp_bar

    if my_bar > 0; can_bear_my = false; end
    if opp_bar > 0; can_bear_opp = false; end

    # Contact: pieces haven't fully passed (or anyone on bar)
    has_contact = (my_bar > 0 || opp_bar > 0 || min_my <= max_opp)

    return (
        my_pips = my_pips,
        opp_pips = opp_pips,
        has_contact = has_contact,
        can_bear_my = can_bear_my,
        can_bear_opp = can_bear_opp,
        my_stragglers = my_stragglers,
        opp_stragglers = opp_stragglers,
        my_off = my_off,
        opp_off = opp_off,
    )
end

"""
    _get_dice_delta(g::BackgammonGame) -> Int

Get absolute difference between original dice values.
Returns 0 if at chance node (no dice rolled).
"""
@inline function _get_dice_delta(g::BackgammonGame)
    d1, d2 = g.dice[1], g.dice[2]
    if d1 == 0 || d2 == 0
        return 0
    end
    return abs(Int(d1) - Int(d2))
end

"""
    _encode_full_features!(obs, g)

Encode full observation features (channels 39-70).
Builds on minimal observation.
"""
function _encode_full_features!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    feat = _compute_full_features(g)
    dice_sum = _get_original_dice_sum(g)
    dice_delta = _get_dice_delta(g)

    pip_diff = clamp((feat.my_pips - feat.opp_pips) / PIP_NORM, -1.0f0, 1.0f0)

    # Scalar features (broadcast)
    _broadcast_scalar!(obs, 39, Float32(dice_sum) / DICE_SUM_NORM)
    _broadcast_scalar!(obs, 40, Float32(dice_delta) / DICE_DELTA_NORM)
    _broadcast_scalar!(obs, 41, Float32(feat.has_contact))
    _broadcast_scalar!(obs, 42, Float32(feat.my_pips) / PIP_NORM)
    _broadcast_scalar!(obs, 43, Float32(feat.opp_pips) / PIP_NORM)
    _broadcast_scalar!(obs, 44, pip_diff)
    _broadcast_scalar!(obs, 45, Float32(feat.can_bear_my))
    _broadcast_scalar!(obs, 46, Float32(feat.can_bear_opp))

    # Threshold-encoded counts
    _encode_threshold_broadcast!(obs, 46, feat.my_stragglers)      # 47-52
    _encode_threshold_broadcast!(obs, 52, feat.opp_stragglers)     # 53-58
    _encode_threshold_broadcast!(obs, 58, 15 - feat.my_off)        # 59-64: remaining
    _encode_threshold_broadcast!(obs, 64, 15 - feat.opp_off)       # 65-70: remaining

    return nothing
end

# --- Biased Observation Helpers ---

"""
    _compute_prime_length(g::BackgammonGame, is_my::Bool) -> Int

Compute longest consecutive run of blocked points (2+ checkers).
"""
function _compute_prime_length(g::BackgammonGame, is_my::Bool)
    max_prime = 0
    current_prime = 0

    for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)
        has_block = is_my ? (my_count >= 2) : (opp_count >= 2)

        if has_block
            current_prime += 1
            max_prime = max(max_prime, current_prime)
        else
            current_prime = 0
        end
    end

    return min(max_prime, 6)  # Cap at 6 (full prime)
end

"""
    _compute_strategic_features(g::BackgammonGame)

Compute strategic features for biased observation.
"""
function _compute_strategic_features(g::BackgammonGame)
    my_prime = _compute_prime_length(g, true)
    opp_prime = _compute_prime_length(g, false)

    my_home_blocks = 0   # Blocks in my home (19-24)
    opp_home_blocks = 0  # Blocks in opp home (1-6)
    my_anchors = 0       # My blocks in opp's home (1-6)
    opp_anchors = 0      # Opp blocks in my home (19-24)
    my_blots = 0         # Total single checkers
    opp_blots = 0
    my_builders = 0      # Points with exactly 2 checkers
    opp_builders = 0

    for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)

        # Blots (exactly 1)
        if my_count == 1; my_blots += 1; end
        if opp_count == 1; opp_blots += 1; end

        # Builders (exactly 2)
        if my_count == 2; my_builders += 1; end
        if opp_count == 2; opp_builders += 1; end

        # Home board blocks (my home is 19-24, opp home is 1-6)
        if pt >= MY_HOME_START && my_count >= 2
            my_home_blocks += 1
        end
        if pt <= OPP_HOME_END && opp_count >= 2
            opp_home_blocks += 1
        end

        # Anchors (blocks in opponent's home territory)
        if pt <= OPP_HOME_END && my_count >= 2
            my_anchors += 1
        end
        if pt >= MY_HOME_START && opp_count >= 2
            opp_anchors += 1
        end
    end

    return (
        my_prime = my_prime,
        opp_prime = opp_prime,
        my_home_blocks = my_home_blocks,
        opp_home_blocks = opp_home_blocks,
        my_anchors = my_anchors,
        opp_anchors = opp_anchors,
        my_blots = my_blots,
        opp_blots = opp_blots,
        my_builders = my_builders,
        opp_builders = opp_builders,
    )
end

"""
    _encode_biased_features!(obs, g)

Encode biased/strategic features (channels 71-130).
Builds on full observation.
"""
function _encode_biased_features!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    strat = _compute_strategic_features(g)

    # Prime length (max 6, no overflow needed)
    _encode_threshold_capped!(obs, 70, strat.my_prime, 6)       # 71-76
    _encode_threshold_capped!(obs, 76, strat.opp_prime, 6)      # 77-82

    # Home board blocks (max 6)
    _encode_threshold_capped!(obs, 82, strat.my_home_blocks, 6)  # 83-88
    _encode_threshold_capped!(obs, 88, strat.opp_home_blocks, 6) # 89-94

    # Anchors (max 6)
    _encode_threshold_capped!(obs, 94, strat.my_anchors, 6)      # 95-100
    _encode_threshold_capped!(obs, 100, strat.opp_anchors, 6)    # 101-106

    # Blot count (can exceed 6, use overflow)
    _encode_threshold_broadcast!(obs, 106, strat.my_blots)       # 107-112
    _encode_threshold_broadcast!(obs, 112, strat.opp_blots)      # 113-118

    # Builder count (can exceed 6, use overflow)
    _encode_threshold_broadcast!(obs, 118, strat.my_builders)    # 119-124
    _encode_threshold_broadcast!(obs, 124, strat.opp_builders)   # 125-130

    return nothing
end

# ============================================================================
# Public API
# ============================================================================

"""
    observe_minimal(g::BackgammonGame) -> Array{Float32,3}

Generate minimal observation (38 channels). Shape: (38, 1, 25).

The network must learn all strategic concepts from raw state.

# Channels (1-indexed)
- 1-6: My checker thresholds (>=1, >=2, >=3, >=4, >=5, 6+)
- 7-12: Opponent checker thresholds
- 13-36: Dice one-hot (4 slots × 6 values, ordered high-to-low)
- 37-38: Off counts (/15)

# Spatial Dimension (for CNN topology)
Width 25 = [Bar, Point1, Point2, ..., Point24]
- Index 1: Bar (adjacent to entry points for 1D convolutions)
- Indices 2-25: Points 1-24 in canonical order (entry → home)

See also: [`observe_full`](@ref), [`observe_biased`](@ref)
"""
function observe_minimal(g::BackgammonGame)
    obs = zeros(Float32, OBS_CHANNELS_MINIMAL, 1, OBS_WIDTH)
    observe_minimal!(obs, g)
    return obs
end

"""
    observe_minimal!(obs::AbstractArray{Float32,3}, g::BackgammonGame)

In-place version of `observe_minimal`. Fills channels 1-38.
"""
function observe_minimal!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    # Zero out minimal channels
    @inbounds for ch in 1:OBS_CHANNELS_MINIMAL, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    _encode_board!(obs, g)
    _encode_dice_onehot!(obs, g)
    _encode_off!(obs, g)

    return obs
end

"""
    observe_full(g::BackgammonGame) -> Array{Float32,3}

Generate full observation (70 channels). Shape: (70, 1, 25).

Includes all minimal features plus pre-computed arithmetic features.
No strategic bias - only saves the network from doing math.

# Additional Channels (beyond minimal, 39-70)
- 39: dice_sum (/12)
- 40: dice_delta |d1-d2| (/5)
- 41: Contact indicator (1=contact, 0=race)
- 42-44: Pip counts (my, opp, diff)
- 45-46: Can bear off flags
- 47-58: Stragglers (outside home) threshold encoded
- 59-70: Remaining checkers threshold encoded

See also: [`observe_minimal`](@ref), [`observe_biased`](@ref)
"""
function observe_full(g::BackgammonGame)
    obs = zeros(Float32, OBS_CHANNELS_FULL, 1, OBS_WIDTH)
    observe_full!(obs, g)
    return obs
end

"""
    observe_full!(obs::AbstractArray{Float32,3}, g::BackgammonGame)

In-place version of `observe_full`. Fills channels 1-70.
"""
function observe_full!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    # Zero out full channels
    @inbounds for ch in 1:OBS_CHANNELS_FULL, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    # Minimal features
    _encode_board!(obs, g)
    _encode_dice_onehot!(obs, g)
    _encode_off!(obs, g)

    # Full features
    _encode_full_features!(obs, g)

    return obs
end

"""
    observe_biased(g::BackgammonGame) -> Array{Float32,3}

Generate biased observation (130 channels). Shape: (130, 1, 25).

Includes all full features plus hand-crafted strategic features
inspired by TD-Gammon and gnubg.

# Additional Channels (beyond full, 71-130)
- 71-82: Prime length (longest consecutive blocks)
- 83-94: Home board blocks (trapping power)
- 95-106: Anchors (blocks in opponent's home)
- 107-118: Blot count (exposure)
- 119-130: Builder count (flexibility)

See also: [`observe_minimal`](@ref), [`observe_full`](@ref)
"""
function observe_biased(g::BackgammonGame)
    obs = zeros(Float32, OBS_CHANNELS_BIASED, 1, OBS_WIDTH)
    observe_biased!(obs, g)
    return obs
end

"""
    observe_biased!(obs::AbstractArray{Float32,3}, g::BackgammonGame)

In-place version of `observe_biased`. Fills channels 1-130.
"""
function observe_biased!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    # Zero out all channels
    @inbounds for ch in 1:OBS_CHANNELS_BIASED, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    # Minimal features
    _encode_board!(obs, g)
    _encode_dice_onehot!(obs, g)
    _encode_off!(obs, g)

    # Full features
    _encode_full_features!(obs, g)

    # Biased features
    _encode_biased_features!(obs, g)

    return obs
end

# --- Legacy API Compatibility ---

# Keep old names working (deprecated)
const OBS_SIZE_FAST = OBS_CHANNELS_MINIMAL
const OBS_SIZE_FULL = OBS_CHANNELS_FULL

"""
    observe_fast(g::BackgammonGame) -> Vector{Float32}

DEPRECATED: Use `observe_minimal(g)` instead.
Returns flattened minimal observation for backward compatibility.
"""
function observe_fast(g::BackgammonGame)
    obs3d = observe_minimal(g)
    return vec(obs3d)
end

"""
    observe_fast!(obs::AbstractVector{Float32}, g::BackgammonGame)

DEPRECATED: Use `observe_minimal!(obs, g)` with 3D array instead.
"""
function observe_fast!(obs::AbstractVector{Float32}, g::BackgammonGame)
    obs3d = reshape(obs, OBS_CHANNELS_MINIMAL, 1, OBS_WIDTH)
    observe_minimal!(obs3d, g)
    return obs
end

# Default observation function
const vector_observation = observe_full

# Export channel counts for external use
const OBSERVATION_SIZES = (
    minimal = OBS_CHANNELS_MINIMAL,
    full = OBS_CHANNELS_FULL,
    biased = OBS_CHANNELS_BIASED,
    width = OBS_WIDTH,
)
