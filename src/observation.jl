# ============================================================================
# Observation Module - Three-Tier Neural Network Input Encoding
# ============================================================================
#
# Provides three observation types that build on each other:
#   - Minimal (30 channels): Raw board + dice + move count + off counts
#   - Full (62 channels): + arithmetic features (no strategic bias)
#   - Biased (122 channels): + hand-crafted strategic features
#
# Shape: (C, 1, 26) where width = 2 bars + 24 board points
#
# Spatial Layout (1-indexed, Julia convention):
#   Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
#   Indices 2-25: Points 1-24 in canonical order (entry → home)
#   Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)
#
# Design Philosophy:
#   - Minimal: Network must learn everything from raw state
#   - Full: Pre-compute arithmetic to save network capacity
#   - Biased: Include TD-Gammon/gnubg strategic features
#
# ============================================================================

# --- Observation Dimensions ---
const OBS_WIDTH = 26  # My bar at 1, points 1-24 at 2-25, opponent bar at 26

# Channel counts for each tier (3D observations)
const OBS_CHANNELS_MINIMAL = 30   # 12 board + 12 dice (2 slots) + 4 move count + 2 off
const OBS_CHANNELS_FULL = 62      # 30 minimal + 32 full features (31-62)
const OBS_CHANNELS_BIASED = 122   # 62 full + 60 biased features (63-122)

# Flat observation sizes (same features, globals appear once instead of broadcast)
# Minimal: 312 board (2×26×6) + 12 dice (2×6) + 4 move count + 2 off = 330
const OBS_FLAT_MINIMAL = 330
# Full adds: 8 scalars + 24 thresholds (4×6) = 32 → 330 + 32 = 362
const OBS_FLAT_FULL = 362
# Biased adds: 60 thresholds (10×6) = 60 → 362 + 60 = 422
const OBS_FLAT_BIASED = 422

# Hybrid observation sizes: board (12×26) + globals (flat)
# Board is spatial for conv1d, globals concatenated after conv
const OBS_HYBRID_BOARD = (12, 26)  # (channels, width)
const OBS_HYBRID_GLOBALS_MINIMAL = 18   # 12 dice + 4 move count + 2 off
const OBS_HYBRID_GLOBALS_FULL = 50      # 18 + 32 full features
const OBS_HYBRID_GLOBALS_BIASED = 110   # 50 + 60 biased features

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
# MINIMAL (30 channels, 1-30):
#   1-6:   My checker thresholds (>=1, >=2, >=3, >=4, >=5, 6+)
#   7-12:  Opponent checker thresholds
#   13-18: Dice slot 0 one-hot (high die, values 1-6)
#   19-24: Dice slot 1 one-hot (low die, values 1-6)
#   25-28: Move count one-hot (bins 1, 2, 3, 4 moves)
#   29:    My off count (/15)
#   30:    Opponent off count (/15)
#
# Move Count Encoding Design Philosophy:
#   - 4-bin one-hot encoding for number of moves this turn (1, 2, 3, or 4)
#   - Allows network to distinguish "strong doubles" (4 moves) from "weak doubles" (3 moves)
#   - Doubles: Compute exact playable moves (1-4) and activate corresponding bin
#   - Non-doubles: Compute exact playable moves (0, 1, or 2) and activate corresponding bin
#   - Chance node (no dice): All bins zero
#   - Completely blocked (0 moves): All bins zero
#
# FULL adds 32 channels (31-62):
#   31:    dice_sum (/12)
#   32:    dice_delta |d1-d2| (/5)
#   33:    Contact indicator (1=contact, 0=race)
#   34:    My pip count (/167)
#   35:    Opponent pip count (/167)
#   36:    Pip difference (my-opp, /167, clipped [-1,1])
#   37:    Can bear off (me)
#   38:    Can bear off (opponent)
#   39-44: My stragglers (outside home) threshold
#   45-50: Opponent stragglers threshold
#   51-56: My remaining (15-off) threshold
#   57-62: Opponent remaining threshold
#
# BIASED adds 60 channels (63-122):
#   63-68:   My prime length threshold
#   69-74:   Opponent prime length threshold
#   75-80:   My home board blocks threshold
#   81-86:   Opponent home board blocks threshold
#   87-92:   My anchors (in opponent's home) threshold
#   93-98:   Opponent anchors threshold
#   99-104:  My blot count threshold
#   105-110: Opponent blot count threshold
#   111-116: My builder count threshold
#   117-122: Opponent builder count threshold
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

Spatial layout: [MyBar, Point1, ..., Point24, OppBar]
- Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
- Indices 2-25: Points 1-24 in canonical order
- Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)
"""
function _encode_board!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    my_bar, opp_bar = _get_bar_counts(g)

    # My bar at spatial index 1 (adjacent to my entry points 1-6)
    _encode_threshold_6!(obs, 0, 1, my_bar)
    _encode_threshold_6!(obs, 6, 1, 0)  # No opponent checkers on my bar

    # Points 1-24 at spatial indices 2-25
    @inbounds for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)
        _encode_threshold_6!(obs, 0, pt + 1, my_count)
        _encode_threshold_6!(obs, 6, pt + 1, opp_count)
    end

    # Opponent bar at spatial index 26 (adjacent to their entry points 19-24)
    _encode_threshold_6!(obs, 0, 26, 0)  # No my checkers on opponent bar
    _encode_threshold_6!(obs, 6, 26, opp_bar)

    return nothing
end

"""
    _get_dice_pair(g::BackgammonGame) -> Tuple{Int, Int}

Get rolled dice values as (high, low) pair.
Returns (0, 0) if at chance node (no dice rolled).
Always shows the original rolled values regardless of remaining_actions.

Note: dice[1] is always high die, dice[2] is always low die (guaranteed by DICE_OUTCOMES).
"""
@inline function _get_dice_pair(g::BackgammonGame)
    d1, d2 = g.dice[1], g.dice[2]
    if d1 == 0 || d2 == 0
        return (0, 0)
    end
    return (Int(d1), Int(d2))  # Already (high, low) from DICE_OUTCOMES
end

"""
    _compute_playable_dice_doubles(g::BackgammonGame, actions=nothing) -> Int

Compute total playable dice for doubles at first action (0-4).
Returns 0 if not doubles or not first action.

Uses optional pre-computed `actions` to avoid duplicate legal_actions computation.
This is the "precise" approach that simulates moves to determine actual playability.
"""
function _compute_playable_dice_doubles(g::BackgammonGame, actions::Union{Nothing, Vector{Int}}=nothing)
    d1, d2 = g.dice[1], g.dice[2]

    # Not doubles or not first action -> not applicable
    if d1 != d2 || d1 == 0 || g.remaining_actions != 2
        return 0
    end

    d = Int(d1)
    cp = g.current_player
    p0, p1 = g.p0, g.p1

    # Get legal actions (reuse if provided)
    acts = actions === nothing ? legal_actions(g) : actions

    # Find max dice playable from any action path
    max_dice = 0

    for act in acts
        l1, l2 = decode_action(act)

        if l1 == PASS_LOC && l2 == PASS_LOC
            # 0 dice playable from this path
            continue
        elseif l1 == PASS_LOC || l2 == PASS_LOC
            # 1 die in first action
            if max_dice < 1; max_dice = 1; end

            # Simulate and check second action
            s = l1 == PASS_LOC ? l2 : l1
            p0_next, p1_next = apply_move_internal(p0, p1, cp, s, d)

            # Check if any moves in second action (single die check)
            get_legal_source_locs!(g._sources_buffer1, p0_next, p1_next, cp, d)
            if !isempty(g._sources_buffer1)
                # At least 2 dice playable (1 + 1)
                if max_dice < 2; max_dice = 2; end

                # Check if 3 dice playable (1 + 2)
                for s2 in g._sources_buffer1
                    p0_n2, p1_n2 = apply_move_internal(p0_next, p1_next, cp, s2, d)
                    get_legal_source_locs!(g._sources_buffer2, p0_n2, p1_n2, cp, d)
                    if !isempty(g._sources_buffer2)
                        max_dice = 3  # 1 first action + 2 second action
                        break
                    end
                end
            end
        else
            # 2 dice in first action
            # Simulate both moves and check second action
            p0_mid, p1_mid = apply_move_internal(p0, p1, cp, l1, d)
            p0_next, p1_next = apply_move_internal(p0_mid, p1_mid, cp, l2, d)

            # Check if any moves in second action (die 3)
            get_legal_source_locs!(g._sources_buffer1, p0_next, p1_next, cp, d)

            if isempty(g._sources_buffer1)
                # Only 2 dice playable
                if max_dice < 2; max_dice = 2; end
            else
                # At least 3 dice playable
                # Check if 4 dice playable
                for s3 in g._sources_buffer1
                    p0_n3, p1_n3 = apply_move_internal(p0_next, p1_next, cp, s3, d)
                    get_legal_source_locs!(g._sources_buffer2, p0_n3, p1_n3, cp, d)
                    if !isempty(g._sources_buffer2)
                        return 4  # Max possible, early exit
                    end
                end
                max_dice = 3
            end
        end

        if max_dice == 4
            break  # Already at max
        end
    end

    return max_dice
end

"""
    _compute_playable_dice_non_doubles(g::BackgammonGame, actions=nothing) -> Int

Compute playable dice count for non-doubles (0, 1, or 2).
Returns 0 if doubles or no dice rolled.

For non-doubles, actions encode (high_die_loc, low_die_loc) where PASS_LOC
indicates that die wasn't used. We check:
- If any action uses both dice (neither is PASS) → 2 playable
- If all actions have at least one PASS but some move exists → 1 playable
- If only PASS|PASS available → 0 playable

Uses optional pre-computed `actions` to avoid duplicate legal_actions computation.
"""
function _compute_playable_dice_non_doubles(g::BackgammonGame, actions::Union{Nothing, Vector{Int}}=nothing)
    d1, d2 = g.dice[1], g.dice[2]

    # Not applicable for doubles or chance node
    if d1 == d2 || d1 == 0 || d2 == 0
        return 0
    end

    # Get legal actions (reuse if provided)
    acts = actions === nothing ? legal_actions(g) : actions

    # Check actions to determine playable count
    has_any_move = false
    for act in acts
        l1, l2 = decode_action(act)

        if l1 != PASS_LOC && l2 != PASS_LOC
            # Both dice used → 2 playable
            return 2
        elseif l1 != PASS_LOC || l2 != PASS_LOC
            # At least one die used
            has_any_move = true
        end
        # l1 == PASS_LOC && l2 == PASS_LOC → 0 dice in this action
    end

    # If we found moves but none used both dice → 1 playable
    # If no moves found (only PASS|PASS) → 0 playable
    return has_any_move ? 1 : 0
end

"""
    _compute_playable_moves_current(g::BackgammonGame, actions=nothing) -> Int

Compute playable moves for the current action (0, 1, or 2).

Used for the second action of doubles where player may be blocked and unable
to use all remaining dice. Checks legal actions for PASS locations:
- Both locs non-PASS → 2 playable
- One loc non-PASS → 1 playable
- Both locs PASS → 0 playable

Uses optional pre-computed `actions` to avoid duplicate legal_actions computation.
"""
function _compute_playable_moves_current(g::BackgammonGame, actions::Union{Nothing, Vector{Int}}=nothing)
    acts = actions === nothing ? legal_actions(g) : actions

    has_any_move = false
    for act in acts
        l1, l2 = decode_action(act)

        if l1 != PASS_LOC && l2 != PASS_LOC
            return 2  # Both moves used
        elseif l1 != PASS_LOC || l2 != PASS_LOC
            has_any_move = true
        end
    end

    return has_any_move ? 1 : 0
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
    _encode_dice!(obs, g, actions=nothing)

Encode dice using 2 one-hot slots (channels 13-24) + move count (channels 25-28).

Slots are ordered high-to-low (largest die value first).
Each slot gets 6 channels for values 1-6. All zeros if at chance node.

Move count is a 4-bin one-hot encoding (bins 1, 2, 3, 4):
- Doubles: Compute exact playable moves (1-4) and activate corresponding bin
- Non-doubles: Compute exact playable moves (0, 1, or 2) and activate corresponding bin
- Chance node or 0 moves: All bins zero

Optional `actions` parameter avoids recomputing legal_actions if already available.
"""
function _encode_dice!(obs::AbstractArray{Float32,3}, g::BackgammonGame,
                       actions::Union{Nothing, Vector{Int}}=nothing)
    high, low = _get_dice_pair(g)

    # Slot 0 (high die): channels 13-18
    if high > 0
        ch = 12 + high  # 1-indexed: 13-18 for values 1-6
        @inbounds for w in 1:OBS_WIDTH
            obs[ch, 1, w] = 1.0f0
        end
    end

    # Slot 1 (low die): channels 19-24
    if low > 0
        ch = 18 + low  # 1-indexed: 19-24 for values 1-6
        @inbounds for w in 1:OBS_WIDTH
            obs[ch, 1, w] = 1.0f0
        end
    end

    # Move count one-hot: channels 25-28 (bins 1, 2, 3, 4)
    if high > 0
        move_count = 0
        if high == low
            # Doubles: compute exact playable moves
            if g.remaining_actions == 2
                # First action: 0-4 moves playable
                move_count = _compute_playable_dice_doubles(g, actions)
            else
                # Second action: 0-2 moves playable (may be blocked by prime)
                move_count = _compute_playable_moves_current(g, actions)
            end
        else
            # Non-doubles: compute exact playable moves (0, 1, or 2)
            move_count = _compute_playable_dice_non_doubles(g, actions)
        end

        # Activate corresponding bin (1-4 maps to channels 25-28)
        if move_count >= 1 && move_count <= 4
            ch = 24 + move_count  # 25, 26, 27, or 28
            @inbounds for w in 1:OBS_WIDTH
                obs[ch, 1, w] = 1.0f0
            end
        end
        # If move_count == 0 (completely blocked), all bins stay 0
    end
    # Chance node (high == 0): all bins stay 0

    return nothing
end

"""
    _encode_off!(obs, g)

Encode borne-off counts (channels 29-30).
"""
function _encode_off!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    my_off, opp_off = _get_off_counts(g)

    @inbounds for w in 1:OBS_WIDTH
        obs[29, 1, w] = Float32(my_off) / OFF_NORM
        obs[30, 1, w] = Float32(opp_off) / OFF_NORM
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

Encode full observation features (channels 31-62).
Builds on minimal observation.
"""
function _encode_full_features!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    feat = _compute_full_features(g)
    dice_sum = _get_original_dice_sum(g)
    dice_delta = _get_dice_delta(g)

    pip_diff = clamp((feat.my_pips - feat.opp_pips) / PIP_NORM, -1.0f0, 1.0f0)

    # Scalar features (broadcast)
    _broadcast_scalar!(obs, 31, Float32(dice_sum) / DICE_SUM_NORM)
    _broadcast_scalar!(obs, 32, Float32(dice_delta) / DICE_DELTA_NORM)
    _broadcast_scalar!(obs, 33, Float32(feat.has_contact))
    _broadcast_scalar!(obs, 34, Float32(feat.my_pips) / PIP_NORM)
    _broadcast_scalar!(obs, 35, Float32(feat.opp_pips) / PIP_NORM)
    _broadcast_scalar!(obs, 36, pip_diff)
    _broadcast_scalar!(obs, 37, Float32(feat.can_bear_my))
    _broadcast_scalar!(obs, 38, Float32(feat.can_bear_opp))

    # Threshold-encoded counts
    _encode_threshold_broadcast!(obs, 38, feat.my_stragglers)      # 39-44
    _encode_threshold_broadcast!(obs, 44, feat.opp_stragglers)     # 45-50
    _encode_threshold_broadcast!(obs, 50, 15 - feat.my_off)        # 51-56: remaining
    _encode_threshold_broadcast!(obs, 56, 15 - feat.opp_off)       # 57-62: remaining

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

Encode biased/strategic features (channels 63-122).
Builds on full observation.
"""
function _encode_biased_features!(obs::AbstractArray{Float32,3}, g::BackgammonGame)
    strat = _compute_strategic_features(g)

    # Prime length (max 6, no overflow needed)
    _encode_threshold_capped!(obs, 62, strat.my_prime, 6)       # 63-68
    _encode_threshold_capped!(obs, 68, strat.opp_prime, 6)      # 69-74

    # Home board blocks (max 6)
    _encode_threshold_capped!(obs, 74, strat.my_home_blocks, 6)  # 75-80
    _encode_threshold_capped!(obs, 80, strat.opp_home_blocks, 6) # 81-86

    # Anchors (max 6)
    _encode_threshold_capped!(obs, 86, strat.my_anchors, 6)      # 87-92
    _encode_threshold_capped!(obs, 92, strat.opp_anchors, 6)     # 93-98

    # Blot count (can exceed 6, use overflow)
    _encode_threshold_broadcast!(obs, 98, strat.my_blots)        # 99-104
    _encode_threshold_broadcast!(obs, 104, strat.opp_blots)      # 105-110

    # Builder count (can exceed 6, use overflow)
    _encode_threshold_broadcast!(obs, 110, strat.my_builders)    # 111-116
    _encode_threshold_broadcast!(obs, 116, strat.opp_builders)   # 117-122

    return nothing
end

# ============================================================================
# Public API
# ============================================================================

"""
    observe_minimal(g::BackgammonGame; actions=nothing) -> Array{Float32,3}

Generate minimal observation (30 channels). Shape: (30, 1, 26).

The network must learn all strategic concepts from raw state.

# Channels (1-indexed)
- 1-6: My checker thresholds (>=1, >=2, >=3, >=4, >=5, 6+)
- 7-12: Opponent checker thresholds
- 13-24: Dice one-hot (2 slots × 6 values, ordered high-to-low)
- 25-28: Move count one-hot (bins 1, 2, 3, 4 moves)
- 29-30: Off counts (/15)

# Spatial Dimension (1-indexed, Julia convention)
Width 26 = [MyBar, Point1, ..., Point24, OppBar]
- Index 1: My bar (adjacent to my entry points 1-6 at indices 2-7)
- Indices 2-25: Points 1-24 in canonical order (entry → home)
- Index 26: Opponent bar (adjacent to their entry points 19-24 at indices 20-25)

# Optional `actions` parameter
Pass pre-computed legal_actions to avoid duplicate computation when
the caller already has them (e.g., for training pipelines).

See also: [`observe_full`](@ref), [`observe_biased`](@ref)
"""
function observe_minimal(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_CHANNELS_MINIMAL, 1, OBS_WIDTH)
    observe_minimal!(obs, g; actions=actions)
    return obs
end

"""
    observe_minimal!(obs::AbstractArray{Float32,3}, g::BackgammonGame; actions=nothing)

In-place version of `observe_minimal`. Fills channels 1-30.
"""
function observe_minimal!(obs::AbstractArray{Float32,3}, g::BackgammonGame;
                          actions::Union{Nothing, Vector{Int}}=nothing)
    # Zero out minimal channels
    @inbounds for ch in 1:OBS_CHANNELS_MINIMAL, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    _encode_board!(obs, g)
    _encode_dice!(obs, g, actions)
    _encode_off!(obs, g)

    return obs
end

"""
    observe_full(g::BackgammonGame; actions=nothing) -> Array{Float32,3}

Generate full observation (62 channels). Shape: (62, 1, 26).

Includes all minimal features plus pre-computed arithmetic features.
No strategic bias - only saves the network from doing math.

# Additional Channels (beyond minimal, 31-62)
- 31: dice_sum (/12)
- 32: dice_delta |d1-d2| (/5)
- 33: Contact indicator (1=contact, 0=race)
- 34-36: Pip counts (my, opp, diff)
- 37-38: Can bear off flags
- 39-50: Stragglers (outside home) threshold encoded
- 51-62: Remaining checkers threshold encoded

See also: [`observe_minimal`](@ref), [`observe_biased`](@ref)
"""
function observe_full(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_CHANNELS_FULL, 1, OBS_WIDTH)
    observe_full!(obs, g; actions=actions)
    return obs
end

"""
    observe_full!(obs::AbstractArray{Float32,3}, g::BackgammonGame; actions=nothing)

In-place version of `observe_full`. Fills channels 1-62.
"""
function observe_full!(obs::AbstractArray{Float32,3}, g::BackgammonGame;
                       actions::Union{Nothing, Vector{Int}}=nothing)
    # Zero out full channels
    @inbounds for ch in 1:OBS_CHANNELS_FULL, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    # Minimal features
    _encode_board!(obs, g)
    _encode_dice!(obs, g, actions)
    _encode_off!(obs, g)

    # Full features
    _encode_full_features!(obs, g)

    return obs
end

"""
    observe_biased(g::BackgammonGame; actions=nothing) -> Array{Float32,3}

Generate biased observation (122 channels). Shape: (122, 1, 26).

Includes all full features plus hand-crafted strategic features
inspired by TD-Gammon and gnubg.

# Additional Channels (beyond full, 63-122)
- 63-74: Prime length (longest consecutive blocks)
- 75-86: Home board blocks (trapping power)
- 87-98: Anchors (blocks in opponent's home)
- 99-110: Blot count (exposure)
- 111-122: Builder count (flexibility)

See also: [`observe_minimal`](@ref), [`observe_full`](@ref)
"""
function observe_biased(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_CHANNELS_BIASED, 1, OBS_WIDTH)
    observe_biased!(obs, g; actions=actions)
    return obs
end

"""
    observe_biased!(obs::AbstractArray{Float32,3}, g::BackgammonGame; actions=nothing)

In-place version of `observe_biased`. Fills channels 1-122.
"""
function observe_biased!(obs::AbstractArray{Float32,3}, g::BackgammonGame;
                         actions::Union{Nothing, Vector{Int}}=nothing)
    # Zero out all channels
    @inbounds for ch in 1:OBS_CHANNELS_BIASED, w in 1:OBS_WIDTH
        obs[ch, 1, w] = 0.0f0
    end

    # Minimal features
    _encode_board!(obs, g)
    _encode_dice!(obs, g, actions)
    _encode_off!(obs, g)

    # Full features
    _encode_full_features!(obs, g)

    # Biased features
    _encode_biased_features!(obs, g)

    return obs
end

# ============================================================================
# Flat Observations (same features, globals appear once)
# ============================================================================

"""
    _encode_board_flat!(obs, g, offset) -> Int

Encode board state into flat vector starting at offset.
Returns new offset after encoding.

Layout: 26 positions × 2 players × 6 thresholds = 312 values
Order: [MyBar×12, Point1×12, ..., Point24×12, OppBar×12]
Each position: [my_thresh1..6, opp_thresh1..6]
"""
function _encode_board_flat!(obs::AbstractVector{Float32}, g::BackgammonGame, offset::Int)
    my_bar, opp_bar = _get_bar_counts(g)

    # My bar at position 0 (index 1 in 1-indexed obs)
    idx = offset
    @inbounds begin
        # My threshold at my bar
        obs[idx + 1] = Float32(my_bar >= 1)
        obs[idx + 2] = Float32(my_bar >= 2)
        obs[idx + 3] = Float32(my_bar >= 3)
        obs[idx + 4] = Float32(my_bar >= 4)
        obs[idx + 5] = Float32(my_bar >= 5)
        obs[idx + 6] = my_bar >= 6 ? Float32(my_bar - 5) / OVERFLOW_NORM : 0.0f0
        # Opp threshold at my bar (always 0)
        obs[idx + 7] = 0.0f0
        obs[idx + 8] = 0.0f0
        obs[idx + 9] = 0.0f0
        obs[idx + 10] = 0.0f0
        obs[idx + 11] = 0.0f0
        obs[idx + 12] = 0.0f0
    end
    idx += 12

    # Points 1-24
    @inbounds for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)
        # My threshold
        obs[idx + 1] = Float32(my_count >= 1)
        obs[idx + 2] = Float32(my_count >= 2)
        obs[idx + 3] = Float32(my_count >= 3)
        obs[idx + 4] = Float32(my_count >= 4)
        obs[idx + 5] = Float32(my_count >= 5)
        obs[idx + 6] = my_count >= 6 ? Float32(my_count - 5) / OVERFLOW_NORM : 0.0f0
        # Opp threshold
        obs[idx + 7] = Float32(opp_count >= 1)
        obs[idx + 8] = Float32(opp_count >= 2)
        obs[idx + 9] = Float32(opp_count >= 3)
        obs[idx + 10] = Float32(opp_count >= 4)
        obs[idx + 11] = Float32(opp_count >= 5)
        obs[idx + 12] = opp_count >= 6 ? Float32(opp_count - 5) / OVERFLOW_NORM : 0.0f0
        idx += 12
    end

    # Opp bar at position 25
    @inbounds begin
        # My threshold at opp bar (always 0)
        obs[idx + 1] = 0.0f0
        obs[idx + 2] = 0.0f0
        obs[idx + 3] = 0.0f0
        obs[idx + 4] = 0.0f0
        obs[idx + 5] = 0.0f0
        obs[idx + 6] = 0.0f0
        # Opp threshold at opp bar
        obs[idx + 7] = Float32(opp_bar >= 1)
        obs[idx + 8] = Float32(opp_bar >= 2)
        obs[idx + 9] = Float32(opp_bar >= 3)
        obs[idx + 10] = Float32(opp_bar >= 4)
        obs[idx + 11] = Float32(opp_bar >= 5)
        obs[idx + 12] = opp_bar >= 6 ? Float32(opp_bar - 5) / OVERFLOW_NORM : 0.0f0
    end

    return offset + 312
end

"""
    _encode_dice_flat!(obs, g, offset, actions) -> Int

Encode dice and move count into flat vector starting at offset.
Returns new offset after encoding.

Layout: 12 dice one-hot (2×6) + 4 move count = 16 values
"""
function _encode_dice_flat!(obs::AbstractVector{Float32}, g::BackgammonGame, offset::Int,
                            actions::Union{Nothing, Vector{Int}}=nothing)
    high, low = _get_dice_pair(g)
    idx = offset

    # High die one-hot (6 values)
    @inbounds for v in 1:6
        obs[idx + v] = Float32(high == v)
    end
    idx += 6

    # Low die one-hot (6 values)
    @inbounds for v in 1:6
        obs[idx + v] = Float32(low == v)
    end
    idx += 6

    # Move count one-hot (4 values, bins 1,2,3,4)
    move_count = 0
    if high > 0
        if high == low
            # Doubles
            if g.remaining_actions == 2
                move_count = _compute_playable_dice_doubles(g, actions)
            else
                move_count = _compute_playable_moves_current(g, actions)
            end
        else
            # Non-doubles
            move_count = _compute_playable_dice_non_doubles(g, actions)
        end
    end

    @inbounds for bin in 1:4
        obs[idx + bin] = Float32(move_count == bin)
    end

    return offset + 16
end

"""
    _encode_off_flat!(obs, g, offset) -> Int

Encode off counts into flat vector starting at offset.
Returns new offset after encoding.

Layout: 2 values (my_off, opp_off normalized by 15)
"""
function _encode_off_flat!(obs::AbstractVector{Float32}, g::BackgammonGame, offset::Int)
    my_off, opp_off = _get_off_counts(g)

    @inbounds begin
        obs[offset + 1] = Float32(my_off) / OFF_NORM
        obs[offset + 2] = Float32(opp_off) / OFF_NORM
    end

    return offset + 2
end

"""
    _encode_threshold_flat!(obs, offset, count) -> Int

Encode count using 6-threshold scheme at offset.
Returns new offset after encoding.
"""
@inline function _encode_threshold_flat!(obs::AbstractVector{Float32}, offset::Int, count::Int)
    @inbounds begin
        obs[offset + 1] = Float32(count >= 1)
        obs[offset + 2] = Float32(count >= 2)
        obs[offset + 3] = Float32(count >= 3)
        obs[offset + 4] = Float32(count >= 4)
        obs[offset + 5] = Float32(count >= 5)
        obs[offset + 6] = count >= 6 ? Float32(count - 5) / OVERFLOW_NORM : 0.0f0
    end
    return offset + 6
end

"""
    _encode_full_features_flat!(obs, g, offset) -> Int

Encode full features into flat vector starting at offset.
Returns new offset after encoding.

Layout: 8 scalars + 4×6 thresholds = 32 values
"""
function _encode_full_features_flat!(obs::AbstractVector{Float32}, g::BackgammonGame, offset::Int)
    feat = _compute_full_features(g)
    dice_sum = _get_original_dice_sum(g)
    dice_delta = _get_dice_delta(g)
    pip_diff = clamp((feat.my_pips - feat.opp_pips) / PIP_NORM, -1.0f0, 1.0f0)

    idx = offset

    # 8 scalar features
    @inbounds begin
        obs[idx + 1] = Float32(dice_sum) / DICE_SUM_NORM
        obs[idx + 2] = Float32(dice_delta) / DICE_DELTA_NORM
        obs[idx + 3] = Float32(feat.has_contact)
        obs[idx + 4] = Float32(feat.my_pips) / PIP_NORM
        obs[idx + 5] = Float32(feat.opp_pips) / PIP_NORM
        obs[idx + 6] = pip_diff
        obs[idx + 7] = Float32(feat.can_bear_my)
        obs[idx + 8] = Float32(feat.can_bear_opp)
    end
    idx += 8

    # 4×6 threshold features
    idx = _encode_threshold_flat!(obs, idx, feat.my_stragglers)
    idx = _encode_threshold_flat!(obs, idx, feat.opp_stragglers)
    idx = _encode_threshold_flat!(obs, idx, 15 - feat.my_off)
    idx = _encode_threshold_flat!(obs, idx, 15 - feat.opp_off)

    return idx
end

"""
    _encode_biased_features_flat!(obs, g, offset) -> Int

Encode biased features into flat vector starting at offset.
Returns new offset after encoding.

Layout: 10×6 thresholds = 60 values
"""
function _encode_biased_features_flat!(obs::AbstractVector{Float32}, g::BackgammonGame, offset::Int)
    strat = _compute_strategic_features(g)
    idx = offset

    idx = _encode_threshold_flat!(obs, idx, strat.my_prime)
    idx = _encode_threshold_flat!(obs, idx, strat.opp_prime)
    idx = _encode_threshold_flat!(obs, idx, strat.my_home_blocks)
    idx = _encode_threshold_flat!(obs, idx, strat.opp_home_blocks)
    idx = _encode_threshold_flat!(obs, idx, strat.my_anchors)
    idx = _encode_threshold_flat!(obs, idx, strat.opp_anchors)
    idx = _encode_threshold_flat!(obs, idx, strat.my_blots)
    idx = _encode_threshold_flat!(obs, idx, strat.opp_blots)
    idx = _encode_threshold_flat!(obs, idx, strat.my_builders)
    idx = _encode_threshold_flat!(obs, idx, strat.opp_builders)

    return idx
end

# ============================================================================
# Flat Public API
# ============================================================================

"""
    observe_minimal_flat(g::BackgammonGame; actions=nothing) -> Vector{Float32}

Generate minimal flat observation (330 values).

Same features as `observe_minimal` but without spatial broadcasting.
Global features (dice, move count, off counts) appear once instead of 26 times.

# Layout (330 total)
- Board: 312 values (26 positions × 12 thresholds per position)
- Dice: 12 values (2 slots × 6 one-hot)
- Move count: 4 values (bins 1,2,3,4 one-hot)
- Off counts: 2 values (my, opp normalized)

See also: [`observe_minimal`](@ref), [`observe_full_flat`](@ref)
"""
function observe_minimal_flat(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_FLAT_MINIMAL)
    observe_minimal_flat!(obs, g; actions=actions)
    return obs
end

"""
    observe_minimal_flat!(obs::AbstractVector{Float32}, g::BackgammonGame; actions=nothing)

In-place version of `observe_minimal_flat`.
"""
function observe_minimal_flat!(obs::AbstractVector{Float32}, g::BackgammonGame;
                               actions::Union{Nothing, Vector{Int}}=nothing)
    fill!(obs, 0.0f0)
    idx = _encode_board_flat!(obs, g, 0)
    idx = _encode_dice_flat!(obs, g, idx, actions)
    idx = _encode_off_flat!(obs, g, idx)
    return obs
end

"""
    observe_full_flat(g::BackgammonGame; actions=nothing) -> Vector{Float32}

Generate full flat observation (362 values).

Same features as `observe_full` but without spatial broadcasting.

# Layout (362 total)
- Minimal: 330 values
- Full additions: 32 values (8 scalars + 4×6 thresholds)

See also: [`observe_full`](@ref), [`observe_minimal_flat`](@ref)
"""
function observe_full_flat(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_FLAT_FULL)
    observe_full_flat!(obs, g; actions=actions)
    return obs
end

"""
    observe_full_flat!(obs::AbstractVector{Float32}, g::BackgammonGame; actions=nothing)

In-place version of `observe_full_flat`.
"""
function observe_full_flat!(obs::AbstractVector{Float32}, g::BackgammonGame;
                            actions::Union{Nothing, Vector{Int}}=nothing)
    fill!(obs, 0.0f0)
    idx = _encode_board_flat!(obs, g, 0)
    idx = _encode_dice_flat!(obs, g, idx, actions)
    idx = _encode_off_flat!(obs, g, idx)
    idx = _encode_full_features_flat!(obs, g, idx)
    return obs
end

"""
    observe_biased_flat(g::BackgammonGame; actions=nothing) -> Vector{Float32}

Generate biased flat observation (422 values).

Same features as `observe_biased` but without spatial broadcasting.

# Layout (422 total)
- Full: 362 values
- Biased additions: 60 values (10×6 thresholds)

See also: [`observe_biased`](@ref), [`observe_full_flat`](@ref)
"""
function observe_biased_flat(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs = zeros(Float32, OBS_FLAT_BIASED)
    observe_biased_flat!(obs, g; actions=actions)
    return obs
end

"""
    observe_biased_flat!(obs::AbstractVector{Float32}, g::BackgammonGame; actions=nothing)

In-place version of `observe_biased_flat`.
"""
function observe_biased_flat!(obs::AbstractVector{Float32}, g::BackgammonGame;
                              actions::Union{Nothing, Vector{Int}}=nothing)
    fill!(obs, 0.0f0)
    idx = _encode_board_flat!(obs, g, 0)
    idx = _encode_dice_flat!(obs, g, idx, actions)
    idx = _encode_off_flat!(obs, g, idx)
    idx = _encode_full_features_flat!(obs, g, idx)
    idx = _encode_biased_features_flat!(obs, g, idx)
    return obs
end

# ============================================================================
# Hybrid Observations (board spatial + globals flat)
# ============================================================================
# For networks that use conv1d on board, then concatenate globals before dense.
# Returns NamedTuple with :board (12×26 matrix) and :globals (flat vector).

"""
    _encode_board_hybrid!(board, g)

Encode board state into 12×26 matrix for hybrid observations.
Channels 1-6: my thresholds, channels 7-12: opponent thresholds.
"""
function _encode_board_hybrid!(board::AbstractMatrix{Float32}, g::BackgammonGame)
    fill!(board, 0.0f0)
    my_bar, opp_bar = _get_bar_counts(g)

    # My bar at width index 1
    @inbounds begin
        board[1, 1] = Float32(my_bar >= 1)
        board[2, 1] = Float32(my_bar >= 2)
        board[3, 1] = Float32(my_bar >= 3)
        board[4, 1] = Float32(my_bar >= 4)
        board[5, 1] = Float32(my_bar >= 5)
        board[6, 1] = my_bar >= 6 ? Float32(my_bar - 5) / OVERFLOW_NORM : 0.0f0
        # Opp thresholds at my bar (always 0, already filled)
    end

    # Points 1-24 at width indices 2-25
    @inbounds for pt in 1:24
        my_count, opp_count = _get_checker_counts(g, pt)
        w = pt + 1
        board[1, w] = Float32(my_count >= 1)
        board[2, w] = Float32(my_count >= 2)
        board[3, w] = Float32(my_count >= 3)
        board[4, w] = Float32(my_count >= 4)
        board[5, w] = Float32(my_count >= 5)
        board[6, w] = my_count >= 6 ? Float32(my_count - 5) / OVERFLOW_NORM : 0.0f0
        board[7, w] = Float32(opp_count >= 1)
        board[8, w] = Float32(opp_count >= 2)
        board[9, w] = Float32(opp_count >= 3)
        board[10, w] = Float32(opp_count >= 4)
        board[11, w] = Float32(opp_count >= 5)
        board[12, w] = opp_count >= 6 ? Float32(opp_count - 5) / OVERFLOW_NORM : 0.0f0
    end

    # Opp bar at width index 26
    @inbounds begin
        # My thresholds at opp bar (always 0, already filled)
        board[7, 26] = Float32(opp_bar >= 1)
        board[8, 26] = Float32(opp_bar >= 2)
        board[9, 26] = Float32(opp_bar >= 3)
        board[10, 26] = Float32(opp_bar >= 4)
        board[11, 26] = Float32(opp_bar >= 5)
        board[12, 26] = opp_bar >= 6 ? Float32(opp_bar - 5) / OVERFLOW_NORM : 0.0f0
    end

    return board
end

"""
    _encode_globals_minimal!(globals, g, actions) -> Int

Encode minimal global features: dice (12) + move count (4) + off (2) = 18.
Returns new offset after encoding.
"""
function _encode_globals_minimal!(globals::AbstractVector{Float32}, g::BackgammonGame,
                                  actions::Union{Nothing, Vector{Int}}=nothing)
    high, low = _get_dice_pair(g)
    idx = 0

    # High die one-hot (6 values)
    @inbounds for v in 1:6
        globals[idx + v] = Float32(high == v)
    end
    idx += 6

    # Low die one-hot (6 values)
    @inbounds for v in 1:6
        globals[idx + v] = Float32(low == v)
    end
    idx += 6

    # Move count one-hot (4 values, bins 1,2,3,4)
    move_count = 0
    if high > 0
        if high == low
            if g.remaining_actions == 2
                move_count = _compute_playable_dice_doubles(g, actions)
            else
                move_count = _compute_playable_moves_current(g, actions)
            end
        else
            move_count = _compute_playable_dice_non_doubles(g, actions)
        end
    end

    @inbounds for bin in 1:4
        globals[idx + bin] = Float32(move_count == bin)
    end
    idx += 4

    # Off counts (2 values)
    my_off, opp_off = _get_off_counts(g)
    @inbounds begin
        globals[idx + 1] = Float32(my_off) / OFF_NORM
        globals[idx + 2] = Float32(opp_off) / OFF_NORM
    end

    return idx + 2
end

"""
    observe_minimal_hybrid(g::BackgammonGame; actions=nothing) -> NamedTuple

Generate minimal hybrid observation.

Returns NamedTuple with:
- `board`: 12×26 Float32 matrix (spatial, for conv1d)
- `globals`: 18-element Float32 vector (dice + move count + off)

# Example
```julia
obs = observe_minimal_hybrid(g)
board_features = conv1d(obs.board)  # Process spatial
combined = vcat(flatten(board_features), obs.globals)  # Concatenate
output = dense(combined)
```

See also: [`observe_full_hybrid`](@ref), [`observe_biased_hybrid`](@ref)
"""
function observe_minimal_hybrid(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    board = zeros(Float32, 12, 26)
    globals = zeros(Float32, OBS_HYBRID_GLOBALS_MINIMAL)
    observe_minimal_hybrid!(board, globals, g; actions=actions)
    return (board=board, globals=globals)
end

"""
    observe_minimal_hybrid!(board, globals, g; actions=nothing)

In-place version of `observe_minimal_hybrid`.
"""
function observe_minimal_hybrid!(board::AbstractMatrix{Float32}, globals::AbstractVector{Float32},
                                 g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    _encode_board_hybrid!(board, g)
    fill!(globals, 0.0f0)
    _encode_globals_minimal!(globals, g, actions)
    return (board=board, globals=globals)
end

"""
    observe_full_hybrid(g::BackgammonGame; actions=nothing) -> NamedTuple

Generate full hybrid observation.

Returns NamedTuple with:
- `board`: 12×26 Float32 matrix (spatial)
- `globals`: 50-element Float32 vector (18 minimal + 32 full features)

See also: [`observe_minimal_hybrid`](@ref), [`observe_biased_hybrid`](@ref)
"""
function observe_full_hybrid(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    board = zeros(Float32, 12, 26)
    globals = zeros(Float32, OBS_HYBRID_GLOBALS_FULL)
    observe_full_hybrid!(board, globals, g; actions=actions)
    return (board=board, globals=globals)
end

"""
    observe_full_hybrid!(board, globals, g; actions=nothing)

In-place version of `observe_full_hybrid`.
"""
function observe_full_hybrid!(board::AbstractMatrix{Float32}, globals::AbstractVector{Float32},
                              g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    _encode_board_hybrid!(board, g)
    fill!(globals, 0.0f0)
    idx = _encode_globals_minimal!(globals, g, actions)
    _encode_full_features_flat!(globals, g, idx)
    return (board=board, globals=globals)
end

"""
    observe_biased_hybrid(g::BackgammonGame; actions=nothing) -> NamedTuple

Generate biased hybrid observation.

Returns NamedTuple with:
- `board`: 12×26 Float32 matrix (spatial)
- `globals`: 110-element Float32 vector (50 full + 60 biased features)

See also: [`observe_minimal_hybrid`](@ref), [`observe_full_hybrid`](@ref)
"""
function observe_biased_hybrid(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    board = zeros(Float32, 12, 26)
    globals = zeros(Float32, OBS_HYBRID_GLOBALS_BIASED)
    observe_biased_hybrid!(board, globals, g; actions=actions)
    return (board=board, globals=globals)
end

"""
    observe_biased_hybrid!(board, globals, g; actions=nothing)

In-place version of `observe_biased_hybrid`.
"""
function observe_biased_hybrid!(board::AbstractMatrix{Float32}, globals::AbstractVector{Float32},
                                g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    _encode_board_hybrid!(board, g)
    fill!(globals, 0.0f0)
    idx = _encode_globals_minimal!(globals, g, actions)
    idx = _encode_full_features_flat!(globals, g, idx)
    _encode_biased_features_flat!(globals, g, idx)
    return (board=board, globals=globals)
end

# ============================================================================
# Game-Level Observation API (dispatches on g.obs_type)
# ============================================================================

"""
    observe(g::BackgammonGame; actions=nothing)

Generate observation based on game's configured `obs_type` field.

Dispatches to the appropriate observation function based on g.obs_type:
- `:minimal` → `observe_minimal(g)` (30×1×26 tensor)
- `:full` → `observe_full(g)` (62×1×26 tensor)
- `:biased` → `observe_biased(g)` (122×1×26 tensor)
- `:minimal_flat` → `observe_minimal_flat(g)` (330 vector)
- `:full_flat` → `observe_full_flat(g)` (362 vector)
- `:biased_flat` → `observe_biased_flat(g)` (422 vector)
- `:minimal_hybrid` → `observe_minimal_hybrid(g)` (board=12×26, globals=18)
- `:full_hybrid` → `observe_full_hybrid(g)` (board=12×26, globals=50)
- `:biased_hybrid` → `observe_biased_hybrid(g)` (board=12×26, globals=110)

# Example
```julia
g = initial_state(obs_type=:minimal_flat)
obs = observe(g)  # Returns 330-element vector

g2 = initial_state(obs_type=:full)
obs2 = observe(g2)  # Returns 62×1×26 tensor

g3 = initial_state(obs_type=:minimal_hybrid)
obs3 = observe(g3)  # Returns (board=12×26 matrix, globals=18-vector)
```

See also: [`obs_dims`](@ref), [`set_obs_type!`](@ref)
"""
function observe(g::BackgammonGame; actions::Union{Nothing, Vector{Int}}=nothing)
    obs_type = g.obs_type
    if obs_type === :minimal
        return observe_minimal(g; actions=actions)
    elseif obs_type === :full
        return observe_full(g; actions=actions)
    elseif obs_type === :biased
        return observe_biased(g; actions=actions)
    elseif obs_type === :minimal_flat
        return observe_minimal_flat(g; actions=actions)
    elseif obs_type === :full_flat
        return observe_full_flat(g; actions=actions)
    elseif obs_type === :biased_flat
        return observe_biased_flat(g; actions=actions)
    elseif obs_type === :minimal_hybrid
        return observe_minimal_hybrid(g; actions=actions)
    elseif obs_type === :full_hybrid
        return observe_full_hybrid(g; actions=actions)
    elseif obs_type === :biased_hybrid
        return observe_biased_hybrid(g; actions=actions)
    else
        error("Unknown observation type: $obs_type. Valid types: :minimal, :full, :biased, :minimal_flat, :full_flat, :biased_flat, :minimal_hybrid, :full_hybrid, :biased_hybrid")
    end
end

"""
    obs_dims(g::BackgammonGame) -> Tuple or Int

Return the dimensions of observations for game's configured `obs_type`.

Returns a tuple (channels, height, width) for 3D observations,
or an integer for flat observations.

# Example
```julia
g = initial_state(obs_type=:minimal_flat)
dims = obs_dims(g)  # Returns 330

g2 = initial_state(obs_type=:full)
dims2 = obs_dims(g2)  # Returns (62, 1, 26)
```

See also: [`observe`](@ref), [`obs_dims`](@ref) (Symbol version)
"""
function obs_dims(g::BackgammonGame)
    return obs_dims(g.obs_type)
end

"""
    obs_dims(obs_type::Symbol) -> Tuple or Int

Return the dimensions for a given observation type.

# Example
```julia
obs_dims(:minimal)       # Returns (30, 1, 26)
obs_dims(:minimal_flat)  # Returns 330
```
"""
function obs_dims(obs_type::Symbol)
    if obs_type === :minimal
        return (OBS_CHANNELS_MINIMAL, 1, OBS_WIDTH)
    elseif obs_type === :full
        return (OBS_CHANNELS_FULL, 1, OBS_WIDTH)
    elseif obs_type === :biased
        return (OBS_CHANNELS_BIASED, 1, OBS_WIDTH)
    elseif obs_type === :minimal_flat
        return OBS_FLAT_MINIMAL
    elseif obs_type === :full_flat
        return OBS_FLAT_FULL
    elseif obs_type === :biased_flat
        return OBS_FLAT_BIASED
    elseif obs_type === :minimal_hybrid
        return (board=OBS_HYBRID_BOARD, globals=OBS_HYBRID_GLOBALS_MINIMAL)
    elseif obs_type === :full_hybrid
        return (board=OBS_HYBRID_BOARD, globals=OBS_HYBRID_GLOBALS_FULL)
    elseif obs_type === :biased_hybrid
        return (board=OBS_HYBRID_BOARD, globals=OBS_HYBRID_GLOBALS_BIASED)
    else
        error("Unknown observation type: $obs_type. Valid types: :minimal, :full, :biased, :minimal_flat, :full_flat, :biased_flat, :minimal_hybrid, :full_hybrid, :biased_hybrid")
    end
end

"""
    set_obs_type!(g::BackgammonGame, obs_type::Symbol)

Change the observation type for an existing game.

Valid types: `:minimal`, `:full`, `:biased`, `:minimal_flat`, `:full_flat`, `:biased_flat`

# Example
```julia
g = initial_state()
set_obs_type!(g, :full_flat)
obs = observe(g)  # Now returns 362-element vector
```
"""
function set_obs_type!(g::BackgammonGame, obs_type::Symbol)
    # Validate obs_type
    valid_types = (:minimal, :full, :biased, :minimal_flat, :full_flat, :biased_flat,
                   :minimal_hybrid, :full_hybrid, :biased_hybrid)
    if obs_type ∉ valid_types
        error("Unknown observation type: $obs_type. Valid types: $valid_types")
    end
    g.obs_type = obs_type
    return g
end

# Export channel counts for external use
const OBSERVATION_SIZES = (
    minimal = OBS_CHANNELS_MINIMAL,
    full = OBS_CHANNELS_FULL,
    biased = OBS_CHANNELS_BIASED,
    width = OBS_WIDTH,
    minimal_flat = OBS_FLAT_MINIMAL,
    full_flat = OBS_FLAT_FULL,
    biased_flat = OBS_FLAT_BIASED,
    hybrid_board = OBS_HYBRID_BOARD,
    hybrid_globals_minimal = OBS_HYBRID_GLOBALS_MINIMAL,
    hybrid_globals_full = OBS_HYBRID_GLOBALS_FULL,
    hybrid_globals_biased = OBS_HYBRID_GLOBALS_BIASED,
)
