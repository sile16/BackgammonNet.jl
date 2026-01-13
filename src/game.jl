using StaticArrays
using Random

# --- Game Constants ---
const NUM_POINTS = 24           # Standard backgammon board points
const MAX_CHECKERS = 15         # Each player has 15 checkers

# --- Bitboard Constants & Layout ---
# Nibble Indices (0-27)
const IDX_P1_OFF = 0
const IDX_P0_OFF = 25
const IDX_P0_BAR = 26
const IDX_P1_BAR = 27
# Indices 1-24 are standard physical points

# Actions
const PASS_LOC = 25
const BAR_LOC = 0

# --- Precomputed Bitmasks for Bearing-Off Validation ---
# Each nibble is 4 bits, so index i is at bit position (i << 2)
# Mask for indices 1-18 (P0 must clear these before bearing off)
const MASK_1_18 = reduce(|, UInt128(0xF) << (i << 2) for i in 1:18)
# Mask for indices 7-24 (P1 must clear these before bearing off)
const MASK_7_24 = reduce(|, UInt128(0xF) << (i << 2) for i in 7:24)

# Precomputed masks for over-bear validation (checking higher backgammon points)
#
# IMPORTANT: "Higher point" in backgammon means FURTHER from bearing off, not higher physical index!
# - P0's home board: physical 19-24, where 19 is the 6-point (furthest) and 24 is the 1-point (closest)
# - P1's home board: physical 1-6, where 6 is the 6-point (furthest) and 1 is the 1-point (closest)
#
# Over-bear rule: Can only bear off with a die larger than needed if NO checkers exist
# on points with HIGHER backgammon point numbers (i.e., further from off).
#
# For P0 (moves 1→24→off): higher backgammon points = LOWER physical indices (19 to src_idx-1)
# MASKS_HIGHER_P0[i] masks indices 19 to i-1 (points further from off than i)
const MASKS_HIGHER_P0 = ntuple(i -> i <= 19 ? UInt128(0) : reduce(|, UInt128(0xF) << (j << 2) for j in 19:(i-1); init=UInt128(0)), 24)

# For P1 (moves 24→1→off): higher backgammon points = HIGHER physical indices (src_idx+1 to 6)
# MASKS_HIGHER_P1[i] masks indices i+1 to 6 (points further from off than i)
const MASKS_HIGHER_P1 = ntuple(i -> i >= 6 ? UInt128(0) : reduce(|, UInt128(0xF) << (j << 2) for j in (i+1):6; init=UInt128(0)), 6)

# Helper to check if any checkers exist in masked region
@inline has_checkers(board::UInt128, mask::UInt128) = (board & mask) != 0

# --- Standard Initial Board Position ---
# P0: 2 on point 1, 5 on point 12, 3 on point 17, 5 on point 19
const INIT_P0_STANDARD = (UInt128(2) << (1<<2)) | (UInt128(5) << (12<<2)) | (UInt128(3) << (17<<2)) | (UInt128(5) << (19<<2))
# P1: 5 on point 6, 3 on point 8, 5 on point 13, 2 on point 24
const INIT_P1_STANDARD = (UInt128(5) << (6<<2)) | (UInt128(3) << (8<<2)) | (UInt128(5) << (13<<2)) | (UInt128(2) << (24<<2))

# --- Short Game Initial Board Position ---
# From pgx: [0, -1, -3, 0, 2, -3, 0, -3, -2, 0, 0, -1, 1, 0, 0, 2, 3, 0, 3, -2, 0, 3, 1, 0, 0, 0, 0, 0]
# P0: 2 on 4, 1 on 12, 2 on 15, 3 on 16, 3 on 18, 3 on 21, 1 on 22
const INIT_P0_SHORT = (UInt128(2) << (4<<2)) | (UInt128(1) << (12<<2)) | (UInt128(2) << (15<<2)) |
                      (UInt128(3) << (16<<2)) | (UInt128(3) << (18<<2)) | (UInt128(3) << (21<<2)) | (UInt128(1) << (22<<2))
# P1: 1 on 1, 3 on 2, 3 on 5, 3 on 7, 2 on 8, 1 on 11, 2 on 19
const INIT_P1_SHORT = (UInt128(1) << (1<<2)) | (UInt128(3) << (2<<2)) | (UInt128(3) << (5<<2)) |
                      (UInt128(3) << (7<<2)) | (UInt128(2) << (8<<2)) | (UInt128(1) << (11<<2)) | (UInt128(2) << (19<<2))

mutable struct BackgammonGame
    p0::UInt128 # Player 0 Checkers
    p1::UInt128 # Player 1 Checkers
    dice::SVector{2, Int8}
    remaining_actions::Int8
    current_player::Int8 # 0 or 1
    terminated::Bool
    reward::Float32
    history::Vector{Int}
    doubles_only::Bool # If true, all dice rolls are doubles
    _actions_buffer::Vector{Int}    # Pre-allocated buffer for legal_actions (reduces GC)
    _sources_buffer1::Vector{Int}   # Pre-allocated buffer for source locations
    _sources_buffer2::Vector{Int}   # Second buffer for nested source lookups
end

function BackgammonGame(p0, p1, dice, remaining, cp, term, rew)
    actions_buf = Int[]; sizehint!(actions_buf, 200)
    src_buf1 = Int[]; sizehint!(src_buf1, 25)
    src_buf2 = Int[]; sizehint!(src_buf2, 25)
    BackgammonGame(p0, p1, dice, remaining, cp, term, rew, Int[], false, actions_buf, src_buf1, src_buf2)
end

function BackgammonGame(p0, p1, dice, remaining, cp, term, rew, history)
    actions_buf = Int[]; sizehint!(actions_buf, 200)
    src_buf1 = Int[]; sizehint!(src_buf1, 25)
    src_buf2 = Int[]; sizehint!(src_buf2, 25)
    BackgammonGame(p0, p1, dice, remaining, cp, term, rew, history, false, actions_buf, src_buf1, src_buf2)
end

Base.show(io::IO, g::BackgammonGame) = print(io, "BackgammonGame(p=$(g.current_player), dice=$(g.dice))")

# --- Accessors ---

function current_player(g::BackgammonGame)
    return g.current_player
end

function game_terminated(g::BackgammonGame)
    return g.terminated
end

function winner(g::BackgammonGame)
    if !g.terminated
        return nothing
    end
    return g.reward > 0 ? Int8(0) : Int8(1)
end

function reset!(g::BackgammonGame; first_player::Union{Nothing, Integer}=nothing,
                short_game::Bool=false, doubles_only::Bool=false)
    p0 = short_game ? INIT_P0_SHORT : INIT_P0_STANDARD
    p1 = short_game ? INIT_P1_SHORT : INIT_P1_STANDARD

    cp = isnothing(first_player) ? rand(Random.default_rng(), 0:1) : Int(first_player)

    g.p0 = p0
    g.p1 = p1
    g.dice = SVector{2, Int8}(0, 0)
    g.remaining_actions = 1
    g.current_player = cp
    g.terminated = false
    g.reward = 0.0f0
    g.doubles_only = doubles_only
    empty!(g.history)
    sizehint!(g.history, 120)  # Pre-allocate for typical game length
    return g
end

# --- Bit Manipulation Helpers ---
@inline function get_count(board::UInt128, idx::Int)
    return (board >> (idx << 2)) & 0xF
end

@inline function incr_count(board::UInt128, idx::Int)
    return board + (UInt128(1) << (idx << 2))
end

@inline function decr_count(board::UInt128, idx::Int)
    return board - (UInt128(1) << (idx << 2))
end

# --- Sanity Check Helpers ---
# TODO: Remove sanity checks for large-scale training (set ENABLE_SANITY_CHECKS = false)
# This is a compile-time constant. Changing it requires recompiling the module.
const ENABLE_SANITY_CHECKS = true

"""
    sanity_check_bitboard(p0::UInt128, p1::UInt128)

Validates bitboard integrity:
1. Points 1-24: only one player should have checkers at each point
2. No player should have more than 15 total checkers (catches overflow)

Throws an error if corruption is detected. This catches bugs like:
- incr_count overflow (if point has 15 checkers and we add one, it corrupts the nibble)
- Both players occupying the same point (impossible in backgammon)

Note: This does NOT catch underflow (decr_count on 0 checkers) since the nibble would
wrap to 15 which is still valid. Underflow bugs would show up as overflow elsewhere
when the "missing" checker is eventually double-counted.

Test boards with fewer than 15 checkers are allowed and still get same-point validation.
"""
function sanity_check_bitboard(p0::UInt128, p1::UInt128)
    @static if !ENABLE_SANITY_CHECKS
        return
    end

    p0_total = 0
    p1_total = 0

    # Check all 28 nibble positions (indices 0-27)
    for idx in 0:27
        p0_count = Int(get_count(p0, idx))
        p1_count = Int(get_count(p1, idx))

        # Points 1-24: only one player should have checkers at each point
        if idx >= 1 && idx <= NUM_POINTS
            if p0_count > 0 && p1_count > 0
                error("Bitboard corruption: Both players have checkers at physical point $idx (P0: $p0_count, P1: $p1_count)")
            end
        end

        p0_total += p0_count
        p1_total += p1_count
    end

    # Check for overflow (more than 15 checkers for any player)
    if p0_total > MAX_CHECKERS || p1_total > MAX_CHECKERS
        error("Bitboard corruption: P0 has $p0_total checkers, P1 has $p1_total checkers. " *
              "Max is $MAX_CHECKERS. Possible overflow from incr_count.")
    end
    # Note: p0_total < 15 or p1_total < 15 is allowed for test boards
end

"""
    sanity_check_game(g::BackgammonGame)

Validates game state integrity. Wrapper around sanity_check_bitboard.
"""
function sanity_check_game(g::BackgammonGame)
    @static if !ENABLE_SANITY_CHECKS
        return
    end
    sanity_check_bitboard(g.p0, g.p1)
end

# --- Canonical Board Access ---
function Base.getindex(g::BackgammonGame, i::Int)
    cp = g.current_player
    p_my = cp == 0 ? g.p0 : g.p1
    p_opp = cp == 0 ? g.p1 : g.p0
    
    if i <= 24
        # Points
        if cp == 0
            idx = i
            my_c = Int(get_count(p_my, idx))
            opp_c = Int(get_count(p_opp, idx))
            return Int8(my_c > 0 ? my_c : -opp_c)
        else
            idx = 25 - i
            my_c = Int(get_count(p_my, idx))
            opp_c = Int(get_count(p_opp, idx))
            return Int8(my_c > 0 ? my_c : -opp_c)
        end
    elseif i == 25 # My Bar
        idx = cp == 0 ? IDX_P0_BAR : IDX_P1_BAR
        return Int8(get_count(p_my, idx))
    elseif i == 26 # Opp Bar
        idx = cp == 0 ? IDX_P1_BAR : IDX_P0_BAR
        return Int8(-Int(get_count(p_opp, idx)))
    elseif i == 27 # My Off
        idx = cp == 0 ? IDX_P0_OFF : IDX_P1_OFF
        return Int8(get_count(p_my, idx))
    elseif i == 28 # Opp Off
        idx = cp == 0 ? IDX_P1_OFF : IDX_P0_OFF
        return Int8(-Int(get_count(p_opp, idx)))
    end
    return Int8(0)
end

"""
    initial_state(; first_player=nothing, short_game=false, doubles_only=false) -> BackgammonGame

Create a new backgammon game in the initial position (chance node awaiting dice roll).

# Keyword Arguments
- `first_player`: Set to `0` or `1` to choose starting player, or `nothing` for random.
- `short_game`: Use modified board position with pieces closer to bearing off (faster games).
- `doubles_only`: All dice rolls are doubles (1-1 through 6-6 with uniform probability).

Call `sample_chance!(g)` to roll dice before the first move, or use `step!` which
handles dice rolls automatically.
"""
function initial_state(; first_player::Union{Nothing, Integer}=nothing,
                       short_game::Bool=false, doubles_only::Bool=false)
    p0 = short_game ? INIT_P0_SHORT : INIT_P0_STANDARD
    p1 = short_game ? INIT_P1_SHORT : INIT_P1_STANDARD

    cp = isnothing(first_player) ? rand(Random.default_rng(), 0:1) : Int(first_player)

    history = Int[]
    sizehint!(history, 120)  # Pre-allocate for typical game length

    actions_buf = Int[]
    sizehint!(actions_buf, 200)  # Pre-allocate for legal actions

    src_buf1 = Int[]
    sizehint!(src_buf1, 25)  # Pre-allocate for source locations

    src_buf2 = Int[]
    sizehint!(src_buf2, 25)  # Pre-allocate for nested source lookups

    return BackgammonGame(
        p0, p1,
        SVector{2, Int8}(0, 0),
        Int8(1),
        Int8(cp),
        false,
        0.0f0,
        history,
        doubles_only,
        actions_buf,
        src_buf1,
        src_buf2
    )
end

# --- Stochastic Actions (Dice Rolls) ---
# 21 unique outcomes for two 6-sided dice (d1 <= d2)
const DICE_OUTCOMES = [
    (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
    (2,2), (2,3), (2,4), (2,5), (2,6),
    (3,3), (3,4), (3,5), (3,6),
    (4,4), (4,5), (4,6),
    (5,5), (5,6),
    (6,6)
]

const DICE_PROBS = Float32[
    1/36, 2/36, 2/36, 2/36, 2/36, 2/36,
    1/36, 2/36, 2/36, 2/36, 2/36,
    1/36, 2/36, 2/36, 2/36,
    1/36, 2/36, 2/36,
    1/36, 2/36,
    1/36
]

# Indices into DICE_OUTCOMES for doubles only (1-1, 2-2, 3-3, 4-4, 5-5, 6-6)
const DOUBLES_INDICES = [1, 7, 12, 16, 19, 21]

# Precomputed chance outcomes for doubles_only mode (length 21, zeros for non-doubles)
const DOUBLES_ONLY_PROBS = ntuple(i -> i in DOUBLES_INDICES ? (1.0f0 / 6.0f0) : 0.0f0, 21)
const DOUBLES_ONLY_OUTCOMES = collect(zip(1:21, DOUBLES_ONLY_PROBS))

function switch_turn!(g::BackgammonGame)
    g.current_player = 1 - g.current_player
    # Set dice to 0 to indicate waiting for roll
    g.dice = SVector{2, Int8}(0, 0)
    g.remaining_actions = 1 # Will be updated when dice are set
end

# --- Chance / Stochastic Interface ---

function is_chance_node(g::BackgammonGame)
    # Check both dice for robustness against state corruption
    return g.dice[1] == 0 && g.dice[2] == 0 && !g.terminated
end

# Precomputed standard chance outcomes (avoids allocation on each call)
const STANDARD_OUTCOMES = collect(zip(1:21, DICE_PROBS))

function chance_outcomes(g::BackgammonGame)
    # Returns vector of (outcome_idx, probability)
    if g.doubles_only
        return DOUBLES_ONLY_OUTCOMES
    else
        return STANDARD_OUTCOMES
    end
end

"""
    apply_action!(g::BackgammonGame, action_idx::Integer)

Apply a deterministic action to the game state.

Validates that the action is legal before applying. Throws an error if the action
is not in `legal_actions(g)`.

!!! note "Performance"
    Action validation is controlled by ENABLE_SANITY_CHECKS. Set to false for
    production RL training once the legal action mask is thoroughly tested.
"""
function apply_action!(g::BackgammonGame, action_idx::Integer)
    if g.terminated; return; end
    if is_chance_node(g)
        error("Cannot apply deterministic action on a chance node. Use apply_chance! or sample_chance!")
    end

    # Validate action - controlled by ENABLE_SANITY_CHECKS for performance
    @static if ENABLE_SANITY_CHECKS
        if !is_action_valid(g, action_idx)
            error("Invalid action $action_idx. Valid actions: $(legal_actions(g))")
        end
    end

    push!(g.history, Int(action_idx))
    
    idx0 = Int(action_idx - 1)
    loc1 = div(idx0, 26)
    loc2 = idx0 % 26
    
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    
    legal1 = is_move_legal(g, loc1, d1)

    if legal1
        # Backup full state that apply_single_move! can modify
        p0_bak, p1_bak = g.p0, g.p1
        terminated_bak, reward_bak = g.terminated, g.reward
        apply_single_move!(g, loc1, d1)

        if is_move_legal(g, loc2, d2)
            apply_single_move!(g, loc2, d2)
        else
            # Restore full state before trying alternate ordering
            g.p0, g.p1 = p0_bak, p1_bak
            g.terminated, g.reward = terminated_bak, reward_bak
            if is_move_legal(g, loc2, d2)
                apply_single_move!(g, loc2, d2)
                apply_single_move!(g, loc1, d1)
            else
                # WARNING: This branch is only reached when ENABLE_SANITY_CHECKS=false
                # and an invalid action is passed. Neither move ordering is legal.
                # We apply moves anyway as a defensive fallback, but this WILL corrupt
                # the game state (potential nibble overflow/underflow).
                #
                # In production, ensure your policy only selects from legal_actions()
                # or use is_action_valid() to validate before calling apply_action!.
                apply_single_move!(g, loc1, d1)
                apply_single_move!(g, loc2, d2)
            end
        end
    else
        apply_single_move!(g, loc2, d2)
        apply_single_move!(g, loc1, d1)
    end
    
    if g.terminated; return; end
    
    g.remaining_actions -= 1
    if g.remaining_actions <= 0
        switch_turn!(g)
    end
end

function apply_chance!(g::BackgammonGame, outcome_idx::Integer)
    if !is_chance_node(g)
        error("Cannot apply chance action on a deterministic node.")
    end
    if g.doubles_only && !(outcome_idx in DOUBLES_INDICES)
        error("In doubles_only mode, only doubles outcomes are valid. Got index $outcome_idx.")
    end
    d1, d2 = DICE_OUTCOMES[outcome_idx]
    g.dice = SVector{2, Int8}(Int8(d1), Int8(d2))
    g.remaining_actions = (d1 == d2) ? 2 : 1
end

function sample_chance!(g::BackgammonGame, rng::AbstractRNG=Random.default_rng())
    # Continuously apply random chance actions until a deterministic state is returned
    # with at least one valid move (not just PASS|PASS)
    iters = 0
    max_iters = 1000

    while true
        if g.terminated
            break
        end

        iters += 1
        if iters > max_iters
            error("Infinite loop detected in sample_chance! (exceeded $max_iters iterations)")
        end

        if is_chance_node(g)
            # Roll dice
            local idx::Int
            if g.doubles_only
                # In doubles_only mode, uniformly sample from the 6 doubles outcomes
                idx = DOUBLES_INDICES[rand(rng, 1:6)]
            else
                # Sample outcome using standard probabilities
                r = rand(rng, Float32)
                c = 0.0f0
                idx = 21
                for (i, p) in enumerate(DICE_PROBS)
                    c += p
                    if r <= c
                        idx = i
                        break
                    end
                end
            end
            apply_chance!(g, idx)
        else
            # Check if only PASS|PASS is available (no valid moves)
            actions = legal_actions(g)
            pass_pass = encode_action(PASS_LOC, PASS_LOC)
            if length(actions) == 1 && actions[1] == pass_pass
                # Auto-apply PASS|PASS and continue to roll for next player
                apply_action!(g, pass_pass)
            else
                # Player has valid moves, exit loop
                break
            end
        end
    end

    return g
end

"""
    step!(g::BackgammonGame, action::Integer, rng=Random.default_rng()) -> BackgammonGame

High-level step function for RL environments. Applies `action` and automatically
rolls dice for the next player, ensuring the returned state is always deterministic
(not a chance node) with valid moves available.

Use this for standard RL training loops where you don't need explicit control
over dice rolls. For MCTS or explicit stochastic control, use `apply_action!`
and `apply_chance!` / `sample_chance!` separately.
"""
function step!(g::BackgammonGame, action::Integer, rng::AbstractRNG=Random.default_rng())
    apply_action!(g, action)
    sample_chance!(g, rng)
    return g
end

# Internal apply move for play!
# Uses apply_move_internal from actions.jl for move logic, then handles termination.
function apply_single_move!(g::BackgammonGame, loc::Integer, die::Integer)
    if loc == PASS_LOC; return; end

    # Delegate move logic to apply_move_internal (shared with legal action generation)
    # Note: apply_move_internal already calls sanity_check_bitboard
    g.p0, g.p1 = apply_move_internal(g.p0, g.p1, g.current_player, loc, die)

    # Check for game termination (15 pieces borne off)
    cp = g.current_player
    if cp == 0
        if get_count(g.p0, IDX_P0_OFF) == 15
            g.terminated = true
            multiplier = 1.0f0
            if get_count(g.p1, IDX_P1_OFF) == 0
                multiplier = 2.0f0 # Gammon
                if get_count(g.p1, IDX_P1_BAR) > 0
                    multiplier = 3.0f0 # Backgammon (on bar)
                else
                    for i in 19:24
                        if get_count(g.p1, i) > 0; multiplier = 3.0f0; break; end # Backgammon (in home)
                    end
                end
            end
            g.reward = multiplier
        end
    else
        if get_count(g.p1, IDX_P1_OFF) == 15
            g.terminated = true
            multiplier = 1.0f0
            if get_count(g.p0, IDX_P0_OFF) == 0
                multiplier = 2.0f0 # Gammon
                if get_count(g.p0, IDX_P0_BAR) > 0
                    multiplier = 3.0f0 # Backgammon (on bar)
                else
                    for i in 1:6
                        if get_count(g.p0, i) > 0; multiplier = 3.0f0; break; end # Backgammon (in home)
                    end
                end
            end
            g.reward = -multiplier
        end
    end
end

# Pure function to check move legality on bitboards (shared with actions.jl)
function is_move_legal_bits(p0::UInt128, p1::UInt128, cp::Integer, loc::Integer, die::Integer)
    if loc == PASS_LOC; return true; end

    # 1. Source Check
    p_my = cp == 0 ? p0 : p1
    p_opp = cp == 0 ? p1 : p0
    bar_idx = cp == 0 ? IDX_P0_BAR : IDX_P1_BAR

    src_idx = 0
    if loc == BAR_LOC
        src_idx = bar_idx
        if get_count(p_my, src_idx) == 0; return false; end
    else
        canon = loc
        src_idx = (cp == 0) ? canon : (25 - canon)
        if get_count(p_my, src_idx) == 0; return false; end
        if get_count(p_my, bar_idx) > 0; return false; end
    end

    # 2. Target Check
    tgt_idx = 0
    is_off = false

    if cp == 0
        tgt_idx = (loc == BAR_LOC) ? Int(die) : (src_idx + Int(die))
        if tgt_idx > 24; is_off = true; end
    else
        tgt_idx = (loc == BAR_LOC) ? (25 - Int(die)) : (src_idx - Int(die))
        if tgt_idx < 1; is_off = true; end
    end

    if !is_off
        # Block check
        if get_count(p_opp, tgt_idx) >= 2; return false; end
        return true
    else
        # Bearing Off - using precomputed bitmasks
        if cp == 0
            if get_count(p_my, IDX_P0_BAR) > 0; return false; end
            if has_checkers(p_my, MASK_1_18); return false; end

            if tgt_idx == 25; return true; end
            # Over-bear: Can only over-bear from the HIGHEST backgammon point (furthest from off).
            # For P0, higher backgammon points = lower physical indices (19 to src_idx-1).
            if src_idx >= 20 && has_checkers(p_my, MASKS_HIGHER_P0[src_idx]); return false; end
            return true
        else
            if get_count(p_my, IDX_P1_BAR) > 0; return false; end
            if has_checkers(p_my, MASK_7_24); return false; end

            if tgt_idx == 0; return true; end
            # Over-bear: Can only over-bear from the HIGHEST backgammon point (furthest from off).
            # For P1, higher backgammon points = higher physical indices (src_idx+1 to 6).
            if src_idx <= 5 && has_checkers(p_my, MASKS_HIGHER_P1[src_idx]); return false; end
            return true
        end
    end
end

# Wrapper for game object (calls the pure bitboard function)
@inline function is_move_legal(g::BackgammonGame, loc::Integer, die::Integer)
    return is_move_legal_bits(g.p0, g.p1, g.current_player, loc, die)
end
