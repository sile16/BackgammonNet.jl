using StaticArrays
using Random

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

# Precomputed masks for over-bear validation (checking higher points)
# For P0: indices 19-24 are home board
const MASKS_19_TO = ntuple(i -> i < 19 ? UInt128(0) : reduce(|, UInt128(0xF) << (j << 2) for j in 19:(i-1); init=UInt128(0)), 25)
# For P1: indices 1-6 are home board
const MASKS_TO_6 = ntuple(i -> i > 6 ? UInt128(0) : reduce(|, UInt128(0xF) << (j << 2) for j in (i+1):6; init=UInt128(0)), 6)

# Helper to check if any checkers exist in masked region
@inline has_checkers(board::UInt128, mask::UInt128) = (board & mask) != 0

mutable struct BackgammonGame
    p0::UInt128 # Player 0 Checkers
    p1::UInt128 # Player 1 Checkers
    dice::SVector{2, Int8}
    remaining_actions::Int8
    turn::Int8
    current_player::Int8 # 0 or 1
    terminated::Bool
    reward::Float32
    history::Vector{Int}
end

function BackgammonGame(p0, p1, dice, remaining, turn, cp, term, rew)
    BackgammonGame(p0, p1, dice, remaining, turn, cp, term, rew, Int[])
end

Base.show(io::IO, g::BackgammonGame) = print(io, "BackgammonGame(p=$(g.current_player), dice=$(g.dice), turn=$(g.turn))")

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

function reset!(g::BackgammonGame)
    p0 = (UInt128(2) << (1<<2)) | (UInt128(5) << (12<<2)) | (UInt128(3) << (17<<2)) | (UInt128(5) << (19<<2))
    p1 = (UInt128(5) << (6<<2)) | (UInt128(3) << (8<<2)) | (UInt128(5) << (13<<2)) | (UInt128(2) << (24<<2))

    rng = Random.default_rng()
    cp = rand(rng, 0:1)

    g.p0 = p0
    g.p1 = p1
    g.dice = SVector{2, Int8}(0, 0)
    g.remaining_actions = 1
    g.turn = 0
    g.current_player = cp
    g.terminated = false
    g.reward = 0.0f0
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

# --- Legacy/Test Support: Canonical Board Access ---
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

function initial_state()
    p0 = (UInt128(2) << (1<<2)) | (UInt128(5) << (12<<2)) | (UInt128(3) << (17<<2)) | (UInt128(5) << (19<<2))
    p1 = (UInt128(5) << (6<<2)) | (UInt128(3) << (8<<2)) | (UInt128(5) << (13<<2)) | (UInt128(2) << (24<<2))

    rng = Random.default_rng()
    current_player = rand(rng, 0:1)

    history = Int[]
    sizehint!(history, 120)  # Pre-allocate for typical game length

    return BackgammonGame(
        p0, p1,
        SVector{2, Int8}(0, 0),
        Int8(1),
        Int8(0),
        Int8(current_player),
        false,
        0.0f0,
        history
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

function switch_turn!(g::BackgammonGame)
    g.current_player = 1 - g.current_player
    g.turn = 1 - g.turn
    # We now leave dice as-is or set to 0 to indicate waiting for roll
    g.dice = SVector{2, Int8}(0, 0)
    g.remaining_actions = 1 # Will be updated when dice are set
end

# --- Chance / Stochastic Interface ---

function is_chance_node(g::BackgammonGame)
    return g.dice[1] == 0 && !g.terminated
end

function chance_outcomes(g::BackgammonGame)
    # Returns vector of (outcome_idx, probability)
    # Since probabilities are constant for dice rolls in Backgammon:
    return collect(zip(1:21, DICE_PROBS))
end

"""
    apply_action!(g::BackgammonGame, action_idx::Integer)

Apply a deterministic action to the game state.

!!! warning "Safety"
    This function assumes `action_idx` is a valid action from `legal_actions(g)`.
    Passing an illegal action will corrupt the bitboard state. This design is
    intentional for RL performance—use `legal_actions(g)` to get valid actions.
"""
function apply_action!(g::BackgammonGame, action_idx::Integer)
    if g.terminated; return; end
    if is_chance_node(g)
        error("Cannot apply deterministic action on a chance node. Use apply_chance! or sample_chance!")
    end

    push!(g.history, Int(action_idx))
    
    idx0 = Int(action_idx - 1)
    loc1 = div(idx0, 26)
    loc2 = idx0 % 26
    
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    
    legal1 = is_move_legal(g, loc1, d1)
    
    if legal1
        p0_bak, p1_bak = g.p0, g.p1
        apply_single_move!(g, loc1, d1)
        
        if is_move_legal(g, loc2, d2)
            apply_single_move!(g, loc2, d2)
        else
            g.p0, g.p1 = p0_bak, p1_bak
            if is_move_legal(g, loc2, d2)
                apply_single_move!(g, loc2, d2)
                apply_single_move!(g, loc1, d1)
            else
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
    d1, d2 = DICE_OUTCOMES[outcome_idx]
    g.dice = SVector{2, Int8}(Int8(d1), Int8(d2))
    g.remaining_actions = (d1 == d2) ? 2 : 1
end

function sample_chance!(g::BackgammonGame, rng::AbstractRNG=Random.default_rng())
    # Continously apply random chance actions until a deterministic state is returned
    iters = 0
    max_iters = 1000
    
    while is_chance_node(g)
        iters += 1
        if iters > max_iters
            error("Infinite loop detected in sample_chance! (exceeded $max_iters iterations)")
        end
        
        # Sample outcome
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
        
        apply_chance!(g, idx)
    end
    
    return g
end

function step!(g::BackgammonGame, action::Integer, rng::AbstractRNG=Random.default_rng())
    # 1. Apply the deterministic action
    apply_action!(g, action)
    
    # 2. Resolve any resulting chance nodes to ensure we return a deterministic state
    sample_chance!(g, rng)
    
    return g
end

# Internal apply move for play!
function apply_single_move!(g::BackgammonGame, loc::Integer, die::Integer)
    if loc == PASS_LOC; return; end
    
    cp = g.current_player
    
    src_idx = 0
    tgt_idx = 0
    to_off = false
    
    if cp == 0
        src_idx = (loc == BAR_LOC) ? IDX_P0_BAR : loc
        if loc == BAR_LOC
            tgt_idx = Int(die)
        else
            tgt_idx = src_idx + Int(die)
            if tgt_idx > 24; tgt_idx = IDX_P0_OFF; to_off = true; end
        end
        
        g.p0 = decr_count(g.p0, src_idx)
        if to_off
            g.p0 = incr_count(g.p0, IDX_P0_OFF)
            if get_count(g.p0, IDX_P0_OFF) == 15
                g.terminated = true
                multiplier = 1.0f0
                if get_count(g.p1, IDX_P1_OFF) == 0
                    multiplier = 2.0f0 # Gammon
                    if get_count(g.p1, IDX_P1_BAR) > 0
                        multiplier = 3.0f0
                    else
                        for i in 19:24
                            if get_count(g.p1, i) > 0; multiplier = 3.0f0; break; end
                        end
                    end
                end
                g.reward = multiplier
            end
        else
            if get_count(g.p1, tgt_idx) == 1
                g.p1 = decr_count(g.p1, tgt_idx)
                g.p1 = incr_count(g.p1, IDX_P1_BAR)
            end
            g.p0 = incr_count(g.p0, tgt_idx)
        end
    else
        src_idx = (loc == BAR_LOC) ? IDX_P1_BAR : (25 - loc)
        if loc == BAR_LOC
            tgt_idx = 25 - Int(die)
        else
            tgt_idx = src_idx - Int(die)
            if tgt_idx < 1; tgt_idx = IDX_P1_OFF; to_off = true; end
        end
        
        g.p1 = decr_count(g.p1, src_idx)
        if to_off
            g.p1 = incr_count(g.p1, IDX_P1_OFF)
            if get_count(g.p1, IDX_P1_OFF) == 15
                g.terminated = true
                multiplier = 1.0f0
                if get_count(g.p0, IDX_P0_OFF) == 0
                    multiplier = 2.0f0 # Gammon
                    if get_count(g.p0, IDX_P0_BAR) > 0
                        multiplier = 3.0f0
                    else
                        for i in 1:6
                            if get_count(g.p0, i) > 0; multiplier = 3.0f0; break; end
                        end
                    end
                end
                g.reward = -multiplier
            end
        else
            if get_count(g.p0, tgt_idx) == 1
                g.p0 = decr_count(g.p0, tgt_idx)
                g.p0 = incr_count(g.p0, IDX_P0_BAR)
            end
            g.p1 = incr_count(g.p1, tgt_idx)
        end
    end
end

function play!(g::BackgammonGame, action_idx::Integer)
    step!(g, action_idx)
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
            # Over-bear: Check 19..src_idx-1 using precomputed mask
            if src_idx >= 20 && has_checkers(p_my, MASKS_19_TO[src_idx]); return false; end
            return true
        else
            if get_count(p_my, IDX_P1_BAR) > 0; return false; end
            if has_checkers(p_my, MASK_7_24); return false; end

            if tgt_idx == 0; return true; end
            # Over-bear: Check src+1..6 using precomputed mask
            if src_idx <= 5 && has_checkers(p_my, MASKS_TO_6[src_idx]); return false; end
            return true
        end
    end
end

# Wrapper for game object (calls the pure bitboard function)
@inline function is_move_legal(g::BackgammonGame, loc::Integer, die::Integer)
    return is_move_legal_bits(g.p0, g.p1, g.current_player, loc, die)
end