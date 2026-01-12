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
const PASS_LOC = 0
const BAR_LOC = 1

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
    p0 = UInt128(0)
    p0 = incr_count(p0, 1) + incr_count(UInt128(0), 1) 
    # Or cleaner:
    p0 = (UInt128(2) << (1<<2)) | (UInt128(5) << (12<<2)) | (UInt128(3) << (17<<2)) | (UInt128(5) << (19<<2))
    
    p1 = (UInt128(5) << (6<<2)) | (UInt128(3) << (8<<2)) | (UInt128(5) << (13<<2)) | (UInt128(2) << (24<<2))
    
    rng = Random.default_rng()
    current_player = rand(rng, 0:1)
    
    d1, d2 = 0, 0
    while d1 == d2
        d1 = rand(rng, 1:6)
        d2 = rand(rng, 1:6)
    end
    
    return BackgammonGame(
        p0, p1,
        SVector{2, Int8}(d1, d2),
        Int8(1),
        Int8(0),
        Int8(current_player),
        false,
        0.0f0
    )
end

function switch_turn!(g::BackgammonGame)
    g.current_player = 1 - g.current_player
    g.turn = 1 - g.turn
    
    rng = Random.default_rng()
    d1 = rand(rng, 1:6)
    d2 = rand(rng, 1:6)
    if d1 > d2; d1, d2 = d2, d1; end
    
    g.dice = SVector{2, Int8}(Int8(d1), Int8(d2))
    g.remaining_actions = (d1 == d2) ? 2 : 1
end

# Internal apply move for play!
function apply_single_move!(g::BackgammonGame, loc::Integer, die::Integer)
    # Just update fields using the pure function from actions.jl?
    # No, we can't depend on actions.jl here (circular dependency).
    # Copy paste logic or include.
    # Usually modules handle this.
    # Since I duplicated logic in actions.jl, I can just use the mutation version here.
    
    if loc == PASS_LOC; return; end
    
    cp = g.current_player
    
    src_idx = 0
    tgt_idx = 0
    to_off = false
    
    if cp == 0
        src_idx = (loc == BAR_LOC) ? IDX_P0_BAR : (loc - 1)
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
                g.reward = 1.0f0
            end
        else
            if get_count(g.p1, tgt_idx) == 1
                g.p1 = decr_count(g.p1, tgt_idx)
                g.p1 = incr_count(g.p1, IDX_P1_BAR)
            end
            g.p0 = incr_count(g.p0, tgt_idx)
        end
    else
        src_idx = (loc == BAR_LOC) ? IDX_P1_BAR : (25 - (loc - 1))
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
                g.reward = 1.0f0
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
    if g.terminated; return; end
    push!(g.history, Int(action_idx))
    
    idx0 = Int(action_idx - 1)
    loc1 = div(idx0, 26)
    loc2 = idx0 % 26
    
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    
    # We don't have is_move_legal here unless we duplicate it or use actions.jl
    # But actions.jl imports game.jl.
    # We should define is_move_legal in game.jl? 
    # Yes, it's safer.
    
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

# Duplicated helper for internal usage (identical to actions.jl logic but operates on Game)
function is_move_legal(g::BackgammonGame, loc::Integer, die::Integer)
    # 1. Source Check
    cp = g.current_player
    p_my = cp == 0 ? g.p0 : g.p1
    p_opp = cp == 0 ? g.p1 : g.p0
    bar_idx = cp == 0 ? IDX_P0_BAR : IDX_P1_BAR
    
    src_idx = 0
    if loc == BAR_LOC
        src_idx = bar_idx
        if get_count(p_my, src_idx) == 0; return false; end
    else
        canon = loc - 1
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
        if get_count(p_opp, tgt_idx) >= 2; return false; end
        return true
    else
        # Bearing Off
        if cp == 0
            if get_count(p_my, IDX_P0_BAR) > 0; return false; end
            for i in 1:18; if get_count(p_my, i) > 0; return false; end; end
            
            if tgt_idx == 25; return true; end
            # Over-bear: Check 19..src-1
            for i in 19:(src_idx-1); if get_count(p_my, i) > 0; return false; end; end
            return true
        else
            if get_count(p_my, IDX_P1_BAR) > 0; return false; end
            for i in 7:24; if get_count(p_my, i) > 0; return false; end; end
            
            if tgt_idx == 0; return true; end
            # Over-bear: Check src+1..6
            for i in (src_idx+1):6; if get_count(p_my, i) > 0; return false; end; end
            return true
        end
    end
end