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

# --- Buffer Size Constants ---
const ACTIONS_BUFFER_SIZE = 200     # Max legal actions in any position
const SOURCES_BUFFER_SIZE = 25      # Max source locations (24 points + bar)
const HISTORY_BUFFER_SIZE = 120     # Typical game length in actions
const MAX_CHANCE_ITERATIONS = 1000  # Safety limit for sample_chance! loop

# --- Game Phase ---
@enum GamePhase::Int8 begin
    PHASE_CHANCE = 0        # Waiting for dice roll
    PHASE_CUBE_DECISION = 1 # Before roll: player may double
    PHASE_CUBE_RESPONSE = 2 # Opponent must take or pass
    PHASE_CHECKER_PLAY = 3  # Normal checker move selection
end

"""
    BackgammonGame

Mutable game state for backgammon, using bitboard representation.

# Fields
- `p0::UInt128`: Player 0's checker positions (4-bit nibbles per location)
- `p1::UInt128`: Player 1's checker positions
- `dice::SVector{2, Int8}`: Current dice values (0,0 indicates chance node)
- `remaining_actions::Int8`: Actions remaining this turn (2 for doubles, 1 otherwise)
- `current_player::Int8`: Current player (0 or 1)
- `terminated::Bool`: Whether game has ended
- `reward::Float32`: Final reward from P0's perspective (±1 single, ±2 gammon, ±3 backgammon)
- `history::Vector{Int}`: Action history for this game
- `doubles_only::Bool`: If true, all dice rolls are doubles (for training variants)
- `obs_type::Symbol`: Observation type for observe(g) dispatch

# Cube State
- `cube_value::Int16`: Current cube value (1, 2, 4, 8, ...)
- `cube_owner::Int8`: Cube ownership (-1=opponent, 0=centered, +1=current player)
- `phase::GamePhase`: Current game phase (CHANCE, CUBE_DECISION, CUBE_RESPONSE, CHECKER_PLAY)

# Match State (my_away=0 and opp_away=0 means money play)
- `my_away::Int8`: Points current player needs to win match (0=money play)
- `opp_away::Int8`: Points opponent needs to win match
- `is_crawford::Bool`: Crawford game (no doubling allowed)
- `is_post_crawford::Bool`: Post-Crawford game
- `jacoby_enabled::Bool`: Jacoby rule (money play: gammons don't count unless cube turned)

# Internal Buffers (pre-allocated to reduce GC pressure)
- `_actions_buffer`: Buffer for legal action generation
- `_actions_cached`: Whether `_actions_buffer` contains valid cached actions
- `_sources_buffer1`, `_sources_buffer2`: Buffers for source location lookups
"""
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
    obs_type::Symbol   # Observation type: :minimal, :full, :biased, :minimal_flat, :full_flat, :biased_flat
    # Cube state
    cube_value::Int16
    cube_owner::Int8       # -1=opponent, 0=centered, +1=current player
    phase::GamePhase
    cube_enabled::Bool     # Whether cube decisions are active (false = money play without cube)
    # Match state (0/0 = money play)
    my_away::Int8
    opp_away::Int8
    is_crawford::Bool
    is_post_crawford::Bool
    jacoby_enabled::Bool
    # Internal buffers
    _actions_buffer::Vector{Int}    # Pre-allocated buffer for legal_actions (reduces GC)
    _actions_cached::Bool           # True if _actions_buffer contains valid cached actions
    _sources_buffer1::Vector{Int}   # Pre-allocated buffer for source locations
    _sources_buffer2::Vector{Int}   # Second buffer for nested source lookups
end

"""
    _create_game_buffers() -> (Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int})

Internal helper to create pre-allocated buffers for a new game.

Returns a tuple of four vectors with pre-allocated capacity:
- `history`: Action history buffer (capacity: 120)
- `actions_buf`: Legal actions buffer (capacity: 200)
- `src_buf1`: Source locations buffer 1 (capacity: 25)
- `src_buf2`: Source locations buffer 2 (capacity: 25)

These buffers are reused during gameplay to reduce GC pressure.
"""
function _create_game_buffers()
    history = Int[]
    sizehint!(history, HISTORY_BUFFER_SIZE)

    actions_buf = Int[]
    sizehint!(actions_buf, ACTIONS_BUFFER_SIZE)

    src_buf1 = Int[]
    sizehint!(src_buf1, SOURCES_BUFFER_SIZE)

    src_buf2 = Int[]
    sizehint!(src_buf2, SOURCES_BUFFER_SIZE)

    return history, actions_buf, src_buf1, src_buf2
end

function BackgammonGame(p0, p1, dice, remaining, cp, term, rew; obs_type::Symbol=:minimal_flat)
    history, actions_buf, src_buf1, src_buf2 = _create_game_buffers()
    # Infer phase from dice: if dice are set, we're in checker play
    phase = (dice[1] == 0 && dice[2] == 0) ? PHASE_CHANCE : PHASE_CHECKER_PLAY
    BackgammonGame(p0, p1, dice, remaining, cp, term, rew, history, false, obs_type,
        Int16(1), Int8(0), phase, false, Int8(0), Int8(0), false, false, false,
        actions_buf, false, src_buf1, src_buf2)
end

function BackgammonGame(p0, p1, dice, remaining, cp, term, rew, history; obs_type::Symbol=:minimal_flat)
    _, actions_buf, src_buf1, src_buf2 = _create_game_buffers()
    phase = (dice[1] == 0 && dice[2] == 0) ? PHASE_CHANCE : PHASE_CHECKER_PLAY
    BackgammonGame(p0, p1, dice, remaining, cp, term, rew, history, false, obs_type,
        Int16(1), Int8(0), phase, false, Int8(0), Int8(0), false, false, false,
        actions_buf, false, src_buf1, src_buf2)
end

"""
    clone(g::BackgammonGame) -> BackgammonGame

Create a deep copy of a game state with fresh internal buffers.

This is the recommended way to copy game states for MCTS or other algorithms
that need independent game copies. The returned game has:
- Identical game state (board, dice, player, etc.)
- Fresh pre-allocated buffers (not shared with original)
- Invalidated action cache (will recompute on next legal_actions call)

# Example
```julia
g_copy = clone(g)
step!(g_copy, action)  # Doesn't affect original g
```
"""
function clone(g::BackgammonGame)
    history, actions_buf, src_buf1, src_buf2 = _create_game_buffers()
    append!(history, g.history)
    return BackgammonGame(
        g.p0, g.p1, g.dice, g.remaining_actions,
        g.current_player, g.terminated, g.reward,
        history, g.doubles_only, g.obs_type,
        g.cube_value, g.cube_owner, g.phase, g.cube_enabled,
        g.my_away, g.opp_away, g.is_crawford, g.is_post_crawford, g.jacoby_enabled,
        actions_buf, false, src_buf1, src_buf2
    )
end

Base.show(io::IO, g::BackgammonGame) = print(io, "BackgammonGame(p=$(g.current_player), dice=$(g.dice))")

"""
    Base.==(g1::BackgammonGame, g2::BackgammonGame)

Value-based equality for game states. Compares all game state fields except
internal buffers and history (which don't affect the game state).

This enables using BackgammonGame as dictionary keys in MCTS trees.
"""
function Base.:(==)(g1::BackgammonGame, g2::BackgammonGame)
    return g1.p0 == g2.p0 &&
           g1.p1 == g2.p1 &&
           g1.dice == g2.dice &&
           g1.remaining_actions == g2.remaining_actions &&
           g1.current_player == g2.current_player &&
           g1.terminated == g2.terminated &&
           g1.reward == g2.reward &&
           g1.doubles_only == g2.doubles_only &&
           g1.cube_value == g2.cube_value &&
           g1.cube_owner == g2.cube_owner &&
           g1.phase == g2.phase &&
           g1.cube_enabled == g2.cube_enabled &&
           g1.my_away == g2.my_away &&
           g1.opp_away == g2.opp_away &&
           g1.is_crawford == g2.is_crawford
end

"""
    Base.hash(g::BackgammonGame, h::UInt)

Hash function for game states, consistent with `==`.
Combines hashes of all game state fields.

This enables using BackgammonGame as dictionary keys in MCTS trees.
"""
function Base.hash(g::BackgammonGame, h::UInt)
    h = hash(g.p0, h)
    h = hash(g.p1, h)
    h = hash(g.dice, h)
    h = hash(g.remaining_actions, h)
    h = hash(g.current_player, h)
    h = hash(g.terminated, h)
    h = hash(g.reward, h)
    h = hash(g.doubles_only, h)
    h = hash(g.cube_value, h)
    h = hash(g.cube_owner, h)
    h = hash(g.phase, h)
    h = hash(g.cube_enabled, h)
    h = hash(g.my_away, h)
    h = hash(g.opp_away, h)
    h = hash(g.is_crawford, h)
    return h
end

"""
    _get_initial_boards(short_game::Bool) -> (UInt128, UInt128)

Internal helper to retrieve initial bitboard positions.

Returns `(p0, p1)` tuple with player 0 and player 1 starting positions.
If `short_game` is true, returns positions with pieces closer to bearing off
(faster games for training). Otherwise, returns standard backgammon starting positions.
"""
@inline function _get_initial_boards(short_game::Bool)
    p0 = short_game ? INIT_P0_SHORT : INIT_P0_STANDARD
    p1 = short_game ? INIT_P1_SHORT : INIT_P1_STANDARD
    return p0, p1
end

"""
    _resolve_first_player(first_player::Union{Nothing, Integer}) -> Int

Internal helper to determine the starting player.

If `first_player` is `nothing`, randomly selects 0 or 1.
If `first_player` is 0 or 1, returns that value.
Throws `ArgumentError` if `first_player` is any other integer value.
"""
@inline function _resolve_first_player(first_player::Union{Nothing, Integer})
    if isnothing(first_player)
        return rand(Random.default_rng(), 0:1)
    end
    fp = Int(first_player)
    if fp != 0 && fp != 1
        throw(ArgumentError("first_player must be 0, 1, or nothing (got $fp)"))
    end
    return fp
end

# --- Accessors ---

function current_player(g::BackgammonGame)
    return g.current_player
end

function game_terminated(g::BackgammonGame)
    return g.terminated
end

"""
    winner(g::BackgammonGame) -> Union{Nothing, Int8}

Returns the winning player (0 or 1) if the game is terminated, or `nothing` if:
- The game is not terminated, or
- The game state is invalid (reward == 0 on a terminated game)

In normal gameplay, a terminated game always has reward != 0.
"""
function winner(g::BackgammonGame)
    if !g.terminated
        return nothing
    end
    if g.reward > 0
        return Int8(0)
    elseif g.reward < 0
        return Int8(1)
    else
        # Invalid state: terminated but no reward set
        return nothing
    end
end

"""
    reset!(g::BackgammonGame; first_player=nothing, short_game=false, doubles_only=false) -> BackgammonGame

Reset an existing game to initial state without reallocating buffers.

This is more efficient than creating a new game with `initial_state()` when running
many games in sequence, as it reuses the pre-allocated internal buffers.

# Keyword Arguments
- `first_player`: Set to `0` or `1` to choose starting player, or `nothing` for random.
- `short_game`: Use modified board position with pieces closer to bearing off (faster games).
- `doubles_only`: All dice rolls are doubles (1-1 through 6-6 with uniform probability).

# Returns
The same game object `g`, reset to initial state (a chance node awaiting dice roll).
"""
function reset!(g::BackgammonGame; first_player::Union{Nothing, Integer}=nothing,
                short_game::Bool=false, doubles_only::Bool=false,
                obs_type::Union{Nothing, Symbol}=nothing)
    p0, p1 = _get_initial_boards(short_game)
    cp = _resolve_first_player(first_player)

    g.p0 = p0
    g.p1 = p1
    g.dice = SVector{2, Int8}(0, 0)
    g.remaining_actions = 1
    g.current_player = cp
    g.terminated = false
    g.reward = 0.0f0
    g.doubles_only = doubles_only
    if obs_type !== nothing
        g.obs_type = obs_type
    end
    # Reset cube/match state
    g.cube_value = Int16(1)
    g.cube_owner = Int8(0)
    g.phase = PHASE_CHANCE
    g.cube_enabled = false
    g.my_away = Int8(0)
    g.opp_away = Int8(0)
    g.is_crawford = false
    g.is_post_crawford = false
    g.jacoby_enabled = false
    g._actions_cached = false  # Invalidate legal actions cache
    empty!(g.history)
    sizehint!(g.history, HISTORY_BUFFER_SIZE)
    return g
end

# --- Bit Manipulation Helpers ---

"""
    get_count(board::UInt128, idx::Integer) -> UInt128

Extract the checker count at position `idx` from a bitboard.
Each position uses a 4-bit nibble, so index `i` is at bit position `i << 2`.
Returns 0-15 (though valid counts are 0-15 checkers).
"""
@inline function get_count(board::UInt128, idx::Integer)
    return (board >> (idx << 2)) & 0xF
end

"""
    incr_count(board::UInt128, idx::Integer) -> UInt128

Increment the checker count at position `idx` by 1.
Returns a new bitboard with the updated count.
Warning: Does not check for overflow (count > 15).
"""
@inline function incr_count(board::UInt128, idx::Integer)
    return board + (UInt128(1) << (idx << 2))
end

"""
    decr_count(board::UInt128, idx::Integer) -> UInt128

Decrement the checker count at position `idx` by 1.
Returns a new bitboard with the updated count.
Warning: Does not check for underflow (count == 0).
"""
@inline function decr_count(board::UInt128, idx::Integer)
    return board - (UInt128(1) << (idx << 2))
end

# --- Sanity Check Helpers ---
# NOTE: Sanity checks are controlled by ENABLE_SANITY_CHECKS (compile-time constant).
# Set to false for large-scale training once legal action generation is thoroughly tested.
# Changing this constant requires recompiling the module.
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
function Base.getindex(g::BackgammonGame, i::Integer)
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
    else # i == 28, Opp Off
        idx = cp == 0 ? IDX_P1_OFF : IDX_P0_OFF
        return Int8(-Int(get_count(p_opp, idx)))
    end
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
                       short_game::Bool=false, doubles_only::Bool=false,
                       obs_type::Symbol=:minimal_flat)
    p0, p1 = _get_initial_boards(short_game)
    cp = _resolve_first_player(first_player)
    history, actions_buf, src_buf1, src_buf2 = _create_game_buffers()

    return BackgammonGame(
        p0, p1,
        SVector{2, Int8}(0, 0),
        Int8(1),
        Int8(cp),
        false,
        0.0f0,
        history,
        doubles_only,
        obs_type,
        Int16(1),       # cube_value
        Int8(0),        # cube_owner (centered)
        PHASE_CHANCE,   # phase
        false,          # cube_enabled
        Int8(0),        # my_away (money play)
        Int8(0),        # opp_away
        false,          # is_crawford
        false,          # is_post_crawford
        false,          # jacoby_enabled
        actions_buf,
        false,  # _actions_cached
        src_buf1,
        src_buf2
    )
end

# --- Stochastic Actions (Dice Rolls) ---
# 21 unique outcomes for two 6-sided dice, stored as (high, low) where high >= low.
# This ordering aligns with action encoding: loc1 uses dice[1] (high), loc2 uses dice[2] (low).
# The observation dice slots also show (high, low) in slots 0 and 1.
const DICE_OUTCOMES = [
    (1,1), (2,1), (3,1), (4,1), (5,1), (6,1),
    (2,2), (3,2), (4,2), (5,2), (6,2),
    (3,3), (4,3), (5,3), (6,3),
    (4,4), (5,4), (6,4),
    (5,5), (6,5),
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

"""
    switch_turn!(g::BackgammonGame)

Internal function to switch the current player and reset dice to indicate a chance node.
Sets `remaining_actions = 1` as a placeholder; this value is overwritten by `apply_chance!`
when dice are actually rolled (to 2 for doubles, 1 for non-doubles).
"""
function switch_turn!(g::BackgammonGame)
    g.current_player = 1 - g.current_player
    g.dice = SVector{2, Int8}(0, 0)  # Zero dice indicates chance node (waiting for roll)
    g.remaining_actions = 1  # Placeholder; overwritten by apply_chance!
    # If cube is available, go to cube decision first; otherwise straight to dice roll
    g.phase = may_double(g) ? PHASE_CUBE_DECISION : PHASE_CHANCE
    g._actions_cached = false  # Invalidate cache (player changed)
end

# --- Chance / Stochastic Interface ---

function is_chance_node(g::BackgammonGame)
    return g.phase == PHASE_CHANCE && !g.terminated
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

    # --- Cube actions (677-680) ---
    if is_cube_action(action_idx)
        g._actions_cached = false
        push!(g.history, Int(action_idx))
        if action_idx == ACTION_CUBE_NO_DOUBLE
            # Proceed to dice roll
            g.phase = PHASE_CHANCE
        elseif action_idx == ACTION_CUBE_DOUBLE
            # Switch to opponent for take/pass response
            g.current_player = 1 - g.current_player
            g.phase = PHASE_CUBE_RESPONSE
        elseif action_idx == ACTION_CUBE_TAKE
            # Accept double: cube value doubles, taker owns cube
            g.cube_value *= Int16(2)
            g.cube_owner = Int8(1)  # Taker owns (from their perspective)
            # Switch back to doubler for their turn
            g.current_player = 1 - g.current_player
            g.phase = PHASE_CHANCE
        elseif action_idx == ACTION_CUBE_PASS
            # Decline double: game ends, doubler wins current cube value
            g.terminated = true
            # Doubler = 1 - current_player (we switched player on DOUBLE)
            doubler = 1 - g.current_player
            g.reward = doubler == 0 ? Float32(g.cube_value) : Float32(-g.cube_value)
        end
        return
    end

    # --- Checker move actions (1-676) ---

    # Invalidate legal actions cache (state is about to change)
    g._actions_cached = false

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
                if !is_move_legal(g, loc1, d1)
                    g.p0, g.p1 = p0_bak, p1_bak
                    g.terminated, g.reward = terminated_bak, reward_bak
                    error("Invalid action $action_idx: neither move ordering is legal")
                end
                apply_single_move!(g, loc1, d1)
            else
                # Neither move ordering is legal - this is an invalid action.
                # Always throw an error to prevent state corruption, even when
                # ENABLE_SANITY_CHECKS=false. The sanity check flag controls
                # upfront validation overhead, not corruption protection.
                error("Invalid action $action_idx: neither move ordering is legal")
            end
        end
    else
        # loc1 not legal first, try loc2 first
        # Must verify legality to prevent state corruption (even when ENABLE_SANITY_CHECKS=false)
        if !is_move_legal(g, loc2, d2)
            error("Invalid action $action_idx: neither move ordering is legal")
        end
        p0_bak, p1_bak = g.p0, g.p1
        terminated_bak, reward_bak = g.terminated, g.reward
        apply_single_move!(g, loc2, d2)
        if !is_move_legal(g, loc1, d1)
            g.p0, g.p1 = p0_bak, p1_bak
            g.terminated, g.reward = terminated_bak, reward_bak
            error("Invalid action $action_idx: neither move ordering is legal")
        end
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
    g.phase = PHASE_CHECKER_PLAY
    g._actions_cached = false  # Invalidate legal actions cache (new dice)
end

function sample_chance!(g::BackgammonGame, rng::AbstractRNG=Random.default_rng())
    # Continuously apply random chance actions until a deterministic state is returned
    # with at least one valid move (not just PASS|PASS)
    iters = 0

    while true
        if g.terminated
            break
        end

        iters += 1
        if iters > MAX_CHANCE_ITERATIONS
            error("Infinite loop detected in sample_chance! (exceeded $MAX_CHANCE_ITERATIONS iterations)")
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

"""
    _compute_win_multiplier(opp_board::UInt128, opp_off_idx::Integer, opp_bar_idx::Integer,
                            home_range::UnitRange{Int}) -> Float32

Internal helper to compute the win multiplier (1=single, 2=gammon, 3=backgammon).

Checks opponent's board state to determine scoring:
- Single (1.0): Opponent has at least one checker borne off
- Gammon (2.0): Opponent has no checkers borne off
- Backgammon (3.0): Gammon + opponent has checker on bar or in winner's home board
"""
@inline function _compute_win_multiplier(opp_board::UInt128, opp_off_idx::Integer,
                                         opp_bar_idx::Integer, home_range::UnitRange{Int})
    # Check if opponent has any pieces off (if so, only single win)
    if get_count(opp_board, opp_off_idx) > 0
        return 1.0f0
    end

    # Gammon - check for backgammon conditions
    if get_count(opp_board, opp_bar_idx) > 0
        return 3.0f0  # Backgammon (opponent on bar)
    end

    # Check if opponent has checkers in winner's home board
    for i in home_range
        if get_count(opp_board, i) > 0
            return 3.0f0  # Backgammon (opponent in winner's home)
        end
    end

    return 2.0f0  # Gammon
end

"""
    compute_game_reward(g::BackgammonGame, winner::Int8, base_multiplier::Float32) -> Float32

Compute final reward from P0's perspective, accounting for cube value and match rules.

- `winner`: 0 (P0 wins) or 1 (P1 wins)
- `base_multiplier`: 1.0 (single), 2.0 (gammon), 3.0 (backgammon)

**Jacoby rule**: In money play with `jacoby_enabled`, gammons/backgammons are
reduced to singles if the cube has not been turned (`cube_value == 1`).
"""
function compute_game_reward(g::BackgammonGame, winner::Int8, base_multiplier::Float32)::Float32
    multiplier = base_multiplier

    # Jacoby rule: gammons/backgammons don't count unless cube was turned
    if g.jacoby_enabled && g.my_away == Int8(0)  # Money play with Jacoby
        if g.cube_value == Int16(1) && multiplier > 1.0f0
            multiplier = 1.0f0
        end
    end

    # Final points = base_multiplier * cube_value
    points = multiplier * Float32(g.cube_value)

    return winner == Int8(0) ? points : -points
end

"""
    init_match_game!(g::BackgammonGame; my_score::Int=0, opp_score::Int=0,
                     match_length::Int=0, is_crawford::Bool=false)

Initialize a game within a match context. Sets away scores, Crawford flags,
enables cube, and resets cube state.

- `my_score`, `opp_score`: Current match scores
- `match_length`: Points needed to win the match
- `is_crawford`: Whether this is the Crawford game (no doubling allowed)

Post-Crawford is inferred: if a player is 1-away and `is_crawford=false`,
it's post-Crawford.
"""
function init_match_game!(g::BackgammonGame;
                          my_score::Int=0, opp_score::Int=0,
                          match_length::Int=0, is_crawford::Bool=false)
    g.my_away = Int8(match_length - my_score)
    g.opp_away = Int8(match_length - opp_score)
    g.is_crawford = is_crawford
    g.is_post_crawford = !is_crawford && (g.my_away == 1 || g.opp_away == 1)
    g.jacoby_enabled = false  # Jacoby off in match play
    g.cube_value = Int16(1)
    g.cube_owner = Int8(0)
    g.cube_enabled = !is_crawford  # No doubling in Crawford game
end

"""
    apply_single_move!(g::BackgammonGame, loc::Integer, die::Integer)

Internal function to apply a single checker move and check for game termination.

Delegates the actual move logic to `apply_move_internal` (from actions.jl), then
checks if the current player has borne off all 15 checkers. If so, sets
`g.terminated = true` and computes the reward (1 for single, 2 for gammon,
3 for backgammon).

# Arguments
- `loc`: Source location (0=bar, 1-24=points, 25=pass)
- `die`: Die value used for this move (1-6)
"""
function apply_single_move!(g::BackgammonGame, loc::Integer, die::Integer)
    if loc == PASS_LOC; return; end

    # Delegate move logic to apply_move_internal (shared with legal action generation)
    # Note: apply_move_internal already calls sanity_check_bitboard
    g.p0, g.p1 = apply_move_internal(g.p0, g.p1, g.current_player, loc, die)

    # Check for game termination (15 pieces borne off)
    cp = g.current_player
    if cp == 0
        if get_count(g.p0, IDX_P0_OFF) == MAX_CHECKERS
            g.terminated = true
            base_mult = _compute_win_multiplier(g.p1, IDX_P1_OFF, IDX_P1_BAR, 19:24)
            g.reward = compute_game_reward(g, Int8(0), base_mult)
        end
    else
        if get_count(g.p1, IDX_P1_OFF) == MAX_CHECKERS
            g.terminated = true
            base_mult = _compute_win_multiplier(g.p0, IDX_P0_OFF, IDX_P0_BAR, 1:6)
            g.reward = compute_game_reward(g, Int8(1), base_mult)
        end
    end
end

"""
    is_move_legal_bits(p0::UInt128, p1::UInt128, cp::Integer, loc::Integer, die::Integer) -> Bool

Pure function to check if a single move is legal on the given bitboard state.

Validates:
1. Source has checkers belonging to current player
2. Bar priority (must move from bar first if checkers are there)
3. Target is not blocked (opponent has < 2 checkers)
4. Bearing off rules (all checkers in home board, over-bear from highest point only)

# Arguments
- `p0`, `p1`: Bitboard states for player 0 and player 1
- `cp`: Current player (0 or 1)
- `loc`: Source location in canonical coordinates (0=bar, 1-24=points, 25=pass)
- `die`: Die value (1-6)

# Returns
`true` if the move is legal, `false` otherwise. Always returns `true` for pass (loc=25).
"""
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
