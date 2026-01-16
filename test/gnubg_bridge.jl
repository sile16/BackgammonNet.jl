# gnubg_bridge.jl - Bridge between BackgammonNet.jl and gnubg
#
# This module provides functions to:
# 1. Convert Julia board states to gnubg "simple" format
# 2. Run gnubg and parse hint output to get all legal moves
# 3. Simulate moves to compute final board states
# 4. Compare Julia's legal moves with gnubg's

using BackgammonNet

# File-based batch approach for fast gnubg queries

"""
    julia_to_gnubg_simple(g::BackgammonGame; perspective::Int=-1) -> Vector{Int}

Convert Julia BackgammonGame to gnubg "simple" format.
Returns 26 integers in gnubg's format:
- Position 0 (index 1 in output): current player's bar
- Positions 1-24 (indices 2-25 in output): points 1-24
- Position 25 (index 26 in output): opponent's bar

Positive = perspective player's checkers, negative = opponent's.

If perspective is -1 (default), use g.current_player.
Otherwise use the specified player (0 or 1) for consistent comparison.

gnubg coordinate system (from current player's perspective):
- Points 1-6: Current player's home board
- Points 19-24: Opponent's home board
- Current player moves from high points to low points (24→1→off)
"""
function julia_to_gnubg_simple(g::BackgammonGame; perspective::Int=-1)
    simple = zeros(Int, 26)
    cp = perspective == -1 ? g.current_player : perspective

    # gnubg simple format (discovered through testing):
    # - simple[1] (1st number) = current player's bar
    # - simple[2..25] (numbers 2-25) = points 1-24
    # - simple[26] (26th number) = opponent's bar
    # Note: Julia arrays are 1-indexed, so we use indices 1-26

    # In Julia:
    # - P0 moves from low indices to high (1→24→off at IDX_P0_OFF=25)
    # - P1 moves from high indices to low (24→1→off at IDX_P1_OFF=0)

    # gnubg: Current player always moves 24→1→off

    p0, p1 = g.p0, g.p1

    if cp == 0
        # Perspective of P0: P0's checkers are positive, P1's are negative
        # gnubg point i maps to Julia index (25-i) because P0 moves opposite to gnubg
        for gnubg_pt in 1:24
            julia_idx = 25 - gnubg_pt
            p0_count = Int((p0 >> (julia_idx << 2)) & 0xF)
            p1_count = Int((p1 >> (julia_idx << 2)) & 0xF)
            simple[gnubg_pt + 1] = p0_count > 0 ? p0_count : -p1_count  # +1 for bar offset
        end
        # Bars: simple[1] = my bar, simple[26] = opponent bar
        simple[1] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
        simple[26] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    else
        # Perspective of P1: P1's checkers are positive, P0's are negative
        # gnubg point i maps to Julia index i directly (P1 moves same as gnubg)
        for gnubg_pt in 1:24
            julia_idx = gnubg_pt
            p1_count = Int((p1 >> (julia_idx << 2)) & 0xF)
            p0_count = Int((p0 >> (julia_idx << 2)) & 0xF)
            simple[gnubg_pt + 1] = p1_count > 0 ? p1_count : -p0_count  # +1 for bar offset
        end
        # Bars: simple[1] = my bar, simple[26] = opponent bar
        simple[1] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
        simple[26] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    end

    return simple
end

"""
    gnubg_simple_to_julia(simple::Vector{Int}, current_player::Int) -> Tuple{UInt128, UInt128}

Convert gnubg simple format back to Julia (p0, p1) bitboards.
"""
function gnubg_simple_to_julia(simple::Vector{Int}, current_player::Int)
    p0 = UInt128(0)
    p1 = UInt128(0)

    # Reverse the mapping from julia_to_gnubg_simple
    if current_player == 0
        # gnubg point i = Julia point (25-i) for P0
        for gnubg_i in 1:24
            julia_i = 25 - gnubg_i
            v = simple[gnubg_i]
            if v > 0
                # Current player (P0) checkers at julia_i
                p0 |= UInt128(v) << (julia_i << 2)
            elseif v < 0
                # Opponent (P1) checkers at julia_i
                p1 |= UInt128(-v) << (julia_i << 2)
            end
        end
        # Bars
        if simple[25] > 0  # Current player (P0) bar
            p0 |= UInt128(simple[25]) << (BackgammonNet.IDX_P0_BAR << 2)
        end
        if simple[26] > 0  # Opponent (P1) bar
            p1 |= UInt128(simple[26]) << (BackgammonNet.IDX_P1_BAR << 2)
        end
    else
        # gnubg point i = Julia point i for P1
        for gnubg_i in 1:24
            julia_i = gnubg_i
            v = simple[gnubg_i]
            if v > 0
                # Current player (P1) checkers
                p1 |= UInt128(v) << (julia_i << 2)
            elseif v < 0
                # Opponent (P0) checkers
                p0 |= UInt128(-v) << (julia_i << 2)
            end
        end
        # Bars
        if simple[25] > 0  # Current player (P1) bar
            p1 |= UInt128(simple[25]) << (BackgammonNet.IDX_P1_BAR << 2)
        end
        if simple[26] > 0  # Opponent (P0) bar
            p0 |= UInt128(simple[26]) << (BackgammonNet.IDX_P0_BAR << 2)
        end
    end

    return p0, p1
end

"""
    run_gnubg_hint(simple::Vector{Int}, d1::Int, d2::Int) -> Vector{String}

Run gnubg with given board position and dice, return all legal moves as strings.

Note: gnubg doesn't allow doubles on the "first roll" of a game. To work around this,
we first make a dummy move with 3-1 dice, then set up our desired position and dice.
"""
function run_gnubg_hint(simple::Vector{Int}, d1::Int, d2::Int)
    # Build the simple board string
    board_str = join(simple, " ")

    # Build gnubg commands
    # Workaround: Start game, make a move to get past "first roll" state,
    # then set our desired board and dice.
    # IMPORTANT: After the move, gnubg prompts for dice. We must set dice
    # BEFORE set board simple, otherwise the board command is ignored.
    commands = """
new game
set automatic roll off
set dice 3 1
move 8/5 6/5
set dice $d1 $d2
set board simple $board_str
hint 999
"""

    # Run gnubg
    output = read(pipeline(`echo $commands`, `gnubg -t -q`), String)

    # Parse hint output to extract moves
    moves = String[]
    for line in split(output, '\n')
        # Match lines like "    1. Cubeful 0-ply    8/5 6/5"
        m = match(r"^\s*\d+\.\s+\w+\s+[\d-]+ply\s+(.+?)\s+Eq\.:", line)
        if m !== nothing
            push!(moves, strip(m.captures[1]))
        end
    end

    return moves
end

# Note: Batch functions (run_gnubg_hint_batch, get_gnubg_final_states_batch) are defined
# in gnubg_hybrid.jl with optimized implementations. This file provides the core single-query
# functions that gnubg_hybrid.jl builds upon.

"""
    parse_gnubg_move(move_str::String) -> Vector{Tuple{Int, Int}}

Parse a gnubg move string like "8/5 6/5" or "24/20" or "bar/22" into
a list of (from, to) tuples. Returns gnubg coordinates (1-24, 0=bar, 25=off).

Handles:
- Basic moves: "8/5", "24/20"
- Bar entry: "bar/22"
- Bearing off: "6/off"
- Multipliers: "8/5(2)" means two checkers do this move
- Hit markers: "24/23*" means landing on a blot (we ignore the *)
- Combined moves: "13/10/7" means 13→10 then 10→7
"""
function parse_gnubg_move(move_str::String)
    moves = Tuple{Int, Int}[]

    # Split by spaces to get individual checker moves
    parts = split(move_str)

    for part in parts
        part = String(part)  # Ensure it's a String, not SubString

        # Remove hit markers (*) - they don't affect the move, just indicate a hit
        part = replace(part, "*" => "")

        # Handle patterns like "8/5", "24/20", "bar/22", "6/off", "8/5(2)"
        # Remove multiplier like "(2)" at the end
        count = 1
        m_count = match(r"\((\d+)\)$", part)
        if m_count !== nothing
            count = parse(Int, m_count.captures[1])
            part = replace(part, r"\((\d+)\)$" => "")
        end

        # Parse the move itself - simple case first
        m = match(r"^(bar|\d+)/(off|\d+)$"i, part)
        if m !== nothing
            from_str, to_str = m.captures
            from_pt = lowercase(from_str) == "bar" ? 0 : parse(Int, from_str)
            to_pt = lowercase(to_str) == "off" ? 25 : parse(Int, to_str)

            # Add the move 'count' times
            for _ in 1:count
                push!(moves, (from_pt, to_pt))
            end
        else
            # Handle combined moves like "13/10/7" (two moves: 13→10, 10→7)
            # Also handles "bar/19/13(2)" meaning do the whole sequence twice
            pts = split(part, "/")
            segment_moves = Tuple{Int, Int}[]
            for i in 1:length(pts)-1
                from_str = String(pts[i])
                to_str = String(pts[i+1])

                # Remove any trailing multiplier from to_str (already handled above)
                to_str = replace(to_str, r"\(\d+\)$" => "")

                from_pt = lowercase(from_str) == "bar" ? 0 : parse(Int, from_str)
                to_pt = lowercase(to_str) == "off" ? 25 : parse(Int, to_str)
                push!(segment_moves, (from_pt, to_pt))
            end
            # Apply the count multiplier to the entire combined move sequence
            for _ in 1:count
                append!(moves, segment_moves)
            end
        end
    end

    return moves
end

"""
    apply_gnubg_move!(simple::Vector{Int}, from_pt::Int, to_pt::Int)

Apply a single checker move to gnubg simple format board.
from_pt: 0=bar, 1-24=points, to_pt: 1-24=points, 25=off

gnubg simple format indices (Julia 1-indexed):
- simple[1] = current player's bar
- simple[2..25] = points 1-24
- simple[26] = opponent's bar
"""
function apply_gnubg_move!(simple::Vector{Int}, from_pt::Int, to_pt::Int)
    # Move checker from 'from' to 'to'
    if from_pt == 0
        # From bar (simple[1])
        simple[1] -= 1
    else
        # From point (simple[from_pt + 1])
        simple[from_pt + 1] -= 1
    end

    if to_pt == 25
        # Bear off - don't track off checkers in simple format for comparison
        # (we only care about board position)
    else
        # To point (simple[to_pt + 1])
        # Check for hit
        if simple[to_pt + 1] == -1
            # Hit opponent's blot - send to their bar (simple[26])
            simple[to_pt + 1] = 0
            simple[26] += 1
        end
        simple[to_pt + 1] += 1
    end
end

"""
    get_gnubg_final_states(g::BackgammonGame) -> Set{Vector{Int}}

Get all final board states from gnubg for the given game state.
Returns a set of board states (as 26-element vectors in gnubg simple format).
"""
function get_gnubg_final_states(g::BackgammonGame)
    simple = julia_to_gnubg_simple(g)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    # Get all legal moves from gnubg
    move_strs = run_gnubg_hint(simple, d1, d2)

    final_states = Set{Vector{Int}}()

    # If no legal moves, the player passes and board stays the same
    if isempty(move_strs)
        push!(final_states, simple[1:26])
        return final_states
    end

    for move_str in move_strs
        # Copy the board
        board = copy(simple)

        # Parse and apply all individual moves
        moves = parse_gnubg_move(move_str)
        for (from_pt, to_pt) in moves
            apply_gnubg_move!(board, from_pt, to_pt)
        end

        # Add to set (only board positions 1-24, not bars since they might differ)
        # Actually, include bars for full comparison
        push!(final_states, board[1:26])
    end

    return final_states
end

"""
    copy_game(g::BackgammonGame) -> BackgammonGame

Create a copy of the game state (without history for efficiency).
"""
function copy_game(g::BackgammonGame)
    BackgammonGame(
        g.p0, g.p1, g.dice, g.remaining_actions,
        g.current_player, g.terminated, g.reward,
        Int[]
    )
end

"""
    get_julia_final_states(g::BackgammonGame) -> Set{Vector{Int}}

Get all final board states from Julia implementation.
For doubles (remaining_actions=2), enumerate all combinations of 2 actions.
For non-doubles (remaining_actions=1), enumerate all single actions.
Returns board states in gnubg simple format for comparison.

Uses the original player's perspective consistently (before turn switches).
"""
function get_julia_final_states(g::BackgammonGame)
    final_states = Set{Vector{Int}}()
    original_player = Int(g.current_player)  # Lock in the perspective

    actions1 = BackgammonNet.legal_actions(g)

    if g.remaining_actions == 2
        # Doubles: need to enumerate both actions
        for a1 in actions1
            g1 = copy_game(g)
            BackgammonNet.apply_action!(g1, a1)

            if g1.terminated
                # Game ended (rare: bearing off last checkers)
                push!(final_states, julia_to_gnubg_simple(g1; perspective=original_player)[1:26])
                continue
            end

            # Get second action
            actions2 = BackgammonNet.legal_actions(g1)
            for a2 in actions2
                g2 = copy_game(g1)
                BackgammonNet.apply_action!(g2, a2)
                push!(final_states, julia_to_gnubg_simple(g2; perspective=original_player)[1:26])
            end
        end
    else
        # Non-doubles: single action
        for a in actions1
            g1 = copy_game(g)
            BackgammonNet.apply_action!(g1, a)
            push!(final_states, julia_to_gnubg_simple(g1; perspective=original_player)[1:26])
        end
    end

    return final_states
end

"""
    compare_legal_moves(g::BackgammonGame) -> NamedTuple

Compare legal moves between Julia and gnubg for a given position.
Returns (match, julia_only, gnubg_only, julia_count, gnubg_count).
"""
function compare_legal_moves(g::BackgammonGame)
    julia_states = get_julia_final_states(g)
    gnubg_states = get_gnubg_final_states(g)

    julia_only = setdiff(julia_states, gnubg_states)
    gnubg_only = setdiff(gnubg_states, julia_states)

    return (
        match = isempty(julia_only) && isempty(gnubg_only),
        julia_only = julia_only,
        gnubg_only = gnubg_only,
        julia_count = length(julia_states),
        gnubg_count = length(gnubg_states)
    )
end

# Test helpers
function print_board_simple(simple::Vector{Int})
    println("Points 1-24:")
    for i in 1:24
        if simple[i] != 0
            println("  Point $i: $(simple[i])")
        end
    end
    println("Bars: current=$(simple[25]), opponent=$(simple[26])")
end
