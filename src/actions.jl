# Actions.jl - Optimized for Bitboards

# Hint size for source location vectors in allocating functions.
# Typically only a few locations are valid sources per die (checkers spread across 2-8 points).
const SOURCES_HINT_SIZE = 8

"""
    encode_action(loc1::Integer, loc2::Integer) -> Int

Encode two source locations into a single action index (1-676).

# Location Values
- 0: Bar (BAR_LOC)
- 1-24: Board points
- 25: Pass (PASS_LOC)

# Formula
`action = loc1 * 26 + loc2 + 1`

See also: `decode_action`, `action_string`
"""
function encode_action(loc1::Integer, loc2::Integer)
    return Int((loc1 * 26) + loc2 + 1)
end

"""
    decode_action(action_idx::Integer) -> (Int, Int)

Decode an action index (1-676) into two source locations.

Returns `(loc1, loc2)` where each location is 0-25:
- 0: Bar (BAR_LOC)
- 1-24: Board points
- 25: Pass (PASS_LOC)

See also: `encode_action`, `action_string`
"""
function decode_action(action_idx::Integer)
    idx0::Int = Int(action_idx - 1)
    loc1::Int = div(idx0, 26)
    loc2::Int = idx0 % 26
    return loc1, loc2
end

"""
    action_string(action_idx::Integer) -> String

Convert an action index to a human-readable string.

# Examples
```julia
action_string(encode_action(1, 2))      # "1 | 2"
action_string(encode_action(BAR_LOC, 5)) # "Bar | 5"
action_string(encode_action(PASS_LOC, PASS_LOC)) # "Pass | Pass"
```

See also: `encode_action`, `decode_action`
"""
function action_string(action_idx::Integer)
    l1, l2 = decode_action(action_idx)
    l_str(l) = l == PASS_LOC ? "Pass" : (l == BAR_LOC ? "Bar" : string(l))
    return "$(l_str(l1)) | $(l_str(l2))"
end

"""
    apply_move_internal(p0::UInt128, p1::UInt128, cp::Integer, loc::Integer, die::Integer) -> (UInt128, UInt128)

Pure function to apply a single move on bitboards, returning the new state.

Used by both legal action generation (to simulate moves) and `apply_single_move!`
in game.jl (to actually execute moves). This ensures move logic is consistent
between validation and execution.

# Arguments
- `p0`, `p1`: Current bitboard states for player 0 and player 1
- `cp`: Current player (0 or 1)
- `loc`: Source location in canonical coordinates (0=bar, 1-24=points, 25=pass)
- `die`: Die value (1-6)

# Returns
A tuple `(new_p0, new_p1)` with the updated bitboard states after the move.
Handles hitting (sending opponent checker to bar) automatically.
"""
@inline function apply_move_internal(p0::UInt128, p1::UInt128, cp::Integer, loc::Integer, die::Integer)
    if loc == PASS_LOC; return p0, p1; end

    src_idx::Int = 0
    tgt_idx::Int = 0
    to_off::Bool = false

    # Logic mirrors game.jl but returns new state instead of mutating
    # Determine indices
    if cp == 0
        src_idx = (loc == BAR_LOC) ? IDX_P0_BAR : loc
        if loc == BAR_LOC
            tgt_idx = Int(die)
        else
            tgt_idx = src_idx + Int(die)
            if tgt_idx > 24; tgt_idx = IDX_P0_OFF; to_off = true; end
        end
        
        # Apply P0 Move
        p0 = decr_count(p0, src_idx)
        if to_off
            p0 = incr_count(p0, IDX_P0_OFF)
        else
            # Hit?
            if get_count(p1, tgt_idx) == 1
                p1 = decr_count(p1, tgt_idx)
                p1 = incr_count(p1, IDX_P1_BAR)
            end
            p0 = incr_count(p0, tgt_idx)
        end
    else
        src_idx = (loc == BAR_LOC) ? IDX_P1_BAR : (25 - loc)
        if loc == BAR_LOC
            tgt_idx = 25 - Int(die)
        else
            tgt_idx = src_idx - Int(die)
            if tgt_idx < 1; tgt_idx = IDX_P1_OFF; to_off = true; end
        end
        
        # Apply P1 Move
        p1 = decr_count(p1, src_idx)
        if to_off
            p1 = incr_count(p1, IDX_P1_OFF)
        else
            # Hit?
            if get_count(p0, tgt_idx) == 1
                p0 = decr_count(p0, tgt_idx)
                p0 = incr_count(p0, IDX_P0_BAR)
            end
            p1 = incr_count(p1, tgt_idx)
        end
    end

    # NOTE: Controlled by ENABLE_SANITY_CHECKS in game.jl (set to false for large-scale training)
    sanity_check_bitboard(p0, p1)

    return p0, p1
end

"""
    get_legal_source_locs!(locs::Vector{Int}, p0::UInt128, p1::UInt128, cp::Integer, die::Integer)

In-place version that fills `locs` with legal source locations.
Clears `locs` before filling. Returns `locs` for convenience.
"""
function get_legal_source_locs!(locs::Vector{Int}, p0::UInt128, p1::UInt128, cp::Integer, die::Integer)
    empty!(locs)
    p_my = cp == 0 ? p0 : p1
    bar_idx = cp == 0 ? IDX_P0_BAR : IDX_P1_BAR

    # Bar Check
    if get_count(p_my, bar_idx) > 0
        if is_move_legal_bits(p0, p1, cp, BAR_LOC, die)
            push!(locs, BAR_LOC)
        end
        return locs
    end

    # Points Check (Canon 1..24)
    for loc in 1:24
        # Optimization: only check if checker exists
        canon = loc
        src_idx = cp == 0 ? canon : 25 - canon
        if get_count(p_my, src_idx) > 0
            if is_move_legal_bits(p0, p1, cp, loc, die)
                push!(locs, loc)
            end
        end
    end

    return locs
end

# Allocating version for backwards compatibility and use in is_action_valid
function get_legal_source_locs(p0::UInt128, p1::UInt128, cp::Integer, die::Integer)
    locs = Int[]
    sizehint!(locs, SOURCES_HINT_SIZE)
    return get_legal_source_locs!(locs, p0, p1, cp, die)
end

const CHANCE_ACTIONS = collect(1:21)  # Pre-allocated chance node actions
const PASS_PASS_ACTION = [encode_action(PASS_LOC, PASS_LOC)]  # Pre-allocated for no-move case

"""
    legal_actions(g::BackgammonGame) -> Vector{Int}

Returns valid action indices for the current state.

At chance nodes: Returns outcome indices 1-21. Use `chance_outcomes(g)` to get probabilities
(non-doubles have probability 0 in `doubles_only` mode).
At player nodes: Returns action indices (1-676) encoding two source locations.

Action encoding: `action = loc1*26 + loc2 + 1` where locations are 0-25
(0=bar, 1-24=points, 25=pass). Use `decode_action(action)` to get `(loc1, loc2)`.

Note: Returns a reference to an internal buffer. Do not mutate the returned vector.
The result is cached until the game state changes; subsequent calls return the cached buffer.
"""
function legal_actions(g::BackgammonGame)
    if is_chance_node(g)
        return CHANCE_ACTIONS  # Always return all 21 indices; use chance_outcomes() for probabilities
    end

    # Return cached result if available
    if g._actions_cached
        return g._actions_buffer
    end

    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    cp = g.current_player
    p0, p1 = g.p0, g.p1

    # Reuse pre-allocated buffer to avoid GC pressure
    actions = g._actions_buffer
    empty!(actions)

    # Track max usage during generation to avoid second pass
    max_usage = 0

    # Get initial sources into buffer1, copy to stack-allocated MVector
    get_legal_source_locs!(g._sources_buffer1, p0, p1, cp, d1)
    n1 = length(g._sources_buffer1)
    sources1 = MVector{25, Int}(undef)
    @inbounds for i in 1:n1
        sources1[i] = g._sources_buffer1[i]
    end

    # Use buffer2 for all nested source lookups (avoids ~50 allocations)
    sub_buf = g._sources_buffer2

    if d1 == d2
        # Doubles
        @inbounds for i in 1:n1
            s1 = sources1[i]
            p0_next, p1_next = apply_move_internal(p0, p1, cp, s1, d1)
            get_legal_source_locs!(sub_buf, p0_next, p1_next, cp, d1)

            if !isempty(sub_buf)
                for s2 in sub_buf
                    push!(actions, encode_action(s1, s2))
                end
                max_usage = 2
            else
                push!(actions, encode_action(s1, PASS_LOC))
                if max_usage < 1; max_usage = 1; end
            end
        end
    else
        # Non-doubles - get d2 sources into stack-allocated MVector
        get_legal_source_locs!(g._sources_buffer1, p0, p1, cp, d2)
        n2 = length(g._sources_buffer1)
        sources2 = MVector{25, Int}(undef)
        @inbounds for i in 1:n2
            sources2[i] = g._sources_buffer1[i]
        end

        # Path A: D1 then D2
        @inbounds for i in 1:n1
            s1 = sources1[i]
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s1, d1)
            get_legal_source_locs!(sub_buf, p0_n, p1_n, cp, d2)
            if !isempty(sub_buf)
                for s2 in sub_buf
                    push!(actions, encode_action(s1, s2))
                end
                max_usage = 2
            else
                push!(actions, encode_action(s1, PASS_LOC))
                if max_usage < 1; max_usage = 1; end
            end
        end

        # Path B: D2 then D1
        @inbounds for i in 1:n2
            s2 = sources2[i]
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s2, d2)
            get_legal_source_locs!(sub_buf, p0_n, p1_n, cp, d1)
            if !isempty(sub_buf)
                for s1 in sub_buf
                    push!(actions, encode_action(s1, s2))
                end
                max_usage = 2
            else
                push!(actions, encode_action(PASS_LOC, s2))
                if max_usage < 1; max_usage = 1; end
            end
        end
    end

    unique!(actions)

    if isempty(actions)
        # Add PASS|PASS to the buffer so caching works correctly
        push!(actions, PASS_PASS_ACTION[1])
        g._actions_cached = true
        return actions
    end

    # Filter by max usage - avoid allocating intermediate array
    # NOTE: This implements the "maximize dice" and "higher die" rules.
    # These same rules are also implemented in is_action_valid().
    # If you modify these rules, update is_action_valid() as well.
    if max_usage == 2
        # Fast path: filter in-place, no decode needed for 2-usage check
        filter!(actions) do act
            l1, l2 = decode_action(act)
            l1 != PASS_LOC && l2 != PASS_LOC
        end
    elseif max_usage == 1
        # Keep only 1-usage actions
        filter!(actions) do act
            l1, l2 = decode_action(act)
            (l1 != PASS_LOC) ⊻ (l2 != PASS_LOC)  # XOR: exactly one is not PASS
        end

        # For non-doubles, must use higher die if both options exist
        if d1 != d2
            has_d1 = false
            has_d2 = false
            @inbounds for act in actions
                l1, l2 = decode_action(act)
                if l1 != PASS_LOC; has_d1 = true; end
                if l2 != PASS_LOC; has_d2 = true; end
                has_d1 && has_d2 && break
            end

            if has_d1 && has_d2
                if d1 > d2
                    filter!(a -> decode_action(a)[1] != PASS_LOC, actions)
                else
                    filter!(a -> decode_action(a)[2] != PASS_LOC, actions)
                end
            end
        end
    end
    # max_usage == 0: keep all (all are PASS|PASS)

    g._actions_cached = true
    return actions
end

"""
    is_action_valid(g::BackgammonGame, action_idx::Integer) -> Bool

Validates that a specific action is legal for the current game state.

Returns `false` if the game is at a chance node (dice not yet rolled).

Checks:
1. Individual moves are legal (source has pieces, target not blocked, etc.)
2. At least one ordering of the moves works
3. Maximize dice rule is respected (uses both dice if possible, higher die if only one)

Note: Complexity is O(num_legal_sources) in the common case, up to O(num_legal_sources²)
when validating the maximize-dice rule. Still faster than generating all legal actions
for membership testing.
"""
function is_action_valid(g::BackgammonGame, action_idx::Integer)
    # Guard against chance nodes - no deterministic actions are valid
    if is_chance_node(g)
        return false
    end

    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    cp = g.current_player
    p0, p1 = g.p0, g.p1

    loc1, loc2 = decode_action(action_idx)

    # Both PASS - only valid if no moves are possible
    if loc1 == PASS_LOC && loc2 == PASS_LOC
        sources1 = get_legal_source_locs(p0, p1, cp, d1)
        sources2 = d1 == d2 ? sources1 : get_legal_source_locs(p0, p1, cp, d2)
        return isempty(sources1) && isempty(sources2)
    end

    # Check if move sequence is executable
    can_use_both = false
    can_use_d1_only = false
    can_use_d2_only = false

    if d1 == d2
        # Doubles: loc1 and loc2 both use the same die value
        # For doubles, only accept the canonical ordering (loc1 then loc2)
        # to match legal_actions() which generates (first_move, second_move)
        if loc1 != PASS_LOC && loc2 != PASS_LOC
            # Only try loc1 then loc2 (canonical form)
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                p0_n, p1_n = apply_move_internal(p0, p1, cp, loc1, d1)
                if is_move_legal_bits(p0_n, p1_n, cp, loc2, d1)
                    can_use_both = true
                end
            end
        elseif loc1 != PASS_LOC
            # loc2 is PASS - this is the only valid single-die format for doubles
            # (legal_actions generates loc1|PASS, not PASS|loc2 for doubles)
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                can_use_d1_only = true
            end
        end
        # Note: PASS|loc2 for doubles is NOT valid - legal_actions never generates it
    else
        # Non-doubles: loc1 uses d1, loc2 uses d2
        if loc1 != PASS_LOC && loc2 != PASS_LOC
            # Try loc1(d1) then loc2(d2)
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                p0_n, p1_n = apply_move_internal(p0, p1, cp, loc1, d1)
                if is_move_legal_bits(p0_n, p1_n, cp, loc2, d2)
                    can_use_both = true
                end
            end
            # Try loc2(d2) then loc1(d1)
            if !can_use_both && is_move_legal_bits(p0, p1, cp, loc2, d2)
                p0_n, p1_n = apply_move_internal(p0, p1, cp, loc2, d2)
                if is_move_legal_bits(p0_n, p1_n, cp, loc1, d1)
                    can_use_both = true
                end
            end
        elseif loc1 != PASS_LOC
            # Using d1 only (loc2 is PASS)
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                can_use_d1_only = true
            end
        elseif loc2 != PASS_LOC
            # Using d2 only (loc1 is PASS)
            if is_move_legal_bits(p0, p1, cp, loc2, d2)
                can_use_d2_only = true
            end
        end
    end

    # Validate against maximize dice rule
    # NOTE: This implements the "maximize dice" and "higher die" rules.
    # These same rules are also implemented in legal_actions() filtering.
    # If you modify these rules, update legal_actions() as well.
    if can_use_both
        return true
    end

    if can_use_d1_only || can_use_d2_only
        # Check if using both dice was possible
        sources1 = get_legal_source_locs(p0, p1, cp, d1)
        for s1 in sources1
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s1, d1)
            d2_val = d1 == d2 ? d1 : d2
            sub = get_legal_source_locs(p0_n, p1_n, cp, d2_val)
            if !isempty(sub)
                return false  # Could use both dice, but action only uses one
            end
        end
        if d1 != d2
            sources2 = get_legal_source_locs(p0, p1, cp, d2)
            for s2 in sources2
                p0_n, p1_n = apply_move_internal(p0, p1, cp, s2, d2)
                sub = get_legal_source_locs(p0_n, p1_n, cp, d1)
                if !isempty(sub)
                    return false  # Could use both dice, but action only uses one
                end
            end
        end

        # Only one die can be used - check higher die rule for non-doubles
        if d1 != d2
            other_die_sources = can_use_d1_only ? get_legal_source_locs(p0, p1, cp, d2) : sources1
            if !isempty(other_die_sources)
                # Both single-die options exist, must use higher
                higher_is_d1 = d1 > d2
                if (can_use_d1_only && !higher_is_d1) || (can_use_d2_only && higher_is_d1)
                    return false  # Using lower die when higher is available
                end
            end
        end

        return true
    end

    return false
end
