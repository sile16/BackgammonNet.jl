# Actions.jl - Optimized for Bitboards

function encode_action(loc1::Integer, loc2::Integer)
    return Int((loc1 * 26) + loc2 + 1)
end

function decode_action(action_idx::Integer)
    idx0 = Int(action_idx - 1)
    loc1 = div(idx0, 26)
    loc2 = idx0 % 26
    return loc1, loc2
end

function action_string(action_idx::Integer)
    l1, l2 = decode_action(action_idx)
    l_str(l) = l == PASS_LOC ? "Pass" : (l == BAR_LOC ? "Bar" : string(l))
    return "$(l_str(l1)) | $(l_str(l2))"
end

# Pure function to apply a move on bitboards
@inline function apply_move_internal(p0::UInt128, p1::UInt128, cp::Integer, loc::Integer, die::Integer)
    if loc == PASS_LOC; return p0, p1; end
    
    src_idx = 0
    tgt_idx = 0
    to_off = false
    
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
    
    return p0, p1
end

# Note: is_move_legal_bits is now defined in game.jl

function get_legal_source_locs(p0::UInt128, p1::UInt128, cp::Integer, die::Integer)
    locs = Int[]
    sizehint!(locs, 8)  # Pre-allocate for typical 4-8 legal sources
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

const CHANCE_ACTIONS = collect(1:21)  # Pre-allocated chance node actions

function get_legal_actions(g::BackgammonGame)
    if is_chance_node(g)
        return CHANCE_ACTIONS
    end

    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    cp = g.current_player
    p0, p1 = g.p0, g.p1

    actions = Int[]
    sizehint!(actions, 30)

    # Track max usage during generation to avoid second pass
    max_usage = 0

    sources1 = get_legal_source_locs(p0, p1, cp, d1)

    if d1 == d2
        # Doubles
        for s1 in sources1
            p0_next, p1_next = apply_move_internal(p0, p1, cp, s1, d1)
            sub_sources = get_legal_source_locs(p0_next, p1_next, cp, d1)

            if !isempty(sub_sources)
                for s2 in sub_sources
                    push!(actions, encode_action(s1, s2))
                end
                max_usage = 2
            else
                push!(actions, encode_action(s1, PASS_LOC))
                if max_usage < 1; max_usage = 1; end
            end
        end
    else
        # Non-doubles
        sources2 = get_legal_source_locs(p0, p1, cp, d2)

        # Path A: D1 then D2
        for s1 in sources1
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s1, d1)
            sub2 = get_legal_source_locs(p0_n, p1_n, cp, d2)
            if !isempty(sub2)
                for s2 in sub2
                    push!(actions, encode_action(s1, s2))
                end
                max_usage = 2
            else
                push!(actions, encode_action(s1, PASS_LOC))
                if max_usage < 1; max_usage = 1; end
            end
        end

        # Path B: D2 then D1
        for s2 in sources2
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s2, d2)
            sub1 = get_legal_source_locs(p0_n, p1_n, cp, d1)
            if !isempty(sub1)
                for s1 in sub1
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
        return [encode_action(PASS_LOC, PASS_LOC)]
    end

    # Filter by max usage - avoid allocating intermediate array
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

    return actions
end

const legal_actions = get_legal_actions

"""
    is_action_valid(g::BackgammonGame, action_idx::Integer) -> Bool

Fast validation that checks if a specific action is valid without generating all legal actions.
This is O(1) vs O(n) for `action in legal_actions(g)`.

Checks:
1. Individual moves are legal (source has pieces, target not blocked, etc.)
2. At least one ordering of the moves works
3. Maximize dice rule is respected (uses both dice if possible, higher die if only one)
"""
function is_action_valid(g::BackgammonGame, action_idx::Integer)
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
        if loc1 != PASS_LOC && loc2 != PASS_LOC
            # Try loc1 then loc2
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                p0_n, p1_n = apply_move_internal(p0, p1, cp, loc1, d1)
                if is_move_legal_bits(p0_n, p1_n, cp, loc2, d1)
                    can_use_both = true
                end
            end
            # Try loc2 then loc1
            if !can_use_both && is_move_legal_bits(p0, p1, cp, loc2, d1)
                p0_n, p1_n = apply_move_internal(p0, p1, cp, loc2, d1)
                if is_move_legal_bits(p0_n, p1_n, cp, loc1, d1)
                    can_use_both = true
                end
            end
        elseif loc1 != PASS_LOC
            # loc2 is PASS
            if is_move_legal_bits(p0, p1, cp, loc1, d1)
                can_use_d1_only = true
            end
        elseif loc2 != PASS_LOC
            # loc1 is PASS
            if is_move_legal_bits(p0, p1, cp, loc2, d1)
                can_use_d2_only = true
            end
        end
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
