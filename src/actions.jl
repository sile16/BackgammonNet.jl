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
    l_str(l) = l == 0 ? "Pass" : (l == 1 ? "Bar" : string(l - 1))
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
        src_idx = (loc == BAR_LOC) ? IDX_P0_BAR : (loc - 1)
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
        src_idx = (loc == BAR_LOC) ? IDX_P1_BAR : (25 - (loc - 1))
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

# Helper to check legality without a full game object
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
        # Block check
        if get_count(p_opp, tgt_idx) >= 2; return false; end
        return true
    else
        # Bearing Off
        if cp == 0
            if get_count(p_my, IDX_P0_BAR) > 0; return false; end
            # 1-18 empty?
            for i in 1:18; if get_count(p_my, i) > 0; return false; end; end
            
            if tgt_idx == 25; return true; end # Exact
            # Over-bear:
            # P0 moves 1->24. Home 19-24.
            # "Higher points" are those further from Off. i.e. < src_idx.
            # Check 19 .. src_idx-1.
            # (1..18 already checked).
            for i in 19:(src_idx-1); if get_count(p_my, i) > 0; return false; end; end
            return true
        else
            if get_count(p_my, IDX_P1_BAR) > 0; return false; end
            # 7-24 empty? (Home is 1-6)
            for i in 7:24; if get_count(p_my, i) > 0; return false; end; end
            
            if tgt_idx == 0; return true; end # Exact
            # Over-bear:
            # P1 moves 24->1. Home 1-6. Off 0.
            # "Higher points" are > src_idx.
            for i in (src_idx+1):6; if get_count(p_my, i) > 0; return false; end; end
            return true
        end
    end
end

function get_legal_source_locs(p0::UInt128, p1::UInt128, cp::Integer, die::Integer)
    locs = Int[]
    p_my = cp == 0 ? p0 : p1
    bar_idx = cp == 0 ? IDX_P0_BAR : IDX_P1_BAR
    
    # Bar Check
    if get_count(p_my, bar_idx) > 0
        if is_move_legal_bits(p0, p1, cp, BAR_LOC, die)
            push!(locs, BAR_LOC)
        end
        return locs
    end
    
    # Points Check (Canon 1..24 -> Loc 2..25)
    for loc in 2:25
        # Optimization: only check if checker exists
        canon = loc - 1
        src_idx = cp == 0 ? canon : 25 - canon
        if get_count(p_my, src_idx) > 0
            if is_move_legal_bits(p0, p1, cp, loc, die)
                push!(locs, loc)
            end
        end
    end
    
    return locs
end

function get_legal_actions(g::BackgammonGame)
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    cp = g.current_player
    p0, p1 = g.p0, g.p1
    
    actions = Int[]
    
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
            else
                push!(actions, encode_action(s1, PASS_LOC))
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
            else
                push!(actions, encode_action(s1, PASS_LOC))
            end
        end
        
        # Path B: D2 then D1
        for s2 in sources2
            p0_n, p1_n = apply_move_internal(p0, p1, cp, s2, d2)
            sub1 = get_legal_source_locs(p0_n, p1_n, cp, d1)
            if !isempty(sub1)
                for s1 in sub1
                    # Encoding always (L1, L2) -> L1 uses D1, L2 uses D2
                    push!(actions, encode_action(s1, s2))
                end
            else
                push!(actions, encode_action(PASS_LOC, s2))
            end
        end
    end
    
    unique!(actions)
    
    if isempty(actions)
        return [encode_action(PASS_LOC, PASS_LOC)]
    end
    
    # Filter for Max Dice Usage & Higher Value
    function usage(act)
        l1, l2 = decode_action(act)
        u = 0
        if l1 != 0; u += 1; end
        if l2 != 0; u += 1; end
        return u
    end
    
    max_u = maximum(usage.(actions))
    filter!(a -> usage(a) == max_u, actions)
    
    if max_u == 1 && d1 != d2
        can_use_d1 = any(a -> decode_action(a)[1] != 0, actions)
        can_use_d2 = any(a -> decode_action(a)[2] != 0, actions)
        
        if can_use_d1 && can_use_d2
            if d1 > d2
                filter!(a -> decode_action(a)[1] != 0, actions)
            elseif d2 > d1
                filter!(a -> decode_action(a)[2] != 0, actions)
            end
        end
    end
    
    return actions
end

const legal_actions = get_legal_actions
