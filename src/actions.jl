function encode_action(loc1::Integer, loc2::Integer)
    # locs are 0..25 (0=Pass, 1=Bar, 2..25=Points)
    # Action = loc1 * 26 + loc2 + 1 (1-based)
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
    
    function l_str(l)
        if l == 0; return "Pass"; end
        if l == 1; return "Bar"; end
        return string(l - 1) # 2->1
    end
    
    return "$(l_str(l1)) | $(l_str(l2))"
end

# Helper to get all legal source LOCATIONS for a specific die
function get_legal_source_locs(board::SVector{28, Int8}, die::Integer)
    locs = Int[]
    
    # Check Bar (Loc 1)
    if board[BAR_IDX] > 0
        if is_move_legal(board, BAR_LOC, die)
            push!(locs, BAR_LOC)
        end
        # If Bar has checkers, ONLY Bar moves are allowed?
        # Yes, must enter.
        return locs
    end
    
    # Check Points (Loc 2..25 -> indices 1..24)
    for i in 1:24
        if board[i] > 0
            loc = i + 1
            if is_move_legal(board, loc, die)
                push!(locs, loc)
            end
        end
    end
    
    return locs
end

function apply_move_svector(board::SVector{28, Int8}, loc::Integer, die::Integer)
    # Need immutable update
    b = MVector{28, Int8}(board)
    apply_single_move!(b, loc, die)
    return SVector{28, Int8}(b)
end

function get_legal_actions(g::BackgammonGame)
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    
    # Non-doubles (d1 < d2 usually, but generic logic)
    # Doubles (d1 == d2)
    
    # Valid sources for D1
    sources1 = get_legal_source_locs(g.board, d1)
    # Valid sources for D2
    sources2 = get_legal_source_locs(g.board, d2)
    
    actions = Int[]
    
    if d1 != d2
        # Non-doubles: Need pair (L1, L2)
        # L1 uses D1, L2 uses D2.
        
        # Try all L1 in sources1
        for s1 in sources1
            # Apply s1 with d1
            b_next = apply_move_svector(g.board, s1, d1)
            # Find valid s2 for d2 in b_next
            sub_sources2 = get_legal_source_locs(b_next, d2)
            
            if !isempty(sub_sources2)
                for s2 in sub_sources2
                    push!(actions, encode_action(s1, s2))
                end
            else
                # Can play s1(d1), but then no move for d2.
                # Is this legal?
                # Only if (s1, Pass) is max dice usage.
                push!(actions, encode_action(s1, PASS_LOC))
            end
        end
        
        # Try all L2 in sources2 (Start with D2)
        for s2 in sources2
            b_next = apply_move_svector(g.board, s2, d2)
            sub_sources1 = get_legal_source_locs(b_next, d1)
            
            if !isempty(sub_sources1)
                for s1 in sub_sources1
                    # Pair (s1, s2) is valid via reverse execution
                    push!(actions, encode_action(s1, s2))
                end
            else
                # Can play s2(d2), but then no move for d1.
                # Action: (Pass, s2). L1=Pass (d1 unused), L2=s2 (d2 used).
                push!(actions, encode_action(PASS_LOC, s2))
            end
        end
        
    else
        # Doubles: d1 == d2.
        # Actions are (s1, s2).
        # Both use same die value.
        
        sources = get_legal_source_locs(g.board, d1)
        
        for s1 in sources
            b_next = apply_move_svector(g.board, s1, d1)
            sub_sources = get_legal_source_locs(b_next, d1)
            
            if !isempty(sub_sources)
                for s2 in sub_sources
                    push!(actions, encode_action(s1, s2))
                end
            else
                # Only 1 move possible
                push!(actions, encode_action(s1, PASS_LOC))
            end
        end
    end
    
    unique!(actions)
    
    if isempty(actions)
        return [encode_action(PASS_LOC, PASS_LOC)]
    end
    
    # Filter for Max Dice Usage / Max Value
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
        # Check if we have any actions using d2 (L2 != 0)
        has_d2 = any(a -> decode_action(a)[2] != 0, actions)
        if has_d2
            # Must use d2
            filter!(a -> decode_action(a)[2] != 0, actions)
        end
    end
    
    return actions
end