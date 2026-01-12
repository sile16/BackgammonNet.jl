using StaticArrays

function observe_fast(g::BackgammonGame)
    # 34 elements
    # Board (28) / 15.0
    # Playable Dice (6) / 4.0
    
    obs = zeros(Float32, 34)
    
    # Board
    @inbounds for i in 1:28
        obs[i] = Float32(g[i]) / 15.0f0
    end
    
    # Dice
    # pgx logic:
    # is_doubles = dice[0] == dice[1]
    # remaining_dice = is_doubles ? remaining_actions * 2 : 2 (actually just remaining_actions count?)
    # Wait, pgx logic:
    # remaining_actions_from_dice: doubles->2, single->1.
    # _remaining_dice_count:
    #   if doubles: return remaining_actions * 2. (So 2 actions left -> 4 dice. 1 action -> 2 dice).
    #   if single: count dice values.
    # Wait, single: "remaining_actions" is global.
    # If 1 remaining action (single), we check which dice are still valid?
    # Actually, in single mode, we assume dice are consumed by action.
    # But `remaining_actions` is 1.
    # For single, we have 2 dice. We play 1 action (using both).
    # So if remaining=1, we have both dice?
    # pgx `_remaining_dice_count`:
    #   "(dice == die_val).sum()" for non-doubles.
    #   So it just counts the dice in `state._dice`.
    #   Since `state._dice` doesn't change in pgx (constant for turn), 
    #   and `remaining_actions` is 1 (start) or 0 (end).
    #   So if remaining=1, show all dice.
    
    d1 = g.dice[1]
    d2 = g.dice[2]
    is_doubles = (d1 == d2)
    
    dice_counts = zeros(Int, 6)
    
    if g.remaining_actions > 0
        if is_doubles
            # remaining_actions: 2 -> 4 dice. 1 -> 2 dice.
            count = Int(g.remaining_actions) * 2
            dice_counts[d1] = count
        else
            # Non-doubles. remaining_actions should be 1.
            # Show both dice.
            dice_counts[d1] += 1
            dice_counts[d2] += 1
        end
    end
    
    @inbounds for i in 1:6
        obs[28 + i] = Float32(dice_counts[i]) / 4.0f0
    end
    
    return obs
end

function observe_full(g::BackgammonGame)
    # 86 elements
    obs = zeros(Float32, 86)
    
    # 1. Fast part
    fast = observe_fast(g)
    obs[1:34] = fast
    
    # 2. Heuristics (Same as before)
    min_my = 25
    max_opp = 0
    my_pip = 0
    opp_pip = 0
    
    @inbounds for i in 1:24
        val = g[i]
        if val > 0
            if i < min_my; min_my = i; end
            my_pip += val * (25 - i)
        elseif val < 0
            if i > max_opp; max_opp = i; end
            opp_pip += (-val) * i
        end
    end
    my_pip += g[BAR_IDX] * 25
    opp_pip += (-g[OPP_BAR_IDX]) * 25
    
    is_race = (g[BAR_IDX] == 0 && g[OPP_BAR_IDX] == 0 && min_my > max_opp)
    obs[35] = Float32(is_race)
    
    can_bear_my = (g[BAR_IDX] == 0)
    if can_bear_my
        for i in 1:18
            if g[i] > 0
                can_bear_my = false; break;
            end
        end
    end
    obs[36] = Float32(can_bear_my)
    
    can_bear_opp = (g[OPP_BAR_IDX] == 0)
    if can_bear_opp
        for i in 7:24
            if g[i] < 0
                can_bear_opp = false; break;
            end
        end
    end
    obs[37] = Float32(can_bear_opp)
    
    diff = (my_pip - opp_pip) / 375.0f0
    obs[38] = diff
    
    idx_blot = 39
    idx_block = 63
    
    @inbounds for i in 1:24
        val = g[i]
        if val == 1; obs[idx_blot] = 1.0f0; elseif val == -1; obs[idx_blot] = -1.0f0; end
        idx_blot += 1
        if val >= 2; obs[idx_block] = 1.0f0; elseif val <= -2; obs[idx_block] = -1.0f0; end
        idx_block += 1
    end
    
    return obs
end

const vector_observation = observe_full
const observe = observe_full