using StaticArrays

# Constants for observation (Canonical indices)
const BAR_IDX = 25
const OPP_BAR_IDX = 26

function observe_fast(g::BackgammonGame)
    # 34 elements
    # Board (28) / 15.0
    # Playable Dice (6) / 4.0

    obs = zeros(Float32, 34)

    # Board
    @inbounds for i in 1:28
        obs[i] = Float32(g[i]) / 15.0f0
    end

    # Dice - use stack-allocated MVector to avoid allocation
    d1 = g.dice[1]
    d2 = g.dice[2]

    if g.remaining_actions > 0 && d1 > 0 && d2 > 0
        if d1 == d2
            # Doubles: remaining_actions: 2 -> 4 dice. 1 -> 2 dice.
            count = Float32(Int(g.remaining_actions) * 2) / 4.0f0
            @inbounds obs[28 + d1] = count
        else
            # Non-doubles
            @inbounds obs[28 + d1] = 0.25f0
            @inbounds obs[28 + d2] = 0.25f0
        end
    end

    return obs
end

function observe_full(g::BackgammonGame)
    # 86 elements
    obs = zeros(Float32, 86)

    # Extract all 28 board values ONCE into stack-allocated array
    vals = MVector{28, Int8}(undef)
    @inbounds for i in 1:28
        vals[i] = g[i]
        obs[i] = Float32(vals[i]) / 15.0f0
    end

    # Dice (same as observe_fast)
    d1 = g.dice[1]
    d2 = g.dice[2]
    if g.remaining_actions > 0 && d1 > 0 && d2 > 0
        if d1 == d2
            count = Float32(Int(g.remaining_actions) * 2) / 4.0f0
            @inbounds obs[28 + d1] = count
        else
            @inbounds obs[28 + d1] = 0.25f0
            @inbounds obs[28 + d2] = 0.25f0
        end
    end

    # Heuristics - use cached vals instead of g[i]
    min_my = 25
    max_opp = 0
    my_pip = 0
    opp_pip = 0
    can_bear_my = (vals[BAR_IDX] == 0)
    can_bear_opp = (vals[OPP_BAR_IDX] == 0)

    @inbounds for i in 1:24
        val = vals[i]
        if val > 0
            if i < min_my; min_my = i; end
            my_pip += val * (25 - i)
            # Check bearing off for my checkers (need all in home 19-24)
            if can_bear_my && i < 19
                can_bear_my = false
            end
        elseif val < 0
            if i > max_opp; max_opp = i; end
            opp_pip += (-val) * i
            # Check bearing off for opp checkers (need all in home 1-6)
            if can_bear_opp && i > 6
                can_bear_opp = false
            end
        end

        # Blot/Block detection
        idx_blot = 38 + i
        idx_block = 62 + i
        if val == 1
            obs[idx_blot] = 1.0f0
        elseif val == -1
            obs[idx_blot] = -1.0f0
        end
        if val >= 2
            obs[idx_block] = 1.0f0
        elseif val <= -2
            obs[idx_block] = -1.0f0
        end
    end

    @inbounds begin
        my_pip += vals[BAR_IDX] * 25
        opp_pip += (-vals[OPP_BAR_IDX]) * 25

        is_race = (vals[BAR_IDX] == 0 && vals[OPP_BAR_IDX] == 0 && min_my > max_opp)
        obs[35] = Float32(is_race)
        obs[36] = Float32(can_bear_my)
        obs[37] = Float32(can_bear_opp)
        obs[38] = (my_pip - opp_pip) / 375.0f0
    end

    return obs
end

const vector_observation = observe_full
const observe = observe_full