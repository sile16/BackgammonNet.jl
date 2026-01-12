using StaticArrays
using Random

const NUM_POINTS = 24
const BAR_IDX = 25
const OPP_BAR_IDX = 26
const OFF_IDX = 27
const OPP_OFF_IDX = 28

# Action Constants
const PASS_LOC = 0
const BAR_LOC = 1
# Points 0..23 map to LOC 2..25

mutable struct BackgammonGame
    board::SVector{28, Int8}
    dice::SVector{2, Int8}
    remaining_actions::Int8 # 1 for normal, 2 for doubles
    turn::Int8 # 0 or 1
    current_player::Int8 # 0 or 1
    terminated::Bool
    reward::Float32
    history::Vector{Int}
end

function BackgammonGame(board, dice, remaining, turn, cp, term, rew)
    BackgammonGame(board, dice, remaining, turn, cp, term, rew, Int[])
end

function Base.show(io::IO, g::BackgammonGame)
    print(io, "BackgammonGame(p=$(g.current_player), dice=$(g.dice), rem=$(g.remaining_actions), term=$(g.terminated))")
end

function initial_state()
    b = zeros(MVector{28, Int8})
    b[1] = 2
    b[6] = -5
    b[8] = -3
    b[12] = 5
    b[13] = -5
    b[17] = 3
    b[19] = 5
    b[24] = -2
    
    rng = Random.default_rng()
    current_player = rand(rng, 0:1)
    
    d1, d2 = 0, 0
    while d1 == d2
        d1 = rand(rng, 1:6)
        d2 = rand(rng, 1:6)
    end
    dice = SVector{2, Int8}(d1, d2)
    
    return BackgammonGame(
        SVector{28, Int8}(b),
        dice,
        Int8(1), # Non-doubles start with 1 action (of 2 moves)
        Int8(0),
        Int8(current_player),
        false,
        0.0f0
    )
end

function flip_board(board::SVector{28, Int8})
    b_new = MVector{28, Int8}(-board)
    @inbounds for i in 1:12
        tmp = b_new[i]
        b_new[i] = b_new[25 - i]
        b_new[25 - i] = tmp
    end
    tmp_bar = b_new[25]
    b_new[25] = b_new[26]
    b_new[26] = tmp_bar
    tmp_off = b_new[27]
    b_new[27] = b_new[28]
    b_new[28] = tmp_off
    return SVector{28, Int8}(b_new)
end

function loc_to_index(loc::Integer)
    if loc == PASS_LOC; return -1; end
    if loc == BAR_LOC; return BAR_IDX; end # 25
    return Int(loc - 1) # 2->1, 25->24
end

function apply_single_move!(board::MVector{28, Int8}, loc::Integer, die::Integer)
    src = loc_to_index(loc)
    if src == -1; return; end # Pass
    
    if src == BAR_IDX
        tgt = Int(die) # 1-based: die is 1..6
        if board[tgt] == -1
            board[tgt] = 1
            board[OPP_BAR_IDX] -= 1
        else
            board[tgt] += 1
        end
        board[src] -= 1
    else
        tgt = Int(src + die)
        if tgt > 24
            board[OFF_IDX] += 1
            board[src] -= 1
        else
            if board[tgt] == -1
                board[tgt] = 1
                board[OPP_BAR_IDX] -= 1
            else
                board[tgt] += 1
            end
            board[src] -= 1
        end
    end
end

function play!(g::BackgammonGame, action_idx::Integer)
    if g.terminated; return; end
    push!(g.history, Int(action_idx))
    
    idx0 = Int(action_idx - 1)
    loc1 = div(idx0, 26)
    loc2 = idx0 % 26
    
    new_board = MVector{28, Int8}(g.board)
    d1 = Int(g.dice[1])
    d2 = Int(g.dice[2])
    
    valid_1 = is_move_legal(g.board, loc1, d1)
    if valid_1
        b_temp = copy(new_board)
        apply_single_move!(b_temp, loc1, d1)
        valid_2 = is_move_legal(SVector{28, Int8}(b_temp), loc2, d2)
        if valid_2
            apply_single_move!(new_board, loc1, d1)
            apply_single_move!(new_board, loc2, d2)
        else
            apply_single_move!(new_board, loc2, d2)
            apply_single_move!(new_board, loc1, d1)
        end
    else
        apply_single_move!(new_board, loc2, d2)
        apply_single_move!(new_board, loc1, d1)
    end
    
    g.board = SVector{28, Int8}(new_board)
    
    if g.board[OFF_IDX] == 15
        g.terminated = true
        g.reward = 1.0f0
        return
    end
    
    g.remaining_actions -= 1
    if g.remaining_actions <= 0
        switch_turn!(g)
    end
end





function switch_turn!(g::BackgammonGame)
    g.board = flip_board(g.board)
    g.current_player = 1 - g.current_player
    g.turn = 1 - g.turn
    
    rng = Random.default_rng()
    d1 = rand(rng, 1:6)
    d2 = rand(rng, 1:6)
    if d1 > d2; d1, d2 = d2, d1; end
    
    g.dice = SVector{2, Int8}(Int8(d1), Int8(d2))
    
    if d1 == d2
        g.remaining_actions = 2
    else
        g.remaining_actions = 1
    end
end

function is_move_legal(board::SVector{28, Int8}, loc::Integer, die::Integer)
    src = loc_to_index(loc)
    if src == -1; return true; end
    
    if src == BAR_IDX
        tgt = Int(die)
        if tgt <= 24 && board[tgt] >= -1
            return true
        end
    else
        if board[src] <= 0; return false; end
        tgt = Int(src + die)
        if tgt > 24
            for i in 1:18
                if board[i] > 0; return false; end
            end
            dist = 25 - src
            if die == dist; return true; end
            if die > dist
                for i in 19:(src-1)
                    if board[i] > 0; return false; end
                end
                return true
            end
            return false
        else
            if board[tgt] >= -1
                return true
            end
        end
    end
    return false
end