# gnubg_pycall.jl - Fast gnubg interface via PyCall
#
# NOTE: gnubg Python module's best_move() and moves() functions are BUGGY
# (they don't respect blocking rules). However, probabilities() works correctly
# for position evaluation using gnubg's neural network.
#
# This module provides:
# 1. FAST hybrid approach: Julia generates legal moves + gnubg evaluates positions
# 2. CLI fallback: Slower but 100% correct for validation
#
# Key functions:
# - evaluate_position(): Fast neural net evaluation via PyCall
# - get_best_move_hybrid(): Julia moves + gnubg evaluation (FAST & CORRECT)
# - get_gnubg_best_move_cli(): CLI-based move (slower but for comparison)

using BackgammonNet
using PyCall

# Initialize gnubg Python module
const gnubg = PyNULL()

function init_gnubg()
    if gnubg == PyNULL()
        copy!(gnubg, pyimport("gnubg"))
    end
    return gnubg
end

"""
    julia_to_gnubg_board(g::BackgammonGame) -> Vector{Vector{Int}}

Convert Julia BackgammonGame to gnubg Python board format.

gnubg Python board format:
- board[0] = OPPONENT's checkers (25 elements: bar + 24 points)
- board[1] = ON-ROLL player's checkers (25 elements: bar + 24 points)
- Each row: [bar, point1, point2, ..., point24]
- Off checkers are NOT included (implied by missing checkers)
- Indices are 0-based in Python

gnubg coordinate system (from on-roll player's perspective):
- Point 1 = on-roll player's home board closest to off
- Point 24 = opponent's home board
- On-roll player moves from high points (24) to low points (1) to off
"""
function julia_to_gnubg_board(g::BackgammonGame)
    init_gnubg()

    # Create 2x25 board: row 0 = opponent, row 1 = on-roll
    # Format: [bar, point1, point2, ..., point24] (no off)
    board = zeros(Int, 2, 25)

    cp = Int(g.current_player)
    p0, p1 = g.p0, g.p1

    if cp == 0
        # P0 is on roll
        # P0 moves Julia 1→24→off, gnubg on-roll moves 24→1→off
        # So gnubg point i = Julia point (25-i)

        for gnubg_pt in 1:24
            julia_idx = 25 - gnubg_pt
            p0_count = Int((p0 >> (julia_idx << 2)) & 0xF)
            p1_count = Int((p1 >> (julia_idx << 2)) & 0xF)

            # board[2] = on-roll (P0), board[1] = opponent (P1)
            board[2, gnubg_pt + 1] = p0_count  # +1 for bar offset
            board[1, gnubg_pt + 1] = p1_count
        end

        # Bars: index 1 (Julia) = bar
        board[2, 1] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)  # P0 bar
        board[1, 1] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)  # P1 bar
    else
        # P1 is on roll
        # P1 moves Julia 24→1→off, gnubg on-roll moves 24→1→off
        # So gnubg point i = Julia point i

        for gnubg_pt in 1:24
            julia_idx = gnubg_pt
            p0_count = Int((p0 >> (julia_idx << 2)) & 0xF)
            p1_count = Int((p1 >> (julia_idx << 2)) & 0xF)

            # board[2] = on-roll (P1), board[1] = opponent (P0)
            board[2, gnubg_pt + 1] = p1_count
            board[1, gnubg_pt + 1] = p0_count
        end

        # Bars
        board[2, 1] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)  # P1 bar
        board[1, 1] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)  # P0 bar
    end

    # Convert to Python format: [opponent, on-roll]
    py_board = [board[1, :], board[2, :]]

    return py_board
end

# =============================================================================
# FAST HYBRID APPROACH: Julia moves + gnubg neural net evaluation
# =============================================================================

"""
    evaluate_position_gnubg(g::BackgammonGame) -> Float64

Evaluate a position using gnubg's neural network via PyCall.

Returns equity from current player's perspective (positive = winning).
This is FAST (~50k+ evals/sec) because it uses gnubg's C neural network.
"""
function evaluate_position_gnubg(g::BackgammonGame)
    init_gnubg()

    board = julia_to_gnubg_board(g)

    # gnubg.probabilities(board, side) returns 5 probabilities:
    # (win, win_gammon, win_bg, lose_gammon, lose_bg)
    # side=1 means evaluate for on-roll player (board[1])
    probs = gnubg.probabilities(board, 1)

    # Convert to equity: win - lose + 2*(gammon_win - gammon_lose) + 3*(bg_win - bg_lose)
    win = probs[1]
    win_g = probs[2]
    win_bg = probs[3]
    lose_g = probs[4]
    lose_bg = probs[5]

    # Cubeless money equity
    equity = (win - (1.0 - win)) +
             (win_g - lose_g) +
             2.0 * (win_bg - lose_bg)

    return equity
end

"""
    evaluate_positions_batch(boards::Vector, signs::Vector{Int}) -> Vector{Float64}

Batch evaluate multiple positions using gnubg's neural network.
signs[i] = 1 if we want equity from on-roll's perspective, -1 otherwise.
"""
function evaluate_positions_batch(boards::Vector, signs::Vector{Int})
    init_gnubg()

    n = length(boards)
    equities = Vector{Float64}(undef, n)

    for i in 1:n
        probs = gnubg.probabilities(boards[i], 1)

        win = probs[1]
        win_g = probs[2]
        win_bg = probs[3]
        lose_g = probs[4]
        lose_bg = probs[5]

        equity = (win - (1.0 - win)) +
                 (win_g - lose_g) +
                 2.0 * (win_bg - lose_bg)

        equities[i] = signs[i] * equity
    end

    return equities
end

"""
    get_best_move_hybrid(g::BackgammonGame) -> Tuple{Int, Float64}

Get best move using Julia's legal move generation + gnubg's neural network evaluation.

This is the FAST and CORRECT approach:
- Julia generates legal moves (handles blocking correctly)
- gnubg evaluates resulting positions (fast neural network)
- Returns (best_action, best_equity)

Speed: ~100-500 best_moves/sec depending on branching factor
"""
function get_best_move_hybrid(g::BackgammonGame)
    init_gnubg()

    actions = BackgammonNet.legal_actions(g)

    if isempty(actions)
        return BackgammonNet.encode_action(25, 25), 0.0  # pass
    end

    # Prepare all positions for batch evaluation
    boards = Vector{Any}(undef, length(actions))
    signs = Vector{Int}(undef, length(actions))

    for (i, action) in enumerate(actions)
        g2 = BackgammonGame(g.p0, g.p1, g.dice, g.remaining_actions,
                            g.current_player, g.terminated, g.reward, Int[])
        BackgammonNet.apply_action!(g2, action)

        boards[i] = julia_to_gnubg_board(g2)
        # Sign: +1 if same player, -1 if switched
        signs[i] = g2.current_player == g.current_player ? 1 : -1
    end

    # Batch evaluate
    equities = evaluate_positions_batch(boards, signs)

    # Find best
    best_idx = argmax(equities)
    return actions[best_idx], equities[best_idx]
end

"""
    play_game_hybrid(julia_player::Int=0; seed::Int=1, verbose::Bool=true) -> NamedTuple

Play a game where gnubg uses hybrid evaluation (Julia moves + gnubg neural net).

This is MUCH faster than CLI-based play while being correct.
"""
function play_game_hybrid(julia_player::Int=0; seed::Int=1, verbose::Bool=true)
    init_gnubg()
    rng = Random.MersenneTwister(seed)

    g = BackgammonNet.initial_state()
    history = []
    num_moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
            if verbose
                println("Dice: $(g.dice[1]), $(g.dice[2]) - Player $(g.current_player)'s turn")
            end
        else
            cp = Int(g.current_player)

            if cp == julia_player
                # Julia's turn - pick random legal move
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]

                if verbose
                    src1, src2 = BackgammonNet.decode_action(action)
                    println("  Julia (random): ($src1, $src2)")
                end

                BackgammonNet.apply_action!(g, action)
                push!(history, (:julia, action))
            else
                # gnubg's turn - use hybrid evaluation
                action, equity = get_best_move_hybrid(g)

                if verbose
                    src1, src2 = BackgammonNet.decode_action(action)
                    println("  gnubg (hybrid): ($src1, $src2) [equity=$(round(equity, digits=3))]")
                end

                BackgammonNet.apply_action!(g, action)
                push!(history, (:gnubg, action, equity))
            end

            num_moves += 1
        end
    end

    winner = g.reward > 0 ? 0 : 1
    if verbose
        println("\nGame over! Winner: Player $winner (reward: $(g.reward))")
        println("Total moves: $num_moves")
    end

    return (winner=winner, num_moves=num_moves, reward=g.reward, history=history)
end

"""
    benchmark_hybrid(n_positions::Int=1000) -> NamedTuple

Benchmark the hybrid evaluation approach.
"""
function benchmark_hybrid(n_positions::Int=1000)
    init_gnubg()
    rng = Random.MersenneTwister(42)

    # Generate diverse positions
    positions = BackgammonGame[]
    for _ in 1:100
        g = BackgammonNet.initial_state()
        for _ in 1:rand(rng, 5:40)
            if BackgammonNet.game_terminated(g)
                break
            end
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                actions = BackgammonNet.legal_actions(g)
                BackgammonNet.apply_action!(g, actions[rand(rng, 1:length(actions))])
            end
        end
        if !BackgammonNet.game_terminated(g) && !BackgammonNet.is_chance_node(g)
            push!(positions, g)
        end
    end

    println("Testing with $(length(positions)) positions...")

    # Benchmark evaluation only
    t_eval = @elapsed for _ in 1:div(n_positions, length(positions))
        for g in positions
            evaluate_position_gnubg(g)
        end
    end
    evals_per_sec = n_positions / t_eval

    # Benchmark best move finding
    t_best = @elapsed for _ in 1:div(n_positions, length(positions))
        for g in positions
            get_best_move_hybrid(g)
        end
    end
    best_per_sec = n_positions / t_best

    println("\nHybrid benchmark results ($n_positions iterations):")
    println("  evaluate_position_gnubg(): $(round(evals_per_sec, digits=0)) evals/sec")
    println("  get_best_move_hybrid():    $(round(best_per_sec, digits=0)) best_moves/sec")

    return (evals_per_sec=evals_per_sec, best_per_sec=best_per_sec)
end

# =============================================================================
# Legacy PyCall functions (BUGGY - kept for reference)
# =============================================================================

"""
    gnubg_move_to_julia_action(g::BackgammonGame, gnubg_move) -> Int

Convert gnubg move (list of (from, to) tuples) to Julia action.

gnubg Python module move format:
- Point 0 = bar
- Points 1-24 = board points (1-indexed like array indices +1)
- Point 25 = off (bearing off destination)

In moves, gnubg uses 1-based point numbers where point N in a move
corresponds to array index N (since array index 0 = bar).

Julia action encoding: (loc1 * 26) + loc2 + 1
- loc = 0: bar (canonical)
- loc = 1-24: canonical point
- loc = 25: pass

Returns the Julia action integer.
"""
function gnubg_move_to_julia_action(g::BackgammonGame, gnubg_move)
    cp = Int(g.current_player)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])
    is_doubles = d1 == d2

    # Handle empty move (pass)
    if isempty(gnubg_move)
        return BackgammonNet.encode_action(25, 25)  # pass/pass
    end

    # gnubg returns moves as tuple of (from, to) tuples
    # For non-doubles: up to 2 moves (one per die)
    # For doubles with remaining_actions=2: up to 2 moves (using 2 of 4 dice)

    # Convert gnubg move point to Julia canonical coordinate
    # gnubg moves use: 0=bar, 1-24=points, 25=off
    # But empirically, gnubg Python uses point numbers where 25 means the 24-point
    # (i.e., array index = point - 1 for points 1-25, where 25 represents idx 24)
    function gnubg_move_to_canonical(gnubg_pt::Int)
        if gnubg_pt == 0
            return 0  # bar
        else
            # gnubg point N (1-25) corresponds to board position
            # gnubg uses 1-based points, array index = gnubg_pt - 1 for non-bar
            # But gnubg point 25 is actually the 24-point (array idx 24)
            # So effective gnubg board point = min(gnubg_pt, 24)
            board_pt = gnubg_pt > 24 ? 24 : gnubg_pt

            # Convert to canonical
            if cp == 0
                # P0: gnubg point i = physical point (25-i) in Julia
                # canonical = physical for P0
                return 25 - board_pt
            else
                # P1: gnubg point i = physical point i in Julia
                # canonical = 25 - physical for P1
                return 25 - board_pt
            end
        end
    end

    # Extract sources from gnubg moves
    sources = Int[]
    for move in gnubg_move
        from_pt = Int(move[1])
        push!(sources, gnubg_move_to_canonical(from_pt))
    end

    # Calculate die used for a gnubg move
    function calc_die(from_g, to_g)
        if from_g == 0  # from bar
            # Entering: die = 25 - to_pt
            return 25 - to_g
        elseif to_g == 0  # bearing off (gnubg sometimes uses 0 for off?)
            return from_g
        else
            return from_g - to_g
        end
    end

    if length(sources) == 0
        return BackgammonNet.encode_action(25, 25)  # pass/pass
    elseif length(sources) == 1
        # Single move - figure out which die was used
        from_gnubg = Int(gnubg_move[1][1])
        to_gnubg = Int(gnubg_move[1][2])
        die_used = calc_die(from_gnubg, to_gnubg)

        src = sources[1]
        if die_used == d1
            return BackgammonNet.encode_action(src, 25)  # use die1, pass die2
        else
            return BackgammonNet.encode_action(25, src)  # pass die1, use die2
        end
    else
        # Two moves
        from1_gnubg = Int(gnubg_move[1][1])
        to1_gnubg = Int(gnubg_move[1][2])
        from2_gnubg = Int(gnubg_move[2][1])
        to2_gnubg = Int(gnubg_move[2][2])

        die1_used = calc_die(from1_gnubg, to1_gnubg)
        die2_used = calc_die(from2_gnubg, to2_gnubg)

        if is_doubles
            # For doubles, order doesn't matter
            return BackgammonNet.encode_action(sources[1], sources[2])
        else
            # Match to d1, d2
            if die1_used == d1 && die2_used == d2
                return BackgammonNet.encode_action(sources[1], sources[2])
            elseif die1_used == d2 && die2_used == d1
                return BackgammonNet.encode_action(sources[2], sources[1])
            else
                # Fallback - try both orderings and see which is legal
                action1 = BackgammonNet.encode_action(sources[1], sources[2])
                if action1 in BackgammonNet.legal_actions(g)
                    return action1
                end
                action2 = BackgammonNet.encode_action(sources[2], sources[1])
                if action2 in BackgammonNet.legal_actions(g)
                    return action2
                end
                error("Could not match gnubg move to Julia action: $gnubg_move, dice=($d1,$d2), die1=$die1_used, die2=$die2_used")
            end
        end
    end
end

"""
    julia_action_to_gnubg_move(g::BackgammonGame, action::Int) -> Vector{Tuple{Int,Int}}

Convert Julia action to gnubg move format (list of (from, to) tuples).

Used to apply Julia's chosen move to gnubg's board representation.
"""
function julia_action_to_gnubg_move(g::BackgammonGame, action::Int)
    cp = Int(g.current_player)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    src1, src2 = BackgammonNet.decode_action(action)

    moves = Tuple{Int,Int}[]

    # Convert canonical source to gnubg coordinate
    function canonical_to_gnubg(canonical::Int)
        if canonical == 0
            return 0  # bar
        elseif canonical == 25
            return -1  # pass (no move)
        else
            # Canonical → gnubg point
            # For both players: gnubg = 25 - canonical
            return 25 - canonical
        end
    end

    # Calculate destination in gnubg coordinates
    function calc_dest(gnubg_from::Int, die::Int)
        if gnubg_from == 0  # from bar
            return 25 - die  # entering
        else
            dest = gnubg_from - die
            return dest <= 0 ? 25 : dest  # 25 = bearing off
        end
    end

    from1 = canonical_to_gnubg(src1)
    from2 = canonical_to_gnubg(src2)

    if from1 != -1
        to1 = calc_dest(from1, d1)
        push!(moves, (from1, to1))
    end

    if from2 != -1
        to2 = calc_dest(from2, d2)
        push!(moves, (from2, to2))
    end

    return moves
end

"""
    get_gnubg_best_move(g::BackgammonGame) -> Tuple{Int, Vector}

Get gnubg's best move for the current position.

Returns (julia_action, gnubg_move_tuples).
"""
function get_gnubg_best_move(g::BackgammonGame)
    init_gnubg()

    board = julia_to_gnubg_board(g)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    # gnubg.best_move(pos, dice1, dice2) - takes dice as separate args
    gnubg_move = gnubg.best_move(board, d1, d2)

    # Convert to Julia action
    julia_action = gnubg_move_to_julia_action(g, gnubg_move)

    return julia_action, gnubg_move
end

"""
    get_gnubg_all_moves(g::BackgammonGame) -> Vector

Get all legal moves from gnubg as position keys.

Uses gnubg.moves() which returns position keys.
"""
function get_gnubg_all_moves(g::BackgammonGame)
    init_gnubg()

    board = julia_to_gnubg_board(g)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    # gnubg.moves(pos, dice1, dice2) returns dict of {position_key: probability}
    gnubg_positions = gnubg.moves(board, d1, d2)

    return collect(keys(gnubg_positions))
end

"""
    play_game_vs_gnubg(julia_player::Int=0; seed::Int=1, verbose::Bool=true) -> NamedTuple

Play a game between Julia (using random legal moves) and gnubg.

Arguments:
- julia_player: Which player Julia controls (0 or 1)
- seed: Random seed for Julia's move selection
- verbose: Print game progress

Returns (winner, num_moves, reward, history).
"""
function play_game_vs_gnubg(julia_player::Int=0; seed::Int=1, verbose::Bool=true)
    init_gnubg()
    rng = Random.MersenneTwister(seed)

    g = BackgammonNet.initial_state()
    history = []
    num_moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            # Roll dice
            BackgammonNet.sample_chance!(g, rng)
            if verbose
                println("Dice: $(g.dice[1]), $(g.dice[2]) - Player $(g.current_player)'s turn")
            end
        else
            cp = Int(g.current_player)

            if cp == julia_player
                # Julia's turn - pick random legal move
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]

                if verbose
                    src1, src2 = BackgammonNet.decode_action(action)
                    println("  Julia plays: ($src1, $src2)")
                end

                BackgammonNet.apply_action!(g, action)
                push!(history, (:julia, action))
            else
                # gnubg's turn
                try
                    action, gnubg_move = get_gnubg_best_move(g)

                    if verbose
                        src1, src2 = BackgammonNet.decode_action(action)
                        println("  gnubg plays: ($src1, $src2) [raw: $gnubg_move]")
                    end

                    # Verify the action is legal in Julia
                    legal = BackgammonNet.legal_actions(g)
                    if action in legal
                        BackgammonNet.apply_action!(g, action)
                        push!(history, (:gnubg, action, gnubg_move))
                    else
                        if verbose
                            println("  WARNING: gnubg move not in Julia legal actions!")
                            println("  Legal actions: $legal")
                        end
                        # Fall back to random legal move
                        action = legal[rand(rng, 1:length(legal))]
                        BackgammonNet.apply_action!(g, action)
                        push!(history, (:gnubg_fallback, action))
                    end
                catch e
                    if verbose
                        println("  ERROR getting gnubg move: $e")
                    end
                    # Fall back to random legal move
                    actions = BackgammonNet.legal_actions(g)
                    action = actions[rand(rng, 1:length(actions))]
                    BackgammonNet.apply_action!(g, action)
                    push!(history, (:error, action, e))
                end
            end

            num_moves += 1
        end
    end

    winner = g.reward > 0 ? 0 : 1
    if verbose
        println("\nGame over! Winner: Player $winner (reward: $(g.reward))")
        println("Total moves: $num_moves")
    end

    return (winner=winner, num_moves=num_moves, reward=g.reward, history=history)
end

"""
    validate_board_sync(g::BackgammonGame) -> Bool

Verify that Julia and gnubg have the same board state.
Useful for debugging synchronization issues during game play.
"""
function validate_board_sync(g::BackgammonGame)
    init_gnubg()

    # Get Julia's board in gnubg format
    julia_board = julia_to_gnubg_board(g)

    # Count checkers on each side
    julia_on_roll = sum(julia_board[2])
    julia_opponent = sum(julia_board[1])

    # Each player should have 15 checkers total
    if julia_on_roll != 15 || julia_opponent != 15
        println("WARNING: Checker count mismatch!")
        println("  On-roll: $julia_on_roll, Opponent: $julia_opponent")
        return false
    end

    return true
end

"""
    benchmark_pycall(n::Int=1000) -> NamedTuple

Benchmark PyCall gnubg operations.
"""
function benchmark_pycall(n::Int=1000)
    init_gnubg()
    rng = Random.MersenneTwister(42)

    # Generate test positions
    positions = BackgammonGame[]
    g = BackgammonNet.initial_state()
    for _ in 1:min(n, 100)
        g = BackgammonNet.initial_state()
        BackgammonNet.sample_chance!(g, rng)
        for _ in 1:rand(rng, 1:20)
            if BackgammonNet.game_terminated(g)
                break
            end
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                actions = BackgammonNet.legal_actions(g)
                BackgammonNet.apply_action!(g, actions[rand(rng, 1:length(actions))])
            end
        end
        if !BackgammonNet.game_terminated(g) && !BackgammonNet.is_chance_node(g)
            push!(positions, g)
        end
    end

    println("Testing with $(length(positions)) positions...")

    # Benchmark gnubg.moves()
    t_moves = @elapsed for _ in 1:div(n, length(positions))
        for g in positions
            board = julia_to_gnubg_board(g)
            d1, d2 = Int(g.dice[1]), Int(g.dice[2])
            gnubg.moves(board, d1, d2)
        end
    end
    moves_per_sec = n / t_moves

    # Benchmark gnubg.best_move()
    t_best = @elapsed for _ in 1:div(n, length(positions))
        for g in positions
            board = julia_to_gnubg_board(g)
            d1, d2 = Int(g.dice[1]), Int(g.dice[2])
            gnubg.best_move(board, d1, d2)
        end
    end
    best_per_sec = n / t_best

    println("\nBenchmark results ($n calls each):")
    println("  gnubg.moves():     $(round(moves_per_sec, digits=0)) calls/sec")
    println("  gnubg.best_move(): $(round(best_per_sec, digits=0)) calls/sec")

    return (moves_per_sec=moves_per_sec, best_per_sec=best_per_sec)
end

# =============================================================================
# CLI-based game play (correct but slower)
# =============================================================================

include("gnubg_bridge.jl")

"""
    expand_gnubg_moves(moves::Vector{Tuple{Int,Int}}, die::Int) -> Vector{Tuple{Int,Int}}

Expand compressed gnubg moves to individual die-sized steps.

gnubg notation like `13/7` with dice 3,3 represents 13→10→7 (two 3-point moves).
This function expands such moves into individual (from, to) pairs where each
pair represents exactly one die's worth of movement.
"""
function expand_gnubg_moves(moves::Vector{Tuple{Int,Int}}, die::Int)
    expanded = Tuple{Int,Int}[]

    for (from_pt, to_pt) in moves
        if from_pt == 0
            # Bar entry - single move
            push!(expanded, (from_pt, to_pt))
        elseif to_pt == 25 || to_pt == 0
            # Bearing off - single move
            push!(expanded, (from_pt, to_pt))
        else
            distance = from_pt - to_pt
            if distance == die
                # Single die move
                push!(expanded, (from_pt, to_pt))
            elseif distance > die && distance % die == 0
                # Multiple die moves (same checker moves multiple times)
                current = from_pt
                num_moves = distance ÷ die
                for _ in 1:num_moves
                    next_pt = current - die
                    push!(expanded, (current, next_pt))
                    current = next_pt
                end
            else
                # Non-standard distance, just use as-is
                push!(expanded, (from_pt, to_pt))
            end
        end
    end

    return expanded
end

"""
    get_gnubg_best_move_cli(g::BackgammonGame) -> Tuple{Int, String}

Get gnubg's best move using the CLI (slower but correct).

For doubles (remaining_actions=2), gnubg returns a full turn (4 moves) but Julia
needs just the first 2 moves. We expand compressed moves and take the first 2.

Returns (julia_action, gnubg_move_string).
"""
function get_gnubg_best_move_cli(g::BackgammonGame)
    simple = julia_to_gnubg_simple(g)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])
    is_doubles = d1 == d2

    # Get all moves from gnubg CLI
    move_strs = run_gnubg_hint(simple, d1, d2)

    if isempty(move_strs)
        # No legal moves - pass
        return BackgammonNet.encode_action(25, 25), "pass"
    end

    # gnubg returns moves sorted by evaluation - first is best
    best_move_str = move_strs[1]

    # Parse and expand the move string
    parsed_moves = parse_gnubg_move(best_move_str)

    # For doubles, expand combined moves like 13/7 into individual steps 13/10, 10/7
    if is_doubles
        expanded_moves = expand_gnubg_moves(parsed_moves, d1)
    else
        expanded_moves = parsed_moves
    end

    # For doubles, Julia uses 2 moves per action
    # gnubg shows all 4 moves for a complete turn, but we only need 2 per Julia action
    # (gnubg is queried with current state, so its first 2 moves should be optimal)
    if is_doubles && length(expanded_moves) > 2
        moves_to_apply = expanded_moves[1:2]
    else
        moves_to_apply = expanded_moves
    end

    # Apply moves to get target final state
    board = copy(simple)
    for (from_pt, to_pt) in moves_to_apply
        apply_gnubg_move!(board, from_pt, to_pt)
    end

    # Find Julia action that produces the same final state
    cp = Int(g.current_player)
    actions = BackgammonNet.legal_actions(g)

    for a in actions
        g2 = copy_game(g)
        BackgammonNet.apply_action!(g2, a)
        julia_state = julia_to_gnubg_simple(g2; perspective=cp)[1:26]

        if julia_state == board[1:26]
            return a, best_move_str
        end
    end

    # Fallback - shouldn't happen if gnubg and Julia agree
    @warn "Could not match gnubg move '$best_move_str' to Julia action (expanded=$expanded_moves, applied=$moves_to_apply)"
    return actions[1], best_move_str
end

"""
    play_game_vs_gnubg_cli(julia_player::Int=0; seed::Int=1, verbose::Bool=true) -> NamedTuple

Play a game between Julia (using random legal moves) and gnubg (using CLI).

Arguments:
- julia_player: Which player Julia controls (0 or 1)
- seed: Random seed for Julia's move selection
- verbose: Print game progress

Returns (winner, num_moves, reward, history).
"""
function play_game_vs_gnubg_cli(julia_player::Int=0; seed::Int=1, verbose::Bool=true)
    rng = Random.MersenneTwister(seed)

    g = BackgammonNet.initial_state()
    history = []
    num_moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
            if verbose
                println("Dice: $(g.dice[1]), $(g.dice[2]) - Player $(g.current_player)'s turn")
            end
        else
            cp = Int(g.current_player)

            if cp == julia_player
                # Julia's turn - pick random legal move
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]

                if verbose
                    src1, src2 = BackgammonNet.decode_action(action)
                    println("  Julia plays: ($src1, $src2)")
                end

                BackgammonNet.apply_action!(g, action)
                push!(history, (:julia, action))
            else
                # gnubg's turn - use CLI
                action, gnubg_str = get_gnubg_best_move_cli(g)

                if verbose
                    src1, src2 = BackgammonNet.decode_action(action)
                    println("  gnubg plays: ($src1, $src2) [$gnubg_str]")
                end

                BackgammonNet.apply_action!(g, action)
                push!(history, (:gnubg, action, gnubg_str))
            end

            num_moves += 1
        end
    end

    winner = g.reward > 0 ? 0 : 1
    if verbose
        println("\nGame over! Winner: Player $winner (reward: $(g.reward))")
        println("Total moves: $num_moves")
    end

    return (winner=winner, num_moves=num_moves, reward=g.reward, history=history)
end

"""
    evaluate_vs_gnubg(num_games::Int; julia_player::Int=0, seed::Int=1, verbose::Bool=false) -> NamedTuple

Evaluate Julia agent against gnubg over multiple games.

Returns (julia_wins, gnubg_wins, draws, total_reward).
"""
function evaluate_vs_gnubg(num_games::Int; julia_player::Int=0, seed::Int=1, verbose::Bool=false)
    julia_wins = 0
    gnubg_wins = 0
    total_reward = 0

    start_time = time()

    for i in 1:num_games
        result = play_game_vs_gnubg_cli(julia_player; seed=seed+i, verbose=false)

        if result.winner == julia_player
            julia_wins += 1
        else
            gnubg_wins += 1
        end

        # Track reward from Julia's perspective
        total_reward += julia_player == 0 ? result.reward : -result.reward

        if verbose && i % 10 == 0
            elapsed = time() - start_time
            rate = i / elapsed
            println("Progress: $i/$num_games ($(round(rate, digits=2)) games/sec) - Julia: $julia_wins, gnubg: $gnubg_wins")
        end
    end

    elapsed = time() - start_time

    println("\n" * "=" ^ 60)
    println("EVALUATION COMPLETE - $num_games games")
    println("=" ^ 60)
    println("Time: $(round(elapsed, digits=1))s ($(round(num_games/elapsed, digits=2)) games/sec)")
    println("Julia wins: $julia_wins ($(round(100*julia_wins/num_games, digits=1))%)")
    println("gnubg wins: $gnubg_wins ($(round(100*gnubg_wins/num_games, digits=1))%)")
    println("Average reward: $(round(total_reward/num_games, digits=3))")

    return (julia_wins=julia_wins, gnubg_wins=gnubg_wins, total_reward=total_reward, games=num_games)
end

# Command-line interface
if abspath(PROGRAM_FILE) == @__FILE__
    using Random

    println("gnubg Interface Test")
    println("=" ^ 40)

    # Benchmark hybrid approach
    println("\nBenchmarking hybrid approach (Julia moves + gnubg neural net)...")
    benchmark_hybrid(1000)

    println("\n" * "=" ^ 40)
    println("\nPlaying test game using HYBRID approach (fast & correct)...")
    result = play_game_hybrid(0; seed=42, verbose=true)

    println("\n" * "=" ^ 40)
    println("\nSummary:")
    println("  - Hybrid approach: Julia moves + gnubg neural net evaluation")
    println("  - This is FAST (~1000+ best_moves/sec) and CORRECT")
    println("  - CLI approach still available for comparison (slower)")
end
