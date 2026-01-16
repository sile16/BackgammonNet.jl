# Validate move sequences between Julia and gnubg
# Compares the actual checker moves, not just final board states
#
# Uses CLI gnubg interface (proven to work) instead of Python module

using BackgammonNet
using Random

include("gnubg_bridge.jl")

"""
Convert Julia action to gnubg move representation.
Returns list of (from, to) tuples in gnubg coordinates.

For gnubg:
- Point 0 = bar
- Points 1-24 = board
- Point 25 = off

Julia action encoding: (loc1 * 26) + loc2 + 1
- loc = 0: bar
- loc = 1-24: canonical point (current player's perspective)
- loc = 25: pass

The canonical→gnubg conversion depends on current player:
- P0 on roll: gnubg_pt = 25 - canonical (since gnubg 24→1 = Julia 1→24)
- P1 on roll: gnubg_pt = canonical (since both use 24→1→off direction)
"""
function julia_action_to_gnubg_moves(g::BackgammonGame, action::Int)
    moves = Tuple{Int, Int}[]
    cp = Int(g.current_player)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    src1, src2 = BackgammonNet.decode_action(action)

    # Convert sources to gnubg coordinates
    function to_gnubg_pt(loc::Int)
        if loc == 0
            return 0  # bar
        elseif loc == 25
            return -1  # pass
        else
            # For P0: canonical = physical, gnubg = 25 - physical
            # For P1: canonical i = physical 25-i, gnubg = physical = 25 - canonical
            # So gnubg = 25 - canonical for both players!
            return 25 - loc
        end
    end

    from1 = to_gnubg_pt(src1)
    from2 = to_gnubg_pt(src2)

    # Compute destinations
    function dest(from::Int, die::Int)
        if from == 0  # from bar
            return 25 - die  # entering at 25-die
        elseif from == -1  # pass
            return -1
        else
            d = from - die
            return d <= 0 ? 25 : d  # 25 = bearing off
        end
    end

    to1 = dest(from1, d1)
    to2 = dest(from2, d2)

    if from1 != -1
        push!(moves, (from1, to1))
    end
    if from2 != -1
        push!(moves, (from2, to2))
    end

    return moves
end

"""
Collapse sequential moves where one checker moves multiple times.
E.g., [(13,8), (8,2)] becomes [(13,2)] if same checker moved twice.

This matches gnubg's compressed notation "13/2" for combined moves.
"""
function collapse_sequential_moves(moves::Vector{Tuple{Int, Int}})
    if length(moves) <= 1
        return moves
    end

    # Build a graph: from → to
    # Find chains where to of one move equals from of another
    result = Tuple{Int, Int}[]
    used = falses(length(moves))

    for (i, (from1, to1)) in enumerate(moves)
        if used[i]
            continue
        end

        # Try to extend this move
        current_to = to1
        used[i] = true

        # Keep extending while we find a continuation
        found = true
        while found
            found = false
            for (j, (from2, to2)) in enumerate(moves)
                if !used[j] && from2 == current_to
                    current_to = to2
                    used[j] = true
                    found = true
                    break
                end
            end
        end

        push!(result, (from1, current_to))
    end

    return result
end

"""
Normalize a list of moves for comparison.
1. Collapse sequential same-checker moves (13→8→2 becomes 13→2)
2. Sort for canonical ordering
"""
function normalize_moves(moves::Vector{Tuple{Int, Int}})
    if isempty(moves)
        return ()
    end
    collapsed = collapse_sequential_moves(moves)
    return Tuple(sort(collapsed))
end

"""
Get Julia full turn sequences for a position.
Returns Set of normalized move sequences.

For non-doubles: Each action is a full turn
For doubles: Two actions make a full turn
"""
function get_julia_turn_sequences(g::BackgammonGame)
    turn_seqs = Set{Tuple}()
    is_doubles = g.dice[1] == g.dice[2]

    actions1 = BackgammonNet.legal_actions(g)

    if is_doubles && g.remaining_actions == 2
        # Need to enumerate action pairs
        for a1 in actions1
            moves1 = julia_action_to_gnubg_moves(g, a1)

            g2 = copy_game(g)
            BackgammonNet.apply_action!(g2, a1)

            if BackgammonNet.game_terminated(g2) || BackgammonNet.is_chance_node(g2)
                # Turn ended early
                push!(turn_seqs, normalize_moves(moves1))
            else
                actions2 = BackgammonNet.legal_actions(g2)
                for a2 in actions2
                    moves2 = julia_action_to_gnubg_moves(g2, a2)
                    all_moves = vcat(moves1, moves2)
                    push!(turn_seqs, normalize_moves(all_moves))
                end
            end
        end
    else
        # Single action is a full turn
        for action in actions1
            moves = julia_action_to_gnubg_moves(g, action)
            push!(turn_seqs, normalize_moves(moves))
        end
    end

    return turn_seqs
end

"""
Get gnubg full turn sequences using CLI hint command.
Returns Set of normalized move sequences.
"""
function get_gnubg_turn_sequences(g::BackgammonGame)
    simple = julia_to_gnubg_simple(g)
    d1, d2 = Int(g.dice[1]), Int(g.dice[2])

    # Get move strings from gnubg
    move_strs = run_gnubg_hint(simple, d1, d2)

    turn_seqs = Set{Tuple}()

    if isempty(move_strs)
        # No legal moves = pass
        push!(turn_seqs, ())
    else
        for move_str in move_strs
            moves = parse_gnubg_move(move_str)
            push!(turn_seqs, normalize_moves(moves))
        end
    end

    return turn_seqs
end

function run_validation(num_games::Int; seed::Int=1, verbose::Bool=true)
    rng = MersenneTwister(seed)

    total_comparisons = 0
    total_mismatches = 0

    start_time = time()

    for game_num in 1:num_games
        g = BackgammonNet.initial_state()
        moves = 0

        while !BackgammonNet.game_terminated(g)
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                d1, d2 = Int(g.dice[1]), Int(g.dice[2])
                is_doubles = d1 == d2

                # Only validate at start of turn
                is_turn_start = is_doubles ? (g.remaining_actions == 2) : (g.remaining_actions == 1)

                if is_turn_start
                    julia_seqs = get_julia_turn_sequences(g)
                    gnubg_seqs = get_gnubg_turn_sequences(g)
                    total_comparisons += 1

                    if julia_seqs != gnubg_seqs
                        total_mismatches += 1
                        if verbose && total_mismatches <= 10
                            julia_only = setdiff(julia_seqs, gnubg_seqs)
                            gnubg_only = setdiff(gnubg_seqs, julia_seqs)
                            println("MISMATCH game $game_num move $moves: dice=($d1,$d2) doubles=$is_doubles")
                            println("  Julia: $(length(julia_seqs)) turns, gnubg: $(length(gnubg_seqs)) turns")
                            if !isempty(julia_only)
                                println("  Julia only: $(collect(julia_only)[1:min(3, length(julia_only))])")
                            end
                            if !isempty(gnubg_only)
                                println("  gnubg only: $(collect(gnubg_only)[1:min(3, length(gnubg_only))])")
                            end
                        end
                    end
                end

                # Play random move
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]
                BackgammonNet.apply_action!(g, action)
                moves += 1
            end
        end

        if verbose && game_num % 10 == 0
            elapsed = time() - start_time
            rate = game_num / elapsed
            println("Progress: $game_num/$num_games ($(round(rate, digits=1)) games/sec) mismatches=$total_mismatches")
        end
    end

    elapsed = time() - start_time

    println()
    println("=" ^ 60)
    println("MOVE SEQUENCE VALIDATION - $num_games games")
    println("=" ^ 60)
    println("Time: $(round(elapsed, digits=1))s ($(round(num_games/elapsed, digits=1)) games/sec)")
    println("Comparisons: $total_comparisons")
    println("Mismatches:  $total_mismatches")

    if total_mismatches == 0
        println("\nAll move sequences match gnubg!")
    else
        println("\nFound $total_mismatches mismatches")
    end

    return (games=num_games, comparisons=total_comparisons, mismatches=total_mismatches)
end

if abspath(PROGRAM_FILE) == @__FILE__
    num_games = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 10
    seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
    run_validation(num_games; seed=seed)
end
