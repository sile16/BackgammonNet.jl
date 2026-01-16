# Hybrid gnubg validation - parallel workers with small batches
# Combines batching (fewer process spawns) with parallelization (multiple workers)

using BackgammonNet
using Random

include("gnubg_bridge.jl")

const DEFAULT_BATCH_SIZE = 20  # Small enough to not hang gnubg
const DEFAULT_NUM_WORKERS = 4

"""
    run_gnubg_batch_small(queries::Vector{Tuple{Vector{Int}, Int, Int}}) -> Vector{Vector{String}}

Run a small batch of gnubg queries (<=20) and return all move lists.
"""
function run_gnubg_batch_small(queries::Vector{Tuple{Vector{Int}, Int, Int}})
    isempty(queries) && return Vector{Vector{String}}[]

    # Build command string for pipe input
    commands = IOBuffer()
    println(commands, "set automatic roll off")
    println(commands, "set output matchpc off")
    println(commands, "set output mwc off")
    println(commands, "set output rawboard off")
    println(commands, "new game")
    println(commands, "set dice 3 1")
    println(commands, "move 8/5 6/5")

    for (simple, d1, d2) in queries
        board_str = join(simple, " ")
        println(commands, "set dice $d1 $d2")
        println(commands, "set board simple $board_str")
        println(commands, "hint 999")
    end

    println(commands, "quit")

    cmd_str = String(take!(commands))
    output = read(pipeline(`echo $cmd_str`, `gnubg -t -q`), String)

    # Parse output
    results = Vector{Vector{String}}(undef, length(queries))
    for i in 1:length(queries)
        results[i] = String[]
    end

    current_query = 0

    for line in split(output, '\n')
        m_move = match(r"^\s*(\d+)\.\s+\w+\s+[\d-]+ply\s+(.+?)\s+Eq\.:", line)
        if m_move !== nothing
            move_num = parse(Int, m_move.captures[1])
            move_str = strip(m_move.captures[2])

            if move_num == 1
                current_query += 1
            end

            if current_query >= 1 && current_query <= length(queries)
                push!(results[current_query], move_str)
            end
        end
    end

    return results
end

"""
    get_gnubg_final_states_batch(games::Vector{BackgammonGame}) -> Vector{Set{Vector{Int}}}

Get final states for a batch of games.
"""
function get_gnubg_final_states_batch(games::Vector{BackgammonGame})
    queries = Tuple{Vector{Int}, Int, Int}[]
    for g in games
        simple = julia_to_gnubg_simple(g)
        d1, d2 = Int(g.dice[1]), Int(g.dice[2])
        push!(queries, (simple, d1, d2))
    end

    all_moves = run_gnubg_batch_small(queries)

    results = Set{Vector{Int}}[]
    for (i, move_strs) in enumerate(all_moves)
        simple = queries[i][1]
        final_states = Set{Vector{Int}}()

        if isempty(move_strs)
            push!(final_states, simple[1:26])
        else
            for move_str in move_strs
                board = copy(simple)
                moves = parse_gnubg_move(move_str)
                for (from_pt, to_pt) in moves
                    apply_gnubg_move!(board, from_pt, to_pt)
                end
                push!(final_states, board[1:26])
            end
        end
        push!(results, final_states)
    end

    return results
end

"""
    run_hybrid_validation(num_games::Int; seed::Int=1, verbose::Bool=true,
                          batch_size::Int=20, n_workers::Int=4)

Run validation using parallel workers with small batches.
"""
function run_hybrid_validation(num_games::Int; seed::Int=1, verbose::Bool=true,
                               batch_size::Int=DEFAULT_BATCH_SIZE, n_workers::Int=DEFAULT_NUM_WORKERS)
    rng = MersenneTwister(seed)

    total_comparisons = Threads.Atomic{Int}(0)
    total_mismatches = Threads.Atomic{Int}(0)

    start_time = time()

    # Collect all positions to check
    all_games = BackgammonGame[]

    for game_num in 1:num_games
        g = BackgammonNet.initial_state()
        moves = 0

        while !BackgammonNet.game_terminated(g)
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                d1, d2 = Int(g.dice[1]), Int(g.dice[2])
                is_doubles = d1 == d2
                is_turn_start = is_doubles ? (g.remaining_actions == 2) : (g.remaining_actions == 1)

                if is_turn_start
                    push!(all_games, copy_game(g))
                end

                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]
                BackgammonNet.apply_action!(g, action)
                moves += 1
            end
        end
    end

    println("Total positions to check: $(length(all_games))")
    println("Batch size: $batch_size, Workers: $n_workers")
    println("Starting hybrid validation...")

    # Split into batches
    num_batches = ceil(Int, length(all_games) / batch_size)
    batches = Vector{Vector{BackgammonGame}}(undef, num_batches)
    for i in 1:num_batches
        start_idx = (i-1) * batch_size + 1
        end_idx = min(i * batch_size, length(all_games))
        batches[i] = all_games[start_idx:end_idx]
    end

    # Process batches in parallel
    mismatch_lock = ReentrantLock()
    mismatch_msgs = String[]

    Threads.@threads for batch_idx in 1:length(batches)
        batch = batches[batch_idx]

        # Get gnubg results for entire batch
        gnubg_results = get_gnubg_final_states_batch(batch)

        for (i, g) in enumerate(batch)
            julia_states = get_julia_final_states(g)
            gnubg_states = gnubg_results[i]

            Threads.atomic_add!(total_comparisons, 1)

            if julia_states != gnubg_states
                Threads.atomic_add!(total_mismatches, 1)
                d1, d2 = Int(g.dice[1]), Int(g.dice[2])

                if total_mismatches[] <= 10
                    msg = "MISMATCH dice=($d1,$d2) julia=$(length(julia_states)) gnubg=$(length(gnubg_states))"
                    lock(mismatch_lock) do
                        push!(mismatch_msgs, msg)
                    end
                end
            end
        end

        if verbose && batch_idx % 10 == 0 && Threads.threadid() == 1
            elapsed = time() - start_time
            current = total_comparisons[]
            rate = current / elapsed
            games_rate = num_games * current / length(all_games) / elapsed
            println("Progress: batch $batch_idx/$(length(batches)) (~$current positions, $(round(rate, digits=1)) pos/sec, ~$(round(games_rate, digits=1)) games/sec)")
        end
    end

    elapsed = time() - start_time

    for msg in mismatch_msgs
        println(msg)
    end

    println()
    println("=" ^ 60)
    println("VALIDATION COMPLETE - $num_games games")
    println("=" ^ 60)
    println("Time: $(round(elapsed, digits=1))s")
    println("Positions checked: $(total_comparisons[])")
    println("Rate: $(round(total_comparisons[]/elapsed, digits=1)) positions/sec")
    println("Games rate: $(round(num_games/elapsed, digits=1)) games/sec")
    println("Mismatches: $(total_mismatches[])")

    if total_mismatches[] == 0
        println("\nAll final board states match gnubg!")
    else
        println("\nFound $(total_mismatches[]) mismatches")
    end

    return (games=num_games, comparisons=total_comparisons[], mismatches=total_mismatches[])
end

if abspath(PROGRAM_FILE) == @__FILE__
    num_games = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 50
    seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
    batch_size = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 20
    n_workers = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 4
    run_hybrid_validation(num_games; seed=seed, batch_size=batch_size, n_workers=n_workers)
end
