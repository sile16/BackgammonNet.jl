# Parallel gnubg validation using concurrent processes
# Runs N gnubg instances in parallel for faster validation

using BackgammonNet
using Random

include("gnubg_bridge.jl")

const NUM_WORKERS = Threads.nthreads()

"""
    run_parallel_validation(num_games::Int; seed::Int=1, verbose::Bool=true, n_workers::Int=4)

Run validation using parallel gnubg processes.
"""
function run_parallel_validation(num_games::Int; seed::Int=1, verbose::Bool=true, n_workers::Int=NUM_WORKERS)
    rng = MersenneTwister(seed)

    total_comparisons = Threads.Atomic{Int}(0)
    total_mismatches = Threads.Atomic{Int}(0)

    start_time = time()

    # Collect all positions to check
    positions_to_check = Vector{Tuple{BackgammonGame, Int, Int}}()  # (game, game_num, move_num)

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
                    push!(positions_to_check, (copy_game(g), game_num, moves))
                end

                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]
                BackgammonNet.apply_action!(g, action)
                moves += 1
            end
        end

        if verbose && game_num % 100 == 0
            println("Collected positions from game $game_num...")
        end
    end

    println("Total positions to check: $(length(positions_to_check))")
    println("Starting parallel validation with $n_workers workers...")

    # Process positions in parallel
    mismatch_details = Vector{String}()
    mismatch_lock = ReentrantLock()

    Threads.@threads for i in 1:length(positions_to_check)
        g, game_num, move_num = positions_to_check[i]

        julia_states = get_julia_final_states(g)
        gnubg_states = get_gnubg_final_states(g)

        Threads.atomic_add!(total_comparisons, 1)

        if julia_states != gnubg_states
            Threads.atomic_add!(total_mismatches, 1)
            d1, d2 = Int(g.dice[1]), Int(g.dice[2])

            if verbose && total_mismatches[] <= 10
                julia_only = setdiff(julia_states, gnubg_states)
                gnubg_only = setdiff(gnubg_states, julia_states)
                msg = "MISMATCH game $game_num move $move_num: dice=($d1,$d2) julia=$(length(julia_states)) gnubg=$(length(gnubg_states))"
                lock(mismatch_lock) do
                    push!(mismatch_details, msg)
                end
            end
        end

        # Progress tracking - only one thread prints
        if i % 500 == 0 && Threads.threadid() == 1
            elapsed = time() - start_time
            current = total_comparisons[]
            rate = current / elapsed
            println("Progress: ~$current/$(length(positions_to_check)) ($(round(rate, digits=1)) positions/sec) mismatches=$(total_mismatches[])")
        end
    end

    elapsed = time() - start_time

    # Print any mismatch details
    for msg in mismatch_details
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
    n_workers = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 4
    run_parallel_validation(num_games; seed=seed, n_workers=n_workers)
end
