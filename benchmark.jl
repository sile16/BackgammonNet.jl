using BackgammonNet
using Random
using Statistics
using Printf

function run_benchmark(duration_seconds=30.0)
    println("Running benchmark for ~$duration_seconds seconds on a single CPU...")
    
    start_time = time()
    end_time = start_time + duration_seconds
    
    games = 0
    total_actions = 0
    total_turns = 0
    
    wins_1pt = 0
    wins_2pt = 0
    wins_3pt = 0
    
    # Pre-allocate RNG to avoid thread/allocation overhead if possible, 
    # though Random.default_rng() is usually fine.
    
    while time() < end_time
        g = initial_state()
        sample_chance!(g) # Roll initial dice
        
        last_player = current_player(g)
        
        while !game_terminated(g)
            actions = legal_actions(g)
            
            len = length(actions)
            if len == 0
                break # Should not happen in standard play
            end
            
            # Simple random policy
            idx = rand(1:len)
            a = actions[idx]
            
            step!(g, a)
            total_actions += 1
            
            cp = current_player(g)
            if cp != last_player
                total_turns += 1
                last_player = cp
            end
        end
        
        games += 1
        
        # Analyze reward
        r = abs(g.reward)
        if r ≈ 1.0f0
            wins_1pt += 1
        elseif r ≈ 2.0f0
            wins_2pt += 1
        elseif r ≈ 3.0f0
            wins_3pt += 1
        end
    end
    
    elapsed = time() - start_time
    
    println("\n--- Benchmark Results ---")
    @printf("Elapsed Time:    %.2f s\n", elapsed)
    @printf("Total Games:     %d\n", games)
    @printf("Total Actions:   %d\n", total_actions)
    @printf("Total Turns:     %d\n", total_turns)
    println("")
    @printf("Games/sec:       %.2f\n", games / elapsed)
    @printf("Turns/sec:       %.2f\n", total_turns / elapsed)
    @printf("Actions/sec:     %.2f\n", total_actions / elapsed)
    println("")
    println("Win Distribution:")
    @printf("  1 pt (Single): %d (%.1f%%)\n", wins_1pt, 100 * wins_1pt / games)
    @printf("  2 pt (Gammon): %d (%.1f%%)\n", wins_2pt, 100 * wins_2pt / games)
    @printf("  3 pt (Backg.): %d (%.1f%%)\n", wins_3pt, 100 * wins_3pt / games)
    
    avg_actions_per_game = total_actions / games
    avg_turns_per_game = total_turns / games
    
    println("")
    @printf("Avg Actions/Game: %.1f\n", avg_actions_per_game)
    @printf("Avg Turns/Game:   %.1f\n", avg_turns_per_game)
end

run_benchmark(30.0)
