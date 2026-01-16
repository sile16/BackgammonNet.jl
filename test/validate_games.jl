# validate_games.jl - Full game validation with reward checking
#
# Plays complete games where both Julia and gnubg make moves, then validates:
# 1. Legal moves match between Julia and gnubg each turn
# 2. Final board states match
# 3. Game termination is detected correctly
# 4. Reward values (1/2/3 for normal/gammon/backgammon) are correct

using BackgammonNet
using Random

include("gnubg_bridge.jl")

"""
    validate_game_rewards(g::BackgammonGame) -> NamedTuple

Validate that the reward for a terminated game is correct.
Returns (valid, expected_reward, actual_reward, details).
"""
function validate_game_rewards(g::BackgammonGame)
    if !BackgammonNet.game_terminated(g)
        return (valid=false, expected=0, actual=g.reward, details="Game not terminated")
    end

    # Determine winner (reward > 0 means P0 wins, < 0 means P1 wins)
    winner = g.reward > 0 ? 0 : 1
    loser = 1 - winner

    # Count loser's checkers
    loser_board = loser == 0 ? g.p0 : g.p1
    loser_off_idx = loser == 0 ? BackgammonNet.IDX_P0_OFF : BackgammonNet.IDX_P1_OFF
    loser_bar_idx = loser == 0 ? BackgammonNet.IDX_P0_BAR : BackgammonNet.IDX_P1_BAR

    loser_off = Int((loser_board >> (loser_off_idx << 2)) & 0xF)
    loser_bar = Int((loser_board >> (loser_bar_idx << 2)) & 0xF)

    # Count checkers on board
    loser_on_board = 0
    for i in 1:24
        loser_on_board += Int((loser_board >> (i << 2)) & 0xF)
    end
    loser_on_board += loser_bar

    # Determine expected reward magnitude
    if loser_off > 0
        # Loser has borne off at least one checker - normal win
        expected_magnitude = 1
        details = "Normal win (loser has $loser_off off)"
    else
        # Loser has not borne off any checkers - at least gammon
        # Check for backgammon: loser has checker on bar or in winner's home board
        winner_home_start = winner == 0 ? 19 : 1
        winner_home_end = winner == 0 ? 24 : 6

        loser_in_winner_home = 0
        for i in winner_home_start:winner_home_end
            loser_in_winner_home += Int((loser_board >> (i << 2)) & 0xF)
        end

        if loser_bar > 0 || loser_in_winner_home > 0
            expected_magnitude = 3
            details = "Backgammon (loser has $loser_bar on bar, $loser_in_winner_home in winner's home)"
        else
            expected_magnitude = 2
            details = "Gammon (loser has 0 off, none in winner's home or bar)"
        end
    end

    # Expected reward: positive for P0 win, negative for P1 win
    expected_reward = winner == 0 ? expected_magnitude : -expected_magnitude
    actual_reward = g.reward

    valid = expected_reward == actual_reward

    return (valid=valid, expected=expected_reward, actual=actual_reward, details=details)
end

"""
    run_game_validation(num_games::Int; seed::Int=1, verbose::Bool=true)

Run full game validation with reward checking.
"""
function run_game_validation(num_games::Int; seed::Int=1, verbose::Bool=true)
    rng = MersenneTwister(seed)

    total_games = 0
    move_mismatches = 0
    reward_mismatches = 0
    total_moves = 0

    # Track reward distribution
    reward_counts = Dict{Int, Int}()

    start_time = time()

    for game_num in 1:num_games
        g = BackgammonNet.initial_state()
        game_moves = 0

        while !BackgammonNet.game_terminated(g)
            if BackgammonNet.is_chance_node(g)
                BackgammonNet.sample_chance!(g, rng)
            else
                d1, d2 = Int(g.dice[1]), Int(g.dice[2])
                is_doubles = d1 == d2
                is_turn_start = is_doubles ? (g.remaining_actions == 2) : (g.remaining_actions == 1)

                if is_turn_start
                    # Get Julia's legal final states
                    julia_states = get_julia_final_states(g)

                    # Get gnubg's legal final states
                    gnubg_states = get_gnubg_final_states(g)

                    if julia_states != gnubg_states
                        move_mismatches += 1
                        if verbose && move_mismatches <= 5
                            julia_only = setdiff(julia_states, gnubg_states)
                            gnubg_only = setdiff(gnubg_states, julia_states)
                            println("MOVE MISMATCH game $game_num move $game_moves: dice=($d1,$d2)")
                            println("  Julia: $(length(julia_states)) states, gnubg: $(length(gnubg_states)) states")
                            if !isempty(julia_only)
                                println("  Julia only: $(length(julia_only)) states")
                            end
                            if !isempty(gnubg_only)
                                println("  gnubg only: $(length(gnubg_only)) states")
                            end
                        end
                    end
                end

                # Play random move
                actions = BackgammonNet.legal_actions(g)
                action = actions[rand(rng, 1:length(actions))]
                BackgammonNet.apply_action!(g, action)
                game_moves += 1
            end
        end

        total_moves += game_moves
        total_games += 1

        # Validate reward
        reward_check = validate_game_rewards(g)
        reward_counts[g.reward] = get(reward_counts, g.reward, 0) + 1

        if !reward_check.valid
            reward_mismatches += 1
            if verbose && reward_mismatches <= 5
                println("REWARD MISMATCH game $game_num: expected=$(reward_check.expected) actual=$(reward_check.actual)")
                println("  $(reward_check.details)")
            end
        end

        if verbose && game_num % 100 == 0
            elapsed = time() - start_time
            rate = game_num / elapsed
            println("Progress: $game_num/$num_games ($(round(rate, digits=2)) games/sec) move_mismatches=$move_mismatches reward_mismatches=$reward_mismatches")
        end
    end

    elapsed = time() - start_time

    println()
    println("=" ^ 60)
    println("FULL GAME VALIDATION - $num_games games")
    println("=" ^ 60)
    println("Time: $(round(elapsed, digits=1))s ($(round(num_games/elapsed, digits=2)) games/sec)")
    println("Total moves: $total_moves (avg $(round(total_moves/num_games, digits=1)) per game)")
    println()
    println("Move mismatches: $move_mismatches")
    println("Reward mismatches: $reward_mismatches")
    println()
    println("Reward distribution:")
    for r in sort(collect(keys(reward_counts)))
        count = reward_counts[r]
        pct = round(100 * count / num_games, digits=1)
        label = if r == 1 "P0 normal" elseif r == 2 "P0 gammon" elseif r == 3 "P0 backgammon"
                elseif r == -1 "P1 normal" elseif r == -2 "P1 gammon" elseif r == -3 "P1 backgammon"
                else "unknown" end
        println("  $r ($label): $count ($pct%)")
    end

    if move_mismatches == 0 && reward_mismatches == 0
        println("\nAll games validated successfully!")
    else
        println("\nFound $(move_mismatches + reward_mismatches) total mismatches")
    end

    return (games=num_games, moves=total_moves, move_mismatches=move_mismatches,
            reward_mismatches=reward_mismatches, reward_distribution=reward_counts)
end

if abspath(PROGRAM_FILE) == @__FILE__
    num_games = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 100
    seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
    run_game_validation(num_games; seed=seed)
end
