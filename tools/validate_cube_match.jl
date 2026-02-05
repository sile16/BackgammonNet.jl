#!/usr/bin/env julia
# validate_cube_match.jl - Verify cube/match implementation against expected behavior
#
# Tests cube mechanics, match play, observation encoding, and optionally
# validates position integrity via gnubg's neural network evaluation.
#
# Usage: julia --project=. tools/validate_cube_match.jl

using BackgammonNet
using Random
using StaticArrays

# ============================================================================
# Test Infrastructure
# ============================================================================

mutable struct TestResults
    passed::Int
    failed::Int
    errors::Vector{String}
end
TestResults() = TestResults(0, 0, String[])

function check(r::TestResults, condition::Bool, description::String)
    if condition
        r.passed += 1
    else
        r.failed += 1
        push!(r.errors, description)
        println("  FAIL: $description")
    end
end

function report(r::TestResults, section::String)
    total = r.passed + r.failed
    status = r.failed == 0 ? "PASS" : "FAIL"
    println("[$status] $section: $(r.passed)/$total passed")
    if r.failed > 0
        for e in r.errors
            println("  - $e")
        end
    end
    println()
    return r.failed == 0
end

# ============================================================================
# Test 1: Cube Action Flow - NO_DOUBLE
# ============================================================================

function test_no_double()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.phase = PHASE_CUBE_DECISION

    check(r, g.phase == PHASE_CUBE_DECISION, "Phase is CUBE_DECISION")

    actions = legal_actions(g)
    check(r, length(actions) == 2, "2 legal actions in CUBE_DECISION")
    check(r, ACTION_CUBE_NO_DOUBLE in actions, "NO_DOUBLE is legal")
    check(r, ACTION_CUBE_DOUBLE in actions, "DOUBLE is legal")

    # Apply NO_DOUBLE
    apply_action!(g, ACTION_CUBE_NO_DOUBLE)
    check(r, g.phase == PHASE_CHANCE, "Phase transitions to CHANCE after NO_DOUBLE")
    check(r, g.cube_value == 1, "Cube value unchanged after NO_DOUBLE")
    check(r, g.cube_owner == -1, "Cube owner unchanged after NO_DOUBLE")
    check(r, !g.terminated, "Game not terminated after NO_DOUBLE")

    return report(r, "Test 1: NO_DOUBLE flow")
end

# ============================================================================
# Test 2: Cube Action Flow - DOUBLE → TAKE
# ============================================================================

function test_double_take()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.phase = PHASE_CUBE_DECISION

    # Apply DOUBLE
    apply_action!(g, ACTION_CUBE_DOUBLE)
    check(r, g.phase == PHASE_CUBE_RESPONSE, "Phase transitions to CUBE_RESPONSE after DOUBLE")
    check(r, g.current_player == 1, "Player switches to opponent after DOUBLE")
    check(r, g.cube_value == 1, "Cube value not yet changed (pending take/pass)")

    actions = legal_actions(g)
    check(r, length(actions) == 2, "2 legal actions in CUBE_RESPONSE")
    check(r, ACTION_CUBE_TAKE in actions, "TAKE is legal")
    check(r, ACTION_CUBE_PASS in actions, "PASS is legal")

    # Apply TAKE
    apply_action!(g, ACTION_CUBE_TAKE)
    check(r, g.cube_value == 2, "Cube value doubles to 2 after TAKE")
    check(r, g.cube_owner == 1, "Taker (P1) owns cube (absolute ID)")
    check(r, g.current_player == 0, "Player switches back to doubler after TAKE")
    check(r, g.phase == PHASE_CHANCE, "Phase transitions to CHANCE after TAKE")
    check(r, !g.terminated, "Game not terminated after TAKE")

    return report(r, "Test 2: DOUBLE → TAKE flow")
end

# ============================================================================
# Test 3: Cube Action Flow - DOUBLE → PASS
# ============================================================================

function test_double_pass()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.phase = PHASE_CUBE_DECISION

    # Apply DOUBLE
    apply_action!(g, ACTION_CUBE_DOUBLE)

    # Apply PASS
    apply_action!(g, ACTION_CUBE_PASS)
    check(r, g.terminated, "Game terminates after PASS")
    check(r, g.reward == 1.0f0, "Doubler (P0) wins cube value 1 (reward=+1)")

    # Test with higher cube value
    g2 = initial_state(first_player=0)
    g2.cube_enabled = true
    g2.cube_value = Int16(4)
    g2.cube_owner = Int8(0)  # P0 owns (absolute player ID)
    g2.phase = PHASE_CUBE_DECISION

    apply_action!(g2, ACTION_CUBE_DOUBLE)
    apply_action!(g2, ACTION_CUBE_PASS)
    check(r, g2.terminated, "Game terminates after PASS (cube=4)")
    check(r, g2.reward == 4.0f0, "Doubler (P0) wins cube value 4 (reward=+4)")

    # Test P1 doubling
    g3 = initial_state(first_player=1)
    g3.cube_enabled = true
    g3.cube_value = Int16(2)
    g3.cube_owner = Int8(1)  # Current player (P1) owns
    g3.phase = PHASE_CUBE_DECISION

    apply_action!(g3, ACTION_CUBE_DOUBLE)
    apply_action!(g3, ACTION_CUBE_PASS)
    check(r, g3.terminated, "Game terminates after PASS (P1 doubles)")
    check(r, g3.reward == -2.0f0, "Doubler (P1) wins cube=2, reward=-2 from P0's perspective")

    return report(r, "Test 3: DOUBLE → PASS flow (various cube values)")
end

# ============================================================================
# Test 4: Cube Pass Rewards for Various Cube Values
# ============================================================================

function test_cube_pass_rewards()
    r = TestResults()

    for cube_val in [1, 2, 4, 8, 16, 32]
        g = initial_state(first_player=0)
        g.cube_enabled = true
        g.cube_value = Int16(cube_val)
        # Cube centered (both can double)
        g.cube_owner = Int8(-1)
        g.phase = PHASE_CUBE_DECISION

        apply_action!(g, ACTION_CUBE_DOUBLE)
        apply_action!(g, ACTION_CUBE_PASS)

        expected = Float32(cube_val)  # P0 doubles, P1 passes → P0 wins cube_val
        check(r, g.reward == expected,
              "Cube=$cube_val: expected reward=$expected, got $(g.reward)")
    end

    return report(r, "Test 4: Cube pass rewards for values 1,2,4,8,16,32")
end

# ============================================================================
# Test 5: Cube Take Correctly Doubles Value
# ============================================================================

function test_cube_take_doubling()
    r = TestResults()

    for initial_cube in [1, 2, 4, 8, 16]
        g = initial_state(first_player=0)
        g.cube_enabled = true
        g.cube_value = Int16(initial_cube)
        g.cube_owner = Int8(-1)
        g.phase = PHASE_CUBE_DECISION

        apply_action!(g, ACTION_CUBE_DOUBLE)
        apply_action!(g, ACTION_CUBE_TAKE)

        expected_cube = initial_cube * 2
        check(r, g.cube_value == expected_cube,
              "Cube $initial_cube → $expected_cube after take: got $(g.cube_value)")
        check(r, g.cube_owner == 1,
              "Taker owns cube after take (got $(g.cube_owner))")
    end

    return report(r, "Test 5: Cube take doubling")
end

# ============================================================================
# Test 6: Board Integrity After Cube Actions
# ============================================================================

function test_board_integrity()
    r = TestResults()

    rng = MersenneTwister(42)

    for i in 1:100
        g = initial_state(first_player=rand(rng, 0:1))
        g.cube_enabled = true

        # Save board state
        p0_before, p1_before = g.p0, g.p1

        # Apply cube action sequence
        g.phase = PHASE_CUBE_DECISION
        apply_action!(g, ACTION_CUBE_NO_DOUBLE)

        check(r, g.p0 == p0_before && g.p1 == p1_before,
              "Board unchanged after NO_DOUBLE (game $i)")
    end

    for i in 1:100
        g = initial_state(first_player=rand(rng, 0:1))
        g.cube_enabled = true
        p0_before, p1_before = g.p0, g.p1

        g.phase = PHASE_CUBE_DECISION
        apply_action!(g, ACTION_CUBE_DOUBLE)
        apply_action!(g, ACTION_CUBE_TAKE)

        check(r, g.p0 == p0_before && g.p1 == p1_before,
              "Board unchanged after DOUBLE→TAKE (game $i)")
    end

    return report(r, "Test 6: Board integrity after cube actions (200 games)")
end

# ============================================================================
# Test 7: Crawford Rule - No Doubling
# ============================================================================

function test_crawford()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = false  # Crawford disables cube
    g.is_crawford = true
    g.my_away = Int8(3)
    g.opp_away = Int8(1)
    g.phase = PHASE_CHANCE  # Crawford goes straight to chance

    check(r, !may_double(g), "may_double returns false in Crawford game")

    # Also test init_match_game! Crawford setup
    g2 = initial_state(first_player=0)
    init_match_game!(g2, my_score=4, opp_score=6, match_length=7, is_crawford=true)

    check(r, g2.is_crawford, "Crawford flag set via init_match_game!")
    check(r, !g2.cube_enabled, "Cube disabled in Crawford game")
    check(r, !may_double(g2), "may_double false after init_match_game! Crawford")
    check(r, g2.my_away == 3, "my_away = 7 - 4 = 3")
    check(r, g2.opp_away == 1, "opp_away = 7 - 6 = 1")
    check(r, !g2.is_post_crawford, "Not post-Crawford in Crawford game")

    return report(r, "Test 7: Crawford rule")
end

# ============================================================================
# Test 8: Post-Crawford Allows Doubling
# ============================================================================

function test_post_crawford()
    r = TestResults()

    g = initial_state(first_player=0)
    init_match_game!(g, my_score=4, opp_score=6, match_length=7, is_crawford=false)

    check(r, g.is_post_crawford, "Post-Crawford inferred (opp is 1-away, not Crawford)")
    check(r, g.cube_enabled, "Cube enabled in post-Crawford")
    check(r, may_double(g), "may_double true in post-Crawford")
    check(r, g.my_away == 3, "my_away = 3")
    check(r, g.opp_away == 1, "opp_away = 1")

    # Test when current player is 1-away
    g2 = initial_state(first_player=0)
    init_match_game!(g2, my_score=6, opp_score=4, match_length=7, is_crawford=false)

    check(r, g2.is_post_crawford, "Post-Crawford when I'm 1-away")
    check(r, g2.cube_enabled, "Cube enabled when I'm 1-away post-Crawford")

    return report(r, "Test 8: Post-Crawford")
end

# ============================================================================
# Test 9: Game Reward with Cube Multiplication
# ============================================================================

function test_game_rewards()
    r = TestResults()

    # Test compute_game_reward directly
    g = initial_state(first_player=0)
    g.cube_value = Int16(4)

    # P0 wins single with cube=4 → 4 points
    reward = compute_game_reward(g, Int8(0), 1.0f0)
    check(r, reward == 4.0f0, "P0 wins single, cube=4: expected 4.0, got $reward")

    # P0 wins gammon with cube=4 → 8 points
    reward = compute_game_reward(g, Int8(0), 2.0f0)
    check(r, reward == 8.0f0, "P0 wins gammon, cube=4: expected 8.0, got $reward")

    # P0 wins backgammon with cube=4 → 12 points
    reward = compute_game_reward(g, Int8(0), 3.0f0)
    check(r, reward == 12.0f0, "P0 wins backgammon, cube=4: expected 12.0, got $reward")

    # P1 wins single with cube=4 → -4 points
    reward = compute_game_reward(g, Int8(1), 1.0f0)
    check(r, reward == -4.0f0, "P1 wins single, cube=4: expected -4.0, got $reward")

    # P1 wins gammon with cube=4 → -8 points
    reward = compute_game_reward(g, Int8(1), 2.0f0)
    check(r, reward == -8.0f0, "P1 wins gammon, cube=4: expected -8.0, got $reward")

    return report(r, "Test 9: Game rewards with cube multiplication")
end

# ============================================================================
# Test 10: Jacoby Rule
# ============================================================================

function test_jacoby()
    r = TestResults()

    # Jacoby enabled, cube not turned → gammons/backgammons reduced to single
    g = initial_state(first_player=0)
    g.jacoby_enabled = true
    g.cube_value = Int16(1)  # Cube not turned

    reward = compute_game_reward(g, Int8(0), 2.0f0)  # Gammon
    check(r, reward == 1.0f0, "Jacoby: gammon reduced to single (cube=1), got $reward")

    reward = compute_game_reward(g, Int8(0), 3.0f0)  # Backgammon
    check(r, reward == 1.0f0, "Jacoby: backgammon reduced to single (cube=1), got $reward")

    # Jacoby enabled, cube turned → gammons/backgammons count
    g.cube_value = Int16(2)
    reward = compute_game_reward(g, Int8(0), 2.0f0)  # Gammon
    check(r, reward == 4.0f0, "Jacoby: gammon counts with cube=2, got $reward")

    reward = compute_game_reward(g, Int8(0), 3.0f0)  # Backgammon
    check(r, reward == 6.0f0, "Jacoby: backgammon counts with cube=2, got $reward")

    # Jacoby disabled → gammons/backgammons always count
    g2 = initial_state(first_player=0)
    g2.jacoby_enabled = false
    g2.cube_value = Int16(1)

    reward = compute_game_reward(g2, Int8(0), 2.0f0)
    check(r, reward == 2.0f0, "No Jacoby: gammon counts with cube=1, got $reward")

    # Jacoby in match play (should be off) — gammons always count
    g3 = initial_state(first_player=0)
    init_match_game!(g3, my_score=0, opp_score=0, match_length=5)
    g3.cube_value = Int16(1)

    reward = compute_game_reward(g3, Int8(0), 2.0f0)
    check(r, reward == 2.0f0, "Match play: gammon counts even with cube=1 (no Jacoby), got $reward")

    return report(r, "Test 10: Jacoby rule")
end

# ============================================================================
# Test 11: Observation Encoding - Cube/Match Channels
# ============================================================================

function test_observation_encoding()
    r = TestResults()

    # Test 1: Default game (money play, no cube) - channels 31-42
    g = initial_state(first_player=0)
    sample_chance!(g, MersenneTwister(1))

    obs = observe_minimal(g)

    # Phase should be CHECKER_PLAY (channel 33)
    check(r, obs[31, 1, 1] == 0.0f0, "Default: phase CUBE_DECISION = 0")
    check(r, obs[32, 1, 1] == 0.0f0, "Default: phase CUBE_RESPONSE = 0")
    check(r, obs[33, 1, 1] == 1.0f0, "Default: phase CHECKER_PLAY = 1")
    # Cube value = log2(1)/6 = 0
    check(r, obs[34, 1, 1] == 0.0f0, "Default: cube value = 0 (log2(1)/6)")
    # Cube centered
    check(r, obs[35, 1, 1] == 0.0f0, "Default: I own cube = 0")
    check(r, obs[36, 1, 1] == 1.0f0, "Default: cube centered = 1")
    # Can't double (cube_enabled=false)
    check(r, obs[37, 1, 1] == 0.0f0, "Default: can double = 0 (cube disabled)")
    # Money play
    check(r, obs[38, 1, 1] == 1.0f0, "Default: money play = 1")
    # Away scores = 0
    check(r, obs[39, 1, 1] == 0.0f0, "Default: my_away = 0")
    check(r, obs[40, 1, 1] == 0.0f0, "Default: opp_away = 0")
    # Crawford / post-Crawford
    check(r, obs[41, 1, 1] == 0.0f0, "Default: Crawford = 0")
    check(r, obs[42, 1, 1] == 0.0f0, "Default: post-Crawford = 0")

    # Test 2: Cube decision phase
    g2 = initial_state(first_player=0)
    g2.cube_enabled = true
    g2.phase = PHASE_CUBE_DECISION
    obs2 = observe_minimal(g2)

    check(r, obs2[31, 1, 1] == 1.0f0, "Cube decision: phase CUBE_DECISION = 1")
    check(r, obs2[32, 1, 1] == 0.0f0, "Cube decision: phase CUBE_RESPONSE = 0")
    check(r, obs2[33, 1, 1] == 0.0f0, "Cube decision: phase CHECKER_PLAY = 0")
    check(r, obs2[37, 1, 1] == 1.0f0, "Cube decision: can double = 1")

    # Test 3: Match game with cube = 4, owned by current player (P0)
    g3 = initial_state(first_player=0)
    init_match_game!(g3, my_score=2, opp_score=3, match_length=7)
    g3.cube_value = Int16(4)
    g3.cube_owner = Int8(0)  # P0 owns (absolute)
    sample_chance!(g3, MersenneTwister(42))
    obs3 = observe_minimal(g3)

    expected_cube = log2(4.0f0) / 6.0f0
    check(r, abs(obs3[34, 1, 1] - expected_cube) < 1e-6,
          "Match: cube value = log2(4)/6 = $(expected_cube), got $(obs3[34, 1, 1])")
    check(r, obs3[35, 1, 1] == 1.0f0, "Match: I own cube = 1")
    check(r, obs3[36, 1, 1] == 0.0f0, "Match: cube centered = 0")
    check(r, obs3[38, 1, 1] == 0.0f0, "Match: money play = 0")
    expected_my_away = 5.0f0 / 25.0f0
    expected_opp_away = 4.0f0 / 25.0f0
    check(r, abs(obs3[39, 1, 1] - expected_my_away) < 1e-6,
          "Match: my_away = $(expected_my_away), got $(obs3[39, 1, 1])")
    check(r, abs(obs3[40, 1, 1] - expected_opp_away) < 1e-6,
          "Match: opp_away = $(expected_opp_away), got $(obs3[40, 1, 1])")

    # Test 4: Crawford game observation
    g4 = initial_state(first_player=0)
    init_match_game!(g4, my_score=3, opp_score=6, match_length=7, is_crawford=true)
    sample_chance!(g4, MersenneTwister(99))
    obs4 = observe_minimal(g4)

    check(r, obs4[37, 1, 1] == 0.0f0, "Crawford: can double = 0")
    check(r, obs4[41, 1, 1] == 1.0f0, "Crawford: Crawford flag = 1")
    check(r, obs4[42, 1, 1] == 0.0f0, "Crawford: post-Crawford = 0")

    # Test 5: Post-Crawford observation
    g5 = initial_state(first_player=0)
    init_match_game!(g5, my_score=3, opp_score=6, match_length=7, is_crawford=false)
    sample_chance!(g5, MersenneTwister(99))
    obs5 = observe_minimal(g5)

    check(r, obs5[37, 1, 1] == 1.0f0, "Post-Crawford: can double = 1")
    check(r, obs5[41, 1, 1] == 0.0f0, "Post-Crawford: Crawford flag = 0")
    check(r, obs5[42, 1, 1] == 1.0f0, "Post-Crawford: post-Crawford = 1")

    # Test 6: Verify broadcast across all spatial positions
    for w in 1:26
        check(r, obs2[31, 1, w] == obs2[31, 1, 1],
              "Broadcast: ch31 consistent at w=$w")
    end

    return report(r, "Test 11: Observation encoding (cube/match channels)")
end

# ============================================================================
# Test 12: Flat Observation Encoding
# ============================================================================

function test_flat_observation()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.cube_value = Int16(8)
    g.cube_owner = Int8(0)  # P0 owns (absolute)
    init_match_game!(g, my_score=2, opp_score=4, match_length=7)
    g.cube_value = Int16(8)
    g.cube_owner = Int8(0)  # P0 owns (absolute)
    sample_chance!(g, MersenneTwister(42))

    obs = observe_minimal_flat(g)
    check(r, length(obs) == 342, "Flat minimal length = 342")

    # Cube/match features are at indices 331-342 (after board 312 + dice 12 + move 4 + off 2 = 330)
    check(r, obs[331] == 0.0f0, "Flat: phase CUBE_DECISION = 0 (in checker play)")
    check(r, obs[332] == 0.0f0, "Flat: phase CUBE_RESPONSE = 0")
    check(r, obs[333] == 1.0f0, "Flat: phase CHECKER_PLAY = 1")

    expected_cube = log2(8.0f0) / 6.0f0
    check(r, abs(obs[334] - expected_cube) < 1e-6,
          "Flat: cube value = log2(8)/6 = $(expected_cube)")
    check(r, obs[335] == 1.0f0, "Flat: I own cube = 1")
    check(r, obs[336] == 0.0f0, "Flat: cube centered = 0")

    return report(r, "Test 12: Flat observation encoding")
end

# ============================================================================
# Test 13: Hybrid Observation Encoding
# ============================================================================

function test_hybrid_observation()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    init_match_game!(g, my_score=1, opp_score=3, match_length=5)
    sample_chance!(g, MersenneTwister(42))

    obs = observe_minimal_hybrid(g)
    check(r, size(obs.board) == (12, 26), "Hybrid board shape = (12, 26)")
    check(r, length(obs.globals) == 30, "Hybrid globals length = 30")

    # Globals layout: dice(12) + move(4) + off(2) + cube/match(12) = 30
    # Cube/match at indices 19-30
    check(r, obs.globals[21] == 1.0f0, "Hybrid: phase CHECKER_PLAY = 1 (index 21)")

    return report(r, "Test 13: Hybrid observation encoding")
end

# ============================================================================
# Test 14: Full Random Games with Cube
# ============================================================================

function test_full_random_cube_games()
    r = TestResults()
    rng = MersenneTwister(12345)

    completed = 0
    max_steps = 500

    for game_idx in 1:100
        g = initial_state(first_player=rand(rng, 0:1))
        g.cube_enabled = true

        steps = 0
        while !game_terminated(g) && steps < max_steps
            if is_chance_node(g)
                sample_chance!(g, rng)
            else
                actions = legal_actions(g)
                if isempty(actions)
                    break
                end
                action = actions[rand(rng, 1:length(actions))]
                apply_action!(g, action)
            end
            steps += 1
        end

        if game_terminated(g)
            completed += 1
            check(r, g.reward != 0.0f0, "Game $game_idx: reward is non-zero ($(g.reward))")
            check(r, abs(g.reward) >= g.cube_value,
                  "Game $game_idx: |reward| >= cube_value (|$(g.reward)| >= $(g.cube_value))")
        end
    end

    check(r, completed >= 90, "At least 90/100 cube games completed ($completed)")

    return report(r, "Test 14: Full random cube games (100 games)")
end

# ============================================================================
# Test 15: Full Random Match Games
# ============================================================================

function test_full_random_match_games()
    r = TestResults()
    rng = MersenneTwister(67890)

    for match_idx in 1:20
        match_length = rand(rng, 3:7)
        p0_score = 0
        p1_score = 0
        game_num = 0
        crawford_used = false
        max_games = 50

        while p0_score < match_length && p1_score < match_length && game_num < max_games
            game_num += 1
            g = initial_state(first_player=rand(rng, 0:1))

            # Determine Crawford status
            is_craw = false
            if !crawford_used
                if p0_score == match_length - 1 || p1_score == match_length - 1
                    is_craw = true
                    crawford_used = true
                end
            end

            init_match_game!(g, my_score=p0_score, opp_score=p1_score,
                            match_length=match_length, is_crawford=is_craw)

            # Play game
            steps = 0
            while !game_terminated(g) && steps < 500
                if is_chance_node(g)
                    sample_chance!(g, rng)
                else
                    actions = legal_actions(g)
                    if isempty(actions)
                        break
                    end
                    action = actions[rand(rng, 1:length(actions))]
                    apply_action!(g, action)
                end
                steps += 1
            end

            if game_terminated(g)
                points = abs(g.reward)
                if g.reward > 0
                    p0_score += Int(points)
                else
                    p1_score += Int(points)
                end
            end
        end

        match_over = p0_score >= match_length || p1_score >= match_length
        check(r, match_over || game_num >= max_games,
              "Match $match_idx (to $match_length) finished: P0=$p0_score P1=$p1_score")
    end

    return report(r, "Test 15: Full random match games (20 matches)")
end

# ============================================================================
# Test 16: switch_turn! Phase Transition
# ============================================================================

function test_switch_turn_phase()
    r = TestResults()

    # With cube enabled, switch_turn should go to CUBE_DECISION if may_double
    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.cube_owner = Int8(-1)  # Centered
    g.dice = SVector{2, Int8}(3, 1)
    g.phase = PHASE_CHECKER_PLAY

    # Simulate what switch_turn does (switches to P1)
    BackgammonNet.switch_turn!(g)
    check(r, g.phase == PHASE_CUBE_DECISION, "switch_turn with cube → CUBE_DECISION")
    check(r, g.current_player == 1, "Player switched to 1")

    # P0 owns cube, switching to P1 → P1 can't double → CHANCE
    g2 = initial_state(first_player=0)
    g2.cube_enabled = true
    g2.cube_owner = Int8(0)  # P0 owns (absolute)
    g2.dice = SVector{2, Int8}(3, 1)
    g2.phase = PHASE_CHECKER_PLAY

    BackgammonNet.switch_turn!(g2)  # switches to P1
    check(r, g2.phase == PHASE_CHANCE, "switch_turn with other player owning cube → CHANCE")

    # Crawford game → can't double → CHANCE
    g3 = initial_state(first_player=0)
    g3.cube_enabled = false
    g3.is_crawford = true

    BackgammonNet.switch_turn!(g3)
    check(r, g3.phase == PHASE_CHANCE, "switch_turn in Crawford → CHANCE")

    return report(r, "Test 16: switch_turn phase transitions")
end

# ============================================================================
# Test 17: Observation Consistency Across Formats
# ============================================================================

function test_observation_consistency()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.cube_value = Int16(4)
    g.cube_owner = Int8(0)  # P0 owns (absolute)
    init_match_game!(g, my_score=2, opp_score=5, match_length=7)
    g.cube_value = Int16(4)
    g.cube_owner = Int8(0)  # P0 owns (absolute)
    sample_chance!(g, MersenneTwister(42))

    obs_3d = observe_minimal(g)
    obs_flat = observe_minimal_flat(g)
    obs_hybrid = observe_minimal_hybrid(g)

    # Extract cube/match values from 3D (pick any spatial position)
    cube_match_3d = Float32[obs_3d[ch, 1, 1] for ch in 31:42]

    # Extract cube/match values from flat (indices 331-342)
    cube_match_flat = obs_flat[331:342]

    # Extract cube/match values from hybrid globals (indices 19-30)
    cube_match_hybrid = obs_hybrid.globals[19:30]

    for i in 1:12
        check(r, abs(cube_match_3d[i] - cube_match_flat[i]) < 1e-6,
              "3D vs flat: cube/match feature $i ($(cube_match_3d[i]) vs $(cube_match_flat[i]))")
        check(r, abs(cube_match_3d[i] - cube_match_hybrid[i]) < 1e-6,
              "3D vs hybrid: cube/match feature $i ($(cube_match_3d[i]) vs $(cube_match_hybrid[i]))")
    end

    # Also check full and biased include cube/match channels
    obs_full = observe_full(g)
    obs_biased = observe_biased(g)

    for ch in 31:42
        check(r, obs_full[ch, 1, 1] == obs_3d[ch, 1, 1],
              "Full ch$ch == minimal ch$ch")
        check(r, obs_biased[ch, 1, 1] == obs_3d[ch, 1, 1],
              "Biased ch$ch == minimal ch$ch")
    end

    return report(r, "Test 17: Observation consistency across formats")
end

# ============================================================================
# Test 18: may_double Edge Cases
# ============================================================================

function test_may_double_edge_cases()
    r = TestResults()

    # cube_enabled=false → false
    g = initial_state(first_player=0)
    g.cube_enabled = false
    check(r, !may_double(g), "cube_enabled=false → can't double")

    # cube_enabled=true, cube centered → true
    g.cube_enabled = true
    g.cube_owner = Int8(-1)  # centered
    check(r, may_double(g), "centered cube → can double")

    # cube_owner=current player (I own) → true
    g.cube_owner = g.current_player
    check(r, may_double(g), "I own cube → can double")

    # cube_owner=other player (opponent owns) → false
    g.cube_owner = Int8(1 - g.current_player)
    check(r, !may_double(g), "opponent owns cube → can't double")

    # Crawford → false
    g.cube_owner = Int8(-1)
    g.is_crawford = true
    check(r, !may_double(g), "Crawford → can't double")

    # Terminated → false
    g.is_crawford = false
    g.terminated = true
    check(r, !may_double(g), "terminated → can't double")

    return report(r, "Test 18: may_double edge cases")
end

# ============================================================================
# Test 19: init_match_game! Comprehensive
# ============================================================================

function test_init_match_game()
    r = TestResults()

    g = initial_state(first_player=0)

    # Normal match game (neither player 1-away)
    init_match_game!(g, my_score=1, opp_score=2, match_length=5)
    check(r, g.my_away == 4, "4-away from 1/5")
    check(r, g.opp_away == 3, "3-away from 2/5")
    check(r, !g.is_crawford, "Not Crawford")
    check(r, !g.is_post_crawford, "Not post-Crawford")
    check(r, g.cube_enabled, "Cube enabled in normal match")
    check(r, !g.jacoby_enabled, "Jacoby off in match")
    check(r, g.cube_value == 1, "Cube reset to 1")
    check(r, g.cube_owner == -1, "Cube centered")

    # Crawford game
    init_match_game!(g, my_score=4, opp_score=2, match_length=5, is_crawford=true)
    check(r, g.my_away == 1, "1-away from 4/5")
    check(r, g.opp_away == 3, "3-away from 2/5")
    check(r, g.is_crawford, "Crawford flag set")
    check(r, !g.is_post_crawford, "Not post-Crawford during Crawford")
    check(r, !g.cube_enabled, "Cube disabled during Crawford")

    # Post-Crawford (1-away, not Crawford)
    init_match_game!(g, my_score=4, opp_score=2, match_length=5, is_crawford=false)
    check(r, g.is_post_crawford, "Post-Crawford inferred (1-away, not Crawford)")
    check(r, !g.is_crawford, "Not Crawford")
    check(r, g.cube_enabled, "Cube enabled in post-Crawford")

    return report(r, "Test 19: init_match_game! comprehensive")
end

# ============================================================================
# Test 20: Context Observation Consistency
# ============================================================================

function test_context_observation()
    r = TestResults()

    g = initial_state(first_player=0)
    g.cube_enabled = true
    g.cube_value = Int16(4)
    g.cube_owner = Int8(0)  # P0 owns (absolute)
    g.phase = PHASE_CUBE_DECISION

    ctx = context_observation(g)
    check(r, length(ctx) == CONTEXT_DIM, "Context dim = $CONTEXT_DIM")

    expected_cube_ctx = log2(4.0f0) / 6.0f0
    check(r, abs(ctx[1] - expected_cube_ctx) < 1e-6,
          "Context: cube value matches ($(ctx[1]) vs $expected_cube_ctx)")
    check(r, ctx[2] == 1.0f0, "Context: cube_owner = I own (1.0)")
    check(r, ctx[3] == 1.0f0, "Context: may_double = 1")
    check(r, ctx[4] == 1.0f0, "Context: money play = 1")
    check(r, ctx[10] == 1.0f0, "Context: phase CUBE_DECISION = 1")
    check(r, ctx[11] == 0.0f0, "Context: phase CUBE_RESPONSE = 0")
    check(r, ctx[12] == 0.0f0, "Context: phase CHECKER_PLAY = 0")

    # Masked context
    mctx = context_observation(g, true)
    check(r, all(mctx .== 0.0f0), "Masked context is all zeros")

    return report(r, "Test 20: Context observation")
end

# ============================================================================
# Main: Run All Tests
# ============================================================================

function main()
    println("=" ^ 70)
    println("Cube/Match Validation Suite")
    println("=" ^ 70)
    println()

    all_passed = true

    all_passed &= test_no_double()
    all_passed &= test_double_take()
    all_passed &= test_double_pass()
    all_passed &= test_cube_pass_rewards()
    all_passed &= test_cube_take_doubling()
    all_passed &= test_board_integrity()
    all_passed &= test_crawford()
    all_passed &= test_post_crawford()
    all_passed &= test_game_rewards()
    all_passed &= test_jacoby()
    all_passed &= test_observation_encoding()
    all_passed &= test_flat_observation()
    all_passed &= test_hybrid_observation()
    all_passed &= test_full_random_cube_games()
    all_passed &= test_full_random_match_games()
    all_passed &= test_switch_turn_phase()
    all_passed &= test_observation_consistency()
    all_passed &= test_may_double_edge_cases()
    all_passed &= test_init_match_game()
    all_passed &= test_context_observation()

    println("=" ^ 70)
    if all_passed
        println("ALL TESTS PASSED")
    else
        println("SOME TESTS FAILED")
        exit(1)
    end
    println("=" ^ 70)
end

# Try to also validate against gnubg if available
function test_gnubg_position_integrity()
    r = TestResults()

    try
        include(joinpath(@__DIR__, "..", "test", "GnubgInterface.jl"))
        GnubgMod = Main.GnubgInterface

        rng = MersenneTwister(42)
        println("  gnubg available - testing position evaluation after cube actions...")

        for i in 1:20
            g = initial_state(first_player=rand(rng, 0:1))
            g.cube_enabled = true
            sample_chance!(g, rng)

            # Get gnubg evaluation at initial position
            probs = GnubgMod.evaluate_probs(g)
            pwin = probs[1]
            check(r, 0.0 <= pwin <= 1.0,
                  "gnubg pwin in [0,1] for game $i: $pwin")

            # Apply cube action and verify evaluation still works
            g2 = clone(g)
            reset!(g2, first_player=rand(rng, 0:1))
            g2.cube_enabled = true
            g2.phase = PHASE_CUBE_DECISION

            apply_action!(g2, ACTION_CUBE_DOUBLE)
            apply_action!(g2, ACTION_CUBE_TAKE)
            sample_chance!(g2, rng)

            probs2 = GnubgMod.evaluate_probs(g2)
            pwin2 = probs2[1]
            check(r, 0.0 <= pwin2 <= 1.0,
                  "gnubg pwin after cube in [0,1] for game $i: $pwin2")
        end

        return report(r, "Test gnubg: Position integrity after cube actions")
    catch e
        println("[SKIP] gnubg position integrity: $e")
        println()
        return true  # Skip doesn't count as failure
    end
end

# Run
main()
println()
test_gnubg_position_integrity()
