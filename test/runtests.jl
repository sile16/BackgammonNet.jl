using Test
using StaticArrays
using Random
using BackgammonNet

# Internal bitboard indices (for test setup only)
const IDX_P1_OFF = 0
const IDX_P0_OFF = 25
const IDX_P0_BAR = 26
const IDX_P1_BAR = 27

# Use exported constants from BackgammonNet
const PASS = PASS_LOC
const BAR = BAR_LOC

function make_test_game(; board=nothing, dice=(1, 2), remaining=1, current_player=0)
    # Parse Canonical Board Vector into Bitboards
    p0 = UInt128(0)
    p1 = UInt128(0)
    
    if board !== nothing
        cp = current_player
        
        # Helper to add checkers
        function add_chk(b, idx, count)
            if count <= 0; return b; end
            return b | (UInt128(count) << (idx << 2))
        end

        for i in 1:24
            val = board[i]
            if val == 0; continue; end
            
            if cp == 0
                # P0 is canonical 1..24
                # val > 0 -> P0. val < 0 -> P1.
                if val > 0
                    p0 = add_chk(p0, i, val)
                else
                    p1 = add_chk(p1, i, -val)
                end
            else
                # P1 is canonical 1..24 (Physical 24..1)
                # val > 0 -> P1. val < 0 -> P0.
                phys_idx = 25 - i
                if val > 0
                    p1 = add_chk(p1, phys_idx, val)
                else
                    p0 = add_chk(p0, phys_idx, -val)
                end
            end
        end
        
        # Bars & Offs
        # 25: My Bar, 26: Opp Bar, 27: My Off, 28: Opp Off
        if cp == 0
            p0 = add_chk(p0, IDX_P0_BAR, board[25])
            p1 = add_chk(p1, IDX_P1_BAR, -board[26])
            p0 = add_chk(p0, IDX_P0_OFF, board[27])
            p1 = add_chk(p1, IDX_P1_OFF, -board[28])
        else
            p1 = add_chk(p1, IDX_P1_BAR, board[25])
            p0 = add_chk(p0, IDX_P0_BAR, -board[26])
            p1 = add_chk(p1, IDX_P1_OFF, board[27])
            p0 = add_chk(p0, IDX_P0_OFF, -board[28])
        end
    end

    d = SVector{2, Int8}(dice[1], dice[2])
    # Use the short constructor which creates the buffers
    return BackgammonNet.BackgammonGame(
        p0, p1,
        d,
        Int8(remaining),
        Int8(current_player),
        false,
        0.0f0
    )
end

@testset "BackgammonNet Tests" begin

    @testset "Initialization" begin
        g = initial_state()
        @test is_chance_node(g)
        @test g.dice == [0, 0]
        sample_chance!(g)
        @test !is_chance_node(g)
        @test g[1] == 2
    end
    
    @testset "Stochastic Steps" begin
        g = initial_state()
        @test is_chance_node(g)
        
        # Manually set dice to (3, 4) -> index 13
        # DICE_OUTCOMES[13] is (3,4)
        apply_chance!(g, 13)
        @test g.dice == [3, 4]
        @test !is_chance_node(g)
        @test g.remaining_actions == 1
        
        # Doubles (2, 2) -> index 7
        g = initial_state()
        apply_chance!(g, 7)
        @test g.dice == [2, 2]
        @test g.remaining_actions == 2
        
        # Chance Outcomes
        outcomes = chance_outcomes(g)
        @test length(outcomes) == 21
        @test outcomes[1][1] == 1
        @test outcomes[1][2] ≈ 1/36 atol=1e-5
    end

    @testset "Deterministic State Guarantee" begin
        # sample_chance! must always return a non-chance node with valid moves
        for _ in 1:20
            g = initial_state()
            @test is_chance_node(g)
            sample_chance!(g)
            @test !is_chance_node(g)
            @test g.dice[1] > 0 && g.dice[2] > 0
            # Must have at least one valid move (not just PASS|PASS)
            actions = legal_actions(g)
            pass_pass = BackgammonNet.encode_action(PASS, PASS)
            @test length(actions) >= 1
            @test !(length(actions) == 1 && actions[1] == pass_pass)
        end

        # apply_chance! CAN return a state where only PASS|PASS is available
        # (This is the low-level API that doesn't auto-skip)
        b = zeros(MVector{28, Int8})
        b[1] = 1  # Single checker
        for i in 2:7; b[i] = -2; end  # Block all moves for dice 1-6
        g = make_test_game(board=b, dice=(0, 0), current_player=0)  # Chance node
        g.dice = SVector{2, Int8}(0, 0)  # Force chance node
        # Manually apply chance (simulating apply_chance! behavior)
        g.dice = SVector{2, Int8}(1, 2)
        g.remaining_actions = 1
        @test !is_chance_node(g)
        actions = legal_actions(g)
        pass_pass = BackgammonNet.encode_action(PASS, PASS)
        @test length(actions) == 1
        @test actions[1] == pass_pass

        # sample_chance! auto-applies PASS|PASS and rolls for next player
        b = zeros(MVector{28, Int8})
        b[1] = 1  # P0 single checker at 1
        for i in 2:7; b[i] = -2; end  # Block all P0 moves
        b[24] = -1  # P1 checker at 24 (can move)
        g = make_test_game(board=b, dice=(0, 0), current_player=0)
        g.dice = SVector{2, Int8}(0, 0)  # Force chance node
        sample_chance!(g)
        # Should have auto-skipped P0's blocked turn and rolled for P1
        # Either P0 got a lucky roll that works, or P1 is now playing
        if !game_terminated(g)
            @test !is_chance_node(g)
            actions = legal_actions(g)
            pass_pass = BackgammonNet.encode_action(PASS, PASS)
            # Must have valid moves (not just PASS|PASS)
            @test !(length(actions) == 1 && actions[1] == pass_pass)
        end

        # step! must always return a non-chance node (auto-rolls dice)
        g = initial_state(first_player=0)
        sample_chance!(g)
        for i in 1:50
            if game_terminated(g)
                break
            end
            @test !is_chance_node(g)  # Before step, should be deterministic
            actions = legal_actions(g)
            step!(g, actions[1])
            if !game_terminated(g)
                @test !is_chance_node(g)  # After step, should still be deterministic
                @test g.dice[1] > 0 && g.dice[2] > 0
            end
        end

        # step! with doubles should still return deterministic (even with remaining_actions)
        b = zeros(MVector{28, Int8})
        b[1] = 4  # 4 checkers at point 1
        g = make_test_game(board=b, dice=(3, 3), remaining=2, current_player=0)
        @test !is_chance_node(g)
        @test g.remaining_actions == 2

        # First step with doubles - still same turn
        actions = legal_actions(g)
        step!(g, actions[1])
        if !game_terminated(g)
            # After doubles first action, could be remaining_actions=1 (same turn)
            # or turn switched with new dice
            @test !is_chance_node(g)  # Must be deterministic either way
        end

        # Multiple consecutive games - all step! calls return deterministic
        for _ in 1:5
            g = initial_state()
            sample_chance!(g)
            moves = 0
            while !game_terminated(g) && moves < 200
                @test !is_chance_node(g)
                actions = legal_actions(g)
                step!(g, actions[rand(1:length(actions))])
                moves += 1
            end
            # Game should have terminated or hit max moves
            @test game_terminated(g) || moves == 200
        end
    end

    @testset "Action Encoding" begin
        # Encoding: (loc1 * 26) + loc2 + 1
        # Bar=0, Points=1-24, Pass=25
        
        # Pass | Pass
        @test BackgammonNet.encode_action(PASS, PASS) == (25 * 26) + 25 + 1
        
        # Bar | Pass
        @test BackgammonNet.encode_action(BAR, PASS) == (0 * 26) + 25 + 1
        
        # 1 | 2
        @test BackgammonNet.encode_action(1, 2) == (1 * 26) + 2 + 1
        
        @test BackgammonNet.decode_action(BackgammonNet.encode_action(1, 2)) == (1, 2)
    end
    
    @testset "Legal Actions: Non-Doubles" begin
        b = zeros(MVector{28, Int8})
        b[25] = 1 # Bar
        b[1] = 14 # Point 1
        b[2] = -2 # Block 2 (1+1)
        b[7] = -2 # Block 7 (1+6) and (6+1)
        
        g = make_test_game(board=b, dice=(1, 6))
        
        actions = legal_actions(g)
        
        # Expected: Pass D1, Bar->6 (D2)? 
        # Or Bar->1 (D1), Pass D2?
        # Maximize dice? 6 > 1.
        # Should prefer using D2 (6).
        # So Pass(D1), Bar(D2). -> (PASS, BAR) = (25, 0).
        
        a_pass_bar = BackgammonNet.encode_action(PASS, BAR)
        
        @test length(actions) == 1
        @test actions[1] == a_pass_bar
    end
    
    @testset "Legal Actions: Doubles" begin
        b = zeros(MVector{28, Int8})
        b[6] = 1
        b[8] = 1

        g = make_test_game(board=b, dice=(2, 2), remaining=2)

        actions = legal_actions(g)

        # 6->8 (using 2). 8->10 (using 2).
        a1 = BackgammonNet.encode_action(6, 8)
        # 8->10 (using 2). 6->8 (using 2).
        a2 = BackgammonNet.encode_action(8, 6)
        # 8->10, 10->12?
        a3 = BackgammonNet.encode_action(8, 10)

        @test a1 in actions
        @test a2 in actions
        @test a3 in actions

        # For doubles with single-die usage, legal_actions generates loc|PASS, not PASS|loc
        # is_action_valid should match this behavior
        b2 = zeros(MVector{28, Int8})
        b2[1] = 1   # Single checker at point 1
        b2[5] = -2  # Block at 5 (blocks second die: 1->3, then 3->5 blocked)
        g2 = make_test_game(board=b2, dice=(2, 2), remaining=2, current_player=0)
        actions2 = legal_actions(g2)

        # Should generate (1, PASS) since 1->3 works but 3->5 is blocked
        a_valid = BackgammonNet.encode_action(1, PASS)
        a_invalid = BackgammonNet.encode_action(PASS, 1)  # PASS|1 not in valid space

        @test a_valid in actions2
        @test !(a_invalid in actions2)
        # is_action_valid should agree with legal_actions
        @test is_action_valid(g2, a_valid)
        @test !is_action_valid(g2, a_invalid)
    end

    @testset "Higher Die Rule" begin
        # Rule: When only one die can be used, must use the higher die

        # Scenario 1: Single checker, both dice blocked differently
        # Checker at 5, block at 6 (blocks d1=1), block at 8 (blocks d2=3)
        # Only d2=3 can play (5->8 blocked, but 5->6 blocked too)
        # Actually need: one die works, other doesn't
        b = zeros(MVector{28, Int8})
        b[5] = 1   # Checker at 5
        b[6] = -2  # Block at 6 (blocks die 1: 5+1=6)
        b[10] = -2 # Block at 10 (blocks die 5: 5+5=10)
        g = make_test_game(board=b, dice=(1, 5), current_player=0)
        actions = legal_actions(g)
        # Die 1 blocked (5->6), die 5 blocked (5->10)
        # Neither can play, so PASS|PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, PASS)

        # Scenario 2: One die blocked, other works - must use the working one
        # Need to block both the first die AND subsequent moves after using second die
        b = zeros(MVector{28, Int8})
        b[5] = 1   # Checker at 5
        b[6] = -2  # Block at 6 (blocks die 1: 5+1=6)
        b[11] = -2 # Block at 11 (blocks 10+1 after 5->10)
        # Die 5: 5->10 is open, but 10->11 blocked, so only one die usable
        g = make_test_game(board=b, dice=(1, 5), current_player=0)
        actions = legal_actions(g)
        # Die 1: 5->6 blocked
        # Die 5: 5->10 open, then die 1: 10->11 blocked
        # Only die 5 can be used (higher die anyway)
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 5)  # PASS d1, use d2 from 5

        # Scenario 3: Only one die works due to blocking - must use higher die
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at 1
        b[3] = -2  # Block at 3 (blocks die 2: 1+2=3)
        b[7] = -2  # Block at 7 (blocks 5+2 after using die 4)
        g = make_test_game(board=b, dice=(2, 4), current_player=0)
        actions = legal_actions(g)
        # Die 2: 1->3 blocked. Die 4: 1->5 open, then 5->7 blocked.
        # Only die 4 works (higher), so must use it
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 1)  # PASS d1=2, use d2=4 from point 1

        # Scenario 4: Both dice blocked - PASS|PASS
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[3] = -2  # Block 1+2=3
        b[4] = -2  # Block 1+3=4
        b[6] = -2  # Block 3+3=6 (if we could get to 3)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Die 2: 1->3 blocked
        # Die 3: 1->4 blocked
        # Neither die can play!
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, PASS)

        # Scenario 5: Higher die rule - die 3 works, die 2 doesn't, must use 3
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[3] = -2  # Block at 3 (blocks die 2: 1+2=3)
        b[6] = -2  # Block at 6 (blocks 4+2 after using die 3)
        # Die 3: 1->4 open, then die 2: 4->6 blocked
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Only die 3 works (higher), so must use it
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 1)  # PASS d1=2, use d2=3 from 1

        # Scenario 6: Lower die works, higher doesn't - must use lower (only option)
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[4] = -2  # Block at 4 (blocks die 3: 1+3=4)
        b[6] = -2  # Block at 6 (blocks 3+3 after using die 2)
        # Die 2: 1->3 open, then die 3: 3->6 blocked
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Only die 2 works (lower), so must use it
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(1, PASS)  # use d1=2 from 1, PASS d2=3

        # Scenario 7: Only lower die works (all paths with higher die blocked)
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[4] = -2  # Block at 4 (blocks die 3: 1+3=4)
        b[5] = -2  # Block at 5
        b[6] = -2  # Block at 6 (blocks 3+3=6 after 1->3)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Die 2: 1->3 open, then die 3: 3->6 blocked. Die 3: 1->4 blocked.
        # Only die 2 works, must use lower die as only option
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(1, PASS)

        # Scenario 8: Bar entry - higher die must be used
        b = zeros(MVector{28, Int8})
        b[25] = 1  # On bar
        b[2] = -2  # Block entry point 2 (blocks die 2)
        b[5] = -2  # Block entry point 5 (blocks die 5)
        g = make_test_game(board=b, dice=(2, 5), current_player=0)
        actions = legal_actions(g)
        # Neither entry works
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, PASS)

        # Scenario 9: Bar entry - one blocked, use the other
        b = zeros(MVector{28, Int8})
        b[25] = 1  # On bar
        b[2] = -2  # Block entry point 2 (blocks die 2)
        b[7] = -2  # Block at 7 (blocks 5+2 after bar entry)
        # Entry point 5 open, but subsequent move blocked
        g = make_test_game(board=b, dice=(2, 5), current_player=0)
        actions = legal_actions(g)
        # Die 5 can enter (Bar->5), then die 2: 5->7 blocked
        # Only one die usable (die 5, which is higher)
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, BAR)  # PASS d1=2, Bar with d2=5

        # Scenario 10: Bearing off - must use higher die if only one works
        b = zeros(MVector{28, Int8})
        b[24] = 1  # Single checker at 24 (in home)
        b[27] = 14 # 14 already off
        g = make_test_game(board=b, dice=(1, 3), current_player=0)
        actions = legal_actions(g)
        # Die 1: 24+1=25=off (exact)
        # Die 3: 24+3=27=off (over-bear, valid since 24 is highest)
        # Both work! So must use both... but only one checker!
        # With one checker, can only use one die. Must use higher (3).
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 24)  # PASS d1=1, use d2=3
    end

    @testset "Maximize Dice Rule" begin
        # Rule: Must use both dice if possible

        # Scenario 1: Two checkers, can use both dice independently
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at 1
        b[5] = 1   # Checker at 5
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Can use d1=2 on point 1 (1->3) and d2=3 on point 5 (5->8)
        # Or d1=2 on point 5 (5->7) and d2=3 on point 1 (1->4)
        # All actions should use both dice
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            @test l1 != PASS && l2 != PASS  # Both dice used
        end
        @test length(actions) >= 2

        # Scenario 2: Single checker, sequential moves use both dice
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # 1->3 (d1=2), then 3->6 (d2=3)
        # 1->4 (d2=3), then 4->6 (d1=2)
        # Both paths end at 6, but action encoding differs
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            @test l1 != PASS && l2 != PASS  # Both dice must be used
        end

        # Scenario 3: First move enables second (blocked initially)
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at 1
        b[3] = -1  # Opponent blot at 3 (can be hit)
        b[6] = -2  # Block at 6
        g = make_test_game(board=b, dice=(2, 4), current_player=0)
        actions = legal_actions(g)
        # d1=2: 1->3 (hit)
        # d2=4: 1->5 open, or after hit: 3->7 open
        # Must use both dice
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            @test l1 != PASS && l2 != PASS
        end

        # Scenario 4: Two checkers, one path uses both, other doesn't
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at 1
        b[10] = 1  # Checker at 10
        b[3] = -2  # Block at 3 (blocks 1+2)
        b[13] = -2 # Block at 13 (blocks 10+3)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # d1=2 from 1: 1->3 blocked
        # d1=2 from 10: 10->12 open
        # d2=3 from 1: 1->4 open
        # d2=3 from 10: 10->13 blocked
        # Can we use both? 10->12 (d1), then 1->4 (d2)? Yes!
        # Or 1->4 (d2), then 10->12 (d1)? Yes!
        a_both = BackgammonNet.encode_action(10, 1)  # d1 from 10, d2 from 1
        @test a_both in actions
        # Single die actions should NOT be in results
        a_single1 = BackgammonNet.encode_action(10, PASS)
        a_single2 = BackgammonNet.encode_action(PASS, 1)
        @test !(a_single1 in actions)
        @test !(a_single2 in actions)

        # Scenario 5: Doubles - must use all 4 sub-moves if possible
        b = zeros(MVector{28, Int8})
        b[1] = 4   # 4 checkers at point 1
        g = make_test_game(board=b, dice=(2, 2), remaining=2, current_player=0)
        actions = legal_actions(g)
        # With doubles, each action uses 2 of the 4 moves
        # All actions should move 2 checkers
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            @test l1 != PASS && l2 != PASS
        end

        # Scenario 6: Can only use one die, no maximize violation
        b = zeros(MVector{28, Int8})
        b[1] = 1
        b[3] = -2  # Block 1+2
        b[4] = -2  # Block 1+3
        b[6] = -2  # Block 3+3 (if we could reach 3)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Both individual moves blocked, PASS|PASS is valid
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, PASS)

        # Scenario 7: Bearing off - must bear off with both if possible
        b = zeros(MVector{28, Int8})
        b[23] = 1  # Checker at 23
        b[24] = 1  # Checker at 24
        b[27] = 13 # 13 already off (need 15 total)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # d1=2: 23->25=off, 24->26=off (overbear)
        # d2=3: 23->26=off (overbear), 24->27=off (overbear)
        # Can bear off both!
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            @test l1 != PASS && l2 != PASS
        end

        # Scenario 8: Hit enables bearing off
        b = zeros(MVector{28, Int8})
        b[19] = 1  # Checker at 19 (in home)
        b[22] = -1 # Opponent blot at 22
        b[27] = 14 # 14 already off
        g = make_test_game(board=b, dice=(3, 6), current_player=0)
        actions = legal_actions(g)
        # d1=3: 19->22 (hit opponent)
        # d2=6: after hit, 22->28=off (overbear from 22)
        # Or d2=6 first: 19->25=off, then d1=3 can't play (no checker)
        # So 19->22, then 22->off uses both dice
        for a in actions
            l1, l2 = BackgammonNet.decode_action(a)
            # At least one action should use both
        end
        @test length(actions) >= 1
    end

    @testset "Step" begin
        b = zeros(MVector{28, Int8})
        b[13] = 2      # P0 has 2 pieces at point 13
        b[12] = -2     # P1 has 2 pieces at point 12 (so P1 has valid moves)

        g = make_test_game(board=b, dice=(3, 4))
        
        # 13->16 (D=3), 13->17 (D=4)
        # Action (13, 13)
        act = BackgammonNet.encode_action(13, 13)
        BackgammonNet.step!(g, act)
        
        # Turn switched.
        # Original moves: 13->16, 13->17.
        # g[i] access is Canonical.
        # Flipped indices: 25-16=9, 25-17=8.
        # Values negated (Canonical for Opponent).
        
        @test g[8] == -1
        @test g[9] == -1
        @test !is_chance_node(g)
        @test g.current_player == 1
    end

    @testset "Game Rules" begin
        
        @testset "Bar Entry & Blocking" begin
            b = zeros(MVector{28, Int8})
            b[25] = 1 # Bar
            b[10] = 1
            b[3] = -2 # Blocked
            b[4] = -1 # Blot (Hit)
            
            g = make_test_game(board=b, dice=(3, 4))
            
            actions = legal_actions(g)
            
            # Dice 3, 4.
            # D1(3): Bar->3 (Blocked).
            # D2(4): Bar->4 (Hit).
            
            # Path A (D1 then D2):
            # D1 blocked from Bar.
            # Path B (D2 then D1):
            # Bar->4 (Hit). Rem D1(3).
            # From new state (Bar empty).
            # Can move 4->7? (4+3=7).
            # Can move 10->13? (10+3=13).
            
            # Actions: (4, BAR). 
            # Note: 4 is source for D1? No.
            # Encode is (S1, S2). S1 uses D1. S2 uses D2.
            # We used D2 for Bar. So S2=BAR.
            # S1 used for other move.
            # If we move 4->7 using D1? S1=4. -> (4, BAR).
            # If we move 10->13 using D1? S1=10. -> (10, BAR).
            
            a1 = BackgammonNet.encode_action(4, BAR) 
            a2 = BackgammonNet.encode_action(10, BAR) 
            
            @test length(actions) > 0
            for a in actions
                l1, l2 = BackgammonNet.decode_action(a)
                @test l2 == BAR
            end
            
            @test a1 in actions
            @test a2 in actions
        end
        
        @testset "Bearing Off: Exact & Over" begin
            b = zeros(MVector{28, Int8})
            b[20] = 1  # P0's 5-point (further from off)
            b[24] = 1  # P0's 1-point (closer to off)

            g = make_test_game(board=b, dice=(6, 1))

            actions = legal_actions(g)

            # D1=6, D2=1.
            # Over-bear rule: Can only over-bear from the HIGHEST backgammon point (furthest from off).
            # For P0: physical 20 is the 5-point (higher BG point), physical 24 is the 1-point (lower).
            #
            # Individual moves:
            # 20->Off (d1=6): Over-bear from 5-point. Check 19: none. LEGAL.
            # 24->Off (d1=6): Over-bear from 1-point. Check 19-23: checker at 20! BLOCKED.
            # 24->Off (d2=1): Exact bear-off. LEGAL.
            # 20->21 (d2=1): Normal move. LEGAL.
            #
            # Valid two-die combos:
            # (20, 24): 20->off(d1=6), then 24->off(d2=1). VALID.
            a_20_24 = BackgammonNet.encode_action(20, 24)

            # (24, 20) encoding means d1=6 from 24, d2=1 from 20.
            # - Try d1 first: 24->off(d1=6) blocked (20 exists at higher BG point).
            # - Try d2 first: 20->21(d2=1), then 24->off(d1=6). After move, checkers at 21 and 24.
            #   24->off(d1=6) still blocked (21 exists at higher BG point). INVALID.
            a_24_20 = BackgammonNet.encode_action(24, 20)

            # (21, 20): 20->21(d2=1), then 21->off(d1=6).
            # After 20->21, checkers at 21 and 24. 21->off(d1=6) checks 19-20: none. LEGAL.
            a_21_20 = BackgammonNet.encode_action(21, 20)

            @test a_20_24 in actions
            @test !(a_24_20 in actions)  # BLOCKED: can't bear off 24 while 20 (or 21 after move) exists
            @test a_21_20 in actions     # VALID: move 20->21, then 21 can over-bear
        end

        @testset "Over-Bear Rule: P0 Highest Point Allowed" begin
            # P0 has checker on the 1-point (physical 24) - the LOWEST backgammon point.
            # Since it's the only checker, it IS the highest occupied, so over-bear is allowed.
            b = zeros(MVector{28, Int8})
            b[24] = 1  # Single checker at physical 24 (P0's 1-point, closest to off)
            b[27] = 14 # 14 off

            g = make_test_game(board=b, dice=(6, 5), current_player=0)

            actions = legal_actions(g)

            # 24->off with die 6: over-bear by 5, but no checkers at 19-23. LEGAL.
            # 24->off with die 5: over-bear by 4, but no checkers at 19-23. LEGAL.
            # Must use higher die (6) since only one checker.
            # Higher die rule: must use d1=6 if possible.
            a_24_pass = BackgammonNet.encode_action(24, PASS)  # use d1=6, PASS d2=5

            @test a_24_pass in actions
        end

        @testset "Over-Bear Rule: P0 Non-Highest Blocked" begin
            # P0 has checkers at physical 20 (5-point) and 23 (2-point).
            # Physical 20 is the HIGHER backgammon point (further from off).
            # Over-bear from 23 is blocked because 20 is occupied (higher backgammon point).
            b = zeros(MVector{28, Int8})
            b[20] = 1  # Checker at physical 20 (P0's 5-point, further from off)
            b[23] = 1  # Checker at physical 23 (P0's 2-point, closer to off)
            b[27] = 13 # 13 off

            g = make_test_game(board=b, dice=(6, 6), remaining=2, current_player=0)

            actions = legal_actions(g)

            # 20->off with die 6: over-bear from 5-point. Check 19: none. LEGAL.
            # 23->off with die 6: over-bear from 2-point. Check 19-22: checker at 20! BLOCKED.
            # With doubles, action is (s1, s2) both using die value 6.
            a_23_20 = BackgammonNet.encode_action(23, 20)  # 23->off blocked! Can't start with 23.
            a_20_23 = BackgammonNet.encode_action(20, 23)  # 20->off first, then 23->off. VALID.

            @test !(a_23_20 in actions)  # Can't start with 23->off (20 is higher)
            @test a_20_23 in actions     # Must do 20 first, then 23 works
        end

        @testset "Over-Bear Rule: P1 Highest Point Allowed" begin
            # P1 moves 24->1->off. P1's home is physical 1-6.
            # P1's 1-point is physical 1 (closest to off), 6-point is physical 6 (furthest).
            # "Higher backgammon point" for P1 = higher physical index (further from off).
            b = zeros(MVector{28, Int8})
            b[24] = 1  # P1's checker at canonical 24 (physical 1, P1's 1-point, closest to off)
            b[27] = 14 # 14 off

            g = make_test_game(board=b, dice=(6, 5), current_player=1)

            actions = legal_actions(g)

            # P1's canonical 24 = physical 1. With die 6: physical 1-6 = -5 < 0, over-bear.
            # Check for higher backgammon points (physical 2-6): none. LEGAL.
            a_24_pass = BackgammonNet.encode_action(24, PASS)

            @test a_24_pass in actions
        end

        @testset "Over-Bear Rule: P1 Non-Highest Blocked" begin
            # P1's home is physical 1-6. Physical 1 is P1's 1-point (closest to off),
            # physical 6 is P1's 6-point (furthest from off).
            # "Higher backgammon point" for P1 = higher physical index.
            b = zeros(MVector{28, Int8})
            b[24] = 1  # P1 canonical 24 = physical 1 (P1's 1-point, closest to off)
            b[21] = 1  # P1 canonical 21 = physical 4 (P1's 4-point, further from off)

            g = make_test_game(board=b, dice=(6, 6), remaining=2, current_player=1)

            actions = legal_actions(g)

            # From physical 1 with die 6: 1-6 = -5, over-bear.
            # Check for higher backgammon points (physical 2-6): checker at 4! BLOCKED.
            # From physical 4 with die 6: 4-6 = -2, over-bear.
            # Check for higher backgammon points (physical 5-6): none. LEGAL.

            # Canonical: 24->off blocked (21 at physical 4 is "higher" backgammon point)
            # Canonical: 21->off first, then 24->off (physical 4 gone, physical 1 now highest). VALID.
            a_24_21 = BackgammonNet.encode_action(24, 21)  # 24 (physical 1) first - blocked
            a_21_24 = BackgammonNet.encode_action(21, 24)  # 21 (physical 4) first, then 24. VALID.

            @test !(a_24_21 in actions)  # Can't start with physical 1 (checker at physical 4)
            @test a_21_24 in actions     # Must do physical 4 first, then physical 1 works
        end

        @testset "Over-Bear Rule: Exact Bear-Off Not Affected" begin
            # Exact bear-off (not over-bear) should always be allowed
            b = zeros(MVector{28, Int8})
            b[19] = 1  # Checker at physical 19 (P0's 6-point, furthest from off)
            b[24] = 1  # Checker at physical 24 (P0's 1-point, closest to off)
            b[27] = 13

            g = make_test_game(board=b, dice=(6, 1), current_player=0)

            actions = legal_actions(g)

            # 19->25 with die 6: 19+6=25, EXACT bear-off. Always legal.
            # 24->25 with die 1: 24+1=25, EXACT bear-off. Always legal.
            # Both are exact, so the over-bear rule doesn't apply.
            a_19_24 = BackgammonNet.encode_action(19, 24)

            @test a_19_24 in actions
        end

        @testset "Winning Condition" begin
            b = zeros(MVector{28, Int8})
            b[27] = 14 # Off
            b[24] = 1
            b[28] = -1 # Opponent has one off to avoid Gammon

            g = make_test_game(board=b, dice=(1, 2))

            # Only one checker, must use higher die (2) to bear off
            # 24->Off (2). Pass (1).
            act = BackgammonNet.encode_action(PASS, 24)

            BackgammonNet.step!(g, act)

            @test g[27] == 15
            @test g.terminated == true
            @test g.reward == 1.0f0
        end

        @testset "Hitting" begin
            b = zeros(MVector{28, Int8})
            b[1] = 1
            b[2] = -1 # Opponent Blot

            g = make_test_game(board=b, dice=(1, 6))

            # Both dice can be used: 1->2 (hit with D1), then 2->8 (D2)
            # Action encodes sources for each die: (1, 2)
            act = BackgammonNet.encode_action(1, 2)

            BackgammonNet.step!(g, act)

            # Opponent was hit, now on bar. Turn switched.
            @test g[25] == 1  # Opponent (now current player) has 1 on bar
            # My checker moved 1->2->8, from opponent view that's 25-8=17
            @test g[17] == -1
        end
    end

    @testset "Full Random Game" begin
        # Use seeded RNG for reproducibility
        rng = Random.MersenneTwister(42)
        g = initial_state(first_player=0)
        sample_chance!(g, rng) # Start with rolled dice

        step_count = 0
        max_steps = 10000

        while !game_terminated(g) && step_count < max_steps
            actions = legal_actions(g)
            if isempty(actions)
                break
            end

            a = actions[rand(rng, 1:length(actions))]
            step!(g, a, rng)

            step_count += 1
        end

        @test game_terminated(g)
        @test step_count < max_steps
        @test g.reward != 0.0f0
    end

    @testset "Reset" begin
        g = initial_state()
        sample_chance!(g)
        apply_action!(g, legal_actions(g)[1])
        @test length(g.history) > 0

        reset!(g)
        @test length(g.history) == 0
        @test is_chance_node(g)
    end

    @testset "First Player Selection" begin
        # Test initial_state with explicit first_player
        g0 = initial_state(first_player=0)
        @test g0.current_player == 0

        g1 = initial_state(first_player=1)
        @test g1.current_player == 1

        # Test reset! with explicit first_player
        reset!(g0, first_player=1)
        @test g0.current_player == 1

        reset!(g1, first_player=0)
        @test g1.current_player == 0

        # Test default (random) still works
        g = initial_state()
        @test g.current_player in [0, 1]

        # Test invalid first_player values are rejected
        @test_throws ArgumentError initial_state(first_player=2)
        @test_throws ArgumentError initial_state(first_player=-1)
        @test_throws ArgumentError initial_state(first_player=100)

        g = initial_state(first_player=0)
        @test_throws ArgumentError reset!(g, first_player=2)
        @test_throws ArgumentError reset!(g, first_player=-1)
    end

    @testset "Short Game Mode" begin
        # Test initial_state with short_game
        g = initial_state(short_game=true, first_player=0)
        @test is_chance_node(g)

        # Verify short game board position for P0 (from canonical perspective)
        # P0 at indices: 4(2), 12(1), 15(2), 16(3), 18(3), 21(3), 22(1)
        sample_chance!(g)
        @test g[4] == 2   # P0 has 2 on point 4
        @test g[12] == 1  # P0 has 1 on point 12
        @test g[15] == 2  # P0 has 2 on point 15
        @test g[16] == 3  # P0 has 3 on point 16
        @test g[18] == 3  # P0 has 3 on point 18
        @test g[21] == 3  # P0 has 3 on point 21
        @test g[22] == 1  # P0 has 1 on point 22

        # Verify P1 positions (negative from P0 perspective)
        # P1 at indices: 1(1), 2(3), 5(3), 7(3), 8(2), 11(1), 19(2)
        @test g[1] == -1  # P1 has 1 on point 1
        @test g[2] == -3  # P1 has 3 on point 2
        @test g[5] == -3  # P1 has 3 on point 5
        @test g[7] == -3  # P1 has 3 on point 7
        @test g[8] == -2  # P1 has 2 on point 8
        @test g[11] == -1 # P1 has 1 on point 11
        @test g[19] == -2 # P1 has 2 on point 19

        # Test reset! with short_game
        g2 = initial_state(first_player=0)
        reset!(g2, short_game=true, first_player=0)
        sample_chance!(g2)
        @test g2[4] == 2   # Verify short game position after reset
        @test g2[16] == 3
    end

    @testset "Doubles Only Mode" begin
        # Test initial_state with doubles_only
        g = initial_state(doubles_only=true, first_player=0)
        @test g.doubles_only == true
        @test is_chance_node(g)

        # Sample chance and verify we get doubles
        sample_chance!(g)
        @test g.dice[1] == g.dice[2]  # Must be doubles
        @test g.remaining_actions == 2  # Doubles give 2 actions

        # chance_outcomes should keep 21 entries with zeros for non-doubles
        reset!(g, doubles_only=true, first_player=0)
        outcomes = chance_outcomes(g)
        @test length(outcomes) == 21
        for (idx, prob) in outcomes
            if idx in (1, 7, 12, 16, 19, 21)
                @test prob ≈ 1/6 atol=1e-6
            else
                @test prob == 0.0f0
            end
        end

        # Test multiple rolls are all doubles
        for _ in 1:20
            reset!(g, doubles_only=true, first_player=0)
            sample_chance!(g)
            @test g.dice[1] == g.dice[2]
            @test g.dice[1] in 1:6
        end

        # Test reset! preserves doubles_only setting when specified
        g2 = initial_state(first_player=0)
        @test g2.doubles_only == false
        reset!(g2, doubles_only=true, first_player=0)
        @test g2.doubles_only == true
        sample_chance!(g2)
        @test g2.dice[1] == g2.dice[2]

        # Test that regular game can still roll non-doubles (seeded for determinism)
        rng = Random.MersenneTwister(12345)
        g3 = initial_state(first_player=0)
        non_doubles_found = false
        for _ in 1:100
            reset!(g3, first_player=0)
            sample_chance!(g3, rng)
            if g3.dice[1] != g3.dice[2]
                non_doubles_found = true
                break
            end
        end
        @test non_doubles_found  # Should find at least one non-double in 100 tries

        # Test doubles_only persists through gameplay (step! calls sample_chance!)
        g4 = initial_state(doubles_only=true, first_player=0)
        sample_chance!(g4)
        for _ in 1:10
            if game_terminated(g4)
                break
            end
            actions = legal_actions(g4)
            step!(g4, actions[1])
            if !game_terminated(g4)
                @test g4.dice[1] == g4.dice[2]  # Still doubles after step!
            end
        end

        # Test chance_outcomes returns all 21 in normal mode with standard probs
        g6 = initial_state(first_player=0)
        outcomes_normal = chance_outcomes(g6)
        @test length(outcomes_normal) == 21
        total_prob_normal = sum(p for (_, p) in outcomes_normal)
        @test total_prob_normal ≈ 1.0f0 atol=1e-5

        # Test apply_chance! rejects non-doubles in doubles_only mode
        g7 = initial_state(doubles_only=true, first_player=0)
        @test is_chance_node(g7)
        # Index 2 is (1,2) - a non-double
        @test_throws ErrorException apply_chance!(g7, 2)
        # Index 1 is (1,1) - a double, should work
        g8 = initial_state(doubles_only=true, first_player=0)
        @test_nowarn apply_chance!(g8, 1)
        @test g8.dice == [1, 1]

        # Test legal_actions at chance nodes returns consistent action space
        g9 = initial_state(doubles_only=true, first_player=0)
        @test is_chance_node(g9)
        chance_actions = legal_actions(g9)
        # Always returns all 21 indices; probabilities indicate which are valid
        @test length(chance_actions) == 21
        # Verify chance_outcomes has 0 probability for non-doubles
        outcomes = chance_outcomes(g9)
        for (idx, prob) in outcomes
            if idx in (1, 7, 12, 16, 19, 21)
                @test prob ≈ 1/6 atol=1e-6
            else
                @test prob == 0.0f0
            end
        end

        # Normal mode should still return 21 chance actions
        g10 = initial_state(first_player=0)
        @test length(legal_actions(g10)) == 21
    end

    @testset "Combined short_game and doubles_only" begin
        g = initial_state(short_game=true, doubles_only=true, first_player=0)
        @test g.doubles_only == true
        sample_chance!(g)

        # Verify short game position
        @test g[4] == 2
        @test g[16] == 3

        # Verify doubles
        @test g.dice[1] == g.dice[2]
    end

    # =========================================================================
    # New 3-Tier Observation API Tests
    # =========================================================================

    @testset "Observation Dimensions" begin
        # Test exported constants
        @test OBS_CHANNELS_MINIMAL == 38
        @test OBS_CHANNELS_FULL == 70
        @test OBS_CHANNELS_BIASED == 130
        @test OBS_WIDTH == 26  # My bar at 1, points at 2-25, opponent bar at 26

        @test OBSERVATION_SIZES.minimal == 38
        @test OBSERVATION_SIZES.full == 70
        @test OBSERVATION_SIZES.biased == 130
        @test OBSERVATION_SIZES.width == 26

        # Test observation shapes
        g = initial_state(first_player=0)
        sample_chance!(g)

        obs_min = observe_minimal(g)
        @test size(obs_min) == (38, 1, 26)
        @test eltype(obs_min) == Float32

        obs_full = observe_full(g)
        @test size(obs_full) == (70, 1, 26)
        @test eltype(obs_full) == Float32

        obs_biased = observe_biased(g)
        @test size(obs_biased) == (130, 1, 26)
        @test eltype(obs_biased) == Float32
    end

    @testset "Board Threshold Encoding" begin
        # Test threshold encoding: channels 1-6 for my checkers, 7-12 for opponent
        # Spatial layout: My bar at 1, Point N at N+1, Opponent bar at 26
        b = zeros(MVector{28, Int8})
        b[5] = 3   # 3 of my checkers at point 5 (spatial index 6)
        b[10] = -2  # 2 opponent checkers at point 10 (spatial index 11)
        b[15] = 7   # 7 of my checkers at point 15 (spatial index 16, tests 6+ overflow)

        g = make_test_game(board=b, dice=(3, 4), current_player=0)
        obs = observe_minimal(g)

        # My 3 checkers at point 5 (spatial index 6): channels 1-3 should be 1
        @test obs[1, 1, 6] == 1.0f0  # >=1
        @test obs[2, 1, 6] == 1.0f0  # >=2
        @test obs[3, 1, 6] == 1.0f0  # >=3
        @test obs[4, 1, 6] == 0.0f0  # >=4
        @test obs[5, 1, 6] == 0.0f0  # >=5
        @test obs[6, 1, 6] == 0.0f0  # 6+ overflow

        # Opponent 2 checkers at point 10 (spatial index 11): channels 7-8 should be 1
        @test obs[7, 1, 11] == 1.0f0   # >=1
        @test obs[8, 1, 11] == 1.0f0   # >=2
        @test obs[9, 1, 11] == 0.0f0   # >=3

        # My 7 checkers at point 15 (spatial index 16): test 6+ overflow encoding
        @test obs[1, 1, 16] == 1.0f0  # >=1
        @test obs[5, 1, 16] == 1.0f0  # >=5
        @test obs[6, 1, 16] ≈ (7-5)/10.0f0 atol=1e-6  # 6+ overflow: (7-5)/10 = 0.2

        # Test maximum overflow (15 checkers) at point 20 (spatial index 21)
        b2 = zeros(MVector{28, Int8})
        b2[20] = 15
        g2 = make_test_game(board=b2, dice=(1, 2), current_player=0)
        obs2 = observe_minimal(g2)
        @test obs2[6, 1, 21] ≈ 1.0f0 atol=1e-6  # (15-5)/10 = 1.0

        # Test my bar encoding (spatial index 1)
        b3 = zeros(MVector{28, Int8})
        b3[25] = 2  # My bar (index 25 in test board = my bar)
        g3 = make_test_game(board=b3, dice=(1, 2), current_player=0)
        obs3 = observe_minimal(g3)
        @test obs3[1, 1, 1] == 1.0f0  # >=1 at my bar
        @test obs3[2, 1, 1] == 1.0f0  # >=2 at my bar
        @test obs3[3, 1, 1] == 0.0f0  # >=3 at my bar

        # Test opponent bar encoding (spatial index 26)
        b4 = zeros(MVector{28, Int8})
        b4[26] = -3  # Opponent bar (index 26 in test board = opponent bar)
        g4 = make_test_game(board=b4, dice=(1, 2), current_player=0)
        obs4 = observe_minimal(g4)
        @test obs4[7, 1, 26] == 1.0f0   # >=1 at opponent bar
        @test obs4[8, 1, 26] == 1.0f0   # >=2 at opponent bar
        @test obs4[9, 1, 26] == 1.0f0   # >=3 at opponent bar
        @test obs4[10, 1, 26] == 0.0f0  # >=4 at opponent bar
    end

    @testset "Dice One-Hot Encoding" begin
        # Non-doubles: dice (3, 5) should have slots 0 and 1 active
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g = make_test_game(board=b, dice=(3, 5), current_player=0)
        obs = observe_minimal(g)

        # Slot 0 (channels 13-18): should have channel 17 active (value 5, high die)
        @test obs[17, 1, 1] == 1.0f0  # Slot 0, value 5
        # Slot 1 (channels 19-24): should have channel 21 active (value 3, low die)
        @test obs[21, 1, 1] == 1.0f0  # Slot 1, value 3
        # Slots 2 and 3 should be all zeros
        for ch in 25:36
            @test obs[ch, 1, 1] == 0.0f0
        end

        # Doubles (4, 4) with remaining=2: all 4 slots active with value 4
        g2 = make_test_game(board=b, dice=(4, 4), remaining=2, current_player=0)
        obs2 = observe_minimal(g2)
        # All 4 slots should have channel for value 4 active
        @test obs2[16, 1, 1] == 1.0f0  # Slot 0, value 4 (ch 12+4=16)
        @test obs2[22, 1, 1] == 1.0f0  # Slot 1, value 4 (ch 18+4=22)
        @test obs2[28, 1, 1] == 1.0f0  # Slot 2, value 4 (ch 24+4=28)
        @test obs2[34, 1, 1] == 1.0f0  # Slot 3, value 4 (ch 30+4=34)

        # Doubles with remaining=1: only 2 slots active
        g3 = make_test_game(board=b, dice=(4, 4), remaining=1, current_player=0)
        obs3 = observe_minimal(g3)
        @test obs3[16, 1, 1] == 1.0f0  # Slot 0 active
        @test obs3[22, 1, 1] == 1.0f0  # Slot 1 active
        @test obs3[28, 1, 1] == 0.0f0  # Slot 2 inactive
        @test obs3[34, 1, 1] == 0.0f0  # Slot 3 inactive
    end

    @testset "Off Counts Encoding" begin
        b = zeros(MVector{28, Int8})
        b[27] = 5   # My off = 5
        b[28] = -3  # Opp off = 3
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        obs = observe_minimal(g)

        # Channel 37: my off / 15
        @test obs[37, 1, 1] ≈ 5.0f0/15.0f0 atol=1e-6
        # Channel 38: opp off / 15
        @test obs[38, 1, 1] ≈ 3.0f0/15.0f0 atol=1e-6
    end

    @testset "Full Observation Features" begin
        # Test dice_delta (channel 40)
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g_delta = make_test_game(board=b, dice=(6, 2), current_player=0)
        obs_delta = observe_full(g_delta)
        @test obs_delta[40, 1, 1] ≈ 4.0f0/5.0f0 atol=1e-6  # |6-2|/5 = 0.8

        # Test dice_delta for doubles (should be 0)
        g_doubles = make_test_game(board=b, dice=(3, 3), remaining=2, current_player=0)
        obs_doubles = observe_full(g_doubles)
        @test obs_doubles[40, 1, 1] == 0.0f0  # |3-3|/5 = 0

        # Test contact indicator (channel 41)
        # Race position: no contact
        b_race = zeros(MVector{28, Int8})
        b_race[20] = 5  # My checkers in home (point 20)
        b_race[5] = -5  # Opp checkers far behind (point 5)
        g_race = make_test_game(board=b_race, dice=(1, 2), current_player=0)
        obs_race = observe_full(g_race)
        @test obs_race[41, 1, 1] == 0.0f0  # No contact (is race)

        # Contact position
        b_contact = zeros(MVector{28, Int8})
        b_contact[10] = 1   # My checker
        b_contact[15] = -1  # Opp checker ahead of me
        g_contact = make_test_game(board=b_contact, dice=(1, 2), current_player=0)
        obs_contact = observe_full(g_contact)
        @test obs_contact[41, 1, 1] == 1.0f0  # Has contact

        # Test pip counts (channels 42-44)
        b_pip = zeros(MVector{28, Int8})
        b_pip[1] = 2   # My 2 checkers at point 1: pip value = 2 * (25-1) = 48
        b_pip[24] = -3 # Opp 3 checkers at point 24: pip value = 3 * 24 = 72
        g_pip = make_test_game(board=b_pip, dice=(1, 2), current_player=0)
        obs_pip = observe_full(g_pip)
        @test obs_pip[42, 1, 1] ≈ 48.0f0/167.0f0 atol=1e-5  # My pips
        @test obs_pip[43, 1, 1] ≈ 72.0f0/167.0f0 atol=1e-5  # Opp pips
        @test obs_pip[44, 1, 1] ≈ (48.0f0-72.0f0)/167.0f0 atol=1e-5  # Pip diff

        # Test can bear off (channels 45-46)
        b_bearoff = zeros(MVector{28, Int8})
        b_bearoff[20] = 5  # My checkers in home (19-24)
        b_bearoff[2] = -5  # Opp checkers in their home (1-6)
        g_bearoff = make_test_game(board=b_bearoff, dice=(1, 2), current_player=0)
        obs_bearoff = observe_full(g_bearoff)
        @test obs_bearoff[45, 1, 1] == 1.0f0  # I can bear off
        @test obs_bearoff[46, 1, 1] == 1.0f0  # Opp can bear off

        # Cannot bear off: checker outside home
        b_no_bear = zeros(MVector{28, Int8})
        b_no_bear[10] = 1  # My checker outside home
        b_no_bear[20] = 4
        g_no_bear = make_test_game(board=b_no_bear, dice=(1, 2), current_player=0)
        obs_no_bear = observe_full(g_no_bear)
        @test obs_no_bear[45, 1, 1] == 0.0f0  # I cannot bear off

        # Test stragglers (channels 47-58)
        b_strag = zeros(MVector{28, Int8})
        b_strag[10] = 3  # 3 stragglers (outside home 19-24)
        b_strag[20] = 2  # 2 in home (not stragglers)
        g_strag = make_test_game(board=b_strag, dice=(1, 2), current_player=0)
        obs_strag = observe_full(g_strag)
        # My stragglers = 3: channels 47-49 should be 1
        @test obs_strag[47, 1, 1] == 1.0f0  # >=1
        @test obs_strag[48, 1, 1] == 1.0f0  # >=2
        @test obs_strag[49, 1, 1] == 1.0f0  # >=3
        @test obs_strag[50, 1, 1] == 0.0f0  # >=4

        # Test remaining (channels 59-70)
        b_remain = zeros(MVector{28, Int8})
        b_remain[20] = 5
        b_remain[27] = 10  # 10 off, so 5 remaining
        g_remain = make_test_game(board=b_remain, dice=(1, 2), current_player=0)
        obs_remain = observe_full(g_remain)
        # My remaining = 5: channels 59-63 should be 1
        @test obs_remain[59, 1, 1] == 1.0f0  # >=1
        @test obs_remain[63, 1, 1] == 1.0f0  # >=5
        @test obs_remain[64, 1, 1] == 0.0f0  # 6+ (should be 0)
    end

    @testset "Biased Observation Features" begin
        # Test prime length (channels 71-82)
        b_prime = zeros(MVector{28, Int8})
        b_prime[5] = 2   # Block
        b_prime[6] = 2   # Block
        b_prime[7] = 2   # Block - 3 consecutive blocks
        b_prime[10] = -2 # Opp block (isolated)
        g_prime = make_test_game(board=b_prime, dice=(1, 2), current_player=0)
        obs_prime = observe_biased(g_prime)
        # My prime = 3
        @test obs_prime[71, 1, 1] == 1.0f0  # >=1
        @test obs_prime[73, 1, 1] == 1.0f0  # >=3
        @test obs_prime[74, 1, 1] == 0.0f0  # >=4
        # Opp prime = 1
        @test obs_prime[77, 1, 1] == 1.0f0  # >=1
        @test obs_prime[78, 1, 1] == 0.0f0  # >=2

        # Test home board blocks (channels 83-94)
        b_home = zeros(MVector{28, Int8})
        b_home[20] = 2  # My block in home (19-24)
        b_home[21] = 2  # Another block in home
        b_home[2] = -2  # Opp block in their home (1-6)
        g_home = make_test_game(board=b_home, dice=(1, 2), current_player=0)
        obs_home = observe_biased(g_home)
        # My home blocks = 2
        @test obs_home[83, 1, 1] == 1.0f0  # >=1
        @test obs_home[84, 1, 1] == 1.0f0  # >=2
        @test obs_home[85, 1, 1] == 0.0f0  # >=3
        # Opp home blocks = 1
        @test obs_home[89, 1, 1] == 1.0f0  # >=1
        @test obs_home[90, 1, 1] == 0.0f0  # >=2

        # Test anchors (channels 95-106)
        b_anchor = zeros(MVector{28, Int8})
        b_anchor[3] = 2   # My anchor in opp's home (1-6)
        b_anchor[20] = -2 # Opp anchor in my home (19-24)
        g_anchor = make_test_game(board=b_anchor, dice=(1, 2), current_player=0)
        obs_anchor = observe_biased(g_anchor)
        # My anchors = 1
        @test obs_anchor[95, 1, 1] == 1.0f0
        @test obs_anchor[96, 1, 1] == 0.0f0
        # Opp anchors = 1
        @test obs_anchor[101, 1, 1] == 1.0f0
        @test obs_anchor[102, 1, 1] == 0.0f0

        # Test blot count (channels 107-118)
        b_blot = zeros(MVector{28, Int8})
        b_blot[5] = 1   # My blot
        b_blot[10] = 1  # Another blot
        b_blot[15] = -1 # Opp blot
        g_blot = make_test_game(board=b_blot, dice=(1, 2), current_player=0)
        obs_blot = observe_biased(g_blot)
        # My blots = 2
        @test obs_blot[107, 1, 1] == 1.0f0  # >=1
        @test obs_blot[108, 1, 1] == 1.0f0  # >=2
        @test obs_blot[109, 1, 1] == 0.0f0  # >=3
        # Opp blots = 1
        @test obs_blot[113, 1, 1] == 1.0f0
        @test obs_blot[114, 1, 1] == 0.0f0

        # Test builder count (channels 119-130)
        b_builder = zeros(MVector{28, Int8})
        b_builder[5] = 2   # My builder
        b_builder[10] = 2  # Another builder
        b_builder[15] = 2  # Third builder
        b_builder[20] = -2 # Opp builder
        g_builder = make_test_game(board=b_builder, dice=(1, 2), current_player=0)
        obs_builder = observe_biased(g_builder)
        # My builders = 3
        @test obs_builder[119, 1, 1] == 1.0f0
        @test obs_builder[121, 1, 1] == 1.0f0  # >=3
        @test obs_builder[122, 1, 1] == 0.0f0  # >=4
        # Opp builders = 1
        @test obs_builder[125, 1, 1] == 1.0f0
        @test obs_builder[126, 1, 1] == 0.0f0
    end

    @testset "Observation Hierarchy" begin
        # Test that observations build on each other
        g = initial_state(first_player=0)
        sample_chance!(g)

        obs_min = observe_minimal(g)
        obs_full = observe_full(g)
        obs_biased = observe_biased(g)

        # Full should contain minimal (channels 1-38)
        @test obs_full[1:38, :, :] ≈ obs_min atol=1e-6

        # Biased should contain full (channels 1-70)
        @test obs_biased[1:70, :, :] ≈ obs_full atol=1e-6
    end

    @testset "In-Place Observation Functions" begin
        g = initial_state(first_player=0)
        sample_chance!(g)

        # observe_minimal! matches observe_minimal
        buf_min = zeros(Float32, OBS_CHANNELS_MINIMAL, 1, OBS_WIDTH)
        result_min = observe_minimal!(buf_min, g)
        @test result_min === buf_min
        @test buf_min ≈ observe_minimal(g) atol=1e-6

        # observe_full! matches observe_full
        buf_full = zeros(Float32, OBS_CHANNELS_FULL, 1, OBS_WIDTH)
        result_full = observe_full!(buf_full, g)
        @test result_full === buf_full
        @test buf_full ≈ observe_full(g) atol=1e-6

        # observe_biased! matches observe_biased
        buf_biased = zeros(Float32, OBS_CHANNELS_BIASED, 1, OBS_WIDTH)
        result_biased = observe_biased!(buf_biased, g)
        @test result_biased === buf_biased
        @test buf_biased ≈ observe_biased(g) atol=1e-6

        # Test proper zeroing
        fill!(buf_full, 999.0f0)
        observe_full!(buf_full, g)
        @test all(buf_full .<= 1.5f0)  # No leftover 999s
    end

    @testset "Observation Through Gameplay" begin
        # Test observations remain valid through gameplay
        g = initial_state(first_player=0)
        sample_chance!(g)

        for _ in 1:20
            if game_terminated(g)
                break
            end

            if !is_chance_node(g)
                obs_min = observe_minimal(g)
                obs_full = observe_full(g)
                obs_biased = observe_biased(g)

                # All values should be bounded
                @test all(obs_min .<= 1.5f0) && all(obs_min .>= -1.5f0)
                @test all(obs_full .<= 1.5f0) && all(obs_full .>= -1.5f0)
                @test all(obs_biased .<= 1.5f0) && all(obs_biased .>= -1.5f0)

                # Hierarchy should hold
                @test obs_full[1:38, :, :] ≈ obs_min atol=1e-6
                @test obs_biased[1:70, :, :] ≈ obs_full atol=1e-6
            end

            actions = legal_actions(g)
            step!(g, actions[1])
        end
    end

    @testset "Edge Cases: dice_sum, contact, overflow" begin
        # Test dice_sum on chance node (dice = [0,0])
        g_chance = initial_state(first_player=0)
        # Don't roll dice - game starts at chance node
        @test is_chance_node(g_chance)
        obs_chance = observe_full(g_chance)
        @test obs_chance[39, 1, 1] == 0.0f0  # dice_sum should be 0 at chance node

        # Test dice_sum minimum (1+1=2)
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g_min_dice = make_test_game(board=b, dice=(1, 1), remaining=2, current_player=0)
        obs_min_dice = observe_full(g_min_dice)
        @test obs_min_dice[39, 1, 1] ≈ 2.0f0/12.0f0 atol=1e-6

        # Test dice_sum maximum (6+6=12)
        g_max_dice = make_test_game(board=b, dice=(6, 6), remaining=2, current_player=0)
        obs_max_dice = observe_full(g_max_dice)
        @test obs_max_dice[39, 1, 1] ≈ 1.0f0 atol=1e-6  # 12/12 = 1.0

        # Test dice_delta on chance node (should be 0)
        @test obs_chance[40, 1, 1] == 0.0f0  # dice_delta should be 0 at chance node

        # Test contact on empty board (no checkers)
        b_empty = zeros(MVector{28, Int8})
        g_empty = make_test_game(board=b_empty, dice=(3, 4), current_player=0)
        obs_empty = observe_full(g_empty)
        @test obs_empty[41, 1, 1] == 0.0f0  # No contact on empty board

        # Test contact when only one side has checkers
        b_one_side = zeros(MVector{28, Int8})
        b_one_side[10] = 5  # Only my checkers
        g_one_side = make_test_game(board=b_one_side, dice=(3, 4), current_player=0)
        obs_one_side = observe_full(g_one_side)
        @test obs_one_side[41, 1, 1] == 0.0f0  # No contact (no opponent)

        # Test 6+ overflow at exact boundary (count = 5)
        b_boundary = zeros(MVector{28, Int8})
        b_boundary[10] = 5  # Exactly 5 checkers at point 10
        g_boundary = make_test_game(board=b_boundary, dice=(1, 2), current_player=0)
        obs_boundary = observe_minimal(g_boundary)
        # Point 10 is at spatial index 11 (bar at 1, point N at N+1)
        @test obs_boundary[5, 1, 11] == 1.0f0  # >=5 is true
        @test obs_boundary[6, 1, 11] == 0.0f0  # 6+ overflow = (5-5)/10 = 0

        # Test 6+ overflow at count = 6
        b_six = zeros(MVector{28, Int8})
        b_six[10] = 6  # 6 checkers at point 10
        g_six = make_test_game(board=b_six, dice=(1, 2), current_player=0)
        obs_six = observe_minimal(g_six)
        # Point 10 is at spatial index 11
        @test obs_six[6, 1, 11] ≈ (6-5)/10.0f0 atol=1e-6  # 0.1

        # Test features bounded in [0, 1] range at chance node
        @test all(obs_chance .>= 0.0f0)
        @test all(obs_chance .<= 1.0f0)

        # Test pip counts are 0 on empty board
        @test obs_empty[42, 1, 1] == 0.0f0  # my_pips
        @test obs_empty[43, 1, 1] == 0.0f0  # opp_pips
        @test obs_empty[44, 1, 1] == 0.0f0  # pip_diff

        # Test can_bear_off on empty board (vacuously true - no checkers outside home)
        @test obs_empty[45, 1, 1] == 1.0f0  # can bear off (no checkers to block it)
        @test obs_empty[46, 1, 1] == 1.0f0  # opp can bear off
    end

    @testset "Scoring & Perspective" begin
        # P0 Win - Single
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[1] = -1  # P1 has some off
        b[28] = -1 # P1 has 1 off
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.terminated
        @test g.reward == 1.0f0
        @test winner(g) == 0

        # P1 Win - Gammon
        b = zeros(MVector{28, Int8})
        b[27] = 14 # My Off (P1 Off)
        b[24] = 1  # My checker at Canon 24 (Physical 1)
        b[7] = -1  # Opp (P0) at Canon 7 (Physical 18) - outside P1 home

        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))

        @test g.terminated
        @test g.reward == -2.0f0
        @test winner(g) == 1

        # P0 Win - Backgammon (via bar)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # My Off (P0 Off)
        b[24] = 1  # My checker at Canon 24 (Physical 24)
        b[26] = -1 # Opp (P1) on Bar
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 3.0f0

        # P0 Win - Backgammon (via home board)
        b = zeros(MVector{28, Int8})
        b[27] = 14
        b[24] = 1
        b[19] = -1 # Opp (P1) in P0's home (Physical 19)
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 3.0f0

        # P1 Win - Backgammon (via home board)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # My Off (P1 Off)
        b[24] = 1  # My checker at Canon 24 (Physical 1)
        b[20] = -1 # Opp (P0) in My home (Canon 20 = Physical 5)

        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == -3.0f0

        # P1 Win - Backgammon (P0 on bar)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[24] = 1  # P1 at Canon 24 (Physical 1)
        b[26] = -1 # P0 on bar
        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == -3.0f0

        # P1 Win - Backgammon (P0 at physical 1, edge of home)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[23] = 1  # P1 at Canon 23 (Physical 2)
        b[24] = -1 # P0 at Canon 24 (Physical 1) - edge of P1's home (blot)
        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        # Must use both dice: 23->24 (die1=1, hits P0), then 24->off (die2=2)
        # P0 ends up on bar, still backgammon
        apply_action!(g, BackgammonNet.encode_action(24, 23))
        @test g.reward == -3.0f0

        # P1 Win - Backgammon (P0 at physical 6, other edge of home)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[20] = 1  # P1 at Canon 20 (Physical 5)
        b[19] = -1 # P0 at Canon 19 (Physical 6) - other edge of P1's home
        g = make_test_game(board=b, dice=(5, 1), current_player=1)
        # Must use both dice: 20->21 (die2=1), then 21->off (die1=5)
        apply_action!(g, BackgammonNet.encode_action(21, 20))
        @test g.reward == -3.0f0

        # P1 Win - Single (P0 on bar but has 1 off - not backgammon)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[24] = 1  # P1 at Canon 24 (Physical 1)
        b[26] = -1 # P0 on bar
        b[28] = -1 # P0 has 1 off - this prevents gammon/backgammon
        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == -1.0f0

        # P1 Win - Single (P0 at physical 1 but has 1 off)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[23] = 1  # P1 at Canon 23 (Physical 2)
        b[24] = -1 # P0 at Canon 24 (Physical 1) - blot
        b[28] = -1 # P0 has 1 off
        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        # Must use both dice: 23->24 (die2=1, hits P0), then 24->off (die1=2)
        apply_action!(g, BackgammonNet.encode_action(24, 23))
        @test g.reward == -1.0f0

        # P1 Win - Single (P0 at physical 6 but has 1 off)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[20] = 1  # P1 at Canon 20 (Physical 5)
        b[19] = -1 # P0 at Canon 19 (Physical 6)
        b[28] = -1 # P0 has 1 off
        g = make_test_game(board=b, dice=(5, 1), current_player=1)
        # Must use both dice: 20->21 (die2=1), then 21->off (die1=5)
        apply_action!(g, BackgammonNet.encode_action(21, 20))
        @test g.reward == -1.0f0

        # --- P0 Win additional tests ---

        # P0 Win - Gammon (P1 has 0 off, outside P0's home, not on bar)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[1] = -1  # P1 at physical 1 (outside P0's home 19-24)
        # b[28] = 0 implicitly (P1 has 0 off)
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 2.0f0

        # P0 Win - Backgammon (P1 at physical 24, edge of home)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[23] = 1  # P0 at point 23
        b[24] = -1 # P1 at physical 24 - edge of P0's home (blot)
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        # Must use both dice: 23->24 (die2=1, hits P1), then 24->off (die1=2)
        apply_action!(g, BackgammonNet.encode_action(24, 23))
        @test g.reward == 3.0f0

        # P0 Win - Single (P1 on bar but has 1 off)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[26] = -1 # P1 on bar
        b[28] = -1 # P1 has 1 off - prevents gammon/backgammon
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 1.0f0

        # P0 Win - Single (P1 at physical 19 but has 1 off)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[19] = -1 # P1 at physical 19 (edge of P0's home)
        b[28] = -1 # P1 has 1 off
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 1.0f0

        # P0 Win - Single (P1 at physical 24 but has 1 off)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[23] = 1  # P0 at point 23
        b[24] = -1 # P1 at physical 24 (other edge of P0's home) - blot
        b[28] = -1 # P1 has 1 off
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        # Must use both dice: 23->24 (die2=1, hits P1), then 24->off (die1=2)
        apply_action!(g, BackgammonNet.encode_action(24, 23))
        @test g.reward == 1.0f0

        # --- Boundary tests (verify home board ranges are correct) ---

        # P0 Win - Gammon (P1 at physical 18, just OUTSIDE home 19-24)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[18] = -1 # P1 at physical 18 - just outside P0's home
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 2.0f0  # Gammon, NOT backgammon

        # P1 Win - Gammon (P0 at physical 7, just OUTSIDE home 1-6)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P1 Off
        b[24] = 1  # P1 at Canon 24 (Physical 1)
        b[18] = -1 # P0 at Canon 18 (Physical 7) - just outside P1's home
        g = make_test_game(board=b, dice=(2, 1), current_player=1)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == -2.0f0  # Gammon, NOT backgammon

        # --- Edge cases: multiple checkers and combined positions ---

        # P0 Win - Backgammon (P1 on bar AND in home simultaneously)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[26] = -1 # P1 on bar
        b[19] = -1 # P1 also in P0's home
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 3.0f0

        # P0 Win - Backgammon (multiple P1 checkers on bar)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[26] = -3 # P1 has 3 checkers on bar
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 3.0f0

        # P0 Win - Backgammon (multiple P1 checkers in home)
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[19] = -2 # P1 has 2 checkers at physical 19
        b[22] = -3 # P1 has 3 checkers at physical 22
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.reward == 3.0f0
    end

    @testset "Forced Pass" begin
        # All moves blocked
        b = zeros(MVector{28, Int8})
        b[1] = 1
        for i in 2:7; b[i] = -2; end # Block all exits for 1, 2, 3, 4, 5, 6
        g = make_test_game(board=b, dice=(1, 2), current_player=0)

        actions = legal_actions(g)
        @test length(actions) == 1
        l1, l2 = BackgammonNet.decode_action(actions[1])
        @test l1 == PASS && l2 == PASS
    end

    @testset "Doubles Full Usage" begin
        b = zeros(MVector{28, Int8})
        b[1] = 4
        # Can move 1->3, 3->5, 5->7, 7->9
        g = make_test_game(board=b, dice=(2, 2), current_player=0, remaining=2)

        actions = legal_actions(g)
        # Each action in doubles represents 2 of 4 moves.
        # So we need to apply two actions to finish the turn.
        act = BackgammonNet.encode_action(1, 1) # moves two checkers to 3
        apply_action!(g, act)
        @test g.remaining_actions == 1
        @test !is_chance_node(g)

        act2 = BackgammonNet.encode_action(3, 3) # moves them to 5
        apply_action!(g, act2)
        @test is_chance_node(g) # Turn finished
    end

    @testset "Error Handling" begin
        # apply_action! after termination is a no-op
        b = zeros(MVector{28, Int8})
        b[27] = 14 # P0 Off
        b[24] = 1  # P0 last checker
        b[28] = -1 # P1 has 1 off
        g = make_test_game(board=b, dice=(2, 1), current_player=0)
        apply_action!(g, BackgammonNet.encode_action(24, PASS))
        @test g.terminated == true
        history_len = length(g.history)
        # Calling apply_action! again should be a no-op (no error, just returns)
        apply_action!(g, BackgammonNet.encode_action(1, 2))
        @test g.terminated == true  # Still terminated
        @test length(g.history) == history_len  # History unchanged

        # apply_action! on chance node throws error
        g = initial_state(first_player=0)
        @test is_chance_node(g)
        @test_throws ErrorException apply_action!(g, 1)

        # is_action_valid returns false at chance node (dice not rolled)
        g = initial_state(first_player=0)
        @test is_chance_node(g)
        @test !is_action_valid(g, 1)  # Any action should be invalid
        @test !is_action_valid(g, BackgammonNet.encode_action(PASS, PASS))  # Even PASS|PASS

        # apply_chance! on non-chance node throws error
        g = initial_state(first_player=0)
        sample_chance!(g)
        @test !is_chance_node(g)
        @test_throws ErrorException apply_chance!(g, 1)

        # Invalid action: empty source
        b = zeros(MVector{28, Int8})
        b[1] = 1  # Single checker on point 1
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(10, 10)  # No pieces at 10
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: blocked target
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at point 1
        b[2] = -2  # Blocked at point 2 (2+ opponent checkers)
        b[3] = -2  # Blocked at point 3
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(1, PASS)  # 1->2 blocked, 1->3 blocked
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: bar priority violation
        b = zeros(MVector{28, Int8})
        b[25] = 1  # On bar
        b[10] = 1  # Also have checker at point 10
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(10, PASS)  # Can't move 10, must enter from bar
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: both PASS when moves are possible
        b = zeros(MVector{28, Int8})
        b[1] = 1  # Checker at point 1, can move
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(PASS, PASS)
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: using one die when both can be used
        b = zeros(MVector{28, Int8})
        b[1] = 2  # Two checkers at point 1
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        # Can use both: 1->2 (d1), 1->3 (d2) or 1->2, 2->4, etc.
        invalid_action = BackgammonNet.encode_action(1, PASS)  # Only uses d1
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: using lower die when higher available (non-doubles)
        b = zeros(MVector{28, Int8})
        b[1] = 1  # Single checker
        b[2] = -2 # Block at 2 (blocks d1=1)
        b[3] = -2 # Block at 3 (blocks d2=2)
        b[7] = -2 # Block at 7 (blocks d1+d2 combo via 1->2->... or 1->3->...)
        g = make_test_game(board=b, dice=(5, 6), current_player=0)
        # Only one die usable at a time, must use higher (6)
        invalid_action = BackgammonNet.encode_action(1, PASS)  # Uses d1=5, not d2=6
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: bearing off when pieces outside home board
        b = zeros(MVector{28, Int8})
        b[24] = 1  # Checker in home board (point 24)
        b[10] = 1  # Checker outside home board (point 10)
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        # Can't bear off from 24, must move pieces first
        invalid_action = BackgammonNet.encode_action(24, 10)
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Valid action should pass is_action_valid
        b = zeros(MVector{28, Int8})
        b[1] = 2  # Two checkers
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        valid_action = BackgammonNet.encode_action(1, 1)  # 1->2 (d1), 1->3 (d2)
        @test is_action_valid(g, valid_action)

        # Invalid action triggering else branch in apply_action!
        # This tests the case where loc1 is NOT legal first, so else branch is taken,
        # but loc2 is also not legal - should throw error
        b = zeros(MVector{28, Int8})
        b[5] = 1   # Single checker at point 5
        b[6] = -2  # Block at 6 (blocks d1=1: 5+1=6)
        b[7] = -2  # Block at 7 (blocks d2=2: 5+2=7)
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        # Action (5, 5) means: use d1 from 5, use d2 from 5
        # legal1 = is_move_legal(5, d1=1) = false (5->6 blocked)
        # So else branch is taken
        # is_move_legal(5, d2=2) = false (5->7 blocked)
        # Should throw
        invalid_action = BackgammonNet.encode_action(5, 5)
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: loc1 blocked, loc2 works initially but loc1 still blocked after
        # This exercises the second error path in the else branch
        b = zeros(MVector{28, Int8})
        b[5] = 1   # Single checker at point 5
        b[6] = -2  # Block at 6 (blocks d1=1: 5+1=6)
        # d2=2: 5->7 is open
        # After moving 5->7, d1=1 would need to move from 7->8
        # But we encoded action as (5, 5) which tries to use loc1=5 after moving loc2=5
        # After 5->7, there's no checker at 5 anymore, so loc1=5 is illegal
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(5, 5)  # Both try to move from 5
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)

        # Invalid action: shared source with single checker (state corruption regression test)
        # This tests the bug fix where loc1 was applied without checking legality after loc2
        # Scenario: single checker at 6, dice (1,2), action (6,6)
        # - loc1=6 with d1=1: 6->7 is legal initially
        # - After 6->7, loc2=6 with d2=2 is NOT legal (no checker)
        # - After restore, loc2=6 with d2=2 IS legal (6->8)
        # - After 6->8, loc1=6 with d1=1 is NOT legal (no checker) - must check!
        b = zeros(MVector{28, Int8})
        b[6] = 1  # Single checker at point 6
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        invalid_action = BackgammonNet.encode_action(6, 6)  # Try to use same source twice
        @test !is_action_valid(g, invalid_action)
        @test_throws ErrorException apply_action!(g, invalid_action)
        # Verify no state corruption occurred (checker should still be at 6)
        @test g[6] == 1

        # winner() edge cases
        # Non-terminated game returns nothing
        g = initial_state(first_player=0)
        @test winner(g) === nothing

        # Terminated game with reward==0 (invalid state) returns nothing
        # This can only happen via direct mutation, not normal gameplay
        g = initial_state(first_player=0)
        g.terminated = true
        g.reward = 0.0f0
        @test winner(g) === nothing  # Should NOT return 1
    end

    @testset "Sanity Check Corruption Detection" begin
        # These tests verify that sanity_check_bitboard catches corrupted states.
        # In normal gameplay, these corruptions should never occur, but the checks
        # help catch bugs during development.
        # NOTE: Controlled by ENABLE_SANITY_CHECKS in game.jl (set to false for large-scale training)

        # Test 1: Both players on same point (impossible in backgammon)
        p0_corrupt = UInt128(0)
        p1_corrupt = UInt128(0)
        # Put P0 checker at physical point 5
        p0_corrupt = BackgammonNet.incr_count(p0_corrupt, 5)
        # Put P1 checker at same physical point 5 (corruption!)
        p1_corrupt = BackgammonNet.incr_count(p1_corrupt, 5)
        @test_throws ErrorException BackgammonNet.sanity_check_bitboard(p0_corrupt, p1_corrupt)

        # Test 2: Overflow - more than 15 total checkers for a player
        p0_overflow = UInt128(0)
        p1_normal = UInt128(0)
        # Add 16 checkers to P0 across different points (simulating overflow corruption)
        for i in 1:16
            p0_overflow = BackgammonNet.incr_count(p0_overflow, i)
        end
        @test_throws ErrorException BackgammonNet.sanity_check_bitboard(p0_overflow, p1_normal)

        # Test 3: Normal state should NOT throw
        g = initial_state(first_player=0)
        @test_nowarn BackgammonNet.sanity_check_game(g)

        # Test 4: State after valid moves should NOT throw
        sample_chance!(g)
        actions = legal_actions(g)
        if !isempty(actions)
            step!(g, actions[1])
            @test_nowarn BackgammonNet.sanity_check_game(g)
        end
    end

    @testset "legal_actions and is_action_valid Agreement" begin
        # Verify that is_action_valid agrees with legal_actions membership
        # This catches drift between the two implementations of maximize-dice rules
        Random.seed!(12345)

        for _ in 1:50  # Test 50 random positions
            g = initial_state()
            sample_chance!(g)

            # Play a few random moves to get varied positions
            for _ in 1:rand(0:20)
                if game_terminated(g)
                    break
                end
                acts = legal_actions(g)
                if isempty(acts)
                    break
                end
                step!(g, rand(acts))
            end

            if game_terminated(g) || is_chance_node(g)
                continue
            end

            # Get legal actions
            valid_actions = Set(legal_actions(g))

            # Verify is_action_valid agrees for all possible actions (1-676)
            for action in 1:676
                expected = action in valid_actions
                actual = is_action_valid(g, action)
                @test actual == expected
            end
        end
    end

    @testset "Base.show Method" begin
        g = initial_state(first_player=0)
        output = sprint(show, g)
        @test occursin("BackgammonGame", output)
        @test occursin("p=0", output)
        @test occursin("dice", output)
    end

    @testset "action_string Function" begin
        # Test basic point-to-point moves
        @test BackgammonNet.action_string(BackgammonNet.encode_action(1, 2)) == "1 | 2"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(12, 13)) == "12 | 13"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(24, 24)) == "24 | 24"

        # Test all special locations
        @test BackgammonNet.action_string(BackgammonNet.encode_action(PASS, PASS)) == "Pass | Pass"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(BAR, BAR)) == "Bar | Bar"

        # Test bar entry combinations
        @test BackgammonNet.action_string(BackgammonNet.encode_action(BAR, 5)) == "Bar | 5"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(BAR, PASS)) == "Bar | Pass"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(5, BAR)) == "5 | Bar"

        # Test pass combinations
        @test BackgammonNet.action_string(BackgammonNet.encode_action(10, PASS)) == "10 | Pass"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(PASS, 10)) == "Pass | 10"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(PASS, BAR)) == "Pass | Bar"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(BAR, PASS)) == "Bar | Pass"

        # Test boundary points
        @test BackgammonNet.action_string(BackgammonNet.encode_action(1, 1)) == "1 | 1"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(24, 1)) == "24 | 1"
        @test BackgammonNet.action_string(BackgammonNet.encode_action(1, 24)) == "1 | 24"

        # Test encode/decode/string roundtrip
        for loc1 in [BAR, 1, 12, 24, PASS]
            for loc2 in [BAR, 1, 12, 24, PASS]
                action = BackgammonNet.encode_action(loc1, loc2)
                decoded_loc1, decoded_loc2 = BackgammonNet.decode_action(action)
                @test decoded_loc1 == loc1
                @test decoded_loc2 == loc2

                # Verify string contains expected parts
                str = BackgammonNet.action_string(action)
                @test occursin("|", str)
                if loc1 == PASS
                    @test occursin("Pass", str)
                elseif loc1 == BAR
                    @test occursin("Bar", str)
                else
                    @test occursin(string(loc1), str)
                end
            end
        end

        # Test action index boundaries
        @test BackgammonNet.encode_action(BAR, BAR) == 1  # Minimum action
        @test BackgammonNet.encode_action(PASS, PASS) == 676  # Maximum action

        # Verify all valid action indices produce valid strings
        for action in 1:676
            str = BackgammonNet.action_string(action)
            @test !isempty(str)
            @test occursin("|", str)
        end
    end

    @testset "History Tracking" begin
        g = initial_state(first_player=0)
        @test isempty(g.history)

        sample_chance!(g)
        actions = legal_actions(g)
        action1 = actions[1]
        step!(g, action1)

        @test length(g.history) >= 1
        @test g.history[1] == action1

        # Continue playing and verify history grows
        if !game_terminated(g)
            actions2 = legal_actions(g)
            action2 = actions2[1]
            step!(g, action2)
            @test length(g.history) >= 2
            @test g.history[2] == action2
        end

        # Reset clears history
        reset!(g)
        @test isempty(g.history)
    end

    @testset "Bit Manipulation Helpers" begin
        # Test get_count
        @testset "get_count" begin
            # Empty board
            board = UInt128(0)
            for idx in 0:27
                @test BackgammonNet.get_count(board, idx) == 0
            end

            # Single checker at various positions
            for idx in 0:27
                board = UInt128(1) << (idx << 2)
                @test BackgammonNet.get_count(board, idx) == 1
                # Other positions should be 0
                for other in 0:27
                    if other != idx
                        @test BackgammonNet.get_count(board, other) == 0
                    end
                end
            end

            # Max checkers (15) at a position
            for idx in [0, 10, 25, 27]
                board = UInt128(15) << (idx << 2)
                @test BackgammonNet.get_count(board, idx) == 15
            end

            # Multiple positions with different counts
            board = (UInt128(3) << (5 << 2)) | (UInt128(7) << (10 << 2)) | (UInt128(15) << (20 << 2))
            @test BackgammonNet.get_count(board, 5) == 3
            @test BackgammonNet.get_count(board, 10) == 7
            @test BackgammonNet.get_count(board, 20) == 15
            @test BackgammonNet.get_count(board, 0) == 0
        end

        @testset "incr_count" begin
            # Increment from 0
            board = UInt128(0)
            for idx in [0, 5, 15, 25, 27]
                new_board = BackgammonNet.incr_count(board, idx)
                @test BackgammonNet.get_count(new_board, idx) == 1
                # Original unchanged (pure function)
                @test BackgammonNet.get_count(board, idx) == 0
            end

            # Increment existing count
            board = UInt128(5) << (10 << 2)
            new_board = BackgammonNet.incr_count(board, 10)
            @test BackgammonNet.get_count(new_board, 10) == 6

            # Multiple increments
            board = UInt128(0)
            for i in 1:10
                board = BackgammonNet.incr_count(board, 7)
            end
            @test BackgammonNet.get_count(board, 7) == 10

            # Increment doesn't affect other positions
            board = (UInt128(3) << (5 << 2)) | (UInt128(7) << (10 << 2))
            new_board = BackgammonNet.incr_count(board, 5)
            @test BackgammonNet.get_count(new_board, 5) == 4
            @test BackgammonNet.get_count(new_board, 10) == 7  # Unchanged
        end

        @testset "decr_count" begin
            # Decrement from positive
            board = UInt128(5) << (10 << 2)
            new_board = BackgammonNet.decr_count(board, 10)
            @test BackgammonNet.get_count(new_board, 10) == 4

            # Decrement to 0
            board = UInt128(1) << (15 << 2)
            new_board = BackgammonNet.decr_count(board, 15)
            @test BackgammonNet.get_count(new_board, 15) == 0

            # Multiple decrements
            board = UInt128(10) << (7 << 2)
            for i in 1:5
                board = BackgammonNet.decr_count(board, 7)
            end
            @test BackgammonNet.get_count(board, 7) == 5

            # Decrement doesn't affect other positions
            board = (UInt128(3) << (5 << 2)) | (UInt128(7) << (10 << 2))
            new_board = BackgammonNet.decr_count(board, 10)
            @test BackgammonNet.get_count(new_board, 10) == 6
            @test BackgammonNet.get_count(new_board, 5) == 3  # Unchanged
        end

        @testset "incr/decr roundtrip" begin
            # Incrementing then decrementing should return to original
            board = UInt128(0)
            for idx in [0, 12, 25, 27]
                board1 = BackgammonNet.incr_count(board, idx)
                board2 = BackgammonNet.decr_count(board1, idx)
                @test board2 == board
            end

            # Complex board roundtrip
            board = (UInt128(2) << (1 << 2)) | (UInt128(5) << (12 << 2)) | (UInt128(3) << (17 << 2))
            board1 = BackgammonNet.incr_count(board, 12)
            board2 = BackgammonNet.decr_count(board1, 12)
            @test board2 == board
        end

        @testset "edge cases" begin
            # All nibbles at max (15) - test isolation
            board = UInt128(0)
            for idx in 0:27
                board = board | (UInt128(15) << (idx << 2))
            end
            for idx in 0:27
                @test BackgammonNet.get_count(board, idx) == 15
            end

            # Check nibble isolation - incrementing one doesn't overflow into adjacent
            board = UInt128(14) << (10 << 2)  # 14 at position 10
            new_board = BackgammonNet.incr_count(board, 10)
            @test BackgammonNet.get_count(new_board, 10) == 15
            @test BackgammonNet.get_count(new_board, 9) == 0
            @test BackgammonNet.get_count(new_board, 11) == 0
        end
    end

    @testset "Buffer Stress Tests" begin
        # Stress test: Play many games with buffer reuse via reset!
        @testset "Buffer reuse across many games" begin
            rng = Random.MersenneTwister(54321)
            g = initial_state(first_player=0)

            for game_num in 1:100
                reset!(g, first_player=rand(rng, 0:1))
                sample_chance!(g, rng)

                move_count = 0
                while !game_terminated(g) && move_count < 500
                    actions = legal_actions(g)
                    @test !isempty(actions)
                    step!(g, rand(rng, actions), rng)
                    move_count += 1
                end

                # Verify game completed without buffer issues
                @test game_terminated(g) || move_count == 500
            end
        end

        # Stress test: Verify legal_actions buffer handles max complexity
        @testset "Legal actions buffer capacity" begin
            # Play games specifically to find high-action-count positions
            rng = Random.MersenneTwister(12345)
            max_actions_seen = 0

            for _ in 1:50
                g = initial_state(first_player=0)
                sample_chance!(g, rng)

                for _ in 1:100
                    if game_terminated(g)
                        break
                    end
                    actions = legal_actions(g)
                    max_actions_seen = max(max_actions_seen, length(actions))
                    @test length(actions) <= 200  # Should never exceed buffer size
                    step!(g, rand(rng, actions), rng)
                end
            end

            # Should have seen reasonably diverse action counts
            @test max_actions_seen > 1
        end

        # Stress test: History buffer with long games (doubles only = longer games)
        @testset "History buffer with long games" begin
            rng = Random.MersenneTwister(99999)

            for _ in 1:10
                g = initial_state(first_player=0, doubles_only=true)
                sample_chance!(g, rng)

                while !game_terminated(g) && length(g.history) < 500
                    actions = legal_actions(g)
                    step!(g, rand(rng, actions), rng)
                end

                # History should have grown without issues
                @test length(g.history) > 0
                @test game_terminated(g) || length(g.history) >= 500
            end
        end

        # Stress test: Rapid reset! doesn't leak memory or corrupt state
        @testset "Rapid reset stress test" begin
            g = initial_state(first_player=0)

            for i in 1:1000
                reset!(g, first_player=i % 2, short_game=(i % 3 == 0), doubles_only=(i % 5 == 0))
                @test is_chance_node(g)
                @test isempty(g.history)
                @test g.doubles_only == (i % 5 == 0)

                # Occasionally play a move
                if i % 10 == 0
                    sample_chance!(g)
                    actions = legal_actions(g)
                    @test !isempty(actions)
                end
            end
        end
    end

    @testset "Dice Encoding Edge Cases" begin
        # Test dice encoding at chance node (dice = 0, 0)
        g = initial_state(first_player=0)
        @test is_chance_node(g)
        obs = observe_minimal(g)
        # Dice channels 13-36 should all be zero at chance node
        @test all(obs[13:36, 1, 1] .== 0.0f0)

        # Test with remaining_actions = 0 (edge case that shouldn't occur in normal play)
        b = zeros(MVector{28, Int8})
        b[1] = 1
        g2 = make_test_game(board=b, dice=(3, 4), remaining=0, current_player=0)
        obs2 = observe_minimal(g2)
        # With remaining_actions=0, dice encoding condition fails, so dice should be zero
        @test all(obs2[13:36, 1, 1] .== 0.0f0)

        # Test with d1=0 (partial invalid state)
        g3 = make_test_game(board=b, dice=(0, 4), remaining=1, current_player=0)
        obs3 = observe_minimal(g3)
        @test all(obs3[13:36, 1, 1] .== 0.0f0)

        # Test with d2=0 (partial invalid state)
        g4 = make_test_game(board=b, dice=(3, 0), remaining=1, current_player=0)
        obs4 = observe_minimal(g4)
        @test all(obs4[13:36, 1, 1] .== 0.0f0)
    end

    # NOTE: Old observation tests for blot/block detection and pip count difference
    # have been removed as they tested the legacy observation API which has been
    # replaced by the new 3-tier observation system (minimal, full, biased).

    @testset "Precomputed Bearing-Off Masks" begin
        # Test MASK_1_18: P0 must clear indices 1-18 before bearing off
        # MASK_1_18 should have bits set for nibbles 1-18
        for idx in 1:18
            @test BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASK_1_18)
        end
        # Indices 19-24 should NOT be in MASK_1_18
        for idx in 19:24
            @test !BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASK_1_18)
        end
        # Special indices (0, 25, 26, 27) should NOT be in MASK_1_18
        for idx in [0, 25, 26, 27]
            @test !BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASK_1_18)
        end

        # Test MASK_7_24: P1 must clear indices 7-24 before bearing off
        for idx in 7:24
            @test BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASK_7_24)
        end
        # Indices 1-6 should NOT be in MASK_7_24
        for idx in 1:6
            @test !BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASK_7_24)
        end

        # Test MASKS_HIGHER_P0: For P0, higher backgammon points = lower physical indices
        # MASKS_HIGHER_P0[i] masks indices 19 to i-1 (points further from off than i)
        @testset "MASKS_HIGHER_P0" begin
            # Index 19 has no higher points (it's the 6-point, furthest from off)
            @test BackgammonNet.MASKS_HIGHER_P0[19] == UInt128(0)

            # Index 20: should mask 19 only
            @test BackgammonNet.has_checkers(UInt128(1) << (19 << 2), BackgammonNet.MASKS_HIGHER_P0[20])
            @test !BackgammonNet.has_checkers(UInt128(1) << (20 << 2), BackgammonNet.MASKS_HIGHER_P0[20])

            # Index 24: should mask 19-23
            for idx in 19:23
                @test BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASKS_HIGHER_P0[24])
            end
            @test !BackgammonNet.has_checkers(UInt128(1) << (24 << 2), BackgammonNet.MASKS_HIGHER_P0[24])
        end

        # Test MASKS_HIGHER_P1: For P1, higher backgammon points = higher physical indices
        # MASKS_HIGHER_P1[i] masks indices i+1 to 6 (points further from off than i)
        @testset "MASKS_HIGHER_P1" begin
            # Index 6 has no higher points (it's the 6-point, furthest from off)
            @test BackgammonNet.MASKS_HIGHER_P1[6] == UInt128(0)

            # Index 5: should mask 6 only
            @test BackgammonNet.has_checkers(UInt128(1) << (6 << 2), BackgammonNet.MASKS_HIGHER_P1[5])
            @test !BackgammonNet.has_checkers(UInt128(1) << (5 << 2), BackgammonNet.MASKS_HIGHER_P1[5])

            # Index 1: should mask 2-6
            for idx in 2:6
                @test BackgammonNet.has_checkers(UInt128(1) << (idx << 2), BackgammonNet.MASKS_HIGHER_P1[1])
            end
            @test !BackgammonNet.has_checkers(UInt128(1) << (1 << 2), BackgammonNet.MASKS_HIGHER_P1[1])
        end
    end

end

