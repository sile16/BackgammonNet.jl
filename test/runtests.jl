using Test
using StaticArrays
using BackgammonNet

# Constants for test setup
const IDX_P1_OFF = 0
const IDX_P0_OFF = 25
const IDX_P0_BAR = 26
const IDX_P1_BAR = 27

# Re-implement constants locally for clarity in tests
const PASS = 25
const BAR = 0

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
    return BackgammonNet.BackgammonGame(
        p0, p1,
        d,
        Int8(remaining),
        Int8(0),
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
    end
    
    @testset "Step" begin
        b = zeros(MVector{28, Int8})
        b[13] = 2
        
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
            b[20] = 1
            b[24] = 1
            
            g = make_test_game(board=b, dice=(6, 1))
            
            actions = legal_actions(g)
            
            # D1=6, D2=1.
            # 20->Off (6). Valid (No checkers > 20).
            # 24->Off (1). Valid.
            # Combo (20, 24).
            a_20_24 = BackgammonNet.encode_action(20, 24) 
            
            # 24->Off (6)? Over-bear. Src=24. Check 19..23.
            # 20 is occupied. So 24->Off(6) is Illegal.
            # So (24, 20) is Illegal.
            a_24_20 = BackgammonNet.encode_action(24, 20)
            
            # 20->21 (1). 21->Off (6).
            # S2=20 (D2). S1=21 (D1).
            # Action (21, 20).
            a_21_20 = BackgammonNet.encode_action(21, 20)
            
            @test a_20_24 in actions
            @test a_21_20 in actions
            
            @test !(a_24_20 in actions)
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
                        g = initial_state()
                        sample_chance!(g) # Start with rolled dice
                        
                        step_count = 0
                        max_steps = 10000
                        
                        while !game_terminated(g) && step_count < max_steps
                            actions = legal_actions(g)
                            if isempty(actions)
                                break
                            end
                            
                            a = rand(actions)
                            step!(g, a)
                            
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
                        @test g.turn == 0
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

                        # Test that regular game can still roll non-doubles
                        g3 = initial_state(first_player=0)
                        non_doubles_found = false
                        for _ in 1:100
                            reset!(g3, first_player=0)
                            sample_chance!(g3)
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

                    @testset "Observation" begin
                        g = initial_state()
                        obs = vector_observation(g)
                        @test length(obs) == 86
                        @test all(obs .<= 1.0) && all(obs .>= -1.0)
                        
                        fast = BackgammonNet.observe_fast(g)
                        @test length(fast) == 34
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
                    end

                end
                