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
    
    @testset "Play" begin
        b = zeros(MVector{28, Int8})
        b[13] = 2
        
        g = make_test_game(board=b, dice=(3, 4))
        
        # 13->16 (D=3), 13->17 (D=4)
        # Action (13, 13)
        act = BackgammonNet.encode_action(13, 13)
        BackgammonNet.play!(g, act)
        
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
            
            # 24->Off (1). Pass (2).
            act = BackgammonNet.encode_action(24, PASS) 
            
            BackgammonNet.play!(g, act)
            
            @test g[27] == 15
            @test g.terminated == true
            @test g.reward == 1.0f0
        end
        
        @testset "Hitting" begin
            b = zeros(MVector{28, Int8})
            b[1] = 1
            b[2] = -1 # Opponent Blot
            
            g = make_test_game(board=b, dice=(1, 6))
            
            # 1->2 (Hit) using D1. Pass D2.
            act = BackgammonNet.encode_action(1, PASS) 
            
            BackgammonNet.play!(g, act)
            
            # g[25] is Current Player Bar.
            # Opponent was hit. Opponent is now Current Player.
            # So Opponent should be on Bar.
            @test g[25] == 1 
            # My checker at 17 (from 8, flipped).
            # Wait, 1->2 (Hit). I am now at 2.
            # Opponent view: 25-2 = 23.
            # My view: 2.
            # Turn switched.
            # g[23] should be -1.
                    @test g[23] == -1 
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
                        g = make_test_game(board=b, dice=(1, 2), current_player=0)
                        apply_action!(g, BackgammonNet.encode_action(24, PASS))
                        @test g.terminated
                        @test g.reward == 1.0f0
                        @test winner(g) == 0
                
                                        # P1 Win - Gammon
                        b = zeros(MVector{28, Int8})
                        b[27] = 14 # My Off (P1 Off)
                        b[24] = 1  # My checker at Canon 24 (Physical 1)
                        b[7] = -1  # Opp (P0) at Canon 7 (Physical 18) - outside P1 home
                        
                        g = make_test_game(board=b, dice=(1, 2), current_player=1)
                        apply_action!(g, BackgammonNet.encode_action(24, PASS))
                        
                        @test g.terminated
                        @test g.reward == -2.0f0
                        @test winner(g) == 1
                
                        # P0 Win - Backgammon (via bar)
                        b = zeros(MVector{28, Int8})
                        b[27] = 14 # My Off (P0 Off)
                        b[24] = 1  # My checker at Canon 24 (Physical 24)
                        b[26] = -1 # Opp (P1) on Bar
                        g = make_test_game(board=b, dice=(1, 2), current_player=0)
                        apply_action!(g, BackgammonNet.encode_action(24, PASS))
                        @test g.reward == 3.0f0

                        # P0 Win - Backgammon (via home board)
                        b = zeros(MVector{28, Int8})
                        b[27] = 14
                        b[24] = 1
                        b[19] = -1 # Opp (P1) in P0's home (Physical 19)
                        g = make_test_game(board=b, dice=(1, 2), current_player=0)
                        apply_action!(g, BackgammonNet.encode_action(24, PASS))
                        @test g.reward == 3.0f0

                        # P1 Win - Backgammon (via home board)
                        b = zeros(MVector{28, Int8})
                        b[27] = 14 # My Off (P1 Off)
                        b[24] = 1  # My checker at Canon 24 (Physical 1)
                        b[20] = -1 # Opp (P0) in My home (Canon 20 = Physical 5)
                        
                        g = make_test_game(board=b, dice=(1, 2), current_player=1)
                        apply_action!(g, BackgammonNet.encode_action(24, PASS))
                        @test g.reward == -3.0f0
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
                
                end
                