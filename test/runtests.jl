using Test
using StaticArrays
using BackgammonNet

function make_test_game(; board=nothing, dice=(1, 2), remaining=1, current_player=0)
    b = zeros(MVector{28, Int8})
    if board !== nothing
        for i in 1:28
            b[i] = board[i]
        end
    end
    d = SVector{2, Int8}(dice[1], dice[2])
    return BackgammonNet.BackgammonGame(
        SVector{28, Int8}(b),
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
        @test g.remaining_actions in [1, 2] # 1 for non-doubles, 2 for doubles
        @test g.board[1] == 2
    end
    
    @testset "Action Encoding" begin
        # 26*26
        # Pass, Pass -> 0, 0 -> 1
        @test BackgammonNet.encode_action(0, 0) == 1
        @test BackgammonNet.decode_action(1) == (0, 0)
        
        # Bar, Point 1 (Loc 2) -> 1, 2
        # (1 * 26) + 2 + 1 = 29
        @test BackgammonNet.encode_action(1, 2) == 29
        @test BackgammonNet.decode_action(29) == (1, 2)
    end
    
    @testset "Legal Actions: Non-Doubles" begin
        b = zeros(MVector{28, Int8})
        b[25] = 1 # Bar
        b[1] = 14 # Point 1
        b[2] = -2 # Block 2 (1+1)
        b[7] = -2 # Block 7 (1+6) and (6+1)
        
        # D1=1, D2=6
        g = make_test_game(board=b, dice=(1, 6))
        
        actions = legal_actions(g)
        
        # Encode (Pass, Bar) -> (0, 1).
        a_pass_bar = BackgammonNet.encode_action(0, 1)
        
        @test length(actions) == 1
        @test actions[1] == a_pass_bar
    end
    
    @testset "Legal Actions: Doubles" begin
        b = zeros(MVector{28, Int8})
        b[6] = 1
        b[8] = 1
        
        g = make_test_game(board=b, dice=(2, 2), remaining=2)
        
        actions = legal_actions(g)
        
        a1 = BackgammonNet.encode_action(7, 9)
        a2 = BackgammonNet.encode_action(9, 7)
        a3 = BackgammonNet.encode_action(9, 11)
        
        @test a1 in actions
        @test a2 in actions
        @test a3 in actions
    end
    
    @testset "Play" begin
        b = zeros(MVector{28, Int8})
        b[13] = 2
        
        g = make_test_game(board=b, dice=(3, 4))
        
        act = BackgammonNet.encode_action(14, 14)
        BackgammonNet.play!(g, act)
        
        # Turn switched, so board flipped.
        # Original moves: 13->16, 13->17.
        # Flipped indices: 25-16=9, 25-17=8.
        # Values negated: 1 -> -1.
        @test g.board[8] == -1
        @test g.board[9] == -1
        @test g.remaining_actions in [1, 2] # Reset for next player
        @test g.current_player == 1 # Switched
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
            
            # Die 3 blocked. Die 4 (Hit) legal.
            # Must use Die 4 from Bar.
            # Then Die 3 from board.
            # Action (L1, L2). L1 uses D1(3), L2 uses D2(4).
            # L2 must be Bar (1).
            # L1 must be from board (Loc 5->8 or Loc 11->14).
            
            a1 = BackgammonNet.encode_action(5, 1) # 4->7(d3), Bar->4(d4)
            a2 = BackgammonNet.encode_action(11, 1) # 10->13(d3), Bar->4(d4)
            
            @test length(actions) > 0
            for a in actions
                l1, l2 = BackgammonNet.decode_action(a)
                @test l2 == 1 # Second slot (D2) MUST be from Bar
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
            
            # Action (20, 24) -> L1=20(d6), L2=24(d1). Legal.
            # Action (21, 20) -> L1=21(d6), L2=20(d1). Legal (20->21, 21->Off).
            
            a_20_24 = BackgammonNet.encode_action(21, 25) # (20, 24)
            a_20_21 = BackgammonNet.encode_action(22, 21) # (21, 20)
            
            @test a_20_24 in actions
            @test a_20_21 in actions
            
            # (24, 20) implies L1=24(d6). Illegal.
            a_24_20 = BackgammonNet.encode_action(25, 21)
            @test !(a_24_20 in actions)
        end
        
        @testset "Winning Condition" begin
            b = zeros(MVector{28, Int8})
            b[27] = 14 # Off
            b[24] = 1
            
            g = make_test_game(board=b, dice=(1, 2))
            
            # Move 24->Off (1).
            act = BackgammonNet.encode_action(25, 0) # 25->Off, Pass
            
            BackgammonNet.play!(g, act)
            
            @test g.board[27] == 15
            @test g.terminated == true
            @test g.reward == 1.0f0
        end
        
        @testset "Hitting" begin
            b = zeros(MVector{28, Int8})
            b[1] = 1
            b[2] = -1 # Opponent Blot
            
            g = make_test_game(board=b, dice=(1, 6))
            
            # Move 1->2 (Die 1).
            # Action (2, 3).
            act = BackgammonNet.encode_action(2, 3) 
            
            BackgammonNet.play!(g, act)
            
            # Turn Switched!
            # Opponent Bar was -1. Became 1 (Hit).
            # Flipped: My New Bar (25) = OppBar (26) = -(-1) = 1.
            # My checker at 8 (index 8). Flipped: 25-8=17. Value -1.
            
            @test g.board[25] == 1 # Opponent on Bar
            @test g.board[17] == -1 # My checker (now opponent)
        end
    end
    
end
