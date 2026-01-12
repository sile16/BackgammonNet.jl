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
        # 6-1. Bar(1), Point 1(14). Blocked 2, 7.
        # D1=1, D2=6.
        # Valid Sources for D1: Bar(1).
        # Valid Sources for D2: Bar(1).
        
        # Bar(1) w/ D1(1) -> Target 1. Valid (own stack).
        # Bar(1) w/ D2(6) -> Target 6. Valid (empty).
        
        # Action (L1, L2). L1 uses D1, L2 uses D2.
        # Case 1: L1=Bar. (Enters at 1).
        #   Next Board: Checker at 1.
        #   Next Valid Sources for D2(6):
        #     Bar (empty now).
        #     Point 1 (15 checkers). Can move 1->7? Blocked.
        #     No legal moves for D2?
        #     Wait, from Bar->1. Point 1 has 15.
        #     Target 7 (1+6) is blocked.
        #     So D2 has no moves.
        #     Action: (Bar, Pass).
        
        # Case 2: L2=Bar. (Enters at 6).
        #   Next Board: Checker at 6.
        #   Next Valid Sources for D1(1):
        #     Bar (empty).
        #     Point 1 (14 checkers). Can move 1->2? Blocked.
        #     Point 6 (1 checker). Can move 6->7? Blocked.
        #     No legal moves for D1.
        #     Action: (Pass, Bar).
        
        # Max Dice Usage: Both use 1 die.
        # Must play higher die (D2=6).
        # Action (Pass, Bar) uses D2.
        # Action (Bar, Pass) uses D1.
        # Expect only (Pass, Bar).
        
        b = zeros(MVector{28, Int8})
        b[25] = 1 # Bar
        b[1] = 14 # Point 1
        b[2] = -2 # Block 2 (1+1)
        b[7] = -2 # Block 7 (1+6) and (6+1)
        
        # D1=1, D2=6
        g = make_test_game(board=b, dice=(1, 6))
        
        actions = BackgammonNet.get_legal_actions(g)
        
        # Encode (Pass, Bar) -> (0, 1). 0*26 + 1 + 1 = 2.
        # Encode (Bar, Pass) -> (1, 0). 1*26 + 0 + 1 = 27.
        
        a_pass_bar = BackgammonNet.encode_action(0, 1)
        
        @test length(actions) == 1
        @test actions[1] == a_pass_bar
    end
    
    @testset "Legal Actions: Doubles" begin
        # 2-2. 2 actions of 2 moves.
        # Board: Point 6(1), Point 8(1).
        # D=2.
        
        b = zeros(MVector{28, Int8})
        b[6] = 1
        b[8] = 1
        # 6->8 (blocked? no). 8->10.
        
        g = make_test_game(board=b, dice=(2, 2), remaining=2)
        
        actions = BackgammonNet.get_legal_actions(g)
        
        # Sources for 2: 6, 8.
        # S1=6. Next: 8(2 checkers).
        #   S2 for 2 from Next:
        #     6 (0 checkers).
        #     8 (2 checkers). Valid 8->10.
        #     Action (6, 8).
        # S1=8. Next: 10(1 checker).
        #   S2 for 2 from Next:
        #     6 (1 checker). Valid 6->8.
        #     10 (1 checker). Valid 10->12.
        #     Action (8, 6), (8, 10).
        
        # Expect (6, 8), (8, 6), (8, 10).
        
        # Locations:
        # 6 -> Loc 7.
        # 8 -> Loc 9.
        # 10 -> Loc 11.
        
        a1 = BackgammonNet.encode_action(7, 9)
        a2 = BackgammonNet.encode_action(9, 7)
        a3 = BackgammonNet.encode_action(9, 11)
        
        @test a1 in actions
        @test a2 in actions
        @test a3 in actions
    end
    
    @testset "Play" begin
        # Non-doubles: 3-4.
        # Point 13 (Loc 14).
        # Move 13->16 (3), 13->17 (4).
        b = zeros(MVector{28, Int8})
        b[13] = 2
        
        g = make_test_game(board=b, dice=(3, 4))
        
        # Action (14, 14).
        # 14->17 (3). 14->18 (4).
        # But wait, action implies L1(D1) and L2(D2).
        # D1=3, D2=4.
        # L1=14. 13->16.
        # L2=14. 13->17.
        # Result: Checker at 16, Checker at 17.
        
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

end