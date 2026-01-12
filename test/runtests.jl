using Test
using StaticArrays
using BackgammonNet

# Constants for test setup
const IDX_P1_OFF = 0
const IDX_P0_OFF = 25
const IDX_P0_BAR = 26
const IDX_P1_BAR = 27

function make_test_game(; board=nothing, dice=(1, 2), remaining=1, current_player=0)
    # Parse Canonical Board Vector into Bitboards
    p0 = UInt128(0)
    p1 = UInt128(0)
    
    if board !== nothing
        cp = current_player
        
        # Helper to add checkers
        function add_chk(b, idx, count)
            if count <= 0; return b; end
            # count is usually small, so we can just add
            # Safer: clear then set? Or assumes 0 init.
            # We assume 0 init.
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
        @test g.remaining_actions in [1, 2] # 1 for non-doubles, 2 for doubles
        @test g[1] == 2
    end
    
    @testset "Action Encoding" begin
        @test BackgammonNet.encode_action(0, 0) == 1
        @test BackgammonNet.decode_action(1) == (0, 0)
        @test BackgammonNet.encode_action(1, 2) == 29
        @test BackgammonNet.decode_action(29) == (1, 2)
    end
    
    @testset "Legal Actions: Non-Doubles" begin
        b = zeros(MVector{28, Int8})
        b[25] = 1 # Bar
        b[1] = 14 # Point 1
        b[2] = -2 # Block 2 (1+1)
        b[7] = -2 # Block 7 (1+6) and (6+1)
        
        g = make_test_game(board=b, dice=(1, 6))
        
        actions = legal_actions(g)
        
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
        
        # Turn switched.
        # Original moves: 13->16, 13->17.
        # g[i] access is Canonical.
        # Flipped indices: 25-16=9, 25-17=8.
        # Values negated (Canonical for Opponent).
        
        @test g[8] == -1
        @test g[9] == -1
        @test g.remaining_actions in [1, 2]
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
            
            a1 = BackgammonNet.encode_action(5, 1) 
            a2 = BackgammonNet.encode_action(11, 1) 
            
            @test length(actions) > 0
            for a in actions
                l1, l2 = BackgammonNet.decode_action(a)
                @test l2 == 1 
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
            
            a_20_24 = BackgammonNet.encode_action(21, 25) 
            a_20_21 = BackgammonNet.encode_action(22, 21) 
            
            @test a_20_24 in actions
            @test a_20_21 in actions
            
            a_24_20 = BackgammonNet.encode_action(25, 21)
            @test !(a_24_20 in actions)
        end
        
        @testset "Winning Condition" begin
            b = zeros(MVector{28, Int8})
            b[27] = 14 # Off
            b[24] = 1
            
            g = make_test_game(board=b, dice=(1, 2))
            
            act = BackgammonNet.encode_action(25, 0) # 25->Off, Pass
            
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
            
            act = BackgammonNet.encode_action(2, 3) 
            
            BackgammonNet.play!(g, act)
            
            # g[25] is Current Player Bar.
            # Opponent was hit. Opponent is now Current Player.
            # So Opponent should be on Bar.
            @test g[25] == 1 
            # My checker at 17 (from 8, flipped).
            @test g[17] == -1 
        end
    end
    
end