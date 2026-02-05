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

# ============================================================================
# Observation Channel Constants for Tests
# ============================================================================
# These constants define the observation tensor layout. If the representation
# changes in the future, only these constants need to be updated.
#
# Observation shape: (channels, 1, width)
# Width: 26 = [MyBar, Point1-24, OppBar]

"""
Observation channel layout constants for test assertions.
Centralizes channel indices so tests don't break if layout changes.
"""
module ObsChannels
    # Board encoding: threshold encoding (>=1, >=2, >=3, >=4, >=5, 6+)
    const MY_CHECKER_START = 1      # Channels 1-6: my checker thresholds
    const MY_CHECKER_END = 6
    const OPP_CHECKER_START = 7     # Channels 7-12: opponent checker thresholds
    const OPP_CHECKER_END = 12

    # Dice encoding: 2 one-hot slots (high die, low die)
    const DICE_SLOT0_START = 13     # Channels 13-18: high die (values 1-6)
    const DICE_SLOT0_END = 18
    const DICE_SLOT1_START = 19     # Channels 19-24: low die (values 1-6)
    const DICE_SLOT1_END = 24

    # Move count encoding: 4-bin one-hot (bins 1, 2, 3, 4)
    const MOVE_COUNT_START = 25     # Channels 25-28: move count bins
    const MOVE_COUNT_END = 28
    const MOVE_BIN_1 = 25           # 1 move playable
    const MOVE_BIN_2 = 26           # 2 moves playable
    const MOVE_BIN_3 = 27           # 3 moves playable
    const MOVE_BIN_4 = 28           # 4 moves playable

    # Off counts
    const MY_OFF = 29               # Channel 29: my borne-off count (/15)
    const OPP_OFF = 30              # Channel 30: opponent borne-off count (/15)

    # Minimal observation total channels
    const MINIMAL_CHANNELS = 30

    # Helper: get channel for dice slot 0 (high die) given die value 1-6
    dice_slot0_channel(die_value::Int) = DICE_SLOT0_START - 1 + die_value  # 13-18

    # Helper: get channel for dice slot 1 (low die) given die value 1-6
    dice_slot1_channel(die_value::Int) = DICE_SLOT1_START - 1 + die_value  # 19-24

    # Helper: get move count bin channel given move count 1-4
    move_bin_channel(move_count::Int) = MOVE_COUNT_START - 1 + move_count  # 25-28
end

# Spatial layout constants
module ObsSpatial
    const WIDTH = 26                # Total spatial width
    const MY_BAR = 1                # Index 1: my bar
    const POINT_START = 2           # Indices 2-25: points 1-24
    const POINT_END = 25
    const OPP_BAR = 26              # Index 26: opponent bar

    # Helper: get spatial index for board point 1-24
    point_index(pt::Int) = pt + 1   # Point 1 → index 2, Point 24 → index 25
end

function make_test_game(; board=nothing, dice=(1, 2), remaining=1, current_player=0, obs_type::Symbol=:minimal_flat)
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

    # Store dice as (high, low) to match DICE_OUTCOMES convention
    d_high = max(dice[1], dice[2])
    d_low = min(dice[1], dice[2])
    d = SVector{2, Int8}(d_high, d_low)
    # Use the short constructor which creates the buffers
    return BackgammonNet.BackgammonGame(
        p0, p1,
        d,
        Int8(remaining),
        Int8(current_player),
        false,
        0.0f0;
        obs_type=obs_type
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

        # DICE_OUTCOMES[13] is now (4,3) - stored as (high, low)
        apply_chance!(g, 13)
        @test g.dice == [4, 3]  # (high, low) ordering
        @test !is_chance_node(g)
        @test g.remaining_actions == 1

        # Doubles (2, 2) -> index 7 (same value for doubles)
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
        g.phase = BackgammonNet.PHASE_CHECKER_PLAY
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
        # Stored as (6, 1) - high die first

        actions = legal_actions(g)

        # Must use higher die when only one can be used.
        # dice[1]=6 (high), dice[2]=1 (low). loc1 uses high die, loc2 uses low die.
        # Only 6 can enter (to point 6), 1 is blocked (point 1 has our checker, point 2 blocked)
        # So: loc1=BAR (using die 6), loc2=PASS
        a_bar_pass = BackgammonNet.encode_action(BAR, PASS)

        @test length(actions) == 1
        @test actions[1] == a_bar_pass
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
        # dice=(1,5) stored as (5,1): dice[1]=5 (high), dice[2]=1 (low)
        # Die 5: 5->10 is open, but 10->11 blocked, so only one die usable
        g = make_test_game(board=b, dice=(1, 5), current_player=0)
        actions = legal_actions(g)
        # Die 1: 5->6 blocked
        # Die 5: 5->10 open, then die 1: 10->11 blocked
        # Only die 5 can be used (higher die anyway)
        # loc1=5 (use high die from 5), loc2=PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(5, PASS)

        # Scenario 3: Only one die works due to blocking - must use higher die
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Checker at 1
        b[3] = -2  # Block at 3 (blocks die 2: 1+2=3)
        b[7] = -2  # Block at 7 (blocks 5+2 after using die 4)
        # dice=(2,4) stored as (4,2): dice[1]=4 (high), dice[2]=2 (low)
        g = make_test_game(board=b, dice=(2, 4), current_player=0)
        actions = legal_actions(g)
        # Die 2: 1->3 blocked. Die 4: 1->5 open, then 5->7 blocked.
        # Only die 4 works (higher), so must use it
        # loc1=1 (use high die from 1), loc2=PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(1, PASS)

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
        # dice=(2,3) stored as (3,2): dice[1]=3 (high), dice[2]=2 (low)
        # Die 3: 1->4 open, then die 2: 4->6 blocked
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Only die 3 works (higher), so must use it
        # loc1=1 (use high die from 1), loc2=PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(1, PASS)

        # Scenario 6: Lower die works, higher doesn't - must use lower (only option)
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[4] = -2  # Block at 4 (blocks die 3: 1+3=4)
        b[6] = -2  # Block at 6 (blocks 3+3 after using die 2)
        # dice=(2,3) stored as (3,2): dice[1]=3 (high), dice[2]=2 (low)
        # Die 2: 1->3 open, then die 3: 3->6 blocked
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Only die 2 works (lower), so must use it
        # loc1=PASS (high die blocked), loc2=1 (use low die from 1)
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 1)

        # Scenario 7: Only lower die works (all paths with higher die blocked)
        b = zeros(MVector{28, Int8})
        b[1] = 1   # Single checker at 1
        b[4] = -2  # Block at 4 (blocks die 3: 1+3=4)
        b[5] = -2  # Block at 5
        b[6] = -2  # Block at 6 (blocks 3+3=6 after 1->3)
        # dice=(2,3) stored as (3,2): dice[1]=3 (high), dice[2]=2 (low)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # Die 2: 1->3 open, then die 3: 3->6 blocked. Die 3: 1->4 blocked.
        # Only die 2 works, must use lower die as only option
        # loc1=PASS (high die blocked), loc2=1 (use low die from 1)
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(PASS, 1)

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
        # dice=(2,5) stored as (5,2): dice[1]=5 (high), dice[2]=2 (low)
        # Entry point 5 open, but subsequent move blocked
        g = make_test_game(board=b, dice=(2, 5), current_player=0)
        actions = legal_actions(g)
        # Die 5 can enter (Bar->5), then die 2: 5->7 blocked
        # Only one die usable (die 5, which is higher)
        # loc1=BAR (use high die to enter), loc2=PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(BAR, PASS)

        # Scenario 10: Bearing off - must use higher die if only one works
        b = zeros(MVector{28, Int8})
        b[24] = 1  # Single checker at 24 (in home)
        b[27] = 14 # 14 already off
        # dice=(1,3) stored as (3,1): dice[1]=3 (high), dice[2]=1 (low)
        g = make_test_game(board=b, dice=(1, 3), current_player=0)
        actions = legal_actions(g)
        # Die 1: 24+1=25=off (exact)
        # Die 3: 24+3=27=off (over-bear, valid since 24 is highest)
        # Both work! So must use both... but only one checker!
        # With one checker, can only use one die. Must use higher (3).
        # loc1=24 (use high die from 24), loc2=PASS
        @test length(actions) == 1
        @test actions[1] == BackgammonNet.encode_action(24, PASS)
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
        # dice=(2, 3) stored as (3, 2): dice[1]=3 (high), dice[2]=2 (low)
        g = make_test_game(board=b, dice=(2, 3), current_player=0)
        actions = legal_actions(g)
        # loc1 uses high die (3): 1->4 open, 10->13 blocked
        # loc2 uses low die (2): 1->3 blocked, 10->12 open
        # Can we use both? 1->4 (high), then 10->12 (low)? Yes!
        a_both = BackgammonNet.encode_action(1, 10)  # high die from 1, low die from 10
        @test a_both in actions
        # Single die actions should NOT be in results
        a_single1 = BackgammonNet.encode_action(1, PASS)   # high die only
        a_single2 = BackgammonNet.encode_action(PASS, 10)  # low die only
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

            # dice=(3, 4) stored as (4, 3): dice[1]=4 (high), dice[2]=3 (low)
            g = make_test_game(board=b, dice=(3, 4))

            actions = legal_actions(g)

            # High die (4): Bar->4 (Hit) - LEGAL
            # Low die (3): Bar->3 (Blocked)
            #
            # Must enter from bar first. Only high die (4) can enter.
            # After Bar->4 (using high die), low die (3) can move:
            # - 4->7 (4+3=7)
            # - 10->13 (10+3=13)
            #
            # Actions:
            # loc1=BAR (high die enters), loc2=4 (low die from 4->7)
            # loc1=BAR (high die enters), loc2=10 (low die from 10->13)

            a1 = BackgammonNet.encode_action(BAR, 4)
            a2 = BackgammonNet.encode_action(BAR, 10)

            @test length(actions) > 0
            for a in actions
                l1, l2 = BackgammonNet.decode_action(a)
                @test l1 == BAR  # High die always enters from bar
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

            # dice=(1, 2) stored as (2, 1): dice[1]=2 (high), dice[2]=1 (low)
            g = make_test_game(board=b, dice=(1, 2))

            # Only one checker, must use higher die (2) to bear off
            # loc1 (high die 2): 24->off. loc2 (low die 1): PASS
            act = BackgammonNet.encode_action(24, PASS)

            BackgammonNet.step!(g, act)

            @test g[27] == 15
            @test g.terminated == true
            @test g.reward == 1.0f0
        end

        @testset "Hitting" begin
            b = zeros(MVector{28, Int8})
            b[1] = 1
            b[2] = -1 # Opponent Blot

            # dice=(1, 6) stored as (6, 1): dice[1]=6 (high), dice[2]=1 (low)
            g = make_test_game(board=b, dice=(1, 6))

            # Both dice can be used. Two orderings:
            # 1->2 (low die 1, hit), then 2->8 (high die 6) → encode_action(2, 1)
            # 1->7 (high die 6), then 7->8 (low die 1) → encode_action(1, 7)
            # Using first ordering: loc1=2 (high from 2 after hit), loc2=1 (low from 1)
            act = BackgammonNet.encode_action(2, 1)

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
        # New encoding: 12 board + 12 dice (2 slots) + 4 move count + 2 off + 12 cube/match = 42 minimal
        @test OBS_CHANNELS_MINIMAL == 42
        @test OBS_CHANNELS_FULL == 74     # 42 + 32 full features
        @test OBS_CHANNELS_BIASED == 134  # 74 + 60 biased features
        @test OBS_WIDTH == 26  # My bar at 1, points at 2-25, opponent bar at 26

        @test OBSERVATION_SIZES.minimal == 42
        @test OBSERVATION_SIZES.full == 74
        @test OBSERVATION_SIZES.biased == 134
        @test OBSERVATION_SIZES.width == 26

        # Test observation shapes
        g = initial_state(first_player=0)
        sample_chance!(g)

        obs_min = observe_minimal(g)
        @test size(obs_min) == (42, 1, 26)
        @test eltype(obs_min) == Float32

        obs_full = observe_full(g)
        @test size(obs_full) == (74, 1, 26)
        @test eltype(obs_full) == Float32

        obs_biased = observe_biased(g)
        @test size(obs_biased) == (134, 1, 26)
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

        # Test bar placement with current_player=1 (canonical flip)
        # Board indices 25/26 are perspective-relative: 25=my bar, 26=opp bar
        # When P1 is current player, board[25] is P1's bar, board[26] is P0's bar
        b5 = zeros(MVector{28, Int8})
        b5[25] = 2   # My bar (P1's bar when current_player=1)
        b5[26] = -1  # Opponent bar (P0's bar when current_player=1)
        g5 = make_test_game(board=b5, dice=(1, 2), current_player=1)
        obs5 = observe_minimal(g5)
        # My bar should be at spatial index 1
        @test obs5[1, 1, 1] == 1.0f0  # >=1 at my bar
        @test obs5[2, 1, 1] == 1.0f0  # >=2 at my bar
        @test obs5[3, 1, 1] == 0.0f0  # >=3 at my bar
        # Opponent bar should be at spatial index 26
        @test obs5[7, 1, 26] == 1.0f0   # >=1 at opponent bar
        @test obs5[8, 1, 26] == 0.0f0   # >=2 at opponent bar

        # Test point remapping with current_player=1 (canonical flip + spatial offset)
        # Verifies that points are correctly mapped to spatial indices 2-25 for P1
        b6 = zeros(MVector{28, Int8})
        b6[1] = 3    # My 3 checkers at canonical point 1 (spatial index 2)
        b6[24] = -2  # Opponent 2 checkers at canonical point 24 (spatial index 25)
        b6[12] = 1   # My 1 checker at canonical point 12 (spatial index 13)
        g6 = make_test_game(board=b6, dice=(1, 2), current_player=1)
        obs6 = observe_minimal(g6)
        # My checkers at canonical point 1 → spatial index 2
        @test obs6[1, 1, 2] == 1.0f0  # >=1 at point 1
        @test obs6[2, 1, 2] == 1.0f0  # >=2 at point 1
        @test obs6[3, 1, 2] == 1.0f0  # >=3 at point 1
        @test obs6[4, 1, 2] == 0.0f0  # >=4 at point 1
        # Opponent checkers at canonical point 24 → spatial index 25
        @test obs6[7, 1, 25] == 1.0f0   # >=1 at point 24
        @test obs6[8, 1, 25] == 1.0f0   # >=2 at point 24
        @test obs6[9, 1, 25] == 0.0f0   # >=3 at point 24
        # My checker at canonical point 12 → spatial index 13
        @test obs6[1, 1, 13] == 1.0f0  # >=1 at point 12
        @test obs6[2, 1, 13] == 0.0f0  # >=2 at point 12
    end

    @testset "Dice Encoding - 2 Slots + Move Count" begin
        # New encoding: 2 dice slots (ch 13-24) + move count one-hot (ch 25-28)
        # Dice slots always show rolled values (high, low)
        # Move count: bins 1, 2, 3, 4 moves

        # Non-doubles: dice (3, 5) - show both dice, bin 2 active (2 moves playable)
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g = make_test_game(board=b, dice=(3, 5), current_player=0)
        obs = observe_minimal(g)

        # Slot 0 (channels 13-18): high die = 5, channel 12+5=17
        @test obs[17, 1, 1] == 1.0f0  # Slot 0, value 5
        for v in 1:6
            if v != 5
                @test obs[12+v, 1, 1] == 0.0f0  # Other values in slot 0
            end
        end

        # Slot 1 (channels 19-24): low die = 3, channel 18+3=21
        @test obs[21, 1, 1] == 1.0f0  # Slot 1, value 3
        for v in 1:6
            if v != 3
                @test obs[18+v, 1, 1] == 0.0f0  # Other values in slot 1
            end
        end

        # Move count for non-doubles: bin 2 active (channel 26)
        @test obs[25, 1, 1] == 0.0f0  # Bin 1
        @test obs[26, 1, 1] == 1.0f0  # Bin 2 (both dice playable)
        @test obs[27, 1, 1] == 0.0f0  # Bin 3
        @test obs[28, 1, 1] == 0.0f0  # Bin 4

        # Doubles (4, 4) at initial board - should have 4 dice playable, bin 4 active
        g_init = initial_state(first_player=0)
        g_init.dice = SVector{2, Int8}(4, 4)
        g_init.remaining_actions = Int8(2)
        g_init.phase = BackgammonNet.PHASE_CHECKER_PLAY
        obs_doubles = observe_minimal(g_init)

        # Both slots show value 4
        @test obs_doubles[16, 1, 1] == 1.0f0  # Slot 0, value 4 (ch 12+4=16)
        @test obs_doubles[22, 1, 1] == 1.0f0  # Slot 1, value 4 (ch 18+4=22)
        # Move count: bin 4 active (4 dice playable at initial position)
        @test obs_doubles[25, 1, 1] == 0.0f0  # Bin 1
        @test obs_doubles[26, 1, 1] == 0.0f0  # Bin 2
        @test obs_doubles[27, 1, 1] == 0.0f0  # Bin 3
        @test obs_doubles[28, 1, 1] == 1.0f0  # Bin 4

        # Doubles with remaining=1 (second action): bin 2 active
        g_second = initial_state(first_player=0)
        g_second.dice = SVector{2, Int8}(4, 4)
        g_second.remaining_actions = Int8(1)
        g_second.phase = BackgammonNet.PHASE_CHECKER_PLAY
        obs_second = observe_minimal(g_second)
        @test obs_second[16, 1, 1] == 1.0f0  # Still show dice
        @test obs_second[22, 1, 1] == 1.0f0
        @test obs_second[26, 1, 1] == 1.0f0  # Bin 2 for second action

        # Chance node (no dice): all zeros
        g_chance = initial_state(first_player=0)
        obs_chance = observe_minimal(g_chance)
        for ch in 13:28
            @test obs_chance[ch, 1, 1] == 0.0f0
        end
    end

    @testset "Move Count - Playable Dice Edge Cases" begin
        # Test the precise playable dice computation for doubles
        # Bins: 1, 2, 3, 4 at channels 25, 26, 27, 28

        # Case 1: 4 dice playable (normal doubles) - bin 4
        g_4dice = initial_state(first_player=0)
        g_4dice.dice = SVector{2, Int8}(3, 3)
        g_4dice.remaining_actions = Int8(2)
        g_4dice.phase = BackgammonNet.PHASE_CHECKER_PLAY
        obs_4 = observe_minimal(g_4dice)
        @test obs_4[28, 1, 1] == 1.0f0  # Bin 4 active

        # Case 2: 3 dice playable (blocked after 3 moves) - bin 3
        # Single checker at point 1 with dice 2-2, opponent blocks point 9
        # Checker can move 1→3→5→7 (3 moves) but not 7→9 (blocked)
        b_3dice = zeros(MVector{28, Int8})
        b_3dice[1] = 1   # My single checker at point 1
        b_3dice[9] = -2  # Opponent block at point 9
        g_3dice = make_test_game(board=b_3dice, dice=(2, 2), remaining=2, current_player=0)
        obs_3 = observe_minimal(g_3dice)
        @test obs_3[27, 1, 1] == 1.0f0  # Bin 3 active (channels 25-28 for bins 1-4)
        @test obs_3[28, 1, 1] == 0.0f0  # Bin 4 inactive

        # Case 3: 2 dice playable (blocked doubles - bin 2)
        # Create position where player is blocked after 2 dice
        # 6 consecutive opponent blocks: primed
        b_primed = zeros(MVector{28, Int8})
        b_primed[1] = 1   # My single checker at point 1
        # Opponent prime from points 7-12 (blocks any 6-move)
        for pt in 7:12
            b_primed[pt] = -2
        end
        g_primed = make_test_game(board=b_primed, dice=(6, 6), remaining=2, current_player=0)
        obs_primed = observe_minimal(g_primed)
        # With dice 6,6 and blocked by prime, only 0 dice playable
        # All bins should be 0
        @test all(obs_primed[25:28, 1, 1] .== 0.0f0)  # All bins 0 for completely blocked

        # Case 4: 1 die playable (blocked doubles - bin 1)
        # Checker can move once but then completely blocked
        b_1die = zeros(MVector{28, Int8})
        b_1die[1] = 1   # My checker at point 1
        # Prime from 3-8 (so checker at 1 can move to 2, but then blocked)
        for pt in 3:8
            b_1die[pt] = -2
        end
        g_1die = make_test_game(board=b_1die, dice=(1, 1), remaining=2, current_player=0)
        obs_1die = observe_minimal(g_1die)
        # Can move 1→2, then blocked by prime at 3 - only 1 die playable
        @test obs_1die[25, 1, 1] == 1.0f0  # Bin 1 active
        @test obs_1die[26, 1, 1] == 0.0f0  # Bin 2 inactive

        # Case 5: 0 dice playable (completely blocked - all bins 0)
        b_blocked = zeros(MVector{28, Int8})
        b_blocked[25] = 1  # My checker on bar
        # Block all entry points (1-6)
        for pt in 1:6
            b_blocked[pt] = -2
        end
        g_blocked = make_test_game(board=b_blocked, dice=(3, 3), remaining=2, current_player=0)
        obs_blocked = observe_minimal(g_blocked)
        @test all(obs_blocked[25:28, 1, 1] .== 0.0f0)  # All bins 0 for completely blocked
    end

    @testset "Move Count - Non-Doubles Playable Dice" begin
        # Test that non-doubles correctly compute playable dice count
        # Previously, non-doubles always defaulted to bin 2 (2 moves)
        # Now we compute the actual playable count

        # Use ObsChannels constants for maintainability
        BIN_1 = ObsChannels.MOVE_BIN_1
        BIN_2 = ObsChannels.MOVE_BIN_2

        # Case 1: Both dice playable - bin 2
        # Checker at point 1 with dice (5, 3), both moves possible
        b_both = zeros(MVector{28, Int8})
        b_both[1] = 2  # 2 checkers at point 1
        g_both = make_test_game(board=b_both, dice=(5, 3), remaining=1, current_player=0)
        obs_both = observe_minimal(g_both)
        @test obs_both[BIN_1, 1, 1] == 0.0f0  # Bin 1 inactive
        @test obs_both[BIN_2, 1, 1] == 1.0f0  # Bin 2 active (2 dice playable)

        # Case 2: Only 1 die playable (only high die works) - bin 1
        # Checker at point 5, dice (6, 3)
        # Low die blocked: 5+3=8 blocked
        # High die: 5+6=11 open, but then 11+3=14 blocked
        # Result: only high die playable → bin 1
        b_high_only = zeros(MVector{28, Int8})
        b_high_only[5] = 1    # My checker at point 5
        b_high_only[8] = -2   # Block at 8 (blocks low die: 5→8)
        b_high_only[14] = -2  # Block at 14 (blocks low die after high: 11→14)
        g_high_only = make_test_game(board=b_high_only, dice=(6, 3), remaining=1, current_player=0)
        obs_high_only = observe_minimal(g_high_only)
        @test obs_high_only[BIN_1, 1, 1] == 1.0f0  # Bin 1 active (only 1 die playable)
        @test obs_high_only[BIN_2, 1, 1] == 0.0f0  # Bin 2 inactive

        # Case 3: Only 1 die playable (only low die works) - bin 1
        # Checker at point 2, dice (6, 3)
        # High die blocked: 2+6=8 blocked
        # Low die: 2+3=5 open, but then 5+6=11 blocked
        # Result: only low die playable → bin 1
        b_low_only = zeros(MVector{28, Int8})
        b_low_only[2] = 1    # My checker at point 2
        b_low_only[8] = -2   # Block at 8 (blocks high die: 2→8)
        b_low_only[11] = -2  # Block at 11 (blocks high die after low: 5→11)
        g_low_only = make_test_game(board=b_low_only, dice=(6, 3), remaining=1, current_player=0)
        obs_low_only = observe_minimal(g_low_only)
        @test obs_low_only[BIN_1, 1, 1] == 1.0f0  # Bin 1 active (only 1 die playable)
        @test obs_low_only[BIN_2, 1, 1] == 0.0f0  # Bin 2 inactive

        # Case 4: 0 dice playable (completely blocked non-doubles) - all bins 0
        # Checker on bar, entry points blocked
        b_nd_blocked = zeros(MVector{28, Int8})
        b_nd_blocked[25] = 1   # My checker on bar
        b_nd_blocked[3] = -2   # Block point 3 (entry for die 3)
        b_nd_blocked[5] = -2   # Block point 5 (entry for die 5)
        g_nd_blocked = make_test_game(board=b_nd_blocked, dice=(5, 3), remaining=1, current_player=0)
        obs_nd_blocked = observe_minimal(g_nd_blocked)
        # All move count bins should be 0
        @test all(obs_nd_blocked[ObsChannels.MOVE_COUNT_START:ObsChannels.MOVE_COUNT_END, 1, 1] .== 0.0f0)

        # Case 5: Using one die blocks the other - still bin 2 if both can be used somehow
        # This tests the "must use higher die" rule interaction
        # Checker at 20 (close to bearing off), dice (6, 2)
        # Can bear off with 6, or move 20→22 with 2
        # If we use 6 first, we bear off and can't use 2
        # If we use 2 first (20→22), we can then bear off with 6 (22→off)
        # Per rules, if using both is possible (in either order), we must use both
        # So this should be bin 2
        b_bearoff = zeros(MVector{28, Int8})
        b_bearoff[20] = 1   # My checker at point 20 (5 away from off)
        b_bearoff[21] = 1   # Another checker at 21
        g_bearoff = make_test_game(board=b_bearoff, dice=(6, 2), remaining=1, current_player=0)
        obs_bearoff = observe_minimal(g_bearoff)
        # Should be able to use both dice in correct order
        @test obs_bearoff[BIN_2, 1, 1] == 1.0f0  # Bin 2 active
    end

    @testset "Move Count - Doubles Second Action" begin
        # Test that doubles second action (remaining_actions=1) correctly computes
        # playable moves instead of hard-coding 2
        BIN_1 = ObsChannels.MOVE_BIN_1
        BIN_2 = ObsChannels.MOVE_BIN_2

        # Case 1: Both moves playable in second action - bin 2
        # Two checkers at point 1 with dice 3-3, remaining=1 (second action)
        b_both = zeros(MVector{28, Int8})
        b_both[1] = 2  # 2 checkers at point 1
        g_both = make_test_game(board=b_both, dice=(3, 3), remaining=1, current_player=0)
        obs_both = observe_minimal(g_both)
        @test obs_both[BIN_2, 1, 1] == 1.0f0  # Bin 2 active (2 moves playable)

        # Case 2: Only 1 move playable in second action - bin 1
        # Single checker blocked after one move
        b_1move = zeros(MVector{28, Int8})
        b_1move[1] = 1   # My checker at point 1
        # Prime from 4-9 blocks 2nd move (checker can go 1→4 but then 4→7 blocked)
        for pt in 4:9
            b_1move[pt] = -2
        end
        g_1move = make_test_game(board=b_1move, dice=(3, 3), remaining=1, current_player=0)
        # Wait - this has remaining=1 but the prime starts at 4, so 1+3=4 is blocked
        # Let me reconsider: 1+3=4 is blocked, so 0 moves playable
        # Need a different setup: checker can make 1 move, then blocked

        # Better setup: checker at 1, blocks at 7-12 (6-prime)
        # With dice 3-3: 1→4 (ok), 4→7 (blocked)
        b_1move2 = zeros(MVector{28, Int8})
        b_1move2[1] = 1
        for pt in 7:12
            b_1move2[pt] = -2
        end
        g_1move2 = make_test_game(board=b_1move2, dice=(3, 3), remaining=1, current_player=0)
        obs_1move2 = observe_minimal(g_1move2)
        @test obs_1move2[BIN_1, 1, 1] == 1.0f0  # Bin 1 active (1 move playable)
        @test obs_1move2[BIN_2, 1, 1] == 0.0f0  # Bin 2 inactive

        # Case 3: 0 moves playable in second action - all bins 0
        # Checker on bar, entry points blocked
        b_0move = zeros(MVector{28, Int8})
        b_0move[25] = 1  # My checker on bar
        b_0move[3] = -2  # Block entry point 3 (dice 3-3)
        g_0move = make_test_game(board=b_0move, dice=(3, 3), remaining=1, current_player=0)
        obs_0move = observe_minimal(g_0move)
        @test all(obs_0move[ObsChannels.MOVE_COUNT_START:ObsChannels.MOVE_COUNT_END, 1, 1] .== 0.0f0)
    end

    @testset "Off Counts Encoding" begin
        b = zeros(MVector{28, Int8})
        b[27] = 5   # My off = 5
        b[28] = -3  # Opp off = 3
        g = make_test_game(board=b, dice=(1, 2), current_player=0)
        obs = observe_minimal(g)

        # Channel 29: my off / 15
        @test obs[29, 1, 1] ≈ 5.0f0/15.0f0 atol=1e-6
        # Channel 30: opp off / 15
        @test obs[30, 1, 1] ≈ 3.0f0/15.0f0 atol=1e-6
    end

    @testset "Full Observation Features" begin
        # Test dice_sum (channel 43)
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g_sum = make_test_game(board=b, dice=(4, 3), current_player=0)
        obs_sum = observe_full(g_sum)
        @test obs_sum[43, 1, 1] ≈ 7.0f0/12.0f0 atol=1e-6  # (4+3)/12

        # Test dice_delta (channel 44)
        g_delta = make_test_game(board=b, dice=(6, 2), current_player=0)
        obs_delta = observe_full(g_delta)
        @test obs_delta[44, 1, 1] ≈ 4.0f0/5.0f0 atol=1e-6  # |6-2|/5 = 0.8

        # Test dice_delta for doubles (should be 0)
        g_doubles = make_test_game(board=b, dice=(3, 3), remaining=2, current_player=0)
        obs_doubles = observe_full(g_doubles)
        @test obs_doubles[44, 1, 1] == 0.0f0  # |3-3|/5 = 0

        # Test contact indicator (channel 45)
        # Race position: no contact
        b_race = zeros(MVector{28, Int8})
        b_race[20] = 5  # My checkers in home (point 20)
        b_race[5] = -5  # Opp checkers far behind (point 5)
        g_race = make_test_game(board=b_race, dice=(1, 2), current_player=0)
        obs_race = observe_full(g_race)
        @test obs_race[45, 1, 1] == 0.0f0  # No contact (is race)

        # Contact position
        b_contact = zeros(MVector{28, Int8})
        b_contact[10] = 1   # My checker
        b_contact[15] = -1  # Opp checker ahead of me
        g_contact = make_test_game(board=b_contact, dice=(1, 2), current_player=0)
        obs_contact = observe_full(g_contact)
        @test obs_contact[45, 1, 1] == 1.0f0  # Has contact

        # Test pip counts (channels 46-48)
        b_pip = zeros(MVector{28, Int8})
        b_pip[1] = 2   # My 2 checkers at point 1: pip value = 2 * (25-1) = 48
        b_pip[24] = -3 # Opp 3 checkers at point 24: pip value = 3 * 24 = 72
        g_pip = make_test_game(board=b_pip, dice=(1, 2), current_player=0)
        obs_pip = observe_full(g_pip)
        @test obs_pip[46, 1, 1] ≈ 48.0f0/167.0f0 atol=1e-5  # My pips
        @test obs_pip[47, 1, 1] ≈ 72.0f0/167.0f0 atol=1e-5  # Opp pips
        @test obs_pip[48, 1, 1] ≈ (48.0f0-72.0f0)/167.0f0 atol=1e-5  # Pip diff

        # Test can bear off (channels 49-50)
        b_bearoff = zeros(MVector{28, Int8})
        b_bearoff[20] = 5  # My checkers in home (19-24)
        b_bearoff[2] = -5  # Opp checkers in their home (1-6)
        g_bearoff = make_test_game(board=b_bearoff, dice=(1, 2), current_player=0)
        obs_bearoff = observe_full(g_bearoff)
        @test obs_bearoff[49, 1, 1] == 1.0f0  # I can bear off
        @test obs_bearoff[50, 1, 1] == 1.0f0  # Opp can bear off

        # Cannot bear off: checker outside home
        b_no_bear = zeros(MVector{28, Int8})
        b_no_bear[10] = 1  # My checker outside home
        b_no_bear[20] = 4
        g_no_bear = make_test_game(board=b_no_bear, dice=(1, 2), current_player=0)
        obs_no_bear = observe_full(g_no_bear)
        @test obs_no_bear[49, 1, 1] == 0.0f0  # I cannot bear off

        # Test stragglers (channels 51-62)
        b_strag = zeros(MVector{28, Int8})
        b_strag[10] = 3  # 3 stragglers (outside home 19-24)
        b_strag[20] = 2  # 2 in home (not stragglers)
        g_strag = make_test_game(board=b_strag, dice=(1, 2), current_player=0)
        obs_strag = observe_full(g_strag)
        # My stragglers = 3: channels 51-53 should be 1
        @test obs_strag[51, 1, 1] == 1.0f0  # >=1
        @test obs_strag[52, 1, 1] == 1.0f0  # >=2
        @test obs_strag[53, 1, 1] == 1.0f0  # >=3
        @test obs_strag[54, 1, 1] == 0.0f0  # >=4

        # Test remaining (channels 63-74)
        b_remain = zeros(MVector{28, Int8})
        b_remain[20] = 5
        b_remain[27] = 10  # 10 off, so 5 remaining
        g_remain = make_test_game(board=b_remain, dice=(1, 2), current_player=0)
        obs_remain = observe_full(g_remain)
        # My remaining = 5: channels 63-67 should be 1
        @test obs_remain[63, 1, 1] == 1.0f0  # >=1
        @test obs_remain[67, 1, 1] == 1.0f0  # >=5
        @test obs_remain[68, 1, 1] == 0.0f0  # 6+ (should be 0)
    end

    @testset "Biased Observation Features" begin
        # Test prime length (channels 75-86)
        b_prime = zeros(MVector{28, Int8})
        b_prime[5] = 2   # Block
        b_prime[6] = 2   # Block
        b_prime[7] = 2   # Block - 3 consecutive blocks
        b_prime[10] = -2 # Opp block (isolated)
        g_prime = make_test_game(board=b_prime, dice=(1, 2), current_player=0)
        obs_prime = observe_biased(g_prime)
        # My prime = 3
        @test obs_prime[75, 1, 1] == 1.0f0  # >=1
        @test obs_prime[77, 1, 1] == 1.0f0  # >=3
        @test obs_prime[78, 1, 1] == 0.0f0  # >=4
        # Opp prime = 1
        @test obs_prime[81, 1, 1] == 1.0f0  # >=1
        @test obs_prime[82, 1, 1] == 0.0f0  # >=2

        # Test home board blocks (channels 87-98)
        b_home = zeros(MVector{28, Int8})
        b_home[20] = 2  # My block in home (19-24)
        b_home[21] = 2  # Another block in home
        b_home[2] = -2  # Opp block in their home (1-6)
        g_home = make_test_game(board=b_home, dice=(1, 2), current_player=0)
        obs_home = observe_biased(g_home)
        # My home blocks = 2
        @test obs_home[87, 1, 1] == 1.0f0  # >=1
        @test obs_home[88, 1, 1] == 1.0f0  # >=2
        @test obs_home[89, 1, 1] == 0.0f0  # >=3
        # Opp home blocks = 1
        @test obs_home[93, 1, 1] == 1.0f0  # >=1
        @test obs_home[94, 1, 1] == 0.0f0  # >=2

        # Test anchors (channels 99-110)
        b_anchor = zeros(MVector{28, Int8})
        b_anchor[3] = 2   # My anchor in opp's home (1-6)
        b_anchor[20] = -2 # Opp anchor in my home (19-24)
        g_anchor = make_test_game(board=b_anchor, dice=(1, 2), current_player=0)
        obs_anchor = observe_biased(g_anchor)
        # My anchors = 1
        @test obs_anchor[99, 1, 1] == 1.0f0
        @test obs_anchor[100, 1, 1] == 0.0f0
        # Opp anchors = 1
        @test obs_anchor[105, 1, 1] == 1.0f0
        @test obs_anchor[106, 1, 1] == 0.0f0

        # Test blot count (channels 111-122)
        b_blot = zeros(MVector{28, Int8})
        b_blot[5] = 1   # My blot
        b_blot[10] = 1  # Another blot
        b_blot[15] = -1 # Opp blot
        g_blot = make_test_game(board=b_blot, dice=(1, 2), current_player=0)
        obs_blot = observe_biased(g_blot)
        # My blots = 2
        @test obs_blot[111, 1, 1] == 1.0f0   # >=1
        @test obs_blot[112, 1, 1] == 1.0f0  # >=2
        @test obs_blot[113, 1, 1] == 0.0f0  # >=3
        # Opp blots = 1
        @test obs_blot[117, 1, 1] == 1.0f0
        @test obs_blot[118, 1, 1] == 0.0f0

        # Test builder count (channels 123-134)
        b_builder = zeros(MVector{28, Int8})
        b_builder[5] = 2   # My builder
        b_builder[10] = 2  # Another builder
        b_builder[15] = 2  # Third builder
        b_builder[20] = -2 # Opp builder
        g_builder = make_test_game(board=b_builder, dice=(1, 2), current_player=0)
        obs_builder = observe_biased(g_builder)
        # My builders = 3
        @test obs_builder[123, 1, 1] == 1.0f0
        @test obs_builder[125, 1, 1] == 1.0f0  # >=3
        @test obs_builder[126, 1, 1] == 0.0f0  # >=4
        # Opp builders = 1
        @test obs_builder[129, 1, 1] == 1.0f0
        @test obs_builder[130, 1, 1] == 0.0f0
    end

    @testset "Observation Hierarchy" begin
        # Test that observations build on each other
        g = initial_state(first_player=0)
        sample_chance!(g)

        obs_min = observe_minimal(g)
        obs_full = observe_full(g)
        obs_biased = observe_biased(g)

        # Full should contain minimal (channels 1-42)
        @test obs_full[1:42, :, :] ≈ obs_min atol=1e-6

        # Biased should contain full (channels 1-74)
        @test obs_biased[1:74, :, :] ≈ obs_full atol=1e-6
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
                @test obs_full[1:42, :, :] ≈ obs_min atol=1e-6
                @test obs_biased[1:74, :, :] ≈ obs_full atol=1e-6
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
        @test obs_chance[43, 1, 1] == 0.0f0  # dice_sum should be 0 at chance node

        # Test dice_sum minimum (1+1=2)
        b = zeros(MVector{28, Int8})
        b[1] = 2
        g_min_dice = make_test_game(board=b, dice=(1, 1), remaining=2, current_player=0)
        obs_min_dice = observe_full(g_min_dice)
        @test obs_min_dice[43, 1, 1] ≈ 2.0f0/12.0f0 atol=1e-6

        # Test dice_sum maximum (6+6=12)
        g_max_dice = make_test_game(board=b, dice=(6, 6), remaining=2, current_player=0)
        obs_max_dice = observe_full(g_max_dice)
        @test obs_max_dice[43, 1, 1] ≈ 1.0f0 atol=1e-6  # 12/12 = 1.0

        # Test dice_delta on chance node (should be 0)
        @test obs_chance[44, 1, 1] == 0.0f0  # dice_delta should be 0 at chance node

        # Test contact on empty board (no checkers)
        b_empty = zeros(MVector{28, Int8})
        g_empty = make_test_game(board=b_empty, dice=(3, 4), current_player=0)
        obs_empty = observe_full(g_empty)
        @test obs_empty[45, 1, 1] == 0.0f0  # No contact on empty board

        # Test contact when only one side has checkers
        b_one_side = zeros(MVector{28, Int8})
        b_one_side[10] = 5  # Only my checkers
        g_one_side = make_test_game(board=b_one_side, dice=(3, 4), current_player=0)
        obs_one_side = observe_full(g_one_side)
        @test obs_one_side[45, 1, 1] == 0.0f0  # No contact (no opponent)

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
        @test obs_empty[46, 1, 1] == 0.0f0  # my_pips
        @test obs_empty[47, 1, 1] == 0.0f0  # opp_pips
        @test obs_empty[48, 1, 1] == 0.0f0  # pip_diff

        # Test can_bear_off on empty board (vacuously true - no checkers outside home)
        @test obs_empty[49, 1, 1] == 1.0f0  # can bear off (no checkers to block it)
        @test obs_empty[50, 1, 1] == 1.0f0  # opp can bear off
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

        @testset "clone() creates independent copy" begin
            # Create a game and play a few moves
            g = initial_state(first_player=0)
            sample_chance!(g, MersenneTwister(42))
            actions = legal_actions(g)
            step!(g, actions[1], MersenneTwister(42))

            # Clone it
            g_copy = clone(g)

            # Verify state matches
            @test g_copy.p0 == g.p0
            @test g_copy.p1 == g.p1
            @test g_copy.dice == g.dice
            @test g_copy.remaining_actions == g.remaining_actions
            @test g_copy.current_player == g.current_player
            @test g_copy.terminated == g.terminated
            @test g_copy.reward == g.reward
            @test g_copy.doubles_only == g.doubles_only
            @test g_copy.history == g.history

            # Verify buffers are independent (not same object)
            @test g_copy._actions_buffer !== g._actions_buffer
            @test g_copy._sources_buffer1 !== g._sources_buffer1
            @test g_copy._sources_buffer2 !== g._sources_buffer2
            @test g_copy.history !== g.history

            # Verify cache is invalidated on clone
            @test !g_copy._actions_cached

            # Modify clone and verify original unchanged
            sample_chance!(g_copy, MersenneTwister(1))
            actions_copy = legal_actions(g_copy)
            step!(g_copy, actions_copy[1], MersenneTwister(1))

            @test g_copy.p0 != g.p0 || g_copy.p1 != g.p1 || g_copy.dice != g.dice
            @test length(g_copy.history) > length(g.history)
        end

    end

    @testset "Dice Encoding Edge Cases" begin
        # Test dice encoding at chance node (dice = 0, 0)
        g = initial_state(first_player=0)
        @test is_chance_node(g)
        obs = observe_minimal(g)
        # Dice channels 13-28 should all be zero at chance node (2 slots + move count bins)
        @test all(obs[13:28, 1, 1] .== 0.0f0)

        # Test with remaining_actions = 0 (edge case that shouldn't occur in normal play)
        # Note: with new encoding, dice slots still show rolled values even with remaining=0
        b = zeros(MVector{28, Int8})
        b[1] = 1
        g2 = make_test_game(board=b, dice=(3, 4), remaining=0, current_player=0)
        obs2 = observe_minimal(g2)
        # New encoding shows rolled dice regardless of remaining_actions
        # High die = 4 at channel 12+4=16, Low die = 3 at channel 18+3=21
        @test obs2[16, 1, 1] == 1.0f0  # High die 4
        @test obs2[21, 1, 1] == 1.0f0  # Low die 3
        # Move count bin 2 active (non-doubles: both dice playable)
        @test obs2[26, 1, 1] == 1.0f0  # Bin 2 active

        # Test with d1=0 (partial invalid state)
        g3 = make_test_game(board=b, dice=(0, 4), remaining=1, current_player=0)
        obs3 = observe_minimal(g3)
        @test all(obs3[13:28, 1, 1] .== 0.0f0)

        # Test with d2=0 (partial invalid state)
        g4 = make_test_game(board=b, dice=(3, 0), remaining=1, current_player=0)
        obs4 = observe_minimal(g4)
        @test all(obs4[13:28, 1, 1] .== 0.0f0)
    end

    # NOTE: Old observation tests for blot/block detection and pip count difference
    # have been removed as they tested the legacy observation API which has been
    # replaced by the new 3-tier observation system (minimal, full, biased).

    @testset "Flat Observations" begin
        @testset "observe_minimal_flat dimensions" begin
            g = initial_state(first_player=0)
            sample_chance!(g)
            obs_flat = observe_minimal_flat(g)
            @test length(obs_flat) == OBS_FLAT_MINIMAL
            @test length(obs_flat) == 342
        end

        @testset "observe_full_flat dimensions" begin
            g = initial_state(first_player=0)
            sample_chance!(g)
            obs_flat = observe_full_flat(g)
            @test length(obs_flat) == OBS_FLAT_FULL
            @test length(obs_flat) == 374
        end

        @testset "observe_biased_flat dimensions" begin
            g = initial_state(first_player=0)
            sample_chance!(g)
            obs_flat = observe_biased_flat(g)
            @test length(obs_flat) == OBS_FLAT_BIASED
            @test length(obs_flat) == 434
        end

        @testset "flat vs 3D feature equivalence" begin
            # Test that flat and 3D produce same feature values
            g = initial_state(first_player=0)
            sample_chance!(g)

            obs_3d = observe_minimal(g)
            obs_flat = observe_minimal_flat(g)

            # Check board encoding (312 values)
            # 3D: channels 1-12 across width 26
            # Flat: positions 1-312 as [pos1_my_thresh1-6, pos1_opp_thresh1-6, pos2_..., ...]
            for w in 1:26
                for ch in 1:6
                    # My checker thresholds
                    flat_idx = (w-1)*12 + ch
                    @test obs_flat[flat_idx] == obs_3d[ch, 1, w]
                    # Opp checker thresholds
                    flat_idx_opp = (w-1)*12 + 6 + ch
                    @test obs_flat[flat_idx_opp] == obs_3d[6+ch, 1, w]
                end
            end

            # Check dice encoding (12 values at indices 313-324)
            # High die one-hot (6 values)
            for v in 1:6
                @test obs_flat[312 + v] == obs_3d[12 + v, 1, 1]
            end
            # Low die one-hot (6 values)
            for v in 1:6
                @test obs_flat[318 + v] == obs_3d[18 + v, 1, 1]
            end

            # Check move count encoding (4 values at indices 325-328)
            for bin in 1:4
                @test obs_flat[324 + bin] == obs_3d[24 + bin, 1, 1]
            end

            # Check off counts (2 values at indices 329-330)
            @test obs_flat[329] == obs_3d[29, 1, 1]
            @test obs_flat[330] == obs_3d[30, 1, 1]
        end

        @testset "flat in-place version" begin
            g = initial_state(first_player=0)
            sample_chance!(g)

            obs_alloc = observe_minimal_flat(g)
            obs_inplace = zeros(Float32, OBS_FLAT_MINIMAL)
            observe_minimal_flat!(obs_inplace, g)

            @test obs_alloc == obs_inplace
        end

        @testset "OBSERVATION_SIZES includes flat" begin
            @test OBSERVATION_SIZES.minimal_flat == 342
            @test OBSERVATION_SIZES.full_flat == 374
            @test OBSERVATION_SIZES.biased_flat == 434
        end

        @testset "observe() dispatches on obs_type" begin
            # Test flat observations
            g_flat = initial_state(first_player=0, obs_type=:minimal_flat)
            sample_chance!(g_flat)
            obs_flat = observe(g_flat)
            @test obs_flat isa Vector{Float32}
            @test length(obs_flat) == 342

            # Test 3D observations
            g_3d = initial_state(first_player=0, obs_type=:full)
            sample_chance!(g_3d)
            obs_3d = observe(g_3d)
            @test obs_3d isa Array{Float32, 3}
            @test size(obs_3d) == (74, 1, 26)

            # Test all observation types
            for (obs_type, expected_size) in [
                (:minimal, (42, 1, 26)),
                (:full, (74, 1, 26)),
                (:biased, (134, 1, 26)),
                (:minimal_flat, 342),
                (:full_flat, 374),
                (:biased_flat, 434),
            ]
                g = initial_state(first_player=0, obs_type=obs_type)
                sample_chance!(g)
                obs = observe(g)
                if obs_type in (:minimal, :full, :biased)
                    @test size(obs) == expected_size
                else
                    @test length(obs) == expected_size
                end
            end
        end

        @testset "obs_dims returns correct dimensions" begin
            # Test game-level obs_dims
            g = initial_state(obs_type=:minimal_flat)
            @test obs_dims(g) == 342

            g2 = initial_state(obs_type=:full)
            @test obs_dims(g2) == (74, 1, 26)

            # Test symbol-level obs_dims
            @test obs_dims(:minimal) == (42, 1, 26)
            @test obs_dims(:full) == (74, 1, 26)
            @test obs_dims(:biased) == (134, 1, 26)
            @test obs_dims(:minimal_flat) == 342
            @test obs_dims(:full_flat) == 374
            @test obs_dims(:biased_flat) == 434
        end

        @testset "set_obs_type! changes observation type" begin
            g = initial_state(obs_type=:minimal_flat)
            @test g.obs_type == :minimal_flat
            @test obs_dims(g) == 342

            set_obs_type!(g, :full)
            @test g.obs_type == :full
            @test obs_dims(g) == (74, 1, 26)

            # Test that observe() returns correct type after change
            sample_chance!(g)
            obs = observe(g)
            @test size(obs) == (74, 1, 26)
        end

        @testset "clone preserves obs_type" begin
            g = initial_state(obs_type=:biased_flat)
            sample_chance!(g)
            g_clone = clone(g)
            @test g_clone.obs_type == :biased_flat
            @test obs_dims(g_clone) == 434
        end

        @testset "hybrid observations" begin
            g = initial_state(first_player=0)
            sample_chance!(g)

            # Test minimal hybrid
            obs = observe_minimal_hybrid(g)
            @test obs isa NamedTuple
            @test haskey(obs, :board)
            @test haskey(obs, :globals)
            @test size(obs.board) == (12, 26)
            @test length(obs.globals) == 30

            # Test full hybrid
            obs_full = observe_full_hybrid(g)
            @test size(obs_full.board) == (12, 26)
            @test length(obs_full.globals) == 62

            # Test biased hybrid
            obs_biased = observe_biased_hybrid(g)
            @test size(obs_biased.board) == (12, 26)
            @test length(obs_biased.globals) == 122

            # Test that board matches flat encoding
            obs_flat = observe_minimal_flat(g)
            for w in 1:26
                for ch in 1:6
                    flat_idx = (w-1)*12 + ch
                    @test obs.board[ch, w] == obs_flat[flat_idx]
                    flat_idx_opp = (w-1)*12 + 6 + ch
                    @test obs.board[6+ch, w] == obs_flat[flat_idx_opp]
                end
            end

            # Test that globals match flat encoding
            for i in 1:30
                @test obs.globals[i] == obs_flat[312 + i]
            end
        end

        @testset "hybrid via observe() dispatch" begin
            g = initial_state(obs_type=:minimal_hybrid)
            sample_chance!(g)
            obs = observe(g)
            @test obs isa NamedTuple
            @test size(obs.board) == (12, 26)
            @test length(obs.globals) == 30

            # Test obs_dims for hybrid
            dims = obs_dims(g)
            @test dims.board == (12, 26)
            @test dims.globals == 30

            dims_full = obs_dims(:full_hybrid)
            @test dims_full.board == (12, 26)
            @test dims_full.globals == 62
        end
    end

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

    # ========================================================================
    # Cube, Match Play, and Context Observations
    # ========================================================================

    @testset "Cube State Initialization" begin
        g = initial_state()
        @test g.cube_value == Int16(1)
        @test g.cube_owner == Int8(-1)
        @test g.cube_enabled == false
        @test g.phase == BackgammonNet.PHASE_CHANCE
        @test g.my_away == Int8(0)
        @test g.opp_away == Int8(0)
        @test g.is_crawford == false
        @test g.is_post_crawford == false
        @test g.jacoby_enabled == false
    end

    @testset "may_double Edge Cases" begin
        g = initial_state()

        # Default: cube disabled
        @test BackgammonNet.may_double(g) == false

        # Enable cube
        g.cube_enabled = true
        @test BackgammonNet.may_double(g) == true

        # Cube owned by opponent (other player owns)
        g.cube_owner = Int8(1 - g.current_player)
        @test BackgammonNet.may_double(g) == false

        # Cube owned by current player
        g.cube_owner = g.current_player
        @test BackgammonNet.may_double(g) == true

        # Centered cube
        g.cube_owner = Int8(-1)
        @test BackgammonNet.may_double(g) == true

        # Crawford game
        g.is_crawford = true
        @test BackgammonNet.may_double(g) == false

        # Terminated game
        g.is_crawford = false
        g.terminated = true
        @test BackgammonNet.may_double(g) == false
    end

    @testset "Cube Action Constants" begin
        @test BackgammonNet.ACTION_CUBE_NO_DOUBLE == 677
        @test BackgammonNet.ACTION_CUBE_DOUBLE == 678
        @test BackgammonNet.ACTION_CUBE_TAKE == 679
        @test BackgammonNet.ACTION_CUBE_PASS == 680
        @test BackgammonNet.MAX_ACTIONS == 680

        @test BackgammonNet.is_cube_action(677) == true
        @test BackgammonNet.is_cube_action(680) == true
        @test BackgammonNet.is_cube_action(676) == false
        @test BackgammonNet.is_cube_action(1) == false
    end

    @testset "Cube Decision Legal Actions" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        g.phase = BackgammonNet.PHASE_CUBE_DECISION

        actions = legal_actions(g)
        @test actions == [677, 678]
        @test is_action_valid(g, 677) == true
        @test is_action_valid(g, 678) == true
        @test is_action_valid(g, 679) == false
        @test is_action_valid(g, 680) == false
        @test is_action_valid(g, 1) == false
    end

    @testset "Cube Response Legal Actions" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        g.phase = BackgammonNet.PHASE_CUBE_RESPONSE

        actions = legal_actions(g)
        @test actions == [679, 680]
        @test is_action_valid(g, 679) == true
        @test is_action_valid(g, 680) == true
        @test is_action_valid(g, 677) == false
        @test is_action_valid(g, 678) == false
    end

    @testset "Cube Action Flow: NO_DOUBLE" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        sample_chance!(g)

        # Play moves until cube decision
        while g.phase == BackgammonNet.PHASE_CHECKER_PLAY && !game_terminated(g)
            actions = legal_actions(g)
            apply_action!(g, actions[1])
        end

        if g.phase == BackgammonNet.PHASE_CUBE_DECISION
            old_player = g.current_player
            apply_action!(g, 677)  # NO_DOUBLE
            @test g.phase == BackgammonNet.PHASE_CHANCE
            @test g.cube_value == Int16(1)
            @test is_chance_node(g)
        end
    end

    @testset "Cube Action Flow: DOUBLE -> TAKE" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        sample_chance!(g)

        while g.phase == BackgammonNet.PHASE_CHECKER_PLAY && !game_terminated(g)
            actions = legal_actions(g)
            apply_action!(g, actions[1])
        end

        if g.phase == BackgammonNet.PHASE_CUBE_DECISION
            doubler = g.current_player
            apply_action!(g, 678)  # DOUBLE
            @test g.phase == BackgammonNet.PHASE_CUBE_RESPONSE
            @test g.current_player == 1 - doubler

            taker = g.current_player  # Taker is current player before TAKE
            apply_action!(g, 679)  # TAKE
            @test g.cube_value == Int16(2)
            @test g.cube_owner == taker  # Taker owns (absolute player ID)
            @test g.phase == BackgammonNet.PHASE_CHANCE
            @test g.current_player == doubler  # Switched back
            @test !game_terminated(g)
        end
    end

    @testset "Cube Action Flow: DOUBLE -> PASS" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        sample_chance!(g)

        while g.phase == BackgammonNet.PHASE_CHECKER_PLAY && !game_terminated(g)
            actions = legal_actions(g)
            apply_action!(g, actions[1])
        end

        if g.phase == BackgammonNet.PHASE_CUBE_DECISION
            doubler = g.current_player
            apply_action!(g, 678)  # DOUBLE
            apply_action!(g, 680)  # PASS

            @test game_terminated(g)
            # Reward: doubler wins 1 point (cube was 1 at time of pass)
            expected_reward = doubler == 0 ? 1.0f0 : -1.0f0
            @test g.reward == expected_reward
        end
    end

    @testset "Cube Re-double" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        g.cube_value = Int16(2)
        g.cube_owner = Int8(0)  # Player 0 owns cube (absolute)
        g.phase = BackgammonNet.PHASE_CUBE_DECISION

        # Can double since P0 owns and P0 is current player
        @test BackgammonNet.may_double(g) == true

        apply_action!(g, 678)  # DOUBLE (P0 doubles, switches to P1)
        @test g.phase == BackgammonNet.PHASE_CUBE_RESPONSE
        apply_action!(g, 679)  # TAKE (P1 takes, cube_owner = P1 = 1)
        @test g.cube_value == Int16(4)
        @test g.cube_owner == Int8(1)  # P1 (taker) now owns cube
    end

    @testset "step! with Cube Actions" begin
        g = initial_state(first_player=0)
        g.cube_enabled = true
        sample_chance!(g)

        while g.phase == BackgammonNet.PHASE_CHECKER_PLAY && !game_terminated(g)
            actions = legal_actions(g)
            apply_action!(g, actions[1])
        end

        if g.phase == BackgammonNet.PHASE_CUBE_DECISION
            # step! with DOUBLE should return CUBE_RESPONSE state
            g2 = clone(g)
            step!(g2, 678)
            @test g2.phase == BackgammonNet.PHASE_CUBE_RESPONSE
            @test !is_chance_node(g2)

            # step! with DOUBLE -> TAKE should return playable state
            g3 = clone(g2)
            step!(g3, 679)
            @test g3.cube_value == Int16(2)
            @test !is_chance_node(g3)
            @test !game_terminated(g3)

            # step! with NO_DOUBLE should return playable state
            g4 = clone(g)
            step!(g4, 677)
            @test !is_chance_node(g4)
            @test !game_terminated(g4)
        end
    end

    @testset "Full Random Games with Cube" begin
        for _ in 1:100
            g = initial_state()
            g.cube_enabled = true
            sample_chance!(g)
            steps = 0
            while !game_terminated(g)
                steps += 1
                @test steps <= 5000
                if steps > 5000; break; end
                actions = legal_actions(g)
                step!(g, actions[rand(1:length(actions))])
            end
        end
    end

    @testset "compute_game_reward" begin
        g = initial_state()

        # Basic rewards (cube=1, no Jacoby)
        @test BackgammonNet.compute_game_reward(g, Int8(0), 1.0f0) == 1.0f0
        @test BackgammonNet.compute_game_reward(g, Int8(0), 2.0f0) == 2.0f0
        @test BackgammonNet.compute_game_reward(g, Int8(1), 1.0f0) == -1.0f0
        @test BackgammonNet.compute_game_reward(g, Int8(1), 3.0f0) == -3.0f0

        # Cube multiplication
        g.cube_value = Int16(4)
        @test BackgammonNet.compute_game_reward(g, Int8(0), 2.0f0) == 8.0f0
        @test BackgammonNet.compute_game_reward(g, Int8(1), 3.0f0) == -12.0f0

        # Jacoby rule: cube not turned
        g2 = initial_state()
        g2.jacoby_enabled = true
        @test BackgammonNet.compute_game_reward(g2, Int8(0), 2.0f0) == 1.0f0  # Reduced
        @test BackgammonNet.compute_game_reward(g2, Int8(0), 3.0f0) == 1.0f0  # Reduced
        @test BackgammonNet.compute_game_reward(g2, Int8(0), 1.0f0) == 1.0f0  # Single stays

        # Jacoby rule: cube turned
        g2.cube_value = Int16(2)
        @test BackgammonNet.compute_game_reward(g2, Int8(0), 2.0f0) == 4.0f0  # Not reduced

        # Jacoby disabled in match play
        g3 = initial_state()
        g3.jacoby_enabled = true
        g3.my_away = Int8(5)
        @test BackgammonNet.compute_game_reward(g3, Int8(0), 2.0f0) == 2.0f0
    end

    @testset "init_match_game!" begin
        # Crawford game
        g = initial_state()
        BackgammonNet.init_match_game!(g, my_score=4, opp_score=2, match_length=5, is_crawford=true)
        @test g.my_away == Int8(1)
        @test g.opp_away == Int8(3)
        @test g.is_crawford == true
        @test g.is_post_crawford == false
        @test g.cube_enabled == false
        @test BackgammonNet.may_double(g) == false

        # Post-Crawford
        g2 = initial_state()
        BackgammonNet.init_match_game!(g2, my_score=4, opp_score=2, match_length=5, is_crawford=false)
        @test g2.is_crawford == false
        @test g2.is_post_crawford == true
        @test g2.cube_enabled == true

        # Normal match
        g3 = initial_state()
        BackgammonNet.init_match_game!(g3, my_score=1, opp_score=2, match_length=7)
        @test g3.my_away == Int8(6)
        @test g3.opp_away == Int8(5)
        @test g3.is_crawford == false
        @test g3.is_post_crawford == false
        @test g3.cube_enabled == true
        @test g3.jacoby_enabled == false
    end

    @testset "Full Random Match Games" begin
        for _ in 1:50
            g = initial_state()
            BackgammonNet.init_match_game!(g,
                my_score=rand(0:4), opp_score=rand(0:4),
                match_length=5, is_crawford=rand(Bool))
            sample_chance!(g)
            steps = 0
            while !game_terminated(g)
                steps += 1
                @test steps <= 5000
                if steps > 5000; break; end
                actions = legal_actions(g)
                step!(g, actions[rand(1:length(actions))])
            end
        end
    end

    @testset "Cube/Match Observation Channels" begin
        # Default game: money play, cube disabled, CHANCE phase
        g = initial_state()
        obs = observe_minimal(g)

        # Channels 31-33: phase one-hot (CHANCE = none active)
        @test obs[31, 1, 1] == 0.0f0  # Not CUBE_DECISION
        @test obs[32, 1, 1] == 0.0f0  # Not CUBE_RESPONSE
        @test obs[33, 1, 1] == 0.0f0  # Not CHECKER_PLAY (CHANCE phase)
        # Channel 34: cube_value = log2(1)/6 = 0
        @test obs[34, 1, 1] == 0.0f0
        # Channel 35-36: cube_owner = -1 (centered)
        @test obs[35, 1, 1] == 0.0f0  # Not "I own"
        @test obs[36, 1, 1] == 1.0f0  # Centered
        # Channel 37: may_double = false (cube disabled)
        @test obs[37, 1, 1] == 0.0f0
        # Channel 38: money play = true
        @test obs[38, 1, 1] == 1.0f0
        # Channels 39-40: away scores = 0
        @test obs[39, 1, 1] == 0.0f0
        @test obs[40, 1, 1] == 0.0f0
        # Channels 41-42: Crawford flags = false
        @test obs[41, 1, 1] == 0.0f0
        @test obs[42, 1, 1] == 0.0f0

        # Cube-enabled game in CHECKER_PLAY phase
        g2 = initial_state(first_player=0)
        g2.cube_enabled = true
        sample_chance!(g2)
        obs2 = observe_minimal(g2)
        @test obs2[33, 1, 1] == 1.0f0  # CHECKER_PLAY
        @test obs2[31, 1, 1] == 0.0f0
        @test obs2[32, 1, 1] == 0.0f0

        # Cube decision phase
        g3 = initial_state(first_player=0)
        g3.cube_enabled = true
        g3.phase = BackgammonNet.PHASE_CUBE_DECISION
        obs3 = observe_minimal(g3)
        @test obs3[31, 1, 1] == 1.0f0  # CUBE_DECISION
        @test obs3[37, 1, 1] == 1.0f0  # Can double

        # Cube value = 4
        g4 = initial_state(first_player=0)
        g4.cube_value = Int16(4)
        obs4 = observe_minimal(g4)
        @test obs4[34, 1, 1] ≈ log2(4.0f0) / 6.0f0

        # Cube owner = current player (I own)
        g4.cube_owner = g4.current_player
        obs4b = observe_minimal(g4)
        @test obs4b[35, 1, 1] == 1.0f0  # I own
        @test obs4b[36, 1, 1] == 0.0f0  # Not centered

        # Match play with Crawford
        g5 = initial_state()
        BackgammonNet.init_match_game!(g5, my_score=4, opp_score=2, match_length=5, is_crawford=true)
        obs5 = observe_minimal(g5)
        @test obs5[38, 1, 1] == 0.0f0  # Not money play
        @test obs5[39, 1, 1] ≈ 1.0f0 / 25.0f0  # my_away=1
        @test obs5[40, 1, 1] ≈ 3.0f0 / 25.0f0  # opp_away=3
        @test obs5[41, 1, 1] == 1.0f0  # Crawford
        @test obs5[42, 1, 1] == 0.0f0  # Not post-Crawford
        @test obs5[37, 1, 1] == 0.0f0  # Cannot double in Crawford

        # Verify broadcast: all 26 spatial positions should match
        for w in 2:OBS_WIDTH
            @test obs5[31, 1, w] == obs5[31, 1, 1]
            @test obs5[38, 1, w] == obs5[38, 1, 1]
            @test obs5[39, 1, w] == obs5[39, 1, 1]
        end

        # Verify flat encoding matches
        obs5_flat = observe_minimal_flat(g5)
        @test obs5_flat[331] == obs5[31, 1, 1]  # Phase CUBE_DECISION
        @test obs5_flat[338] == obs5[38, 1, 1]  # Money play
        @test obs5_flat[341] == obs5[41, 1, 1]  # Crawford
        @test obs5_flat[342] == obs5[42, 1, 1]  # Post-Crawford
    end

    @testset "Context Observations" begin
        g = initial_state()
        ctx = BackgammonNet.context_observation(g)
        @test length(ctx) == BackgammonNet.CONTEXT_DIM
        @test ctx[1] == 0.0f0  # cube=1 -> log2(1)/6=0
        @test ctx[2] == 0.0f0  # cube_owner=centered
        @test ctx[3] == 0.0f0  # may_double=false (cube disabled)
        @test ctx[4] == 1.0f0  # is_money=true

        # Cube enabled, cube decision phase
        g2 = initial_state()
        g2.cube_enabled = true
        g2.phase = BackgammonNet.PHASE_CUBE_DECISION
        ctx2 = BackgammonNet.context_observation(g2)
        @test ctx2[3] == 1.0f0  # may_double=true
        @test ctx2[10] == 1.0f0  # CUBE_DECISION
        @test ctx2[11] == 0.0f0
        @test ctx2[12] == 0.0f0

        # Match play context
        g3 = initial_state()
        BackgammonNet.init_match_game!(g3, my_score=2, opp_score=3, match_length=7)
        ctx3 = BackgammonNet.context_observation(g3)
        @test ctx3[4] == 0.0f0  # not money play
        @test ctx3[5] ≈ 5.0f0 / 5.0f0  # my_away=5, max=5
        @test ctx3[6] ≈ 4.0f0 / 5.0f0  # opp_away=4

        # Masked context
        ctx_masked = BackgammonNet.context_observation(g3, true)
        @test all(ctx_masked .== 0.0f0)
        ctx_full = BackgammonNet.context_observation(g3, false)
        @test ctx_full == ctx3
    end

    @testset "Backward Compatibility - Non-Cube Games" begin
        # Default games should work exactly as before
        for _ in 1:100
            g = initial_state()
            @test g.cube_enabled == false
            sample_chance!(g)
            steps = 0
            while !game_terminated(g)
                steps += 1
                @test steps <= 5000
                if steps > 5000; break; end
                actions = legal_actions(g)
                # No cube actions should appear
                for a in actions
                    @test !BackgammonNet.is_cube_action(a)
                end
                step!(g, actions[rand(1:length(actions))])
            end
        end
    end

end

