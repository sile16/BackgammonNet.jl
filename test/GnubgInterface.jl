# GnubgInterface.jl - Clean interface for gnubg integration
#
# Usage:
#   include("test/GnubgInterface.jl")
#   using .GnubgInterface
#
#   # Evaluate a position
#   equity = evaluate(game)
#   probs = evaluate_probs(game)  # (win, win_g, win_bg, lose_g, lose_bg)
#
#   # Get best move
#   action, equity = best_move(game)
#
#   # Play games
#   result = play_game(your_agent, gnubg_agent; verbose=true)
#
#   # Create agents with configurable ply depth
#   gnubg = GnubgAgent()           # default ply (0)
#   gnubg = GnubgAgent(ply=2)      # 2-ply search
#   random = RandomAgent(seed=42)
#   my_agent = CustomAgent(move_fn)
#
#   # Configure global settings
#   set_default_ply!(2)            # set default ply depth
#   get_default_ply()              # get current default

module GnubgInterface

using BackgammonNet
using PyCall
using Random

export evaluate, evaluate_probs, best_move
export play_game, play_match
export AbstractAgent, GnubgAgent, RandomAgent, CustomAgent
export agent_move
export set_default_ply!, get_default_ply, set_rollout_games!
export init_wandb, finish_wandb

# =============================================================================
# Initialization and Settings
# =============================================================================

const _gnubg = PyNULL()
const _initialized = Ref(false)

# Global settings
const _default_ply = Ref(0)           # 0 = neural net only, 1+ = lookahead plies
const _rollout_games = Ref(1296)      # number of games for rollout evaluation

function _init()
    if !_initialized[]
        copy!(_gnubg, pyimport("gnubg"))
        _initialized[] = true
    end
end

"""
    set_default_ply!(ply::Int)

Set the default ply depth for gnubg evaluations.
- 0: Neural net evaluation only (fastest, ~48k evals/sec)
- 1: 1-ply lookahead (slower but more accurate)
- 2: 2-ply lookahead (even slower, even more accurate)

Higher ply = better play but slower evaluation.
"""
function set_default_ply!(ply::Int)
    @assert ply >= 0 "Ply must be non-negative"
    _default_ply[] = ply
end

"""
    get_default_ply() -> Int

Get the current default ply depth.
"""
get_default_ply() = _default_ply[]

"""
    set_rollout_games!(n::Int)

Set the number of games used for rollout evaluation (default: 1296).
Only affects rollout-based evaluation, not standard ply search.
"""
function set_rollout_games!(n::Int)
    @assert n > 0 "Number of games must be positive"
    _rollout_games[] = n
end

# =============================================================================
# Weights & Biases (wandb) Integration
# =============================================================================

const _wandb = PyNULL()
const _wandb_initialized = Ref(false)
const _wandb_run = Ref{PyObject}(PyNULL())

"""
    init_wandb(project::String; name::String="", config::Dict=Dict())

Initialize Weights & Biases logging for monitoring matches.

# Arguments
- `project`: wandb project name
- `name`: run name (optional, wandb will generate one if not provided)
- `config`: dictionary of config parameters to log

# Example
    init_wandb("backgammon-eval", name="gnubg-vs-random", config=Dict("ply"=>2))
"""
function init_wandb(project::String; name::String="", config::Dict=Dict())
    if _wandb_initialized[]
        @warn "wandb already initialized, finishing previous run"
        finish_wandb()
    end

    copy!(_wandb, pyimport("wandb"))

    kwargs = Dict{Symbol, Any}(:project => project)
    if !isempty(name)
        kwargs[:name] = name
    end
    if !isempty(config)
        kwargs[:config] = config
    end

    _wandb_run[] = _wandb.init(; kwargs...)
    _wandb_initialized[] = true
    return _wandb_run[]
end

"""
    finish_wandb()

Finish the current wandb run.
"""
function finish_wandb()
    if _wandb_initialized[]
        _wandb.finish()
        _wandb_initialized[] = false
    end
end

"""
    _log_wandb(data::Dict; step::Union{Int, Nothing}=nothing)

Log metrics to wandb (internal).
"""
function _log_wandb(data::Dict; step::Union{Int, Nothing}=nothing)
    if _wandb_initialized[]
        if step !== nothing
            _wandb.log(data, step=step)
        else
            _wandb.log(data)
        end
    end
end

# =============================================================================
# Board Conversion
# =============================================================================

"""
Convert Julia game to gnubg board format.
Returns [opponent_checkers, on_roll_checkers] each with 25 elements.
"""
function _to_gnubg_board(g::BackgammonGame)
    _init()
    board = zeros(Int, 2, 25)
    cp = Int(g.current_player)
    p0, p1 = g.p0, g.p1

    if cp == 0
        for pt in 1:24
            idx = 25 - pt
            board[2, pt + 1] = Int((p0 >> (idx << 2)) & 0xF)
            board[1, pt + 1] = Int((p1 >> (idx << 2)) & 0xF)
        end
        board[2, 1] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
        board[1, 1] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
    else
        for pt in 1:24
            board[2, pt + 1] = Int((p1 >> (pt << 2)) & 0xF)
            board[1, pt + 1] = Int((p0 >> (pt << 2)) & 0xF)
        end
        board[2, 1] = Int((p1 >> (BackgammonNet.IDX_P1_BAR << 2)) & 0xF)
        board[1, 1] = Int((p0 >> (BackgammonNet.IDX_P0_BAR << 2)) & 0xF)
    end

    return [board[1, :], board[2, :]]
end

# =============================================================================
# Position Evaluation
# =============================================================================

"""
    evaluate_probs(g::BackgammonGame; ply::Int=get_default_ply()) -> NTuple{5, Float64}

Get gnubg's probability estimates for a position.
Returns (win, win_gammon, win_backgammon, lose_gammon, lose_backgammon).

All probabilities are from the current player's perspective.

# Arguments
- `ply`: Search depth (0=neural net only, 1+=lookahead). Default: global setting.
"""
function evaluate_probs(g::BackgammonGame; ply::Int=_default_ply[])
    _init()
    board = _to_gnubg_board(g)
    probs = _gnubg.probabilities(board, ply)
    return (Float64(probs[1]), Float64(probs[2]), Float64(probs[3]),
            Float64(probs[4]), Float64(probs[5]))
end

"""
    evaluate(g::BackgammonGame; ply::Int=get_default_ply()) -> Float64

Get gnubg's equity estimate for a position.
Returns cubeless money equity from current player's perspective.
Positive = winning, negative = losing. Range roughly -3 to +3.

# Arguments
- `ply`: Search depth (0=neural net only, 1+=lookahead). Default: global setting.
"""
function evaluate(g::BackgammonGame; ply::Int=_default_ply[])
    win, win_g, win_bg, lose_g, lose_bg = evaluate_probs(g; ply=ply)
    return (win - (1.0 - win)) + (win_g - lose_g) + 2.0 * (win_bg - lose_bg)
end

# =============================================================================
# Best Move Selection
# =============================================================================

"""
    best_move(g::BackgammonGame; ply::Int=get_default_ply()) -> Tuple{Int, Float64}

Get the best move according to gnubg's evaluation.
Returns (action, equity) where action is the encoded move.

Uses Julia's legal move generation (correct) + gnubg's neural net evaluation.

# Arguments
- `ply`: Search depth for position evaluation (0=neural net only, 1+=lookahead).
"""
function best_move(g::BackgammonGame; ply::Int=_default_ply[])
    _init()
    actions = BackgammonNet.legal_actions(g)

    if isempty(actions)
        return BackgammonNet.encode_action(25, 25), 0.0
    end

    best_action = actions[1]
    best_equity = -Inf

    for action in actions
        g2 = BackgammonGame(g.p0, g.p1, g.dice, g.remaining_actions,
                            g.current_player, g.terminated, g.reward, Int[])
        BackgammonNet.apply_action!(g2, action)

        equity = evaluate(g2; ply=ply)
        # Negate if player switched
        if g2.current_player != g.current_player
            equity = -equity
        end

        if equity > best_equity
            best_equity = equity
            best_action = action
        end
    end

    return best_action, best_equity
end

# =============================================================================
# Agent Interface
# =============================================================================

"""
Abstract type for backgammon agents.
Implement `agent_move(agent, game) -> action` for custom agents.
"""
abstract type AbstractAgent end

"""
    agent_move(agent::AbstractAgent, g::BackgammonGame) -> Int

Return the action the agent chooses for the given game state.
"""
function agent_move end

# --- GnubgAgent ---

"""
Agent that uses gnubg's neural network for move selection.

    GnubgAgent()           # uses default ply (0)
    GnubgAgent(ply=2)      # 2-ply search

Higher ply = stronger play but slower.
"""
struct GnubgAgent <: AbstractAgent
    ply::Int
end

GnubgAgent(; ply::Int=_default_ply[]) = GnubgAgent(ply)

function agent_move(agent::GnubgAgent, g::BackgammonGame)
    action, _ = best_move(g; ply=agent.ply)
    return action
end

# --- RandomAgent ---

"""
Agent that plays random legal moves.
"""
mutable struct RandomAgent <: AbstractAgent
    rng::MersenneTwister
end

RandomAgent(; seed::Int=42) = RandomAgent(MersenneTwister(seed))

function agent_move(agent::RandomAgent, g::BackgammonGame)
    actions = BackgammonNet.legal_actions(g)
    return actions[rand(agent.rng, 1:length(actions))]
end

# --- CustomAgent ---

"""
Agent with a custom move function.

    agent = CustomAgent(my_move_function)

where `my_move_function(game) -> action`
"""
struct CustomAgent <: AbstractAgent
    move_fn::Function
end

function agent_move(agent::CustomAgent, g::BackgammonGame)
    return agent.move_fn(g)
end

# =============================================================================
# Game Playing
# =============================================================================

"""
    play_game(agent0::AbstractAgent, agent1::AbstractAgent;
              seed::Int=1, verbose::Bool=false) -> NamedTuple

Play a single game between two agents.
agent0 plays as player 0, agent1 plays as player 1.

Returns (winner, reward, num_moves, history).
"""
function play_game(agent0::AbstractAgent, agent1::AbstractAgent;
                   seed::Int=1, verbose::Bool=false)
    rng = MersenneTwister(seed)
    g = BackgammonNet.initial_state()
    history = Tuple{Int, Int, Float64}[]  # (player, action, equity)
    num_moves = 0

    while !BackgammonNet.game_terminated(g)
        if BackgammonNet.is_chance_node(g)
            BackgammonNet.sample_chance!(g, rng)
            if verbose
                println("Dice: $(g.dice[1]), $(g.dice[2]) - Player $(g.current_player)'s turn")
            end
        else
            cp = Int(g.current_player)
            agent = cp == 0 ? agent0 : agent1

            action = agent_move(agent, g)
            equity = evaluate(g)

            if verbose
                src1, src2 = BackgammonNet.decode_action(action)
                agent_name = agent isa GnubgAgent ? "gnubg" :
                             agent isa RandomAgent ? "random" : "custom"
                println("  P$cp ($agent_name): ($src1, $src2) [equity=$(round(equity, digits=3))]")
            end

            BackgammonNet.apply_action!(g, action)
            push!(history, (cp, action, equity))
            num_moves += 1
        end
    end

    winner = g.reward > 0 ? 0 : 1
    if verbose
        println("\nGame over! Winner: Player $winner (reward: $(g.reward))")
    end

    return (winner=winner, reward=g.reward, num_moves=num_moves, history=history)
end

"""
    play_match(agent0::AbstractAgent, agent1::AbstractAgent, num_games::Int;
               seed::Int=1, verbose::Bool=false) -> NamedTuple

Play multiple games between two agents.

Returns (agent0_wins, agent1_wins, agent0_points, agent1_points, results).
"""
function play_match(agent0::AbstractAgent, agent1::AbstractAgent, num_games::Int;
                    seed::Int=1, verbose::Bool=false, log_interval::Int=10)
    agent0_wins = 0
    agent1_wins = 0
    agent0_points = 0
    agent1_points = 0
    results = NamedTuple[]

    for i in 1:num_games
        result = play_game(agent0, agent1; seed=seed+i, verbose=false)
        push!(results, result)

        if result.winner == 0
            agent0_wins += 1
            agent0_points += abs(result.reward)
        else
            agent1_wins += 1
            agent1_points += abs(result.reward)
        end

        # Log to wandb at intervals
        if _wandb_initialized[] && i % log_interval == 0
            win_rate = agent0_wins / i
            _log_wandb(Dict(
                "games" => i,
                "agent0_wins" => agent0_wins,
                "agent1_wins" => agent1_wins,
                "agent0_win_rate" => win_rate,
                "agent0_points" => agent0_points,
                "agent1_points" => agent1_points,
                "avg_moves" => sum(r.num_moves for r in results) / length(results)
            ), step=i)
        end

        if verbose && i % 100 == 0
            println("Game $i/$num_games: Agent0 $agent0_wins-$agent1_wins Agent1")
        end
    end

    # Log final results
    if _wandb_initialized[]
        win_rate = agent0_wins / num_games
        _log_wandb(Dict(
            "final/agent0_wins" => agent0_wins,
            "final/agent1_wins" => agent1_wins,
            "final/agent0_win_rate" => win_rate,
            "final/agent0_points" => agent0_points,
            "final/agent1_points" => agent1_points,
            "final/total_games" => num_games
        ))
    end

    if verbose
        println("\n" * "="^50)
        println("Match complete: $num_games games")
        println("Agent0 wins: $agent0_wins ($(round(100*agent0_wins/num_games, digits=1))%)")
        println("Agent1 wins: $agent1_wins ($(round(100*agent1_wins/num_games, digits=1))%)")
        println("Points: Agent0 $agent0_points - $agent1_points Agent1")
    end

    return (agent0_wins=agent0_wins, agent1_wins=agent1_wins,
            agent0_points=agent0_points, agent1_points=agent1_points,
            results=results)
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    gnubg_vs_random(num_games::Int; seed::Int=1, verbose::Bool=false)

Quick benchmark: gnubg agent vs random agent.
"""
function gnubg_vs_random(num_games::Int; seed::Int=1, verbose::Bool=false)
    gnubg = GnubgAgent()
    random = RandomAgent(seed=seed)
    return play_match(gnubg, random, num_games; seed=seed, verbose=verbose)
end

"""
    random_vs_gnubg(num_games::Int; seed::Int=1, verbose::Bool=false)

Quick benchmark: random agent vs gnubg agent.
"""
function random_vs_gnubg(num_games::Int; seed::Int=1, verbose::Bool=false)
    random = RandomAgent(seed=seed)
    gnubg = GnubgAgent()
    return play_match(random, gnubg, num_games; seed=seed, verbose=verbose)
end

end # module
