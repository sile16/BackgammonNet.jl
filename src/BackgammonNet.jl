module BackgammonNet

using StaticArrays

export BackgammonGame, initial_state, play!, current_player, legal_actions, observe, game_terminated, winner, vector_observation, action_string

include("game.jl")
include("actions.jl")
include("observation.jl")

end
