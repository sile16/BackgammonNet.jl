module BackgammonNet

using StaticArrays

export BackgammonGame, initial_state, reset!, current_player, legal_actions, observe, game_terminated, winner, vector_observation, action_string
export step!, apply_action!, apply_chance!, sample_chance!, chance_outcomes, is_chance_node, DICE_PROBS, DICE_OUTCOMES

include("game.jl")
include("actions.jl")
include("observation.jl")

end
