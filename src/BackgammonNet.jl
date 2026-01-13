module BackgammonNet

using StaticArrays
using Random

export BackgammonGame, initial_state, reset!, current_player, legal_actions, game_terminated, winner, vector_observation, action_string
export step!, apply_action!, apply_chance!, sample_chance!, chance_outcomes, is_chance_node, DICE_PROBS, DICE_OUTCOMES, is_action_valid
export observe_fast, observe_full, observe_fast!, observe_full!
export encode_action, decode_action
export PASS_LOC, BAR_LOC

include("game.jl")
include("actions.jl")
include("observation.jl")

end
