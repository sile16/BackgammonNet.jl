module BackgammonNet

using StaticArrays
using Random

export BackgammonGame, initial_state, reset!, clone, current_player, legal_actions, game_terminated, winner, vector_observation, action_string
export step!, apply_action!, apply_chance!, sample_chance!, chance_outcomes, is_chance_node, DICE_PROBS, DICE_OUTCOMES, is_action_valid
export encode_action, decode_action
export PASS_LOC, BAR_LOC

# New observation API (3D tensors)
export observe_minimal, observe_full, observe_biased
export observe_minimal!, observe_full!, observe_biased!
export OBS_CHANNELS_MINIMAL, OBS_CHANNELS_FULL, OBS_CHANNELS_BIASED, OBS_WIDTH
export OBSERVATION_SIZES

# Legacy observation API (deprecated, 1D vectors)
export observe_fast, observe_full!, observe_fast!

include("game.jl")
include("actions.jl")
include("observation.jl")

end
