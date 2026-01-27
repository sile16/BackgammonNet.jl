module BackgammonNet

using StaticArrays
using Random

export BackgammonGame, initial_state, reset!, clone, current_player, legal_actions, game_terminated, winner, action_string
export step!, apply_action!, apply_chance!, sample_chance!, chance_outcomes, is_chance_node, DICE_PROBS, DICE_OUTCOMES, is_action_valid
export encode_action, decode_action
export PASS_LOC, BAR_LOC

# Observation API (3D tensors)
export observe_minimal, observe_full, observe_biased
export observe_minimal!, observe_full!, observe_biased!
export OBS_CHANNELS_MINIMAL, OBS_CHANNELS_FULL, OBS_CHANNELS_BIASED, OBS_WIDTH

# Observation API (flat vectors)
export observe_minimal_flat, observe_full_flat, observe_biased_flat
export observe_minimal_flat!, observe_full_flat!, observe_biased_flat!
export OBS_FLAT_MINIMAL, OBS_FLAT_FULL, OBS_FLAT_BIASED

# Observation API (hybrid: board spatial + globals flat)
export observe_minimal_hybrid, observe_full_hybrid, observe_biased_hybrid
export observe_minimal_hybrid!, observe_full_hybrid!, observe_biased_hybrid!
export OBS_HYBRID_BOARD, OBS_HYBRID_GLOBALS_MINIMAL, OBS_HYBRID_GLOBALS_FULL, OBS_HYBRID_GLOBALS_BIASED

# Game-level observation API (dispatches on g.obs_type)
export observe, obs_dims, set_obs_type!

export OBSERVATION_SIZES

include("game.jl")
include("actions.jl")
include("observation.jl")

end
