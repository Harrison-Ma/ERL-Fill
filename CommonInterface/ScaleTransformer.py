def reset_actor_scaling(agent, new_bounds):
    """
    Reset scaling parameters (scale_params and offset_params) for both the actor
    and its target network based on new action bounds.

    This function is typically used when the environment's action space changes
    (e.g., different target weight or new physical limits), and the actor networks
    must rescale their output accordingly.

    Args:
        agent: DDPGAgent instance that contains actor and actor_target networks.
        new_bounds (dict): A dictionary of new action bounds (e.g., from env.bounds).
                           Each key maps to a (min, max) tuple for the action variable.
    """
    if hasattr(agent.actor, '_init_param_scaling'):
        agent.actor._init_param_scaling(new_bounds)
    if hasattr(agent.actor_target, '_init_param_scaling'):
        agent.actor_target._init_param_scaling(new_bounds)
    print("✅ Actor scaling parameters re-initialized based on new bounds.")
