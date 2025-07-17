def reset_actor_scaling(agent, new_bounds):
    """
    重置 Actor 和 Actor-Target 的参数缩放信息（scale_params 和 offset_params）

    参数:
        agent: DDPGAgent 实例
        new_bounds: dict, 例如 env.bounds
    """
    if hasattr(agent.actor, '_init_param_scaling'):
        agent.actor._init_param_scaling(new_bounds)
    if hasattr(agent.actor_target, '_init_param_scaling'):
        agent.actor_target._init_param_scaling(new_bounds)
    print("✅ Actor 缩放参数已根据新 bounds 重新初始化。")