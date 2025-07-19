import torch
import logging
import numpy as np
import os
import pandas as pd
from CommonInterface.ScaleTransformer import reset_actor_scaling
from WeightEnv import WeightEnv
from EmotionModule import EmotionModuleNone, EmotionModule, EmotionModuleSimple
from DifferentModules.ddpg_agent import DDPGAgent
from DifferentModules.sac_agent import SACAgent
from DifferentModules.ERL_Fill_agent import EmotionSACAgent
from DifferentModules.simple_emotion_sac_agent import SimpleEmotionSACAgent, train_simple_sac
from MutiConditionEnv import MultiConditionWeightEnv
from CommonInterface.Logger import init_logger
from baseline_experiments import (
    build_ddpg_agent, build_td3_agent, build_ppo_agent, build_sac_agent,
    build_td3_bc_agent, build_cql_agent, build_erl_fill_agent, build_fuzzy_agent,
    build_ppol_agent, build_rls_pid_agent
)
from DifferentModules.td3_agent import train_td3
from DifferentModules.td3_bc_agent import train_td3_bc
from DifferentModules.cql_agent import train_cql
from DifferentModules.PPOL import train_ppo_lagrangian
from DifferentModules.rls_pidagent import train_rls_pid
from DifferentModules.ppo_agent import train_ppo
from DifferentModules.sac_agent import train_sac
from DifferentModules.ddpg_agent import train_ddpg
from DifferentModules.ERL_Fill_agent import train_erl_fill


def run_experiment_1(
    episodes=1000, max_steps=500, device='cuda', env_mode='sim',
    emotion_modes=['none', 'simple', 'transformer'],
    algo_list=['ddpg', 'td3', 'sac'],
    train=True,
    lambda_emo=0.1,
    log_prefix=None,
    logger=None
):
    """
    Run a grid experiment combining multiple Emotion Modules and RL Algorithms.

    This function loops through all combinations of emotion modes and RL algorithms,
    initializes the environment and agents accordingly, and either trains or tests the models.
    It supports SAC and DDPG-based algorithms with various emotion augmentation strategies.

    Args:
        episodes (int): Number of episodes to train/test.
        max_steps (int): Maximum steps per episode.
        device (str): Device used for training (e.g., 'cuda' or 'cpu').
        env_mode (str): Mode of environment: 'sim', 'onboard', or 'real'.
        emotion_modes (list): Emotion module options: ['none', 'simple', 'transformer'].
        algo_list (list): Algorithms to run. Currently supports ['ddpg', 'td3', 'sac'].
        train (bool): Whether to train (True) or evaluate (False).
        lambda_emo (float): Emotion loss weighting factor (used in certain algorithms).
        log_prefix (str): Optional prefix for log directory.
        logger (logging.Logger): Optional logger instance for recording output.

    Returns:
        list of tuples: Each entry is (algorithm, emotion_mode, average_test_reward) if not training.
    """
    results = []

    for algo in algo_list:
        for mode in emotion_modes:
            if logger:
                logger.info(f"=== Running {algo.upper()} | Emotion Mode: {mode.upper()} | Train={train} ===")

            # Initialize environment
            env = WeightEnv()
            env.use_offline_sim = {'sim': 1, 'onboard': 0, 'real': 2}.get(env_mode, 1)

            # Initialize emotion module
            if mode == 'transformer':
                emotion_module = EmotionModule(device=device)
            elif mode == 'simple':
                emotion_module = EmotionModuleSimple()
            elif mode == 'none':
                emotion_module = EmotionModuleNone()
            else:
                raise ValueError(f"Unsupported emotion mode: {mode}")

            # Initialize RL agent based on algorithm and emotion mode
            if algo == 'sac':
                if mode == 'none':
                    agent = SACAgent(env, device=device)
                elif mode == 'simple':
                    agent = SimpleEmotionSACAgent(env, device=device)
                else:
                    agent = EmotionSACAgent(env, device=device)
            else:
                raise ValueError(f"Unsupported algorithm: {algo}")

            # Attach emotion module if supported
            if hasattr(agent, 'emotion'):
                agent.emotion = emotion_module
            env.attach_agent(agent)

            # Setup model save/load paths
            model_dir = f"saved_models/exp1_{algo}_emotion_compare"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{algo}_{mode}_{lambda_emo}.pth")

            # Handle pretrained weight loading
            if env_mode == 'sim':
                if logger: logger.info("No pretrained weights loaded. Training from scratch.")
            elif env_mode == 'onboard':
                if logger: logger.info("Onboard training selected.")
                pretrain_path = "pre_train_pth/sac_transformer.pth"
            elif env_mode == 'real':
                if logger: logger.info("Real-world training selected.")
                pretrain_path = "pre_train_pth/AIM_v2_onboard_24900_25000.pth"
                weights = torch.load(pretrain_path, weights_only=False)
                agent.load_weights(weights)
                reset_actor_scaling(agent, env.bounds)
                if logger: logger.info(f"Loaded pretrained weights from: {pretrain_path}")
            else:
                if logger: logger.info("Default pretrained weights loaded.")
                pretrain_path = "pre_train_pth/AIM_v2_sim_24900_25000.pth"
                agent.load(pretrain_path)
                if logger: logger.info(f"Loaded pretrained weights from: {pretrain_path}")

            # === Training Phase ===
            if train:
                if logger: logger.info(f"Start training [{algo.upper()}] with emotion mode [{mode}]")

                if algo == 'ddpg':
                    train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

                elif algo == 'sac':
                    if mode == 'none':
                        train_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, logger=logger)
                    elif mode == "simple":
                        train_simple_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, logger=logger)
                    else:
                        train_erl_fill(env, agent, episodes=episodes, max_steps=max_steps,
                                       log_prefix=log_prefix, lambda_emo=lambda_emo, logger=logger)

                agent.save(model_path)
                if logger: logger.info(f"Training complete. Model saved to {model_path}")

            # === Testing Phase ===
            else:
                if logger: logger.info(f"Start testing [{algo.upper()}] with emotion mode [{mode}]")

                if not os.path.exists(model_path):
                    if logger: logger.info(f"Model not found: {model_path}. Skip.")
                    continue

                agent.load(model_path)

                total_reward = 0
                test_episodes = 10
                test_log_dir = os.path.join("logs", f"exp1_result_{algo}_{mode}_test.log")
                os.makedirs("logs", exist_ok=True)

                # Run test episodes and log results
                with open(test_log_dir, "w", encoding="utf-8") as f_log:
                    for ep in range(test_episodes):
                        state = env.reset()
                        episode_reward = 0
                        for step in range(max_steps):
                            if hasattr(agent, 'act'):
                                if algo in ['er_ddpg', 'ddpg', 'td3', 'emotion_td3']:
                                    action = agent.act(state, add_noise=False)
                                elif algo in ['sac', 'emotion_sac']:
                                    action = agent.act(state, deterministic=True)
                                elif algo == 'ppo':
                                    action, _ = agent.select_action(state)
                                else:
                                    raise ValueError(f"Unsupported algorithm: {algo}")
                            else:
                                action = agent.select_action(state)

                            action = action.squeeze() if isinstance(action, np.ndarray) else action
                            next_state, reward, done, info = env.step(action)
                            state = next_state
                            episode_reward += reward
                            if done:
                                break

                        total_reward += episode_reward

                        # Optional: log current emotion state
                        cur_emotion = None
                        if hasattr(agent, 'emotion') and hasattr(agent.emotion, 'get_emotion'):
                            cur_emotion = agent.emotion.get_emotion()
                            if isinstance(cur_emotion, torch.Tensor):
                                cur_emotion = cur_emotion.detach().cpu().numpy().squeeze()

                        slow_weight = info.get("action", {}).get("slow_weight", 0.0)

                        log_str = (
                            f"[Episode {ep:04d}] "
                            f"TotalReward: {episode_reward:.2f} | "
                            f"WeightErr: {info['weight_error']:.2f}g | "
                            f"Time: {info['total_time']:.2f}s | "
                            f"SlowWeight: {slow_weight:.2f}g | "
                            f"Emotion: {cur_emotion}"
                        )

                        if logger: logger.info(f" Test Episode {ep+1}: Reward = {episode_reward:.2f}")
                        f_log.write(log_str + "\n")

                    avg_reward = total_reward / test_episodes
                    f_log.write(f"\n Avg Test Reward: {avg_reward:.2f}")
                    if logger: logger.info(f" Avg Test Reward for {algo.upper()} [{mode}] = {avg_reward:.2f}")

                results.append((algo, mode, avg_reward))

    # Output summary if testing
    if not train:
        if logger: logger.info("Final Results: Algo × Emotion Mode ===")
        for algo, mode, reward in results:
            if logger: logger.info(f"[{algo.upper()} - {mode}] Avg Test Reward: {reward:.2f}")

    return results


def run_experiment_2(algo='sac', train=True, episodes=1000, max_steps=100, device='cuda', log_prefix=None, logger=None):
    """
    Run a single-agent training or testing experiment for baseline RL algorithms.

    This function supports multiple RL algorithms, builds the appropriate agent and environment,
    trains or evaluates it, and logs results to files.

    Args:
        algo (str): Name of the RL algorithm to use (e.g., 'sac', 'td3', 'ppo', etc.).
        train (bool): Whether to run training or testing.
        episodes (int): Number of episodes to train or test.
        max_steps (int): Max steps per episode.
        device (str): PyTorch device ('cuda' or 'cpu').
        log_prefix (str): Prefix used for saving models and logs.
        logger (logging.Logger): Optional logger object for logging.

    Returns:
        tuple: (model_path (str), average_reward (float))
    """
    # Import necessary agent builders and training functions

    # === Initialize environment ===
    env = WeightEnv()

    # === Mapping algorithm name to corresponding builder function ===
    algo_builders = {
        'ddpg': build_ddpg_agent,
        'td3': build_td3_agent,
        'td3_bc': build_td3_bc_agent,
        'ppo': build_ppo_agent,
        'sac': build_sac_agent,
        'cql': build_cql_agent,
        'ppol': build_ppol_agent,
        'rls_pid': build_rls_pid_agent,
        'erl_fill': build_erl_fill_agent,
        'fuzzy': build_fuzzy_agent  # Fuzzy controller is also supported
    }

    assert algo in algo_builders, f"Unsupported algorithm: {algo}"
    agent = algo_builders[algo](env, device=device)
    env.attach_agent(agent)

    # === Model path for saving/loading ===
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")

    # === Training Phase ===
    if train and algo != 'fuzzy':
        print(f"Start training {algo.upper()}...")

        if algo == 'ddpg':
            train_ddpg(env, agent, episodes=episodes, max_steps=max_steps,
                       log_prefix=log_prefix, logger=logger)
            agent.save(model_path)

        elif algo == 'td3':
            train_td3(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, logger=logger)
            agent.save(model_path)

        elif algo == 'td3_bc':
            train_td3_bc(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                         log_prefix=log_prefix, logger=logger)
            agent.save(model_path)

        elif algo == 'ppo':
            train_ppo(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path, logger=logger)

        elif algo == 'sac':
            train_sac(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path, logger=logger)

        elif algo == 'cql':
            train_cql(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path, logger=logger)

        elif algo == 'ppol':
            train_ppo_lagrangian(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                                 log_prefix=log_prefix, model_path=model_path)

        elif algo == 'rls_pid':
            train_rls_pid(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                          log_prefix=log_prefix, model_path=model_path)

        elif algo == 'erl_fill':
            train_erl_fill(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                           log_prefix=log_prefix, model_path=model_path, logger=logger)

        else:
            print("No such module")

        print(f"Training complete. Model saved at: {model_path}")
        avg_reward = 0

    # === Testing Phase ===
    else:
        print(f"Start testing {algo.upper()}...")

        if algo != 'fuzzy' and not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        if algo != 'fuzzy':
            agent.load(model_path)
            print(f"✅ Model weights loaded from: {model_path}")

        test_episodes = 10
        total_reward = 0

        test_log_path = os.path.join("logs", f"exp2_{algo}.log")
        os.makedirs("logs", exist_ok=True)

        # Helper: safely extract scalar value from numpy arrays
        def extract_scalar(v):
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    return v.item()
                elif v.ndim == 0:
                    return float(v)
                else:
                    raise ValueError(f"❌ Non-scalar ndarray: {v} (shape={v.shape})")
            return float(v)

        # Run test episodes
        with open(test_log_path, "w", encoding="utf-8") as f_log:
            for ep in range(test_episodes):
                state = env.reset()
                episode_reward = 0
                try:
                    for step in range(max_steps):
                        # Select action depending on algorithm
                        if algo in ['ppo', 'sac']:
                            action = agent.act(state, deterministic=True) if hasattr(agent, 'act') else agent.select_action(state)
                        else:
                            action = agent.act(state)

                        # Format action as dict
                        if isinstance(action, dict):
                            action_dict = action
                        else:
                            action_dict = {k: extract_scalar(v) for k, v in zip(env.bounds.keys(), action)}

                        # Apply action in environment
                        next_state, reward, done, info = env.step(action_dict)
                        state = next_state
                        episode_reward += reward
                        if done:
                            break

                    slow_weight = info["action"].get("slow_weight", 0.0)
                    log_str = (
                        f"[{algo.upper()} | Episode {ep:04d}] "
                        f"TotalReward: {episode_reward:.2f} | "
                        f"WeightErr: {info['weight_error']:.2f}g | "
                        f"Time: {info['total_time']:.2f}s | "
                        f"SlowWeight: {slow_weight:.2f}g"
                    )
                    print(f"Test Episode {ep + 1}: {episode_reward:.2f}")
                    f_log.write(log_str + "\n")

                except Exception as e:
                    error_msg = f"[{algo.upper()} | Episode {ep:04d}] ❌ Error during testing: {e}"
                    print(error_msg)
                    f_log.write(error_msg + "\n")

                total_reward += episode_reward

            # Average reward across episodes
            avg_reward = total_reward / test_episodes
            summary = f"\nAvg Test Reward for {algo.upper()}: {avg_reward:.2f}\n"
            print(summary)
            f_log.write(summary)

    return model_path, avg_reward


def run_experiment_3(train=True, episodes=1000, max_steps=100, device='cuda', algo='er_ddpg'):
    """
    Run a multi-condition training or testing experiment for advanced RL algorithms (e.g., ER-DDPG, ERL-Fill).

    This function evaluates RL agents under different environmental settings (e.g., 25kg, 20kg, 15kg variants),
    supporting algorithms with emotion or hybrid mechanisms. It builds the environment and agent, performs training
    or evaluation, saves model and logs, and records detailed results.

    Args:
        train (bool): Whether to run training or testing.
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        device (str): PyTorch device ('cuda' or 'cpu').
        algo (str): Name of the algorithm to use (e.g., 'er_ddpg', 'erl_fill').

    Returns:
        None
    """

    # === 1. Configuration for different test conditions ===
    experiment_configs = {
        "variant_25kg": {
            "name": "Condition Variant - 25kg±25g",
            "env_kwargs": {
                "target_weight": 25000,
                "target_weight_err": 25,
                "target_time": 2.5,
                "bounds": {
                    "fast_weight": (7000, 18000),
                    "medium_weight": (0, 1),
                    "slow_weight": (24900, 25000),
                    "fast_opening": (35, 75),
                    "medium_opening": (3, 5),
                    "slow_opening": (5, 20),
                    "fast_delay": (100, 300),
                    "medium_delay": (100, 200),
                    "slow_delay": (100, 200),
                    "unload_delay": (300, 500)
                }
            },
        },
        "variant_20kg": {
            "name": "Condition Variant - 20kg±20g",
            "env_kwargs": {
                "target_weight": 20000,
                "target_weight_err": 20,
                "target_time": 2.0,
                "bounds": {
                    "fast_weight": (6000, 15000),
                    "medium_weight": (0, 1),
                    "slow_weight": (19800, 20000),
                    "fast_opening": (30, 70),
                    "medium_opening": (3, 5),
                    "slow_opening": (5, 18),
                    "fast_delay": (80, 250),
                    "medium_delay": (80, 180),
                    "slow_delay": (80, 180),
                    "unload_delay": (280, 480)
                }
            },
        },
        "variant_15kg": {
            "name": "Condition Variant - 15kg±15g",
            "env_kwargs": {
                "target_weight": 15000,
                "target_weight_err": 15,
                "target_time": 1.5,
                "bounds": {
                    "fast_weight": (5000, 12000),
                    "medium_weight": (0, 1),
                    "slow_weight": (14800, 15000),
                    "fast_opening": (25, 65),
                    "medium_opening": (2, 4),
                    "slow_opening": (4, 15),
                    "fast_delay": (60, 200),
                    "medium_delay": (60, 150),
                    "slow_delay": (60, 150),
                    "unload_delay": (250, 450)
                }
            },
        }
    }

    # === 2. Agent construction function ===
    def build_agent(env, algo_name, device):
        if algo_name == 'er_ddpg':
            agent = DDPGAgent(env, device=device, use_td_error=True)
            agent.emotion = EmotionModule(device=device)
        elif algo_name == 'erl_fill':
            agent = EmotionSACAgent(env, device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")
        return agent

    # === 3. Training function mapping ===
    train_funcs = {
        'er_ddpg': lambda env, agent, episodes, max_steps, log_prefix, logger:
        train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, logger=logger),
        'erl_fill': lambda env, agent, episodes, max_steps, log_prefix, logger:
        train_erl_fill(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
    }

    # === 4. Create output directory ===
    os.makedirs("analysis_outputs/exp3", exist_ok=True)
    results = []

    # === 5. Loop through each experimental condition ===
    for key, cfg in experiment_configs.items():
        print(f"\n=== Running {algo.upper()} under condition: {cfg['name']} ===")

        # Create environment and agent
        env = MultiConditionWeightEnv(cfg["env_kwargs"])
        agent = build_agent(env, algo, device)
        env.attach_agent(agent)

        log_prefix = f"exp3_{algo}_{key}"
        model_path = f"saved_models/exp3_{algo}/{key}_final.pth"
        log_file_path = f"logs/{log_prefix}_train.log"

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Configure logger
        logger = logging.getLogger(log_prefix)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        # === Training phase ===
        if train:
            print(f"🚀 Start training {algo.upper()} under {cfg['name']}")
            train_funcs[algo](env, agent, episodes, max_steps, log_prefix, logger)
            agent.save(model_path)
            print(f"✅ Model saved to {model_path}")

        # === Testing phase ===
        agent.load(model_path)
        total_reward, total_weight_err, total_time = 0, 0, 0
        results_detail = []

        test_log_file = f"logs/{log_prefix}_test.log"
        with open(test_log_file, "w", encoding="utf-8") as f:
            for ep in range(10):
                state = env.reset()
                ep_reward = 0
                for _ in range(max_steps):
                    if algo in ['ppo', 'sac']:
                        action = agent.act(state, deterministic=True) if hasattr(agent, 'act') else agent.select_action(state)
                    else:
                        action = agent.act(state, add_noise=False)

                    if isinstance(action, tuple):
                        action = action[0]  # Some agents return (action, value), take only action

                    state, reward, done, info = env.step(action)
                    ep_reward += reward
                    if done:
                        break

                # Extract testing results
                weight_err = info.get("weight_error", 0.0)
                fill_time = info.get("total_time", 0.0)
                slow_weight = info.get("action", {}).get("slow_weight", 0.0)
                emotion = agent.emotion.get_emotion()
                if isinstance(emotion, torch.Tensor):
                    emotion = emotion.detach().cpu().numpy().squeeze()
                emotion = np.array(emotion).flatten()

                # Log and print per-episode results
                print(f"🎯 {cfg['name']} Episode {ep + 1}: Reward = {ep_reward:.2f}, "
                      f"WeightErr = {weight_err:.2f}g, Time = {fill_time:.2f}s, SlowWeight = {slow_weight:.2f}g")
                f.write(f"[{cfg['name']} Episode {ep + 1}] Reward: {ep_reward:.2f} | "
                        f"WeightErr: {weight_err:.2f}g | Time: {fill_time:.2f}s | SlowWeight: {slow_weight:.2f}g | "
                        f"Emotion: [Curi: {emotion[0]:.2f}, Cons: {emotion[1]:.2f}, Anx: {emotion[2]:.2f}]\n")

                total_reward += ep_reward
                total_weight_err += weight_err
                total_time += fill_time

                results_detail.append({
                    "Condition": cfg['name'],
                    "Episode": ep + 1,
                    "Reward": ep_reward,
                    "WeightErr": weight_err,
                    "Time": fill_time,
                    "SlowWeight": slow_weight,
                    "Curiosity": float(emotion[0]),
                    "Conservativeness": float(emotion[1]),
                    "Anxiety": float(emotion[2])
                })

            # Average results over episodes
            avg_reward = total_reward / 10
            avg_err = total_weight_err / 10
            avg_time = total_time / 10

            print(f"✅ {cfg['name']} Avg Test Reward: {avg_reward:.2f}, Avg Error: {avg_err:.2f}g, Avg Time: {avg_time:.2f}s")
            f.write(f"\n✅ Avg Test Result for {cfg['name']}: "
                    f"Reward = {avg_reward:.2f}, WeightErr = {avg_err:.2f}g, Time = {avg_time:.2f}s\n")

            results.append((cfg['name'], avg_reward, avg_err, avg_time))

        # Save per-condition detailed CSV results
        df_detail = pd.DataFrame(results_detail)
        df_detail.to_csv(f"analysis_outputs/exp3/test_results_{algo}_{key}.csv", index=False, encoding="utf-8-sig")

    # === Final summary output ===
    result_file = f"logs/exp3_summary.log"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Condition\tReward\tWeightErr(g)\tTime(s)\n")
        for name, score, err, t in results:
            f.write(f"{name}\t{score:.2f}\t{err:.2f}\t{t:.2f}\n")
        print(f"\n📄 Final results of Experiment 4 ({algo.upper()}) written to: {result_file}")


def run_experiment_4(device='cuda', max_steps=100, train=True, use_off_sim=1):
    """
    Run a staged pretraining experiment with different environment modes (simulation, onboard, real).

    This function is designed for modular experimentation across several training stages
    such as simulation, onboard hardware, or real-world deployment. It supports logging,
    directory creation, and reusable model paths.

    Args:
        device (str): Computing device to use (e.g., 'cuda' or 'cpu').
        max_steps (int): Maximum steps per episode.
        train (bool): Whether to enable training mode.
        use_off_sim (int): Stage selection flag. Must be one of:
            - 1: Simulation-only stage
            - 0: Onboard hardware stage
            - 2: Real-world stage
            - 3: Continued training on real-world data

    Returns:
        None. Prints status and saves logs/models to disk.
    """

    # Configuration for each stage. Each entry is a list of dicts defining:
    # - env_mode: environment type
    # - episodes: number of episodes to run
    # - log_prefix: used for naming log/model files
    experiment_configs = {
        1: [{"env_mode": "sim", "episodes": 5000, "log_prefix": "stage1_sim"}],
        0: [{"env_mode": "onboard", "episodes": 500, "log_prefix": "stage2_onboard"}],
        2: [{"env_mode": "real", "episodes": 100, "log_prefix": "stage3_real"}],
        3: [{"env_mode": "continue", "episodes": 5000, "log_prefix": "stage4_real"}]
    }

    # Human-readable names for stages
    stage_names = {1: "stage1", 0: "stage2", 2: "stage3", 3: "stage4"}

    # Validate the selected stage
    if use_off_sim not in experiment_configs:
        print(f"❌ Invalid value for use_off_sim: {use_off_sim}. Must be 0, 1, 2 or 3.")
        return

    selected_stage = experiment_configs[use_off_sim]
    stage_name = stage_names[use_off_sim]

    print(f"\n🚀 Starting {stage_name.upper()} pretraining experiment...\n")

    model_path = None  # Path to the saved model, to be updated per stage

    # Iterate through phases in the selected stage (usually one, but list allows future extension)
    for i, cfg in enumerate(selected_stage):
        log_prefix = f"exp4_erl_fill_{cfg['log_prefix']}"
        log_file_path = f"logs/{log_prefix}_train.log"
        model_path = f"saved_models/exp4_erl_fill/{cfg['log_prefix']}_final.pth"

        # ✅ Ensure that directories for logs and model saving exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # ✅ Configure logging to file
        logger = logging.getLogger(log_prefix)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        print(f"▶️  Phase {i + 1} | Mode: {cfg['env_mode']} | Episodes: {cfg['episodes']}")

        # 🔁 Run the experiment phase with specific configuration
        model_path = run_experiment_1(
            emotion_modes=['transformer'],  # Emotion recognition module mode
            algo_list=['sac'],  # RL algorithm (e.g., Soft Actor-Critic)
            episodes=cfg["episodes"],  # Number of training episodes
            max_steps=max_steps,  # Max steps per episode
            device=device,  # Compute device
            env_mode=cfg["env_mode"],  # Environment mode (sim, real, etc.)
            train=train,  # Training enabled/disabled
            lambda_emo=0.1,  # Emotion-related reward weighting
            log_prefix=f"{log_prefix}_train",  # Prefix for log naming
            logger=logger  # Logger instance for file logging
        )

    print(f"✅ {stage_name.upper()} training complete. Final model saved at: {model_path}")


if __name__ == "__main__":
    # === 🔧 Global Configuration ===
    device = 'cuda'                # Device to use for training/testing
    train_mode = True              # ✅ True: Train mode | False: Test only
    experiment_id = 4              # ✅ Select which experiment to run: 1, 2, 3, or 4
    episodes = 5                   # Number of training episodes per run
    max_steps = 50                 # Maximum steps per episode
    test_episodes = 10             # Number of test episodes per configuration
    use_offline_sim = 1           # Environment selection:
                                  # 1 - offline simulation
                                  # 0 - onboard simulation
                                  # 2 - real-world training

    # === 🧪 Experiment 1: Ablation Study on Emotion Mechanism ===
    if experiment_id == 1:
        logger = init_logger()

        # Define the emotion modes and algorithms used
        emotion_modes = ['none', 'simple', 'transformer']  # Baseline, Simple heuristic, Transformer-based
        algo_list = ['sac']  # Only SAC is used in this experiment
        results = []

        # === Define 5 experiment configurations with various λ_emo settings ===
        experiment_configs = [
            {'name': 'Baseline', 'mode': 'none', 'lambda_emo': 0.0},
            {'name': 'Simple', 'mode': 'simple', 'lambda_emo': 0.05},
            {'name': 'Transformer', 'mode': 'transformer', 'lambda_emo': 0.0},
            {'name': 'Transformer', 'mode': 'transformer', 'lambda_emo': 0.01},
            {'name': 'Transformer-High', 'mode': 'transformer', 'lambda_emo': 0.1},
            {'name': 'Transformer-Low', 'mode': 'transformer', 'lambda_emo': 0.2},
        ]

        # === Loop through each configuration ===
        for config in experiment_configs:
            mode = config['mode']
            lambda_emo = config['lambda_emo']
            group_name = config['name']

            # Prefix for log and result files
            log_prefix = f"exp1_{group_name}_lambda{lambda_emo}".replace(".", "_")

            logger.info(f"=== Running Group: {group_name} | Mode: {mode} | λ_emo: {lambda_emo} ===")

            # === Training Phase ===
            _ = run_experiment_1(
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                env_mode='sim',
                emotion_modes=[mode],
                algo_list=algo_list,
                train=True,
                lambda_emo=lambda_emo,
                log_prefix=f"{log_prefix}_train",
                logger=logger
            )

            # === Evaluation Phase ===
            test_results = run_experiment_1(
                episodes=test_episodes,
                max_steps=max_steps,
                device=device,
                env_mode='sim',
                emotion_modes=[mode],
                algo_list=algo_list,
                train=False,
                lambda_emo=lambda_emo,
                log_prefix=f"{log_prefix}_test",
                logger=logger
            )

            # Collect test results
            for algo, mode_result, avg_reward in test_results:
                results.append((group_name, algo, mode_result, avg_reward))

    # === 🧪 Experiment 2: RL Algorithm Comparison ===
    elif experiment_id == 2:
        logger = init_logger()
        print(f"\n=== Running Algorithm Comparison for All Methods ===")

        algo_list = ['ddpg']  # Replace with other algorithms as needed
        # algo_list = ['td3', 'sac', 'ppo', 'ddpg', 'td3_bc', 'rls_pid', 'cql', 'erl_fill', 'fuzzy']
        results = []

        for algo in algo_list:
            log_prefix = f"exp2_{algo}"  # Prefix for logs

            print(f"\n▶️ Start {algo.upper()} Training & Testing")

            # Run training and testing for the selected algorithm
            model_path, test_reward = run_experiment_2(
                algo=algo,
                train=train_mode,
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                log_prefix=log_prefix,
                logger=logger
            )
            results.append((algo, test_reward))

        # === Output final comparison results ===
        print("\n=== ✅ Experiment 2 Results Summary ===")
        for algo, reward in results:
            print(f"[{algo.upper()}] Average Test Reward: {reward:.2f}")

    # === 🧪 Experiment 3: Multi-Condition Generalization Test ===
    elif experiment_id == 3:
        print(f"\n=== Running Multi-Condition Comparison Experiment ===")

        # Algorithms to evaluate under various conditions
        algo_list = ['erl_fill']  # You can include more algorithms here

        for algo in algo_list:
            print(f"\n=== 🚀 Running {algo.upper()} for Multi-Condition ===")
            run_experiment_3(
                train=True,            # Set False to skip training and only evaluate
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                algo=algo              # Algorithm identifier
            )

    # === 🧪 Experiment 4: Multi-Stage Pretraining ===
    elif experiment_id == 4:
        print("\n=== Running Multi-Stage Pretraining Evaluation ===")
        run_experiment_4(
            device=device,
            train=train_mode,
            use_off_sim=use_offline_sim
        )

    # === ❌ Invalid Experiment ID Handler ===
    else:
        print("❌ Unsupported experiment ID. Please set experiment_id = 1, 2, 3, or 4.")
