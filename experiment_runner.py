import torch
import logging
import numpy as np
import os
import pandas as pd

from CommonInterface.ScaleTransformer import reset_actor_scaling

from WeightEnv import WeightEnv
from EmotionModule import EmotionModuleNone,EmotionModule,EmotionModuleSimple
from DifferentModules.ddpg_agent import DDPGAgent, train_ddpg
from DifferentModules.td3_agent import TD3Agent
from DifferentModules.sac_agent import SACAgent,train_sac
from DifferentModules.ERL_Fill_agent import EmotionSACAgent,train_emotion_sac
from DifferentModules.simple_emotion_sac_agent import SimpleEmotionSACAgent,train_simple_sac
from MutiConditionEnv import MultiConditionWeightEnv

def run_experiment_1(
    episodes=1000, max_steps=500, device='cuda', env_mode='sim',
    emotion_modes=['none', 'simple', 'transformer'],
    algo_list=['ddpg', 'td3', 'sac'],
    train=True, # ✅ 加了开关
    experiment_id = 1,
    lambda_emo=0.05,
    log_prefix="experiment"
):
    """
    Emotion × Algorithm 全组合实验，支持 train/test 分开执行。
    """
    results = []

    for algo in algo_list:
        for mode in emotion_modes:
            print(f"\n=== Running {algo.upper()} | Emotion Mode: {mode.upper()} | Train={train} ===")

            # Step 1: 创建环境
            env = WeightEnv()
            if env_mode == 'sim':
                env.use_offline_sim = 1
            if env_mode == 'onboard':
                env.use_offline_sim = 0
            if env_mode == 'real':
                env.use_offline_sim = 2

            # Step 2: 创建 Emotion 模块
            if mode == 'transformer':
                emotion_module = EmotionModule(device=device)
            elif mode == 'simple':
                emotion_module = EmotionModuleSimple()
            elif mode == 'none':
                emotion_module = EmotionModuleNone()
            else:
                raise ValueError(f"Unsupported emotion mode: {mode}")

            # Step 3: 创建 Agent
            if algo == 'ddpg':
                # from WeightDemo import DDPGAgent
                agent = DDPGAgent(env, device=device)
            elif algo == 'td3':
                # from td3_module import TD3Agent
                agent = TD3Agent(env, device=device)
            elif algo == 'sac':
                # from Emotion_sac import EmotionSACAgent
                # from sac_module import SACAgent
                # from simple_emotion_sac import SimpleEmotionSACAgent
                if mode == 'none':
                    agent = SACAgent(env, device=device)
                elif mode =="simple":
                    agent = SimpleEmotionSACAgent(env, device=device)
                else:
                    agent = EmotionSACAgent(env, device=device)
            else:
                raise ValueError(f"Unsupported algorithm: {algo}")

            # Step 4: 绑定情感模块
            if hasattr(agent, 'emotion'):
                agent.emotion = emotion_module
            env.attach_agent(agent)

            # Step 5: 路径
            if experiment_id == 2:
                model_dir = f"saved_models/{algo}_exp2_emotion_compare"
            else:
                model_dir = f"saved_models/{algo}_emotion_compare"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{algo}_{mode}_{lambda_emo}.pth")
            log_prefix = f"exp1_{algo}_emotion_{mode}_{lambda_emo}"

            if env_mode == 'sim':
                print("已选择不加载预训练权重，将从头开始训练。")
            elif env_mode == 'onboard':
                print("执行板载训练。")
                pretrain_path = "pre_train_pth/sac_transformer.pth"
                # agent.load(pretrain_path)
                # print(f"成功加载预训练权重: {pretrain_path}")
            elif env_mode == 'real':
                print("执行真实训练。")
                pretrain_path = "pre_train_pth/AIM_v2_onboard_24900_25000.pth"
                weights = torch.load(pretrain_path, weights_only=False)
                agent.load_weights(weights)
                reset_actor_scaling(agent, env.bounds)
                print(f"成功加载预训练权重: {pretrain_path}")
            else:
                print("加载现有默认权重，继续执行训练。")
                pretrain_path = "pre_train_pth/AIM_v2_sim_24900_25000.pth"
                agent.load(pretrain_path)
                # reset_actor_scaling(agent, env.bounds)
                print(f"成功加载预训练权重: {pretrain_path}")

            if train:
                # === 训练阶段 ===
                print(f"🚀 Start training [{algo.upper()}] with emotion mode [{mode}]")
                if algo == 'ddpg':
                    train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
                elif algo == 'sac':
                    if mode == 'none':
                        train_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
                    elif mode == "simple":
                        train_simple_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
                    else:
                        train_emotion_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, lambda_emo=lambda_emo)
                agent.save(model_path)
                print(f"✅ Training complete. Model saved to {model_path}")

            else:
                # === 测试阶段 ===
                print(f"🔍 Start testing [{algo.upper()}] with emotion mode [{mode}]")
                if not os.path.exists(model_path):
                    print(f"❌ Model not found: {model_path}. Skip.")
                    continue
                agent.load(model_path)

                total_reward = 0
                test_episodes = 10
                test_log_dir = os.path.join("logs", f"test_result_{algo}_{mode}.log")
                os.makedirs("logs", exist_ok=True)

                with open(test_log_dir, "w", encoding="utf-8") as f_log:
                    for ep in range(test_episodes):
                        state = env.reset()
                        episode_reward = 0
                        for step in range(max_steps):
                            if hasattr(agent, 'act'):
                                # === 通用动作选择（根据算法适配）
                                if algo in ['er_ddpg', 'ddpg', 'td3', 'emotion_td3']:
                                    action = agent.act(state, add_noise=False)

                                elif algo in ['sac', 'emotion_sac']:
                                    action = agent.act(state, deterministic=True)

                                elif algo in ['ppo']:
                                    action, _ = agent.select_action(state)

                                else:
                                    raise ValueError(f"Unsupported algorithm type: {algo}")
                            else:
                                action = agent.select_action(state)
                            action = action.squeeze() if isinstance(action, np.ndarray) else action
                            next_state, reward, done, info = env.step(action)
                            state = next_state
                            episode_reward += reward
                            if done:
                                break

                        total_reward += episode_reward

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

                        print(f"🎯 Test Episode {ep+1}: Reward = {episode_reward:.2f}")
                        f_log.write(log_str + "\n")

                    avg_reward = total_reward / test_episodes
                    f_log.write(f"\n✅ Avg Test Reward: {avg_reward:.2f}\n")
                    print(f"✅ Avg Test Reward for {algo.upper()} [{mode}] = {avg_reward:.2f}")

                results.append((algo, mode, avg_reward))

    # === 汇总测试结果 ===
    if not train:
        print("\n=== 🎯 Final Results: Algo × Emotion Mode ===")
        for algo, mode, reward in results:
            print(f"[{algo.upper()} - {mode}] Avg Test Reward: {reward:.2f}")

    return results


def run_experiment_2(algo='er_ddpg', train=True, episodes=1000, max_steps=100, device='cuda'):
    # from WeightDemo import WeightEnv
    from baseline_experiments import (
        # build_er_ddpg_agent,
        build_ddpg_agent,
        build_td3_agent,
        build_ppo_agent,
        build_sac_agent,
        build_td3_bc_agent, build_cql_agent,
        # build_emotion_td3_agent,
        build_emotion_sac_agent,
        build_fuzzy_agent,
        build_ppol_agent, build_rls_pid_agent
    )
    from DifferentModules.td3_agent import train_td3
    from DifferentModules.td3_bc_agent import train_td3_bc
    from DifferentModules.cql_agent import train_cql
    from DifferentModules.PPOL import train_ppo_lagrangian
    from DifferentModules.rls_pidagent import train_rls_pid
    from DifferentModules.ppo_agent import train_ppo
    from DifferentModules.sac_agent import train_sac
    # from Emotion_TD3 import train_emotion_td3
    import os
    import numpy as np

    env = WeightEnv()
    algo_builders = {
        # 'er_ddpg': build_er_ddpg_agent,
        'ddpg': build_ddpg_agent,
        'td3': build_td3_agent,
        'td3_bc': build_td3_bc_agent,
        'ppo': build_ppo_agent,
        'sac': build_sac_agent,
        'cql': build_cql_agent,
        'ppol': build_ppol_agent,
        'rls_pid': build_rls_pid_agent,
        # 'emotion_td3': build_emotion_td3_agent,
        'emotion_sac': build_emotion_sac_agent,
        'fuzzy': build_fuzzy_agent  # ✅ 添加 fuzzy agent
    }
    assert algo in algo_builders, f"Unsupported algorithm: {algo}"
    agent = algo_builders[algo](env, device=device)
    env.attach_agent(agent)

    model_dir = os.path.join("saved_models", algo)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{algo}_final.pth")
    log_prefix = f"exp3_{algo}"

    if train and algo != 'fuzzy':  # 模糊控制不训练
        print(f"�� Start training {algo.upper()}...")
        if algo == 'ddpg':
            from DifferentModules.ddpg_agent import train_ddpg
            train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
            agent.save(model_path)

        elif algo == 'td3':
            train_td3(env=env, agent=agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
            agent.save(model_path)

        elif algo == 'td3_bc':
            train_td3_bc(env=env, agent=agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
            agent.save(model_path)

        elif algo == 'ppo':
            train_ppo(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path)

        elif algo == 'sac':
            train_sac(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path)

        elif algo == 'cql':
            train_cql(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                      log_prefix=log_prefix, model_path=model_path)

        elif algo == 'ppol':
            train_ppo_lagrangian(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                                 log_prefix=log_prefix, model_path=model_path)

        elif algo == 'rls_pid':
            train_rls_pid(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                          log_prefix=log_prefix, model_path=model_path)

        # elif algo == 'emotion_td3':
        #     train_emotion_td3(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
        #                       log_prefix=log_prefix, model_path=model_path)

        elif algo == 'emotion_sac':
            from DifferentModules.ERL_Fill_agent import train_emotion_sac
            train_emotion_sac(env=env, agent=agent, episodes=episodes, max_steps=max_steps,
                              log_prefix=log_prefix, model_path=model_path)
        else:
            print("No such module")

        print(f"✅ Training complete. Model saved at: {model_path}")
        avg_reward = 0
    else:
        print(f"�� Start testing {algo.upper()}...")
        if algo != 'fuzzy' and not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
        test_episodes = 200
        if algo != 'fuzzy':
            agent.load(model_path)
            test_episodes = 100
            print(f"✅ 模型权重已加载：{model_path}")

        total_reward = 0

        test_log_path = os.path.join("logs", f"test_result_{algo}.log")
        os.makedirs("logs", exist_ok=True)

        def extract_scalar(v):
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    return v.item()
                elif v.ndim == 0:
                    return float(v)
                else:
                    raise ValueError(f"❌ 非标量 ndarray: {v} (shape={v.shape})")
            return float(v)

        with open(test_log_path, "w", encoding="utf-8") as f_log:
            for ep in range(test_episodes):
                state = env.reset()
                episode_reward = 0
                try:
                    for step in range(max_steps):
                        if algo in ['ppo', 'sac']:
                            action = agent.act(state, deterministic=True) if hasattr(agent,
                                                                                     'act') else agent.select_action(
                                state)
                        else:
                            action = agent.act(state)

                        # ✅ 修复 fuzzy 返回 dict 错误
                        if isinstance(action, dict):
                            action_dict = action
                        else:
                            action_dict = {k: extract_scalar(v) for k, v in zip(env.bounds.keys(), action)}

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
                    print(f"�� Test Episode {ep + 1}: {episode_reward:.2f}")
                    f_log.write(log_str + "\n")

                except Exception as e:
                    error_msg = f"[{algo.upper()} | Episode {ep:04d}] ❌ 测试出错: {e}"
                    print(error_msg)
                    f_log.write(error_msg + "\n")

                total_reward += episode_reward

            avg_reward = total_reward / test_episodes
            summary = f"\n✅ Avg Test Reward for {algo.upper()}: {avg_reward:.2f}\n"
            print(summary)
            f_log.write(summary)

    return model_path, avg_reward

def run_experiment_3(train=True, episodes=1000, max_steps=100, device='cuda', algo='er_ddpg'):
    """
    实验四：多工况对比实验（ER-DDPG）
    """
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

    # === 2. Agent 构建函数 ===
    def build_agent(env, algo_name, device):
        if algo_name == 'er_ddpg':
            # from WeightDemo import DDPGAgent, EmotionModule
            agent = DDPGAgent(env, device=device, use_td_error=True)
            agent.emotion = EmotionModule(device=device)
        # elif algo_name == 'emotion_td3':
        #     from Emotion_TD3 import EmotionTD3Agent
        #     agent = EmotionTD3Agent(env, device=device, use_td_error=True)
        elif algo_name == 'emotion_sac':
            # from Emotion_sac import EmotionSACAgent
            agent = EmotionSACAgent(env, device=device)
            # agent.emotion = EmotionModule(device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")
        return agent

    # === 3. Train 函数映射 ===
    train_funcs = {
        'er_ddpg': lambda env, agent, episodes, max_steps, log_prefix, logger:
        train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, logger=logger),
        # 'emotion_td3': lambda env, agent, episodes, max_steps, log_prefix, logger:
        # train_emotion_td3(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix),
        'emotion_sac': lambda env, agent, episodes, max_steps, log_prefix, logger:
        train_emotion_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
    }

    # === 4. 初始化目录 ===
    os.makedirs(f"saved_models/{algo}", exist_ok=True)
    os.makedirs(f"logs/{algo}", exist_ok=True)
    os.makedirs("analysis_outputs/exp4", exist_ok=True)

    results = []

    for key, cfg in experiment_configs.items():
        print(f"\n=== Running {algo.upper()} under condition: {cfg['name']} ===")

        # 创建环境和Agent
        env = MultiConditionWeightEnv(cfg["env_kwargs"])
        agent = build_agent(env, algo, device)
        env.attach_agent(agent)

        log_prefix = f"exp4_{algo}_{key}"
        model_path = f"saved_models/{algo}/{key}_final.pth"
        log_file_path = f"logs/{algo}/train_{log_prefix}.log"

        # Logger配置
        logger = logging.getLogger(log_prefix)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        # === 训练 ===
        if train:
            print(f"🚀 Start training {algo.upper()} under {cfg['name']}")
            train_funcs[algo](env, agent, episodes, max_steps, log_prefix, logger)
            agent.save(model_path)
            print(f"✅ Model saved to {model_path}")

        # === 测试 ===
        agent.load(model_path)
        total_reward, total_weight_err, total_time = 0, 0, 0
        results_detail = []

        test_log_file = f"logs/{algo}/test_{log_prefix}.log"
        with open(test_log_file, "w", encoding="utf-8") as f:
            for ep in range(10):
                state = env.reset()
                ep_reward = 0
                for _ in range(max_steps):
                    if algo in ['ppo', 'sac']:
                        action = agent.act(state, deterministic=True) if hasattr(agent, 'act') else agent.select_action(
                            state)
                    else:
                        action = agent.act(state, add_noise=False)

                    if isinstance(action, tuple):
                        action = action[0]  # 有些agent返回(action, value)的，取第一个

                    state, reward, done, info = env.step(action)
                    ep_reward += reward
                    if done:
                        break

                weight_err = info.get("weight_error", 0.0)
                fill_time = info.get("total_time", 0.0)
                slow_weight = info.get("action", {}).get("slow_weight", 0.0)
                emotion = agent.emotion.get_emotion()
                if isinstance(emotion, torch.Tensor):
                    emotion = emotion.detach().cpu().numpy().squeeze()
                emotion = np.array(emotion).flatten()

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

            avg_reward = total_reward / 10
            avg_err = total_weight_err / 10
            avg_time = total_time / 10

            print(f"✅ {cfg['name']} 平均测试奖励: {avg_reward:.2f}, 平均误差: {avg_err:.2f}g, 平均时间: {avg_time:.2f}s")
            f.write(f"\n✅ Avg Test Result for {cfg['name']}: "
                    f"Reward = {avg_reward:.2f}, WeightErr = {avg_err:.2f}g, Time = {avg_time:.2f}s\n")

            results.append((cfg['name'], avg_reward, avg_err, avg_time))

        # 保存每个variant的详细CSV
        df_detail = pd.DataFrame(results_detail)
        df_detail.to_csv(f"analysis_outputs/exp4/test_results_{algo}_{key}.csv", index=False, encoding="utf-8-sig")

    # 汇总所有Condition的结果
    result_file = f"logs/{algo}/experiment4_summary.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Condition\tReward\tWeightErr(g)\tTime(s)\n")
        for name, score, err, t in results:
            f.write(f"{name}\t{score:.2f}\t{err:.2f}\t{t:.2f}\n")
        print(f"\n📄 实验四({algo.upper()})最终结果写入: {result_file}")


def run_experiment_4(device='cuda', max_steps=100, train=True, use_off_sim=1):
    # 预设阶段训练配置
    experiment_configs = {
        1: [{"env_mode": "sim",     "episodes": 5000, "log_prefix": "stage1_sim"}],
        0: [{"env_mode": "onboard", "episodes": 500,  "log_prefix": "stage2_onboard"}],
        2: [{"env_mode": "real",    "episodes": 100,  "log_prefix": "stage3_real"}],
        3: [{"env_mode": "continue", "episodes": 5000, "log_prefix": "stage3_real"}]
    }

    stage_names = {1: "stage1", 0: "stage2", 2: "stage3", 3:"stage4"}

    if use_off_sim not in experiment_configs:
        print(f"❌ Invalid value for use_off_sim: {use_off_sim}. Must be 0, 1,  2 or 3.")
        return

    selected_stage = experiment_configs[use_off_sim]
    stage_name = stage_names[use_off_sim]

    print(f"\n🚀 Starting {stage_name.upper()} pretraining experiment...\n")

    model_path = None  # 可复用权重路径

    for i, cfg in enumerate(selected_stage):
        print(f"▶️  Phase {i+1} | Mode: {cfg['env_mode']} | Episodes: {cfg['episodes']}")

        model_path = run_experiment_1(
            emotion_modes=['transformer'],
            algo_list=['sac'],
            episodes=cfg["episodes"],
            max_steps=max_steps,
            device=device,
            env_mode=cfg["env_mode"],
            train=train,
            experiment_id=3,
            lambda_emo = 0.05
        )

    print(f"✅ {stage_name.upper()} training complete. Final model saved at: {model_path}")

if __name__ == "__main__":
    # === 全局配置 ===
    device = 'cuda'
    train_mode = True        # ✅ True 开始训练，False 开始测试
    experiment_id = 3        # ✅ 设置为 1、2、3、4 选择实验组
    episodes = 5
    max_steps = 50
    test_episodes = 10
    use_offline_sim = 1 #1-采样离线仿真，0-采用板载仿真，2-真实系统训练

    # === 实验一：情感机制消融实验（5组实验） ===
    if experiment_id == 1:
        emotion_modes = ['none', 'simple', 'transformer']  # Baseline、Simple、Transformer
        algo_list = ['sac']  # 本实验只跑SAC
        results = []

        # === 定义5组实验配置 ===
        experiment_configs = [
            {'name': 'Baseline', 'mode': 'none', 'lambda_emo': 0.0},
            {'name': 'Simple', 'mode': 'simple', 'lambda_emo': 0.05},
            {'name': 'Transformer', 'mode': 'transformer', 'lambda_emo': 0.0},
            {'name': 'Transformer', 'mode': 'transformer', 'lambda_emo': 0.01},
            {'name': 'Transformer-High', 'mode': 'transformer', 'lambda_emo': 0.1},
            {'name': 'Transformer-Low', 'mode': 'transformer', 'lambda_emo': 0.2},
        ]

        for config in experiment_configs:
            mode = config['mode']
            lambda_emo = config['lambda_emo']
            group_name = config['name']

            print(f"\n=== Running Group: {group_name} | Mode: {mode} | λ_emo: {lambda_emo} ===")

            # === 第一步：训练 ===
            _ = run_experiment_1(
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                env_mode='sim',
                emotion_modes=[mode],
                algo_list=algo_list,
                train=True,
                lambda_emo=lambda_emo,
                log_prefix=f"{group_name}_train"
            )

            # === 第二步：测试 ===
            test_results = run_experiment_1(
                episodes=test_episodes,
                max_steps=max_steps,
                device=device,
                env_mode='sim',
                emotion_modes=[mode],
                algo_list=algo_list,
                train=False,
                lambda_emo=lambda_emo,
                log_prefix=f"{group_name}_test"
            )

            # 结果保存
            for algo, mode_result, avg_reward in test_results:
                results.append((group_name, algo, mode_result, avg_reward))

        # === 最终实验结果输出 ===
        print("\n=== ✅ 实验一（Emotion Mechanism Ablation）5组对照结果 ===")
        for group, algo, mode, reward in results:
            print(f"[Group: {group} | Algo: {algo.upper()} | Mode: {mode}] → AvgTestReward: {reward:.2f}")

    # === 实验二：强化学习算法对比实验 ===
    elif experiment_id == 2:
        print(f"\n=== Running Algorithm Comparison for All Methods ===")
        # algo_list = ['er_ddpg', 'ddpg', 'td3','td3_bc','ppo', 'sac', 'cql', 'ppol', 'rls_pid', 'emotion_td3','emotion_sac', 'fuzzy']  # ✅ 包含模糊控制
        algo_list = ['td3','sac','ppo','ddpg','td3_bc','rls_pid', 'cql','emotion_sac', 'fuzzy']  # ✅ 包含模糊控制
        results = []

        for algo in algo_list:
            print(f"\n>>>  Start {algo.upper()} Training & Testing")
            model_path, test_reward = run_experiment_2(
                algo=algo,
                train=train_mode,
                episodes=episodes,
                max_steps=max_steps,
                device=device
            )
            results.append((algo, test_reward))

        print("\n=== ✅ 实验三结果对比 ===")
        for algo, reward in results:
            print(f"[{algo.upper()}] 平均测试奖励: {reward:.2f}")

    # === 实验四：多工况对比实验 ===
    elif experiment_id == 3:
        print(f"\n=== Running Multi-Condition Comparison Experiment ===")

        # 选择要跑的算法
        algo_list = ['emotion_sac']  # 可以任选

        for algo in algo_list:
            print(f"\n=== 🚀 Running {algo.upper()} for Multi-Condition ===")
            run_experiment_3(
                train=True,  # True=训练+测试，False=只测试
                episodes=episodes,
                max_steps=max_steps,
                device=device,
                algo=algo  # ✅ 传入指定算法
            )

    # === 实验四：分阶段预训练实验 ===
    elif experiment_id == 4:
        print("\n=== Running Multi-Stage Pretraining Evaluation ===")
        run_experiment_4(device=device, train=train_mode, use_off_sim=use_offline_sim)

    else:
        print("❌ Unsupported experiment ID. Please set experiment_id = 1, 2, 3, or 4.")