import torch
import logging
import numpy as np
import os
import pandas as pd

from WeightEnv import WeightEnv


def run_experiment_3(algo='er_ddpg', train=True, episodes=1000, max_steps=100, device='cuda'):
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


if __name__ == "__main__":
    # === 全局配置 ===
    device = 'cuda'
    train_mode = True        # ✅ True 开始训练，False 开始测试
    experiment_id = 3        # ✅ 设置为 1、2、3、4 选择实验组
    # selected_algo = 'er_ddpg'  # 实验3、4专用
    episodes = 5
    max_steps = 50
    test_episodes = 10
    # # selected_algo = 'ddpg'
    # # selected_algo = 'td3'
    # # selected_algo = 'ppo'
    # selected_algo = 'sac'
    use_offline_sim = 1 #1-采样离线仿真，0-采用板载仿真，2-真实系统训练

    # === 实验一：情感机制对照组 ===
    # === 实验一：情感机制对照组（5组实验） ===
    if experiment_id == 1:
        emotion_modes = ['none', 'simple', 'transformer']  # Baseline、Simple、Transformer
        # algo_list = ['sac']  # 本实验只跑SAC
        # results = []
        #
        # # === 定义5组实验配置 ===
        # experiment_configs = [
        #     {'name': 'Baseline', 'mode': 'none', 'lambda_emo': 0.0},
        #     {'name': 'Simple', 'mode': 'simple', 'lambda_emo': 0.05},
        #     {'name': 'Transformer', 'mode': 'transformer', 'lambda_emo': 0.05},
        #     {'name': 'Transformer-High', 'mode': 'transformer', 'lambda_emo': 0.2},
        #     {'name': 'Transformer-Low', 'mode': 'transformer', 'lambda_emo': 0.01},
        # ]
        #
        # for config in experiment_configs:
        #     mode = config['mode']
        #     lambda_emo = config['lambda_emo']
        #     group_name = config['name']
        #
        #     print(f"\n=== Running Group: {group_name} | Mode: {mode} | λ_emo: {lambda_emo} ===")
        #
        #     # === 第一步：训练 ===
        #     _ = run_experiment_1(
        #         episodes=episodes,
        #         max_steps=max_steps,
        #         device=device,
        #         env_mode='sim',
        #         emotion_modes=[mode],
        #         algo_list=algo_list,
        #         train=True,
        #         lambda_emo=lambda_emo,
        #         log_prefix=f"{group_name}_train"
        #     )
        #
        #     # === 第二步：测试 ===
        #     test_results = run_experiment_1(
        #         episodes=test_episodes,
        #         max_steps=max_steps,
        #         device=device,
        #         env_mode='sim',
        #         emotion_modes=[mode],
        #         algo_list=algo_list,
        #         train=False,
        #         lambda_emo=lambda_emo,
        #         log_prefix=f"{group_name}_test"
        #     )
        #
        #     # 结果保存
        #     for algo, mode_result, avg_reward in test_results:
        #         results.append((group_name, algo, mode_result, avg_reward))
        #
        # # === 最终实验结果输出 ===
        # print("\n=== ✅ 实验一（Emotion Mechanism Ablation）5组对照结果 ===")
        # for group, algo, mode, reward in results:
        #     print(f"[Group: {group} | Algo: {algo.upper()} | Mode: {mode}] → AvgTestReward: {reward:.2f}")
    # if experiment_id == 1:
    #     # emotion_modes = ['none', 'simple', 'transformer']
    #     # emotion_modes = ['simple', 'transformer']
    #     emotion_modes = ['transformer']
    #     # algo_list = ['ddpg', 'td3', 'sac']  # ✅ 3种算法
    #     algo_list = [ 'sac']  # ✅ 3种算法
    #     results = []
    #     lambda_emo = 0.01 #lambda_emo = 0.01, lambda_emo=0.05, lambda_emo=0.1
    #
    #     for algo in algo_list:
    #         for mode in emotion_modes:
    #             print(f"\n=== Running Algo: {algo.upper()} | Emotion Mode: {mode.upper()} ===")
    #
    #             # 第一步：训练 + 保存模型
    #             _ = run_experiment_1(
    #                 episodes=episodes,
    #                 max_steps=max_steps,
    #                 device=device,
    #                 env_mode='sim',
    #                 emotion_modes=[mode],  # ✅ 只训练这个 mode
    #                 algo_list=[algo],  # ✅ 只训练这个 algo
    #                 train=True,  # ✅ 训练
    #                 lambda_emo = lambda_emo
    #             )
    #
    #             # 第二步：测试 + 加载模型
    #             test_results = run_experiment_1(
    #                 episodes=test_episodes,
    #                 max_steps=max_steps,
    #                 device=device,
    #                 env_mode='sim',
    #                 emotion_modes=[mode],  # ✅ 只测试这个 mode
    #                 algo_list=[algo],  # ✅ 只测试这个 algo
    #                 train=False,  # ✅ 测试
    #                 lambda_emo = lambda_emo
    #             )
    #
    #             results.extend(test_results)
    #
    #     # === 最终实验一结果输出 ===
    #     print("\n=== ✅ 实验一（Emotion Mechanism Ablation）结果对比 ===")
    #     for algo, mode, reward in results:
    #         print(f"[{algo.upper()} | {mode}] 平均测试奖励: {reward:.2f}")

    # # === 实验二：分阶段预训练结构对比 ===
    # elif experiment_id == 2:
    #     print("\n=== Running Multi-Stage Pretraining Evaluation ===")
    #     run_experiment_2(device=device, train=train_mode, use_off_sim=use_offline_sim)

    # # === 实验三：强化学习算法对比 ===
    # elif experiment_id == 3:
    #     print(f"\n=== Running Algorithm Comparison for All Methods ===")
    #     algo_list = ['er_ddpg', 'ddpg', 'td3', 'ppo', 'sac','emotion_td3','emotion_sac']
    #     # algo_list = ['emotion_sac','sac']
    #     # algo_list = ['emotion_sac']
    #     results = []

    #     for algo in algo_list:
    #         print(f"\n>>> 🚀 Start {algo.upper()} Training & Testing")
    #         model_path, test_reward = run_experiment_3(
    #             algo=algo,
    #             train=train_mode,
    #             episodes=episodes,
    #             max_steps=max_steps,
    #             device=device
    #         )
    #         results.append((algo, test_reward))

    #     print("\n=== ✅ 实验三结果对比 ===")
    #     for algo, reward in results:
    #         print(f"[{algo.upper()}] 平均测试奖励: {reward:.2f}")

    # === 实验三：强化学习算法对比 ===
    elif experiment_id == 3:
        print(f"\n=== Running Algorithm Comparison for All Methods ===")
        # algo_list = ['er_ddpg', 'ddpg', 'td3','td3_bc','ppo', 'sac', 'cql', 'ppol', 'rls_pid', 'emotion_td3','emotion_sac', 'fuzzy']  # ✅ 包含模糊控制
        algo_list = ['td3','sac','ppo','ddpg','td3_bc','rls_pid', 'cql','emotion_sac', 'fuzzy']  # ✅ 包含模糊控制
        results = []

        for algo in algo_list:
            print(f"\n>>>  Start {algo.upper()} Training & Testing")
            model_path, test_reward = run_experiment_3(
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

    # # === 实验四：多工况对比实验 ===
    # elif experiment_id == 4:
    #     print(f"\n=== Running Multi-Condition Comparison Experiment ===")
    #
    #     # 选择要跑的算法
    #     algo_list = ['emotion_sac']  # 可以任选
    #
    #     for algo in algo_list:
    #         print(f"\n=== 🚀 Running {algo.upper()} for Multi-Condition ===")
    #         run_experiment_4(
    #             train=True,  # True=训练+测试，False=只测试
    #             episodes=episodes,
    #             max_steps=max_steps,
    #             device=device,
    #             algo=algo  # ✅ 传入指定算法
    #         )

    else:
        print("❌ Unsupported experiment ID. Please set experiment_id = 1, 2, 3, or 4.")