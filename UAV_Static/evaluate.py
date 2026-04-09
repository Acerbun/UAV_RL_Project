import numpy as np
import argparse
from agent import Regular_DDQN_Agent, Modified_DDQN_Agent
from settings import *

def compute_Qs_over_random_states(agent, memory_valid):
    states = []
    for i in range(EVALUATION_STATES):
        state, _ = memory_valid.get_current_and_next_state(i)
        states.append(state)
    states = np.array(states) 
    Qs = agent.main_net_predict(states).squeeze().detach().cpu().numpy()
    return np.mean(np.max(Qs, axis=1))

def compute_scores(agent, env, n_trails, random=False):
    scores_all_trials = []
    for i in range(n_trails):
        score = 0
        state = env.reset()
        while True:
            if not random:
                # 评估模式下，不进行随机探索 (epsilon=0)
                action = agent.act_epsilon_greedy(state, 0.0)
            else:
                action = agent.act_epsilon_greedy(state, 1.0)
            
            # 接收 3 个返回值，与原作者逻辑完美契合
            next_state, reward, episode_done = env.step(action)
            score += reward
            
            if episode_done:
                break
            else:
                state = next_state
                
        scores_all_trials.append(score)
    return np.mean(scores_all_trials)