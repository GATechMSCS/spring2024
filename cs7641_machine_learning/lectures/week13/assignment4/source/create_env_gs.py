import gymnasium as gym
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

seed = 123
np.random.seed(seed)

def make_env(mdp, size, slip, render, seed, prob_frozen, ep_steps):

    match mdp:
        case 'FrozenLake8x8-v1':

            size = generate_random_map(size=size, p=prob_frozen)
    
            env = gym.make(id=mdp,
                   desc=size,
                   render_mode=render,
                   is_slippery=slip,
                   max_episode_steps=ep_steps)

        case 'FrozenLake-v1':

            size = generate_random_map(size=size, p=prob_frozen)

            env = gym.make(id=mdp,
                    desc=size,
                    render_mode=render,
                    is_slippery=slip,
                    max_episode_steps=ep_steps)

        case 'Blackjack-v1':

            env = gym.make(id=mdp,
                       render_mode=render,
                       max_episode_steps=ep_steps)

            env.observation_space = size
            env = BlackjackWrapper(env)
            env.reset(seed=seed)
            
            if render == 'rgb_array': print(env.render())
                
            return env
        
    env.reset(seed=seed)
    print(env.render())

    return env

def run_ql_search(process, n_episodes, gamma=0.99, epsilon_decay_ratio=0.9, init_alpha=0.5):

    results_dict = {}

    if isinstance(gamma, list):
        param = 'gamma'

        for gam in gamma:
        
            np.random.seed(seed)
            process.reset(seed=seed)

            print(f"running q_learning with {param} = {gam}, edr = {epsilon_decay_ratio}, init alpha = {init_alpha}")
            Q, V, pi, Q_track, pi_track = RL(env=process).q_learning(n_episodes=n_episodes,
                                                                    gamma=gam,
                                                                    epsilon_decay_ratio=epsilon_decay_ratio,
                                                                    init_alpha=init_alpha)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_episodes, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[gam] = {'Q': Q, 
                                'V': V,
                                'pi': pi, 
                                'Q_track': Q_track,
                                'pi_track': pi_track, 
                                'episode_rewards': episode_rewards,
                                'average_episode_rewards': avg_ep_rewards}

            print("Avg. episode reward: ", avg_ep_rewards) 
            print("###################\n")
            
    elif isinstance(epsilon_decay_ratio, list):
        param = 'epsilon_decay_ratio'

        for edr in epsilon_decay_ratio:
        
            np.random.seed(seed)
            process.reset(seed=seed)

            print(f"running q_learning with {param} = {edr}; gamma = {gamma}, init alpha = {init_alpha}")
            Q, V, pi, Q_track, pi_track = RL(env=process).q_learning(n_episodes=n_episodes,
                                                                    gamma=gamma,
                                                                    epsilon_decay_ratio=edr,
                                                                    init_alpha=init_alpha)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_episodes, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[edr] = {'Q': Q, 
                                'V': V,
                                'pi': pi, 
                                'Q_track': Q_track,
                                'pi_track': pi_track, 
                                'episode_rewards': episode_rewards,
                                'average_episode_rewards': avg_ep_rewards}

            print("Avg. episode reward: ", avg_ep_rewards) 
            print("###################\n")
        
    elif isinstance(init_alpha, list):
        param = 'init_alpha'

        for alpha in init_alpha:
        
            np.random.seed(seed)
            process.reset(seed=seed)

            print(f"running q_learning with {param} = {alpha}; gamma = {gamma}; edr = {epsilon_decay_ratio}")
            Q, V, pi, Q_track, pi_track = RL(env=process).q_learning(n_episodes=n_episodes,
                                                                    gamma=gamma,
                                                                    epsilon_decay_ratio=epsilon_decay_ratio,
                                                                    init_alpha=alpha)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_episodes, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[alpha] = {'Q': Q, 
                                'V': V,
                                'pi': pi, 
                                'Q_track': Q_track,
                                'pi_track': pi_track, 
                                'episode_rewards': episode_rewards,
                                'average_episode_rewards': avg_ep_rewards}

            print("Avg. episode reward: ", avg_ep_rewards) 
            print("###################\n")

    return results_dict

def run_pi_search(process, n_iters, gamma=1.0, theta=1e-10):

    results_dict = {}

    if isinstance(gamma, list):
        param = 'gamma'

        for gam in gamma:   

            np.random.seed(seed)
            process.reset(seed=seed) 

            print(f"running PI with {param} = {gam}; theta = {theta}")
            V, V_track, pi = Planner(P=process.P).policy_iteration(n_iters=n_iters,
                                                                    gamma=gam,
                                                                    theta=theta)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_iters, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[gam] = {'V': V, 
                                'vi_track': V_track, 
                                'pi': pi,
                                'episode_rewards': episode_rewards,
                                'average_episode_rewards': avg_ep_rewards}
                
            print("Avg. episode reward: ", avg_ep_rewards)
            print("###################\n")

    elif isinstance(theta, list):
        param = 'theta'

        for th in theta:

            np.random.seed(seed)
            process.reset(seed=seed) 

            print(f"running PI with {param} = {th}; gamma = {gamma}")
            V, V_track, pi = Planner(P=process.P).policy_iteration(n_iters=n_iters,
                                                                    gamma=gamma,
                                                                    theta=th)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_iters, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[th] = {'V': V, 
                                'vi_track': V_track, 
                                'pi': pi,
                                'episode_rewards': episode_rewards,
                                'average_episode_rewards': avg_ep_rewards}
                
            print("Avg. episode reward: ", avg_ep_rewards)
            print("###################\n")

    return results_dict
                       
def run_vi_search(process, n_iters, gamma=1.0, theta=1e-10):

    results_dict = {}

    if isinstance(gamma, list):
        param = 'gamma'

        for gam in gamma:   

            np.random.seed(seed)
            process.reset(seed=seed) 

            print(f"running VI with {param} = {gam}; theta = {theta}")            
            V, V_track, pi = Planner(P=process.P).value_iteration(n_iters=n_iters,
                                                                 gamma=gam,
                                                                 theta=theta)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_iters, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[gam] = {'V': V, 
                            'vi_track': V_track, 
                            'pi': pi,
                            'episode_rewards': episode_rewards,
                            'average_episode_rewards': avg_ep_rewards}
        
            print("Avg. episode reward: ", avg_ep_rewards)
            print("###################\n")

    elif isinstance(theta, list):
        param = 'theta'

        for th in theta:   

            np.random.seed(seed)
            process.reset(seed=seed) 

            print(f"running VI with {param} = {th}; gamma = {gamma}")            
            V, V_track, pi = Planner(P=process.P).value_iteration(n_iters=n_iters,
                                                                    gamma=gamma,
                                                                    theta=th)
            episode_rewards = TestEnv.test_env(env=process, n_iters=n_iters, pi=pi)
            avg_ep_rewards = np.mean(episode_rewards)

            results_dict[th] = {'V': V, 
                            'vi_track': V_track, 
                            'pi': pi,
                            'episode_rewards': episode_rewards,
                            'average_episode_rewards': avg_ep_rewards}
        
            print("Avg. episode reward: ", avg_ep_rewards)
            print("###################\n")

    return results_dict

def main():

    pass
    
if __name__ == "__main__":
    main()