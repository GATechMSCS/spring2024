from create_env_gs import (make_env, run_ql_search, run_pi_search, run_vi_search)
import pandas as pd
import numpy as np
from gymnasium.spaces import Tuple, Discrete

seed = 123
np.random.seed(seed)

# make environments
mdp_bj = 'Blackjack-v1'
render_mode_bj = None #'rgb_array'
size_bj = Tuple([Discrete(32), Discrete(11), Discrete(2)])
ep_steps_bj = 100
blackjack = make_env(mdp=mdp_bj,
                     size=size_bj,
                     slip=None,
                     render=render_mode_bj,
                     seed=seed,
                     prob_frozen=None,
                     ep_steps=ep_steps_bj)

# BJ
# QL
iters_ql_bj = 100000
gamma_ql_bj = [0.90, 0.99, 0.999]
epsilon_decay_ql_bj_edr = [0.90, 0.99, 0.999]
init_alpha_ql_bj = [0.30, 0.50, 0.70]

# PI
iters_pi_bj = 100000
gamma_pi_bj = [0.90, 0.99, 0.999]
theta_pi_bj = [1e-5, 1e-7, 1e-9]

# VI
iters_vi_bj = 100000
gamma_vi_bj = [0.90, 0.99, 0.999]
theta_vi_bj = [1e-5, 1e-7, 1e-9]

pathql = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/qlearning/"
pathpi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/pi/"
pathvi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/vi/"

############### Q LEARNING ##################

############### GAMMA ##################
ql_bj_gamma = run_ql_search(process=blackjack,
                        gamma=gamma_ql_bj,
                        n_episodes=iters_ql_bj)

new_fields = {'V': np.append(ql_bj_gamma[0.9]['V'],
                             [0]*99710),
              'pi': list(ql_bj_gamma[0.9]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_gamma[0.9]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_gamma[0.9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma9.csv')

new_fields = {'V': np.append(ql_bj_gamma[0.99]['V'],
                             [0]*99710),
              'pi': list(ql_bj_gamma[0.99]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_gamma[0.99]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_gamma[0.99]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma99.csv')

new_fields = {'V': np.append(ql_bj_gamma[0.999]['V'],
                             [0]*99710),
              'pi': list(ql_bj_gamma[0.999]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_gamma[0.999]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_gamma[0.999]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma999.csv')

############### EDR ##################
ql_bj_edr = run_ql_search(process=blackjack,
                        epsilon_decay_ratio=epsilon_decay_ql_bj_edr,
                        n_episodes=iters_ql_bj)

new_fields = {'V': np.append(ql_bj_edr[0.9]['V'],
                             [0]*99710),
              'pi': list(ql_bj_edr[0.9]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_edr[0.9]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_edr[0.9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr9.csv')

new_fields = {'V': np.append(ql_bj_edr[0.9]['V'],
                             [0]*99710),
              'pi': list(ql_bj_edr[0.9]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_edr[0.9]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_edr[0.9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr99.csv')

new_fields = {'V': np.append(ql_bj_edr[0.99]['V'],
                             [0]*99710),
              'pi': list(ql_bj_edr[0.99]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_edr[0.99]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_edr[0.99]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr999.csv')

############### ALPHA ##################
ql_bj_alpha = run_ql_search(process=blackjack,
                        init_alpha=init_alpha_ql_bj,
                        n_episodes=iters_ql_bj)

new_fields = {'V': np.append(ql_bj_alpha[0.3]['V'],
                             [0]*99710),
              'pi': list(ql_bj_alpha[0.3]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_alpha[0.3]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_alpha[0.3]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha3.csv')

new_fields = {'V': np.append(ql_bj_alpha[0.5]['V'],
                             [0]*99710),
              'pi': list(ql_bj_alpha[0.5]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_alpha[0.5]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_alpha[0.5]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha5.csv')

new_fields = {'V': np.append(ql_bj_alpha[0.7]['V'],
                             [0]*99710),
              'pi': list(ql_bj_alpha[0.7]['pi'].values())+[0]*99710,
              'episode_rewards': ql_bj_alpha[0.7]['episode_rewards'],
              'average_episode_rewards': np.append(ql_bj_alpha[0.7]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha7.csv')

############### POLICY ITERATION ##################

############### GAMMA ##################
bj_pi_gamma = run_pi_search(process=blackjack,
                  gamma=gamma_pi_bj,
                  n_iters=iters_pi_bj)

new_fields = {'V': np.append(bj_pi_gamma[0.9]['V'],
                             [0]*99710),
              'pi': list(bj_pi_gamma[0.9]['pi'].values())+[0]*99710,
              'episode_rewards': bj_pi_gamma[0.9]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_gamma[0.9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam9.csv')

new_fields = {'V': np.append(bj_pi_gamma[0.99]['V'],
                             [0]*99710),
              'pi': list(bj_pi_gamma[0.99]['pi'].values())+[0]*99710,
              'episode_rewards': bj_pi_gamma[0.99]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_gamma[0.99]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam99.csv')

new_fields = {'V': np.append(bj_pi_gamma[0.999]['V'],
                             [0]*99710),
              'pi': list(bj_pi_gamma[0.999]['pi'].values())+[0]*99710,
              'episode_rewards': bj_pi_gamma[0.999]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_gamma[0.999]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam999.csv')

############### THETA ##################
bj_pi_theta = run_pi_search(process=frozenlakeL,
                  theta=theta_pi_bj,
                  n_iters=iters_pi_bj)

new_fields = {'V': np.append(bj_pi_theta[1e-5]['V'],
                             [0]*99744),
              'pi': list(bj_pi_theta[1e-5]['pi'].values())+[0]*99744,
              'episode_rewards': bj_pi_theta[1e-5]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_theta[1e-5]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta5.csv')

new_fields = {'V': np.append(bj_pi_theta[1e-7]['V'],
                             [0]*99744),
              'pi': list(bj_pi_theta[1e-7]['pi'].values())+[0]*99744,
              'episode_rewards': bj_pi_theta[1e-7]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_theta[1e-7]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta7.csv')

new_fields = {'V': np.append(bj_pi_theta[1e-9]['V'],
                             [0]*99744),
              'pi': list(bj_pi_theta[1e-9]['pi'].values())+[0]*99744,
              'episode_rewards': bj_pi_theta[1e-9]['episode_rewards'],
              'average_episode_rewards': np.append(bj_pi_theta[1e-9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta9.csv')

############### VALUE ITERATION ##################

############### GAMMA ##################
bj_vi_gamma = run_vi_search(process=blackjack,
                  gamma=gamma_vi_bj,
                  n_iters=iters_vi_bj)

new_fields = {'V': np.append(bj_vi_gamma[0.9]['V'],
                             [0]*99710),
              'pi': list(bj_vi_gamma[0.9]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_gamma[0.9]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_gamma[0.9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam9.csv')

new_fields = {'V': np.append(bj_vi_gamma[0.99]['V'],
                             [0]*99710),
              'pi': list(bj_vi_gamma[0.99]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_gamma[0.99]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_gamma[0.99]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam99.csv')

new_fields = {'V': np.append(bj_vi_gamma[0.999]['V'],
                             [0]*99710),
              'pi': list(bj_vi_gamma[0.999]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_gamma[0.999]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_gamma[0.999]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam999.csv')

############### THETA ##################
bj_vi_theta = run_vi_search(process=blackjack,
                  theta=theta_vi_bj,
                  n_iters=iters_vi_bj)

new_fields = {'V': np.append(bj_vi_theta[1e-5]['V'],
                             [0]*99710),
              'pi': list(bj_vi_theta[1e-5]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_theta[1e-5]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_theta[1e-5]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta5.csv')

new_fields = {'V': np.append(bj_vi_theta[1e-7]['V'],
                             [0]*99710),
              'pi': list(bj_vi_theta[1e-7]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_theta[1e-7]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_theta[1e-7]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta7.csv')

new_fields = {'V': np.append(bj_vi_theta[1e-9]['V'],
                             [0]*99710),
              'pi': list(bj_vi_theta[1e-9]['pi'].values())+[0]*99710,
              'episode_rewards': bj_vi_theta[1e-9]['episode_rewards'],
              'average_episode_rewards': np.append(bj_vi_theta[1e-9]['average_episode_rewards'],
                                                  [0]*99999)}
pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta9.csv')

############################## FINAL ####################################
from create_env_gs import make_env
from gymnasium.spaces import Tuple, Discrete
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner

seed = 123

pathql = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/qlearning/"
pathpi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/pi/"
pathvi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/bj/vi/"

# make environments
mdp_bj = 'Blackjack-v1'
render_mode_bj = None #'rgb_array'
size_bj = Tuple([Discrete(32), Discrete(11), Discrete(2)])
ep_steps_bj = 100
blackjack = make_env(mdp=mdp_bj,
                     size=size_bj,
                     slip=None,
                     render=render_mode_bj,
                     seed=seed,
                     prob_frozen=None,
                     ep_steps=ep_steps_bj)

iters = 100000

############################### Q LEARNING ###################
np.random.seed(seed)
blackjack.reset(seed=seed)

#print(f"q_learning: gamma={ql_bj_gamma_best}; edr={ql_bj_edr_best}; ialpha={ql_bj_alpha_best}; episodes={iters}")
Q, V, pi, Q_track, pi_track = RL(env=blackjack).q_learning(n_episodes=iters,
                                                        gamma=0.9,
                                                        epsilon_decay_ratio=0.8,
                                                        init_alpha=0.7)
episode_rewards = TestEnv.test_env(env=blackjack, n_iters=iters, pi=pi)
avg_ep_rewards = np.mean(episode_rewards)

results_ql = {'Q': Q, 
                    'V': V,
                    'pi': pi, 
                    'Q_track': Q_track,
                    'pi_track': pi_track, 
                    'episode_rewards': episode_rewards,
                    'average_episode_rewards': avg_ep_rewards}

print("Avg. episode reward: ", avg_ep_rewards) 
print("###################\n")

Q_track = results_ql['Q_track']

# Initialize V_track to zeros
V_track = np.zeros((Q_track.shape[0], Q_track.shape[1]))

# Find indices where all Q-values are initialized to 10
initial_indices = np.all(Q_track == 10, axis=2)

# Replace initialized Q-values with zeros in those indices
Q_track[initial_indices] = 0

# Calculate V_track by taking the maximum Q-value for each state
V_track = np.max(Q_track, axis=2)

# Calculate the mean delta across all states for each episode
mean_delta_V_ql = np.mean(np.abs(V_track[1:] - V_track[:-1]), axis=1)

new_fields = {'V': np.append(results_ql['V'],
                             [0]*99710),
              'pi': list(results_ql['pi'].values())+[0]*99710,
              'episode_rewards': results_ql['episode_rewards'],
              'cumulative_sum': np.cumsum(results_ql['episode_rewards']),
              'average_episode_rewards': np.append(results_ql['average_episode_rewards'],
                                                  [0]*99999)}

pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/results_ql_final.csv')
pd.DataFrame(data=np.trim_zeros(np.mean(V_track, axis=1), 'b')).to_csv(path_or_buf=pathql+'MeanV.csv')
pd.DataFrame(data=V_track).to_csv(path_or_buf=pathql+'/results_v_track.csv')
pd.DataFrame(data=mean_delta_V_ql).to_csv(path_or_buf=pathql+'/results_mean_delta_v.csv')

######################## POLICY ITERATION ######################
np.random.seed(seed)
blackjack.reset(seed=seed) 

#print(f"PI: gamma={bj_pi_gam_best}; theta={bj_pi_theta_best}; iters={iters}")
V,V_track, pi = Planner(P=blackjack.P).policy_iteration(n_iters=iters,
                                                        gamma=0.999,
                                                        theta=1e-5)
episode_rewards = TestEnv.test_env(env=blackjack, n_iters=iters, pi=pi)
avg_ep_rewards = np.mean(episode_rewards)

results_pi = {'V': V, 
                'vi_track': V_track, 
                'pi': pi,
                'episode_rewards': episode_rewards,
                'average_episode_rewards': avg_ep_rewards}
    
print("Avg. episode reward: ", avg_ep_rewards)
print("###################\n")

# Calculate the mean delta across all states for each episode
mean_delta_V_pi = np.mean(np.abs(V_track[1:] - V_track[:-1]), axis=1)

new_fields = {'V': np.append(results_pi['V'],
                             [0]*99710),
              'pi': list(results_pi['pi'].values())+[0]*99710,
              'episode_rewards': results_pi['episode_rewards'],
              'cumulative_sum': np.cumsum(results_pi['episode_rewards']),
              'average_episode_rewards': np.append(results_pi['average_episode_rewards'],
                                                  [0]*99999)}

pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/results_pi_final.csv')
pd.DataFrame(data=np.trim_zeros(np.mean(V_track, axis=1), 'b')).to_csv(path_or_buf=pathpi+'MeanV.csv')
pd.DataFrame(data=results_pi['vi_track']).to_csv(path_or_buf=pathpi+'/results_v_track.csv')
pd.DataFrame(data=mean_delta_V_pi).to_csv(path_or_buf=pathpi+'/results_mean_delta_v.csv')

################# VALUE ITERATION ######################
np.random.seed(seed)
blackjack.reset(seed=seed) 

#print(f"VI: gamma={bj_vi_gam_best}; theta={bj_vi_theta_best}; iters={iters}")            
V, V_track, pi = Planner(P=blackjack.P).value_iteration(n_iters=iters,
                                                        gamma=0.999,
                                                        theta=1e-5)
episode_rewards = TestEnv.test_env(env=blackjack, n_iters=iters, pi=pi)
avg_ep_rewards = np.mean(episode_rewards)

results_vi = {'V': V, 
                'vi_track': V_track, 
                'pi': pi,
                'episode_rewards': episode_rewards,
                'average_episode_rewards': avg_ep_rewards}

print("Avg. episode reward: ", avg_ep_rewards)
print("###################\n")

# Calculate the mean delta across all states for each sode
mean_delta_V_vi = np.mean(np.abs(V_track[1:] - V_track[:-1]), axis=1)

new_fields = {'V': np.append(results_vi['V'],
                             [0]*99710),
              'pi': list(results_vi['pi'].values())+[0]*99710,
              'episode_rewards': results_vi['episode_rewards'],
              'cumulative_sum': np.cumsum(results_ql['episode_rewards']),
              'average_episode_rewards': np.append(results_vi['average_episode_rewards'],
                                                  [0]*99999)}

pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/results_vi_final.csv')
pd.DataFrame(data=np.trim_zeros(np.mean(V_track, axis=1), 'b')).to_csv(path_or_buf=pathvi+'MeanV.csv')
pd.DataFrame(data=results_vi['vi_track']).to_csv(path_or_buf=pathvi+'/results_v_track.csv')
pd.DataFrame(data=mean_delta_V_vi).to_csv(path_or_buf=pathvi+'/results_mean_delta_v.csv')