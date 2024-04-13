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

################ Q LEARNING ##################

################ GAMMA ##################
# ql_bj_gamma = run_ql_search(process=blackjack,
#                         gamma=gamma_ql_bj,
#                         n_episodes=iters_ql_bj)

# new_fields = {'V': np.append(ql_bj_gamma[0.9]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_gamma[0.9]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_gamma[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_gamma[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma9.csv')

# new_fields = {'V': np.append(ql_bj_gamma[0.99]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_gamma[0.99]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_gamma[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_gamma[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma99.csv')

# new_fields = {'V': np.append(ql_bj_gamma[0.999]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_gamma[0.999]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_gamma[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_gamma[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_gamma999.csv')

################ EDR ##################
# ql_bj_edr = run_ql_search(process=blackjack,
#                         epsilon_decay_ratio=epsilon_decay_ql_bj_edr,
#                         n_episodes=iters_ql_bj)

# new_fields = {'V': np.append(ql_bj_edr[0.9]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_edr[0.9]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_edr[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_edr[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr9.csv')

# new_fields = {'V': np.append(ql_bj_edr[0.9]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_edr[0.9]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_edr[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_edr[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr99.csv')

# new_fields = {'V': np.append(ql_bj_edr[0.99]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_edr[0.99]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_edr[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_edr[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_edr999.csv')

################ ALPHA ##################
# ql_bj_alpha = run_ql_search(process=blackjack,
#                         init_alpha=init_alpha_ql_bj,
#                         n_episodes=iters_ql_bj)

# new_fields = {'V': np.append(ql_bj_alpha[0.3]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_alpha[0.3]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_alpha[0.3]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_alpha[0.3]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha3.csv')

# new_fields = {'V': np.append(ql_bj_alpha[0.5]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_alpha[0.5]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_alpha[0.5]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_alpha[0.5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha5.csv')

# new_fields = {'V': np.append(ql_bj_alpha[0.7]['V'],
#                              [0]*99710),
#               'pi': list(ql_bj_alpha[0.7]['pi'].values())+[0]*99710,
#               'episode_rewards': ql_bj_alpha[0.7]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_bj_alpha[0.7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_bj_alpha7.csv')

################ POLICY ITERATION ##################

################ GAMMA ##################
# bj_pi_gamma = run_pi_search(process=blackjack,
#                   gamma=gamma_pi_bj,
#                   n_iters=iters_pi_bj)

# new_fields = {'V': np.append(bj_pi_gamma[0.9]['V'],
#                              [0]*99710),
#               'pi': list(bj_pi_gamma[0.9]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_pi_gamma[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_gamma[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam9.csv')

# new_fields = {'V': np.append(bj_pi_gamma[0.99]['V'],
#                              [0]*99710),
#               'pi': list(bj_pi_gamma[0.99]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_pi_gamma[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_gamma[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam99.csv')

# new_fields = {'V': np.append(bj_pi_gamma[0.999]['V'],
#                              [0]*99710),
#               'pi': list(bj_pi_gamma[0.999]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_pi_gamma[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_gamma[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_gam999.csv')

################ THETA ##################
# bj_pi_theta = run_pi_search(process=frozenlakeL,
#                   theta=theta_pi_bj,
#                   n_iters=iters_pi_bj)

# new_fields = {'V': np.append(bj_pi_theta[1e-5]['V'],
#                              [0]*99744),
#               'pi': list(bj_pi_theta[1e-5]['pi'].values())+[0]*99744,
#               'episode_rewards': bj_pi_theta[1e-5]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_theta[1e-5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta5.csv')

# new_fields = {'V': np.append(bj_pi_theta[1e-7]['V'],
#                              [0]*99744),
#               'pi': list(bj_pi_theta[1e-7]['pi'].values())+[0]*99744,
#               'episode_rewards': bj_pi_theta[1e-7]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_theta[1e-7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta7.csv')

# new_fields = {'V': np.append(bj_pi_theta[1e-9]['V'],
#                              [0]*99744),
#               'pi': list(bj_pi_theta[1e-9]['pi'].values())+[0]*99744,
#               'episode_rewards': bj_pi_theta[1e-9]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_pi_theta[1e-9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/bj_pi_theta9.csv')

################ VALUE ITERATION ##################

################ GAMMA ##################
# bj_vi_gamma = run_vi_search(process=blackjack,
#                   gamma=gamma_vi_bj,
#                   n_iters=iters_vi_bj)

# new_fields = {'V': np.append(bj_vi_gamma[0.9]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_gamma[0.9]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_gamma[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_gamma[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam9.csv')

# new_fields = {'V': np.append(bj_vi_gamma[0.99]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_gamma[0.99]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_gamma[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_gamma[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam99.csv')

# new_fields = {'V': np.append(bj_vi_gamma[0.999]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_gamma[0.999]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_gamma[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_gamma[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_gam999.csv')

################ THETA ##################
# bj_vi_theta = run_vi_search(process=blackjack,
#                   theta=theta_vi_bj,
#                   n_iters=iters_vi_bj)

# new_fields = {'V': np.append(bj_vi_theta[1e-5]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_theta[1e-5]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_theta[1e-5]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_theta[1e-5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta5.csv')

# new_fields = {'V': np.append(bj_vi_theta[1e-7]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_theta[1e-7]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_theta[1e-7]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_theta[1e-7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta7.csv')

# new_fields = {'V': np.append(bj_vi_theta[1e-9]['V'],
#                              [0]*99710),
#               'pi': list(bj_vi_theta[1e-9]['pi'].values())+[0]*99710,
#               'episode_rewards': bj_vi_theta[1e-9]['episode_rewards'],
#               'average_episode_rewards': np.append(bj_vi_theta[1e-9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/bj_vi_theta9.csv')
