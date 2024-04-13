from create_env_gs import (make_env, run_ql_search, run_pi_search, run_vi_search)
import pandas as pd
import numpy as np

seed = 123
np.random.seed(seed)

# make environments
mdp_fl_S = 'FrozenLake-v1'
size_fl_S = 4
is_slippery_fl_S = True
render_mode_fl_S = 'ansi'
prob_frozen_fl_S = 0.9
ep_steps_fl_S = 100
frozenlakeS = make_env(mdp=mdp_fl_S, 
                      size=size_fl_S, 
                      slip=is_slippery_fl_S,
                      render=render_mode_fl_S,
                      prob_frozen=prob_frozen_fl_S,
                      seed=seed,
                      ep_steps=ep_steps_fl_S)

# FL 4x4
# QL
iters_ql_fl_S = 100000
gamma_ql_fl_S = [0.90, 0.99, 0.999]
epsilon_decay_ql_fl_S = [0.80, 0.90, 0.99]
init_alpha_ql_fl_S = [0.30, 0.50, 0.70]

# PI
iters_pi_fl_S = 100000
gamma_pi_fl_S = [0.90, 0.99, 0.999]
theta_pi_fl_S = [1e-5, 1e-7, 1e-9]

# VI
iters_vi_fl_S = 100000
gamma_vi_fl_S = [0.90, 0.99, 0.999]
theta_vi_fl_S = [1e-5, 1e-7, 1e-9]

pathql = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/fl_S/qlearning/"
pathpi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/fl_S/pi/"
pathvi = "/home/leonardo_leads/Documents/SchoolDocs/ga_tech_masters/omscs_ml/spring2024/cs7641_machine_learning/lectures/week13/assignment4/source/csv/fl_S/vi/"

################ Q LEARNING ##################

################ GAMMA ##################
# ql_fl_S_gamma = run_ql_search(process=frozenlakeL,
#                         gamma=gamma_ql_fl_S,
#                         n_episodes=iters_ql_fl_S)

# new_fields = {'V': np.append(ql_fl_S_gamma[0.9]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_gamma[0.9]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_gamma[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_gamma[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_gamma9.csv')

# new_fields = {'V': np.append(ql_fl_S_gamma[0.99]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_gamma[0.99]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_gamma[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_gamma[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_gamma99.csv')

# new_fields = {'V': np.append(ql_fl_S_gamma[0.999]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_gamma[0.999]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_gamma[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_gamma[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_gamma999.csv')

################ EDR ##################
# ql_fl_S_edr = run_ql_search(process=frozenlakeL,
#                         epsilon_decay_ratio=epsilon_decay_ql_fl_S,
#                         n_episodes=iters_ql_fl_S)

# new_fields = {'V': np.append(ql_fl_S_edr[0.8]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_edr[0.8]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_edr[0.8]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_edr[0.8]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_edr8.csv')

# new_fields = {'V': np.append(ql_fl_S_edr[0.9]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_edr[0.9]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_edr[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_edr[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_edr99.csv')

# new_fields = {'V': np.append(ql_fl_S_edr[0.99]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_edr[0.99]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_edr[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_edr[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_edr999.csv')

################ ALPHA ##################
# ql_fl_S_alpha = run_ql_search(process=frozenlakeL,
#                         init_alpha=init_alpha_ql_fl_S,
#                         n_episodes=iters_ql_fl_S)

# new_fields = {'V': np.append(ql_fl_S_alpha[0.3]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_alpha[0.3]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_alpha[0.3]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_alpha[0.3]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_alpha3.csv')

# new_fields = {'V': np.append(ql_fl_S_alpha[0.5]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_alpha[0.5]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_alpha[0.5]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_alpha[0.5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_alpha5.csv')

# new_fields = {'V': np.append(ql_fl_S_alpha[0.7]['V'],
#                              [0]*99744),
#               'pi': list(ql_fl_S_alpha[0.7]['pi'].values())+[0]*99744,
#               'episode_rewards': ql_fl_S_alpha[0.7]['episode_rewards'],
#               'average_episode_rewards': np.append(ql_fl_S_alpha[0.7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathql+'/ql_fl_S_alpha7.csv')

################ POLICY ITERATION ##################

################ GAMMA ##################
# fl_S_pi_gam = run_pi_search(process=frozenlakeL,
#                   gamma=gamma_pi_fl_S,
#                   n_iters=iters_pi_fl_S)

# new_fields = {'V': np.append(fl_S_pi_gam[0.9]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_gam[0.9]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_gam[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_gam[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_gam9.csv')

# new_fields = {'V': np.append(fl_S_pi_gam[0.99]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_gam[0.99]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_gam[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_gam[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_gam99.csv')

# new_fields = {'V': np.append(fl_S_pi_gam[0.999]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_gam[0.999]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_gam[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_gam[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_gam999.csv')

################ THETA ##################
# fl_S_pi_theta = run_pi_search(process=frozenlakeL,
#                   theta=theta_pi_fl_S,
#                   n_iters=iters_pi_fl_S)

# new_fields = {'V': np.append(fl_S_pi_theta[1e-5]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_theta[1e-5]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_theta[1e-5]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_theta[1e-5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_theta5.csv')

# new_fields = {'V': np.append(fl_S_pi_theta[1e-7]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_theta[1e-7]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_theta[1e-7]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_theta[1e-7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_theta7.csv')

# new_fields = {'V': np.append(fl_S_pi_theta[1e-9]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_pi_theta[1e-9]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_pi_theta[1e-9]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_pi_theta[1e-9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathpi+'/fl_S_pi_theta9.csv')

################ VALUE ITERATION ##################

################ GAMMA ##################
# fl_S_vi_gam = run_vi_search(process=frozenlakeL,
#                   gamma=gamma_vi_fl_S,
#                   n_iters=iters_vi_fl_S)

# new_fields = {'V': np.append(fl_S_vi_gam[0.9]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_gam[0.9]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_gam[0.9]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_gam[0.9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_gam9.csv')

# new_fields = {'V': np.append(fl_S_vi_gam[0.99]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_gam[0.99]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_gam[0.99]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_gam[0.99]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_gam99.csv')

# new_fields = {'V': np.append(fl_S_vi_gam[0.999]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_gam[0.999]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_gam[0.999]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_gam[0.999]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_gam999.csv')

################ THETA ##################
# fl_S_vi_theta = run_vi_search(process=frozenlakeL,
#                   theta=theta_vi_fl_S,
#                   n_iters=iters_vi_fl_S)

# new_fields = {'V': np.append(fl_S_vi_theta[1e-5]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_theta[1e-5]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_theta[1e-5]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_theta[1e-5]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_theta5.csv')

# new_fields = {'V': np.append(fl_S_vi_theta[1e-7]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_theta[1e-7]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_theta[1e-7]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_theta[1e-7]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_theta7.csv')

# new_fields = {'V': np.append(fl_S_vi_theta[1e-9]['V'],
#                              [0]*99744),
#               'pi': list(fl_S_vi_theta[1e-9]['pi'].values())+[0]*99744,
#               'episode_rewards': fl_S_vi_theta[1e-9]['episode_rewards'],
#               'average_episode_rewards': np.append(fl_S_vi_theta[1e-9]['average_episode_rewards'],
#                                                   [0]*99999)}
# pd.DataFrame(data=new_fields).to_csv(path_or_buf=pathvi+'/fl_S_vi_theta9.csv')
