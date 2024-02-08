N_ACTION_OUTPUTS = 13
trials = 20
n_combs = 4
sort_eval = False
eval_trials = list(range(trials))
genfight_gens = [0, 2, 5, 10, 14, 19]
genfight_repeats = 10
reward_fn = {'damageEnemyUnit': 0.1,'friendlyFire': 0.1, 'fallDamageTaken': -100, 'damageTaken': -0.05, 'timeScaling': 0.5}
weapon = 'Talons'
tournament_params = {
    'turbo_mode' : True,
    'home_team' : [
        {'primaryColor': '#FF0000', 'slots': [weapon, None, None], 'rewardFunction': reward_fn},
        {'primaryColor': '#FF7F00', 'slots': [weapon, None, None], 'rewardFunction': reward_fn},
        {'primaryColor': '#FFFF00', 'slots': [weapon, None, None], 'rewardFunction': reward_fn}
    ],
    'away_team' : [
        {'primaryColor': '#00FF00', 'slots': [weapon, None, None], 'rewardFunction': reward_fn},
        {'primaryColor': '#0000FF', 'slots': [weapon, None, None], 'rewardFunction': reward_fn},
        {'primaryColor': '#9400D3', 'slots': [weapon, None, None], 'rewardFunction': reward_fn}
    ]
}
evolve_params = {
    'n_agents' : 144,
    'context_size' : 31,
    'action_size' : N_ACTION_OUTPUTS, 
    'num_elites' : 6,
    'topk' : 24
}

results_dir = "results"