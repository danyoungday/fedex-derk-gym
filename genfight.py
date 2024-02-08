from pathlib import Path
from gym_derk.envs import DerkEnv
from prescriptor import God
import pandas as pd
import setup
import numpy as np
import copy

genfight_params = copy.deepcopy(setup.evolve_params)
genfight_params["n_agents"] = 6 * setup.genfight_repeats
god = God(**genfight_params)
temp_god = God(**setup.evolve_params)

print("Evaluating...")
env = DerkEnv(
    n_arenas=setup.genfight_repeats,
    **setup.tournament_params
)

# Load agents
df = pd.read_csv(f"{setup.results_dir}/log.csv", index_col=False)
for i, trial in enumerate(setup.genfight_gens):
    result_path = Path(f"{setup.results_dir}/{trial}")
    temp_god.load_agents(result_path)
    rewards = df.iloc[trial].values
    temp_god.sort_agents(rewards)
    god.agents = np.array(god.agents)
    god.agents[[i + 6 * x for x in range(god.n_agents // 6)]] = temp_god.agents[0]

observation_n = env.reset()
while True:
    action_n = god.step(observation_n)
    observation_n, reward_n, done_n, info = env.step(action_n)
    if all(done_n):
        break

total_reward = np.array(env.total_reward).reshape(-1, 6).mean(axis=0)

print(f"gen {setup.genfight_gens[0]} red: {total_reward[0]:.2f}, gen {setup.genfight_gens[1]} orange: {total_reward[1]:.2f}, gen {setup.genfight_gens[2]} yellow: {total_reward[2]:.2f}, gen {setup.genfight_gens[3]} green: {total_reward[3]:.2f}, gen {setup.genfight_gens[4]} blue: {total_reward[4]:.2f}, gen {setup.genfight_gens[5]} purple: {total_reward[5]:.2f}")
env.close()