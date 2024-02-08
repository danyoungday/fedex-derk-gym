from pathlib import Path
from gym_derk.envs import DerkEnv
from prescriptor import God
import pandas as pd
import setup


god = God(**setup.evolve_params)

print("Evaluating...")
env = DerkEnv(
    n_arenas=1,
    **setup.tournament_params
)

df = pd.read_csv(f"{setup.results_dir}/log.csv", index_col=False)
for trial in setup.eval_trials:
    result_path = Path(f"{setup.results_dir}/{trial}")
    god.load_agents(result_path)
    if setup.sort_eval:
        rewards = df.iloc[trial].values
        god.sort_agents(rewards)
    observation_n = env.reset()
    while True:
        action_n = god.step(observation_n)
        observation_n, reward_n, done_n, info = env.step(action_n)
        if all(done_n):
            break
    print(f"generation {trial}: red: {env.total_reward[0]:.2f}, orange: {env.total_reward[1]:.2f}, yellow: {env.total_reward[2]:.2f}, green: {env.total_reward[3]:.2f}, blue: {env.total_reward[4]:.2f}, purple: {env.total_reward[5]:.2f}")

env.close()
