import numpy as np
from pathlib import Path
import pandas as pd

from gym_derk import ActionKeys, ObservationKeys, TeamStatsKeys
from gym_derk.envs import DerkEnv

from prescriptor import God
from predictor import Predictor, RFPredictor
import setup

results_dir = Path(setup.results_dir)
results_dir.mkdir(exist_ok=True)

god = God(**setup.evolve_params)

env = DerkEnv(n_arenas=24, **setup.tournament_params)
log = []
for trial in range(setup.trials):
    total_reward = np.zeros(god.n_agents)
    for i in range(setup.n_combs):
        unshuffle_idxs = god.shuffle_agents()
        observation_n = env.reset()
        while True:
            action_n = god.step(observation_n)
            observation_n, reward_n, done_n, info = env.step(action_n)
            if all(done_n):
                break
        print(f"combination {i}: red: {env.total_reward[0]:.2f}, orange: {env.total_reward[1]:.2f}, yellow: {env.total_reward[2]:.2f}, green: {env.total_reward[3]:.2f}, blue: {env.total_reward[4]:.2f}, purple: {env.total_reward[5]:.2f}")

        reward = np.array(env.total_reward)
        total_reward += reward[unshuffle_idxs]
        god.unshuffle_agents(unshuffle_idxs)

    #god.tournament_selection(total_reward, env)
    total_reward /= setup.n_combs
    print(f"Trial {trial} complete. Average reward: {total_reward.sum() / len(total_reward)}")
    god.evolve_omniscient(total_reward)
    log.append(np.array(total_reward))
    god.save_agents(results_dir, trial)
    
env.close()

df = pd.DataFrame(log)
df.to_csv(results_dir / "log.csv", index=False)
