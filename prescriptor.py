import random

import torch
import numpy as np

from gym_derk import TeamStatsKeys
from gym_derk.envs import DerkEnv


class Prescriptor(torch.nn.Module):

    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size)
        )
        self.model.eval()

    def copy(self):
        new_agent = Prescriptor(self.in_size, self.out_size, self.hidden_size)
        new_agent.model.load_state_dict(self.model.state_dict())
        return new_agent

    def forward(self, x):
        x = torch.tensor(x[:31], dtype=torch.float32)
        raw = self.model(x)
        movex = torch.tanh(raw[0])
        rotate = torch.tanh(raw[1])
        chasefocus = torch.sigmoid(raw[2])
        castingslot = torch.argmax(raw[3:7]) # TODO: Later we can softmax and sample from this?
        changefocus = torch.argmax(raw[7:13]).item() # Modify so they ignore towers
        if changefocus >= 3:
            changefocus += 2
        elif changefocus >= 1:
            changefocus += 1
        return [movex.item(), rotate.item(), chasefocus.item(), castingslot.item(), changefocus]
    
    def orthogonal_init(self, layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight) 
            layer.bias.data.fill_(0.01)

    def initialize_model(self):
        self.model.apply(self.orthogonal_init)

    def mutate(self, p_mutation):
        for param in self.model.parameters():
            noise = torch.normal(0, 0.1, size=param.data.shape) * (torch.rand(size=param.data.shape) < p_mutation)
            param.data += noise


class God():
    def __init__(self, n_agents, context_size, action_size, p_mutation=0.2, topk=32, num_elites=16):
        self.n_agents = n_agents
        self.agents = [Prescriptor(context_size, action_size) for _ in range(n_agents)]
        for agent in self.agents:
            agent.initialize_model()
        self.p_mutation = p_mutation
        self.topk = topk
        self.num_elites = num_elites

    def save_agents(self, path, gen):
        (path / str(gen)).mkdir(exist_ok=True, parents=True)
        for i, agent in enumerate(self.agents):
            torch.save(agent.state_dict(), path / str(gen) / f"{i}.pt")

    def load_agents(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(torch.load(path / f"{i}.pt"))

    def step(self, observations):
        """
        Creates actions from the first n agents where n is the number of observations.
        """
        assert len(observations) <= len(self.agents), f"Too many observations: {len(observations)} > {len(self.agents)}"
        actions = np.array([agent(observations[i]) for i, agent in enumerate(self.agents[:len(observations)])])
        return actions
    
    def sort_agents(self, rewards):
        agents, _ = zip(*sorted(zip(self.agents, rewards), key=lambda x: x[1], reverse=True))
        agents = list(agents)
        self.agents = agents

    def shuffle_agents(self):
        arr = np.array(self.agents)
        random_idxs = np.random.permutation(len(arr))
        self.agents = arr[random_idxs].tolist()
        unshuffle_idxs = np.argsort(random_idxs)
        return unshuffle_idxs
    
    def unshuffle_agents(self, unshuffle_idxs):
        arr = np.array(self.agents)
        self.agents = arr[unshuffle_idxs].tolist()

    def tournament_selection(self, rewards, env):
        self.sort_agents(rewards)
        best_agents = self.agents[:self.topk]
        # We make this too long to make sure we fill the arena
        parents = random.choices(best_agents, k=len(self.agents))

        observation_n = env.reset()
        while True:
            action_n = [parent(observation) for parent, observation in zip(parents, observation_n)]
            observation_n, reward_n, done_n, info = env.step(action_n)
            if all(done_n):
                break
        
        # Team based win mask
        # win_mask = np.argmax(env.team_stats[:, [TeamStatsKeys.OpponentReward.value, TeamStatsKeys.Reward.value]], axis=1)
        # win_mask = np.repeat(win_mask, 3)
        
        # Top 3 scores in match win mask
        total_reward = np.array([env.total_reward])
        total_reward = total_reward.reshape(-1, 6)
        sorted_idx = np.argsort(total_reward, axis=1)
        win_mask = np.zeros_like(total_reward)
        win_mask[np.arange(len(win_mask))[:,None], sorted_idx[:, -3]] = 1
        win_mask = win_mask.flatten()

        winners = [parent for i, parent in enumerate(parents) if win_mask[i] == 1]
        # If we don't have enough winners (in the case of ties), fill in the rest with random good agents.
        if len(winners) < len(parents) // 2:
            winners.extend(random.choices(best_agents, k=len(parents) - len(winners)))
        random.shuffle(winners)

        children = []
        for i in range(0, len(winners), 2):
            children.extend(self.better_crossover(winners[i], winners[i+1]))
        
        self.agents = self.agents[:self.num_elites] + children
        self.agents = self.agents[:self.n_agents]
        random.shuffle(self.agents)

    def crossover(self, n_children, parent1, parent2):
        assert parent1.in_size == parent2.in_size
        assert parent1.out_size == parent2.out_size
        assert parent1.hidden_size == parent2.hidden_size

        # Cross over the weights by taking the average of the weights
        child = Prescriptor(parent1.in_size, parent1.out_size, parent1.hidden_size)
        for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            child_param.data = (parent1_param.data + parent2_param.data) / 2
        
        # Clone child n times
        children = [child]
        for _ in range(n_children - 1):
            children.append(child.copy())

        # Mutate
        for child in children:
            child.mutate(self.p_mutation)
        return children
    
    def better_crossover(self, parent1, parent2):
        child1 = Prescriptor(parent1.in_size, parent1.out_size, parent1.hidden_size)
        child2 = Prescriptor(parent1.in_size, parent1.out_size, parent1.hidden_size)
        child3 = Prescriptor(parent1.in_size, parent1.out_size, parent1.hidden_size)
        for child1_param, child2_param, child3_param, parent1_param, parent2_param in zip(child1.parameters(), child2.parameters(), child3.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand(size=parent1_param.data.shape) < 0.5
            child1_param.data = torch.where(mask, parent1_param.data, parent2_param.data)
            child2_param.data = torch.where(mask, parent2_param.data, parent1_param.data)
            child3_param.data = (parent1_param.data + parent2_param.data) / 2
    
        child4 = child3.copy()
        child1.mutate(0.2)
        child2.mutate(0.2)
        child3.mutate(0.2)
        child4.mutate(0.2)
        return [child1, child2, child3, child4]

    def evolve_omniscient(self, rewards):
        """
        Select the top p% of agents, shuffle, then create children by crossover. Then shuffle again.
        """
        # TODO: The math behind this is going to have to get sorted out

        # Get the top agents
        self.sort_agents(rewards)
        best_agents = self.agents[:self.topk]
        parents = random.choices(best_agents, k=len(self.agents))
        
        # Randomly breed children from top agents
        random.shuffle(parents)
        children = []
        for i in range(0, len(parents), 2):
            children.extend(self.better_crossover(parents[i], parents[i+1]))
            #children.extend(self.crossover(2, parents[i], parents[i+1]))

        self.agents = self.agents[:self.num_elites] + children
        self.agents = self.agents[:self.n_agents]
        random.shuffle(self.agents)
        assert(len(self.agents) == self.n_agents), f"Agents: {len(self.agents)}, should be {self.n_agents}"

    # def evolve_with_surrogate(self, generations, simulation_data, predictor):
    #     for gen in range(generations):
    #         gen_rewards = [] # steps x agents
    #         all_gen_observations = simulation_data["observations"]
    #         for observations in all_gen_observations:
    #             actions = self.step(observations)
    #             rewards = predictor.predict(observations, actions)
    #             gen_rewards.append(rewards)
    #         gen_rewards = np.array(gen_rewards)
    #         avg_reward_gen = np.mean(gen_rewards, axis=0)
    #         if (gen+1) % 10 == 0:
    #             print(f"Average reward generation {gen}: {np.mean(avg_reward_gen)}")
    #         self.evolve_omniscient(avg_reward_gen)
