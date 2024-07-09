from PatrolOptim import (intercept_prob_direct, optimize_gameDEFEND, 
                         optimize_gameBOTH, min_visit)
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch.distributions import Cauchy
import pickle

number_comparisons = 10000
np.random.seed(17072024)
seeds = np.random.randint(0, 1000000000, number_comparisons)
device = 'mps'
attack_duration = 10
timesteps = 50

# Import graphs
with open('Graphs.pkl', 'rb') as file:
    variables_loaded = pickle.load(file)

# Initialize dictionaries to store results
results = {location: {} for location in variables_loaded}
optimized_vals = {location: {} for location in variables_loaded}

# Generate homogeneous input and results
for location in variables_loaded:
    G = variables_loaded[location]
    size = G.number_of_nodes()
    start = torch.tensor([1.0 for _ in range(size)],
                            dtype=torch.float32, device=device)
    step_hom = torch.tensor(nx.adjacency_matrix(G).todense(), 
                            dtype=torch.float32, device=device)
    for i in range(len(step_hom)):
        step_hom[i] = step_hom[i]/step_hom[i].sum()
    attack = torch.tensor([[1.0 for _ in range(size)] 
                        for _ in range(timesteps-attack_duration+1)], 
                        dtype=torch.float32, device=device)

    start_hom = F.softmax(start, dim=0)
    attack_flattened = attack.view(-1)
    attack_softmax = F.softmax(attack_flattened, dim=0)
    attack_hom = attack_softmax.view(attack.size())

################################################################################
# Numbering:
# 1 = Optimized Quasi Nash
# 2 = Optimized defender with homogeneous attack
# 3 = Maximize Minimum visit

#######################################1########################################
    opt_iterations = 100000
    start1, step1, attack1 = optimize_gameBOTH(G, timesteps, attack_duration, 
                                            opt_iterations, device)
    optimized_vals[location]['start1'] = start1
    optimized_vals[location]['step1'] = step1
    optimized_vals[location]['attack1'] = attack1

#######################################2########################################
    opt_iterations = 5000
    start2, step2 = optimize_gameDEFEND(G, timesteps, attack_duration, 
                                        attack_hom, intercept_prob_direct, 
                                        opt_iterations)
    optimized_vals[location]['start2'] = start2
    optimized_vals[location]['step2'] = step2

#######################################3########################################
    opt_iterations = 25000
    start3, step3 = optimize_gameDEFEND(G, timesteps, attack_duration, 
                                        attack_hom, min_visit, opt_iterations, 
                                        lr=0.0003)
    optimized_vals[location]['start3'] = start3
    optimized_vals[location]['step3'] = step3

##########################Compare different strategies##########################  
    id = torch.eye(size, dtype=torch.float32, device=device)
    Id = id.repeat(attack_duration, 1)
    X2 = torch.eye(size*attack_duration, device=device)
    adj = torch.tensor(nx.adjacency_matrix(G).todense(), device=device)

    # Calculate the minimum visit values
    min_visits = {}
    min_visits[1] = min_visit(timesteps, step1, start1, attack1, id, Id, X2, 
                            attack_duration, size)
    min_visits[2] = min_visit(timesteps, step2, start2, attack1, id, Id, X2, 
                            attack_duration, size)
    min_visits[3] = min_visit(timesteps, step3, start3, attack1, id, Id, X2, 
                            attack_duration, size)
    results[location]['min_visits'] = min_visits

# Compare with Nash attacker
    probsNash = {}
    probsNash[1] = intercept_prob_direct(timesteps, step1, start1, attack1, id, 
                                        Id, X2, attack_duration, size).item()
    probsNash[2] = intercept_prob_direct(timesteps, step2, start2, attack1, id, 
                                        Id, X2, attack_duration, size).item()
    probsNash[3] = intercept_prob_direct(timesteps, step3, start3, attack1, id, 
                                        Id, X2, attack_duration, size).item()
    results[location]['probsNash'] = probsNash

# Compare with Homogeneous attacker
    probsHomogeneous = {}
    probsHomogeneous[1] = intercept_prob_direct(timesteps, step1, start1, 
                                                attack_hom, id, Id, X2, 
                                                attack_duration, size).item()
    probsHomogeneous[2] = intercept_prob_direct(timesteps, step2, start2, 
                                                attack_hom, id, Id, X2, 
                                                attack_duration, size).item()
    probsHomogeneous[3] = intercept_prob_direct(timesteps, step3, start3, 
                                                attack_hom, id, Id, X2, 
                                                attack_duration, size).item()
    results[location]['probsHomogeneous'] = probsHomogeneous

# Compare with random attacker (normal distribution)
    probsRandomNormal = {1: [], 2: [], 3: []}
    for i in range(number_comparisons):
        seed = seeds[i]
        torch.manual_seed(seed)
        attack_random = torch.randn(timesteps-attack_duration+1, size, 
                                    device=device)
        attack_random = (F.softmax(attack_random.view(-1), dim=0)
                         .view(attack_random.size()))
        val1 = intercept_prob_direct(timesteps, step1, start1, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomNormal[1].append(val1)

        val2 = intercept_prob_direct(timesteps, step2, start2, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomNormal[2].append(val2)

        val3 = intercept_prob_direct(timesteps, step3, start3, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomNormal[3].append(val3)
        if i%100 == 0:
            print(f"Iteration: {i+1}")
    results[location]['probsRandomNormal'] = probsRandomNormal

# Compare with random attacker (Cauchy distribution)
    probsRandomCauchy = {1: [], 2: [], 3: []}
    for i in range(number_comparisons):
        seed = seeds[i]
        torch.manual_seed(seed)
        loc = 0  # location parameter (median)
        scale = 1  # scale parameter
        cauchy_dist = Cauchy(loc, scale)
        attack_random = cauchy_dist.sample((timesteps-attack_duration+1, size)
                                        ).to(device)
        attack_random = (F.softmax(attack_random.view(-1), dim=0)
                         .view(attack_random.size()))

        val1 = intercept_prob_direct(timesteps, step1, start1, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomCauchy[1].append(val1)

        val2 = intercept_prob_direct(timesteps, step2, start2, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomCauchy[2].append(val2)

        val3 = intercept_prob_direct(timesteps, step3, start3, attack_random, id, 
                                     Id, X2, attack_duration, size).item()
        probsRandomCauchy[3].append(val3)

        if i%100 == 0:
            print(f"Iteration: {i}")
    results[location]['probsRandomCauchy'] = probsRandomCauchy


##########################Save variables for later use##########################
with open('variables10.pkl', 'wb') as file:
    pickle.dump(optimized_vals, file)

with open('results10.pkl', 'wb') as file:
    pickle.dump(results, file)





