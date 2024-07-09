import torch
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx

def intercept_prob_direct(timesteps: int, Q: torch.Tensor, start: torch.Tensor, 
                          attack: torch.Tensor, id: torch.Tensor, 
                          Id: torch.Tensor, X2: torch.Tensor, 
                          attack_duration: int, size: int) -> torch.Tensor:
    """
    Calculates the interception probability for a patrolling game with:
    - Markov transition matrix:     Q
    - Starting position:            start
    - Attacker's strategy:          attack
    - Number of time steps:         timesteps
    - Attack duration:              attack_duration
    - Number of nodes:              size

    All operations work with the PyTorch autograd framework, so the output is
    differentiable with respect to the input tensors.
    """
    # Precompute powers of Q
    max_power = max(timesteps - attack_duration, attack_duration)
    powers = [id, Q]
    for _ in range(1, max_power):
        powers.append(powers[-1] @ Q)
    B = torch.hstack(powers[:attack_duration])

    # Adjust X2 based on powers
    for i in range(1, attack_duration):
        vec = (torch.diagonal(powers[i])).repeat(attack_duration-i)
        X2 = torch.diagonal_scatter(X2, vec, size*i)

    # Solve for A and compute P
    A = torch.linalg.solve_triangular(X2, B, upper=True, left=False, unitriangular=True)
    P = A @ Id

    # Faster calculation of intercept_prob using batch operations
    stacked_powers = torch.stack([powers[i] for i in range(timesteps - attack_duration + 1)])
    stacked_attack = torch.stack([attack[i] for i in range(timesteps - attack_duration + 1)])

    # Perform batch matrix multiplication and sum for intercept probability
    intermediate_result = torch.matmul(torch.matmul(start.unsqueeze(0), stacked_powers), P)
    final_result = torch.matmul(intermediate_result, stacked_attack.unsqueeze(-1))
    intercept_prob = final_result.sum()

    return intercept_prob

def min_visit(timesteps: int, Q: torch.Tensor, start: torch.Tensor, 
                          attack: torch.Tensor, id: torch.Tensor, 
                          Id: torch.Tensor, X2: torch.Tensor, 
                          attack_duration: int, size: int) -> torch.Tensor:
    """
    Calculates the lowest visit probability over all vertices for all 
    attack_duration sized intervals.
    - Markov transition matrix:     Q
    - Starting position:            start
    - Attacker's strategy:          attack
    - Number of time steps:         timesteps
    - Attack duration:              attack_duration
    - Number of nodes:              size

    All operations work with the PyTorch autograd framework, so the output is
    differentiable with respect to the input tensors.

    The attack duration is the total number of time steps it takes. So, if
    attack_duration=1, the patroller takes 0 steps during the attack (can only 
    intercept at the starting position).
    """
    max_power = max(timesteps - attack_duration, attack_duration)
    powers = [id, Q]
    for _ in range(1, max_power):
        powers.append(powers[-1] @ Q)
    B = torch.hstack(powers[:attack_duration])

    for i in range(1, attack_duration):
        vec = (torch.diagonal(powers[i])).repeat(attack_duration-i)
        X2 = torch.diagonal_scatter(X2, vec, size*i)

    A = torch.linalg.solve_triangular(X2, B, upper=True, left=False, 
                                      unitriangular=True)
    P = A @ Id

    stacked_powers = torch.stack([powers[i] for i in range(timesteps - attack_duration + 1)])
    batch_result = torch.matmul(torch.matmul(start.unsqueeze(0), 
                                             stacked_powers), P)
    min_visits = torch.min(batch_result, dim=1).values

    return torch.min(min_visits)

def get_hook(A):
    def hook(grad):
        return grad * A  # Only keep gradients where A is non-zero
    return hook

def optimize_gameBOTH(Graph, timesteps, attack_duration, iterations=10000, 
                      device='mps'):
    size = Graph.number_of_nodes()
    adj = torch.tensor(nx.adjacency_matrix(Graph).todense(), 
                       dtype=torch.float32, device=device)
    adj.fill_diagonal_(1.0)
    adj.requires_grad = True
    
    mask = adj == 1

    # Initialize variables
    start = torch.tensor([1.0 for _ in range(size)], requires_grad=True, 
                        dtype=torch.float32, device=device)

    step = torch.tensor([[1.0 for _ in range(size)] for _ in range(size)], 
                        dtype=torch.float32, device=device)
    step[~mask] = -float('inf')
    step.requires_grad = True
    step.register_hook(get_hook(mask))

    attack = torch.tensor([[1.0 for _ in range(size)] 
                        for _ in range(timesteps-attack_duration+1)], 
                        requires_grad=True, dtype=torch.float32, device=device)

    id = torch.eye(size, dtype=torch.float32, device=device)
    Idcon = id.repeat(attack_duration, 1)
    X2 = torch.eye(size*attack_duration, device=device)

    # Define optimizer
    optimizer = optim.NAdam([start, step, attack], lr=0.0003)

    # Perform gradient descent
    for i in range(iterations):  
        start_prob = F.softmax(start, dim=0)
        step_prob = F.softmax(step, dim=1)

        attack_flattened = attack.view(-1)
        attack_softmax = F.softmax(attack_flattened, dim=0)
        attack_prob = attack_softmax.view(attack.size())

        # Compute function f()
        output = intercept_prob_direct(timesteps, step_prob, 
                                        start_prob, attack_prob, id, Idcon, X2,
                                        attack_duration, size)

        # Backpropagation
        output.backward()

        # Manually negate gradients for defender (for ascent)
        with torch.no_grad():
            for param in [start, step]:
                param.grad = -param.grad

        optimizer.step()

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: f={output.item()}")
        
        optimizer.zero_grad()

    return (F.softmax(start, dim=0).detach(), F.softmax(step, dim=1).detach(), 
            F.softmax(attack.view(-1), dim=0).view(attack.size()).detach())

def optimize_gameDEFEND(Graph, timesteps, attack_duration, attack, loss_func, 
                        iterations=10000, device='mps', lr=0.05):
    size = Graph.number_of_nodes()
    adj = torch.tensor(nx.adjacency_matrix(Graph).todense(), 
                       dtype=torch.float32, device=device)
    adj.fill_diagonal_(1.0)
    adj.requires_grad = True

    mask = adj == 1

    # Initialize variables
    start = torch.tensor([1.0 for _ in range(size)], requires_grad=True, 
                        dtype=torch.float32, device=device)

    step = torch.tensor([[1.0 for _ in range(size)] for _ in range(size)], 
                        dtype=torch.float32, device=device)
    step[~mask] = -float('inf')
    step.requires_grad = True
    step.register_hook(get_hook(mask))

    id = torch.eye(size, dtype=torch.float32, device=device)
    Idcon = id.repeat(attack_duration, 1)
    X2 = torch.eye(size*attack_duration, device=device)

    # Define optimizer for defender
    optimizer_defender = optim.NAdam([start, step], lr=lr)

    for i in range(iterations):  # Adjust number of iterations as needed
        start_prob = F.softmax(start, dim=0)
        step_prob = F.softmax(step, dim=1)

        # Use provided 'attack' tensor, assuming it's already in the correct 
        # shape and softmax applied
        attack_prob = attack

        output = -loss_func(timesteps, step_prob, start_prob, attack_prob, id, 
                            Idcon, X2, attack_duration, size)

        # Backpropagation
        output.backward()

        # Step optimizer
        optimizer_defender.step()

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: f={-output.item()}")
        
        optimizer_defender.zero_grad()

    return F.softmax(start, dim=0).detach(), F.softmax(step, dim=1).detach()

def optimize_gameATTACK(Graph, timesteps, attack_duration, start, step, 
                        iterations=10000, device='mps'):
    size = Graph.number_of_nodes()

    # Initialize 'attack' variable
    attack = torch.tensor([[1.0 for _ in range(size)] 
                           for _ in range(timesteps-attack_duration+1)], 
                           requires_grad=True, dtype=torch.float32, 
                           device=device)
    attack_flattened = attack.view(-1)
    attack_softmax = F.softmax(attack_flattened, dim=0)
    attack_prob = attack_softmax.view(attack.size())

    id = torch.eye(size, dtype=torch.float32, device=device)
    Idcon = id.repeat(attack_duration, 1)
    X2 = torch.eye(size*attack_duration, device=device)

    # Define optimizer for attacker
    optimizer_attacker = optim.NAdam([attack], lr=0.005)

    for i in range(iterations):  # Adjust number of iterations as needed
        
        # Compute function f()
        output = intercept_prob_direct(timesteps, step, start, attack_prob, id, 
                                       Idcon, X2, attack_duration, size)

        # Backpropagation
        output.backward()

        # Step optimizer
        optimizer_attacker.step()

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: f={output.item()}")

        optimizer_attacker.zero_grad()

        attack_flattened = attack.view(-1)
        attack_softmax = F.softmax(attack_flattened, dim=0)
        attack_prob = attack_softmax.view(attack.size())

    return attack_prob.detach()