import math
import networkx as nx
import numpy as np
import heapq
import time
from scipy import spatial


def ConstructGRNetwork(number_of_nodes, average_neighbors):
    """Constructs a Geometric Random Network with the given number of nodes and average number of neighbors using K-D Tree.

    Args:
        number_of_nodes: Number of nodes in the network.
        average_neighbors: Average number of neighbors of the generated network.

    Returns:
        The neighbor list stores all neighbors of all nodes in the network.
    """
    positions = np.random.rand(number_of_nodes, 2)
    kdtree = spatial.KDTree(positions)
    r = np.sqrt(average_neighbors / number_of_nodes / math.pi)
    pairs = list(kdtree.query_pairs(r))
    G = nx.Graph()
    G.add_nodes_from(range(number_of_nodes))
    G.add_edges_from(pairs)

    return [list(G.neighbors(i)) for i in range(number_of_nodes)]

def ConstructERNetwork(number_of_nodes, average_neighbors):
    """Constructs a Erdos Renyi Network with the given number of nodes and average number of neighbors.

    Args:
        number_of_nodes: Number of nodes in the network.
        average_neighbors: Average number of neighbors of the generated network.

    Returns:
        The neighbor list stores all neighbors of all nodes in the network.
    """
    G = nx.fast_gnp_random_graph(number_of_nodes, average_neighbors / number_of_nodes)
    return [list(G.neighbors(i)) for i in range(number_of_nodes)]

def PoissonSample(rate):
    """Generates a interarrival time between two consecutive events in a Poisson process with the input rate.

    Args:
        rate: The rate for the Poisson process.

    Returns:
        A interarraival time.
    """
    return -np.log(np.random.rand()) / rate

def EstimateSynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability, end_criteria=0.9):
    """Estimates the spreading time for the synchronous gossip.

    Args:
        source_node: The source of the spreading process.
        number_of_nodes: Number of nodes in the network.
        neighbor_list: The list that stores the neighbors of all nodes.
        failure_probability: The probability that an infected node fails to push the information to its chosen.
        end_criteria: The criteria that marks the end of the spreading process, it should be the format of a fraction,
            which represents the fraction of nodes that are infected in the end.

    Returns:
        The estimated the spreading time for the synchronous gossip.
    """
    t = 0
    infected_node_set = set()
    infected_node_set.add(source_node)
    while True:
        newly_infected_node_set = set()
        for active_node in infected_node_set:
            if np.random.rand() >= failure_probability:
                chosen_node = np.random.choice(neighbor_list[active_node])
                newly_infected_node_set.add(chosen_node)
        infected_node_set = infected_node_set.union(newly_infected_node_set)
        t += 1
        if len(infected_node_set) >= end_criteria * len(neighbor_list):
            return t

def FastEstimateSynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability, end_criteria=0.9):
    """Estimates the spreading time for the synchronous gossip fast by filtering out useless nodes.

    This is only useful for estimate the spreading time. Use the above exact synchronous gossip process
    to perform other simulations like the source location attacks.
    Args:
        source_node: The source of the spreading process.
        number_of_nodes: Number of nodes in the network.
        neighbor_list: The list that stores the neighbors of all nodes.
        failure_probability: The probability that an infected node fails to push the information to its chosen.
        end_criteria: The criteria that marks the end of the spreading process, it should be the format of a fraction,
            which represents the fraction of nodes that are infected in the end.

    Returns:
        The estimated the spreading time for the synchronous gossip.
    """
    t = 0
    infected_node_set = set()
    useful_node_set = set()
    infected_node_set.add(source_node)
    useful_node_set.add(source_node)
    while True:
        useless_node_set = set()
        newly_infected_node_set = set()
        for active_node in useful_node_set:
            if set(neighbor_list[active_node]).issubset(infected_node_set):
                useless_node_set.add(active_node)
                continue
            if np.random.rand() >= failure_probability:
                chosen_node = np.random.choice(neighbor_list[active_node])
                infected_node_set.add(chosen_node)
                newly_infected_node_set.add(chosen_node)
        useful_node_set = useful_node_set.union(newly_infected_node_set)
        useful_node_set = useful_node_set.difference(useless_node_set)
        t += 1
        if len(infected_node_set) >= end_criteria * len(neighbor_list):
            return t

def PushAsynchronousEvent(event_heap, time, node):
    """Pushes the event into the event heap in the asynchronous gossip.

    Args:
        event_heap: The heap stores all future events.
        time: The time for the new event to be pushed.
        node: The node for the new event to be pushed.
    """
    heapq.heappush(event_heap, (time, node))

def EstimateAsynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability, end_criteria=0.9):
    """Estimates the spreading time for the asynchronous gossip.

    Args:
        source_node: The source of the spreading process.
        number_of_nodes: Number of nodes in the network.
        neighbor_list: The list that stores the neighbors of all nodes.
        failure_probability: The probability that an infected node fails to push the information to its chosen.
        end_criteria: The criteria that marks the end of the spreading process, it should be the format of a fraction,
            which represents the fraction of nodes that are infected in the end.

    Returns:
        The estimated the spreading time for the asynchronous gossip.
    """
    t = 0
    infected_node_set = set()
    infected_node_set.add(source_node)
    event_heap = []
    heapq.heapify(event_heap)
    PushAsynchronousEvent(event_heap, t + PoissonSample(1.0), source_node)
    while True:
        current_time, active_node = heapq.heappop(event_heap)
        uninf_node_chosen = False
        if np.random.rand() >= failure_probability:
            chosen_node = np.random.choice(neighbor_list[active_node])
            uninf_node_chosen = (chosen_node not in infected_node_set)
            infected_node_set.add(chosen_node)
        if len(infected_node_set) >= end_criteria * len(neighbor_list):
            return current_time
        PushAsynchronousEvent(event_heap, current_time + PoissonSample(1.0), active_node)
        if uninf_node_chosen:
            PushAsynchronousEvent(event_heap, current_time + PoissonSample(1.0), chosen_node)

def FastEstimateAsynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability, end_criteria=0.9):
    """Estimates the spreading time for the asynchronous gossip fast by filtering out unuseful events.

    This is only useful for estimate the spreading time. Use the above exact asynchronous gossip process
    to perform other simulations like the source location attacks.
    Args:
        source_node: The source of the spreading process.
        number_of_nodes: Number of nodes in the network.
        neighbor_list: The list that stores the neighbors of all nodes.
        failure_probability: The probability that an infected node fails to push the information to its chosen.
        end_criteria: The criteria that marks the end of the spreading process, it should be the format of a fraction,
            which represents the fraction of nodes that are infected in the end.

    Returns:
        The estimated the spreading time for the asynchronous gossip.
    """
    t = 0
    infected_node_set = set()
    infected_node_set.add(source_node)
    event_heap = []
    heapq.heapify(event_heap)
    PushAsynchronousEvent(event_heap, t + PoissonSample(1.0), source_node)
    while True:
        current_time, active_node = heapq.heappop(event_heap)
        if set(neighbor_list[active_node]).issubset(infected_node_set):
            continue
        uninf_node_chosen = False
        if np.random.rand() >= failure_probability:
            chosen_node = np.random.choice(neighbor_list[active_node])
            uninf_node_chosen = (chosen_node not in infected_node_set)
            infected_node_set.add(chosen_node)
        if len(infected_node_set) >= end_criteria * number_of_nodes:
            return current_time
        PushAsynchronousEvent(event_heap, current_time + PoissonSample(1.0), active_node)
        if uninf_node_chosen:
            PushAsynchronousEvent(event_heap, current_time + PoissonSample(1.0), chosen_node)

# Example of simulating the Gossip spreading process.
number_of_nodes = 100000
failure_probability = 0.0
neighbor_list = ConstructGRNetwork(number_of_nodes, 10)
source_node = int(number_of_nodes * np.random.rand())
estimate_time = EstimateSynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability)
print(estimate_time)
