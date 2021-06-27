from gossip_spreading_util import ConstructERNetwork, ConstructGRNetwork, FastEstimateAsynchronousGossipTime, FastEstimateSynchronousGossipTime, EstimateSynchronousPrivateGossipTime, EstimateAsynchronousPrivateGossipTime
import numpy as np

failure_probability_list = np.arange(0, 0.91, 0.1)
number_of_nodes = 100000
average_neighbors = 10
monte_carlo_runs = 100
graph_instances = 5

spreading_times = []
total_time = 0
for failure_probability in failure_probability_list:
    print(failure_probability)
    for _ in range(graph_instances):
        neighbor_list = ConstructERNetwork(number_of_nodes, average_neighbors)
        for _ in range(monte_carlo_runs):
            source_node = np.random.choice(number_of_nodes)
            total_time += FastEstimateAsynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability)
    spreading_times.append(total_time / graph_instances / monte_carlo_runs)
    print(spreading_times)