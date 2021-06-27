from gossip_spreading_util import ConstructERNetwork, ConstructGRNetwork, FastEstimateAsynchronousGossipTime, FastEstimateSynchronousGossipTime, EstimateSynchronousPrivateGossipTime, EstimateAsynchronousPrivateGossipTime
import numpy as np

number_of_nodes = 100000
failure_probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for failure_probability in failure_probabilities:
    print(failure_probability)
    network_instances = 5
    monte_runs = 100
    total_estimate_time = 0
    for _ in range(network_instances):
        neighbor_list = ConstructERNetwork(number_of_nodes, 10)
        for run in range(monte_runs):
            source_node = int(number_of_nodes * np.random.rand())
            total_estimate_time += FastEstimateSynchronousGossipTime(source_node, number_of_nodes, neighbor_list, failure_probability)
    print(total_estimate_time / monte_runs / network_instances)