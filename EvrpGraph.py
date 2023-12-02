import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

# Node coordinates
nodes = {
    0: (145, 215),
    1: (151, 264),
    2: (159, 261),
    3: (130, 254),
    4: (128, 252),
    5: (163, 247),
    6: (146, 246),
    7: (161, 242),
    8: (142, 239),
    9: (163, 236),
    10: (148, 232),
    11: (128, 231),
    12: (156, 217),
    13: (129, 214),
    14: (146, 208),
    15: (164, 208),
    16: (141, 206),
    17: (147, 193),
    18: (164, 193),
    19: (129, 189),
    20: (155, 185),
    21: (139, 182),
    22: (130, 225),
    23: (136, 189),
    24: (136, 248),
    25: (152, 208),
    26: (153, 242),
    27: (154, 189),
    28: (154, 254),
}

demands = {
    0: 0,
    1: 1100,
    2: 700,
    3: 800,
    4: 1400,
    5: 2100,
    6: 400,
    7: 800,
    8: 100,
    90: 500,
    10: 600,
    11: 1200,
    12: 1300,
    13: 1300,
    14: 300,
    15: 900,
    16: 2100,
    17: 1000,
    18: 900,
    29: 2500,
    20: 1800,
    21: 700,
}

stations = [22, 23, 24, 25, 26, 27, 28]

max_capacity = 6000
max_charge = 90

num_tours = 4
depo_node = 1

# tours = {
#     0: [depo_node],
#     1: [depo_node],
#     2: [depo_node],
#     3: [depo_node],
# }

tour = {}

for i in range(0, num_tours):
    tour[i] = [depo_node]


def clearTour():
    for i in range(0, num_tours):
        tour[i] = [depo_node]


def getTour(i):
    return tour[i]


def addNodeToTour(i, node):
    tour[i].append(node)

def addNodeToTourLocal(localTour, i, node):
    localTour[i].append(node)


def getAllVisitedNodes():
    all_nodes = []
    for i in range(num_tours):
        all_nodes.extend(tour[i])  # Exclude depot node
    return all_nodes


def getDistance(source_node, destination_node, nodes):
    """
    Get the distance between two nodes based on their coordinates.
    """
    source_coord = np.array(nodes[source_node])
    destination_coord = np.array(nodes[destination_node])

    distance = np.linalg.norm(source_coord - destination_coord)

    return distance


def getTourCost(i):
    cost = 0
    for j in range(0, len(tour[i]) - 1):
        cost += getDistance(tour[i][j], tour[i][j + 1], nodes)
    return cost


def getTourCostLocal(localTour, index):
    cost = 0
    for j in range(0, len(localTour[index]) - 1):
        cost += getDistance(localTour[index][j], localTour[index][j + 1], nodes)
    return cost


def getLastNodeOfATour(i):
    return tour[i][-1]


def findNearestNodes(original_nodes, nodes, source_node, k=1):
    # Add the source node to the nodes dictionary
    nodes_with_source = nodes.copy()
    if source_node not in nodes_with_source.keys():
        nodes_with_source[source_node] = (original_nodes.get(source_node)[0], original_nodes.get(source_node)[1])

    # Get coordinates of the input node
    node_coord = np.array(nodes_with_source.get(source_node))

    # Convert all node coordinates to NumPy array for vectorized operations
    all_coords = np.array(list(nodes_with_source.values()))

    # Calculate Euclidean distances to all other nodes using vectorized operations
    distances = np.linalg.norm(all_coords - node_coord, axis=1)

    # Get indices of k nearest nodes
    nearest_indices = np.argsort(distances)[:k]

    # Get the k nearest nodes and their distances
    nearest_nodes = [list(nodes_with_source.keys())[idx] for idx in nearest_indices]
    nearest_distances = distances[nearest_indices]
    if source_node in nearest_nodes:
        nearest_nodes.remove(source_node)

    return list(zip(nearest_nodes, nearest_distances))


def getTheLowestCostNode(nearest_nodes_and_distances):
    # Find the tuple with the lowest distance (cost)
    lowest_cost_node = min(nearest_nodes_and_distances, key=lambda x: x[1])

    return lowest_cost_node


def getMeanCostNode(nearest_nodes_and_distances):
    """
    Get the node with the mean cost from a list of nodes and their distances.
    """
    # Calculate the mean cost
    mean_cost = sum(distance for _, distance in nearest_nodes_and_distances) / len(nearest_nodes_and_distances)

    # Find the node closest to the mean cost
    mean_cost_node = min(nearest_nodes_and_distances, key=lambda x: abs(x[1] - mean_cost))

    return mean_cost_node


def getHighestCostNode(nearest_nodes_and_distances):
    """
    Get the node with the highest cost from a list of nodes and their distances.
    """
    # Find the tuple with the highest distance (cost)
    highest_cost_node = max(nearest_nodes_and_distances, key=lambda x: x[1])

    return highest_cost_node


def getRandomNode(nearest_nodes_and_distances):
    """
    Get a random node from a list of nodes and their distances.
    """
    # Choose a random index
    random_index = random.randint(0, len(nearest_nodes_and_distances) - 1)

    # Get the node at the random index
    random_node = nearest_nodes_and_distances[random_index]

    return random_node


def getNearestDemandedCustomerNode(nearest_nodes_and_distances, demands):
    """
    Get the nearest demanded customer node from a list of nodes, their distances, and demands.
    """
    # Filter nodes that are demanded customers
    demanded_customer_nodes = [node for node in nearest_nodes_and_distances if
                               node[0] in demands and demands[node[0]] > 0]

    if not demanded_customer_nodes:
        return None  # No demanded customers found in the nearest nodes

    # Find the demanded customer node with the lowest distance (cost)
    nearest_demanded_customer_node = min(demanded_customer_nodes, key=lambda x: x[1])

    return nearest_demanded_customer_node


def findNearestChargingStationNode(nearest_nodes_and_distances, stations):
    """
    Get the nearest charging station node from a list of nodes, their distances, and charging stations.
    """
    # Filter nodes that are charging stations
    charging_station_nodes = [node for node in nearest_nodes_and_distances if node[0] in stations]

    if not charging_station_nodes:
        return None  # No charging stations found in the nearest nodes

    # Find the charging station node with the lowest distance (cost)
    nearest_charging_station_node = min(charging_station_nodes, key=lambda x: x[1])

    return nearest_charging_station_node


def findNearestChargingStationFromAllNode(nodes, stations, source_node):
    """
    Get the nearest charging station node from a list of nodes, charging stations, and a source node.
    """
    # Calculate distances from the source node to charging stations
    distances_to_stations = {station: np.linalg.norm(np.array(nodes[source_node]) - np.array(nodes[station])) for
                             station in stations}

    # Find the charging station with the minimum distance
    nearest_charging_station_node = min(distances_to_stations, key=distances_to_stations.get)

    return nearest_charging_station_node


def getAllUnvisitedNodesWithCoordinates(nodes, visited_nodes):
    """
    Get all unvisited nodes with coordinates from a list of nodes and a list of visited nodes.
    Returns a dictionary where keys are node numbers and values are node coordinates.
    """
    # Filter nodes that have not been visited
    unvisited_nodes = {node: nodes[node] for node in nodes if node not in visited_nodes}

    return unvisited_nodes


def addToLowestCostTour():
    dummy_tour = copy.deepcopy(tour)
    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours
    nearest_nodes = []
    for j in range(0, num_tours):
        # [1,2,4,5]
        l_visited_nodes = getAllVisitedNodes()
        unvisitedNodes = getAllUnvisitedNodesWithCoordinates(nodes, l_visited_nodes)

        nearest_nodes.append(findNearestNodes(nodes, unvisitedNodes, last_nodes[j], k=5))

    # find the lowest cost node
    lowest_cost_node = []
    for j in range(0, num_tours):
        lowest_cost_node.append(getTheLowestCostNode(nearest_nodes[j]))

    # add the lowest_cost_nodes in the current tour and then find the lowest cost tour

    for j in range(0, num_tours):
        dummy_tour[j].append(lowest_cost_node[j][0])

    # find the lowest cost tour
    tour_costs = []
    for j in range(0, num_tours):
        tour_costs.append(getTourCostLocal(dummy_tour, j))

    # find the lowest cost tour
    lowest_cost_tour = tour_costs.index(min(tour_costs))

    # add the lowest cost node to the lowest cost tour
    addNodeToTour(lowest_cost_tour, lowest_cost_node[lowest_cost_tour][0])


def addToHighestCostTour():
    dummy_tour = copy.deepcopy(tour)

    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours

    nearest_nodes = []
    for j in range(0, num_tours):
        # [1,2,4,5]
        unvisited_nodes = getAllVisitedNodes()
        unvisitedNodes = getAllUnvisitedNodesWithCoordinates(nodes, unvisited_nodes)
        nearest_nodes.append(findNearestNodes(nodes, unvisitedNodes, last_nodes[j], k=5))

    highest_cost_node = []
    for j in range(0, num_tours):
        highest_cost_node.append(getHighestCostNode(nearest_nodes[j]))


    for j in range(0, num_tours):
        dummy_tour[j].append(highest_cost_node[j][0])

    # find the lowest cost tour
    tour_costs = []
    for j in range(0, num_tours):
        tour_costs.append(getTourCostLocal(dummy_tour, j))

    # find the lowest cost tour
    lowest_cost_tour = tour_costs.index(min(tour_costs))

    # add the lowest cost node to the lowest cost tour
    addNodeToTour(lowest_cost_tour, highest_cost_node[lowest_cost_tour][0])

    # lowest_cost_tour = highest_cost_node.index(min(highest_cost_node, key=lambda x: x[1]))
    #
    # addNodeToTour(lowest_cost_tour, highest_cost_node[lowest_cost_tour][0])


def addToMeanCostTour():
    dummy_tour = copy.deepcopy(tour)
    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours

    nearest_nodes = []
    for j in range(0, num_tours):
        # [1,2,4,5]
        unvisited_nodes = getAllVisitedNodes()
        unvisitedNodes = getAllUnvisitedNodesWithCoordinates(nodes, unvisited_nodes)
        nearest_nodes.append(findNearestNodes(nodes, unvisitedNodes, last_nodes[j], k=5))

    mean_cost_node = []
    for j in range(0, num_tours):
        mean_cost_node.append(getMeanCostNode(nearest_nodes[j]))

    for j in range(0, num_tours):
        dummy_tour[j].append(mean_cost_node[j][0])

    # find the lowest cost tour
    tour_costs = []
    for j in range(0, num_tours):
        tour_costs.append(getTourCostLocal(dummy_tour, j))

    # find the lowest cost tour
    lowest_cost_tour = tour_costs.index(min(tour_costs))

    # add the lowest cost node to the lowest cost tour
    addNodeToTour(lowest_cost_tour, mean_cost_node[lowest_cost_tour][0])

    # lowest_cost_tour = mean_cost_node.index(min(mean_cost_node, key=lambda x: x[1]))
    #
    # addNodeToTour(lowest_cost_tour, mean_cost_node[lowest_cost_tour][0])


def addToRandomTour():
    dummy_tour = copy.deepcopy(tour)
    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours

    nearest_nodes = []
    for j in range(0, num_tours):
        # [1,2,4,5]
        unvisited_nodes = getAllVisitedNodes()
        unvisitedNodes = getAllUnvisitedNodesWithCoordinates(nodes, unvisited_nodes)
        nearest_nodes.append(findNearestNodes(nodes, unvisitedNodes, last_nodes[j], k=5))

    random_node = []
    for j in range(0, num_tours):
        random_node.append(getRandomNode(nearest_nodes[j]))

    for j in range(0, num_tours):
        dummy_tour[j].append(random_node[j][0])

    # find the lowest cost tour
    tour_costs = []
    for j in range(0, num_tours):
        tour_costs.append(getTourCostLocal(dummy_tour, j))

    # find the lowest cost tour
    lowest_cost_tour = tour_costs.index(min(tour_costs))

    # add the lowest cost node to the lowest cost tour
    addNodeToTour(lowest_cost_tour, random_node[lowest_cost_tour][0])

    # lowest_cost_tour = random_node.index(min(random_node, key=lambda x: x[1]))
    #
    # addNodeToTour(lowest_cost_tour, random_node[lowest_cost_tour][0])


def addToNearestChargingStationTour():
    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours

    nearest_charging_station_node = []
    for j in range(0, num_tours):
        nearest_charging_station_node.append(findNearestChargingStationFromAllNode(nodes, stations, last_nodes[j]))

    #  Use lamda to transform the nearest_charging_station_node like nodes
    nearest_charging_station_node = list(
        map(lambda x: (x, getDistance(last_nodes[0], x, nodes)), nearest_charging_station_node))

    lowest_cost_tour = nearest_charging_station_node.index(min(nearest_charging_station_node, key=lambda x: x[1]))

    addNodeToTour(lowest_cost_tour, nearest_charging_station_node[lowest_cost_tour][0])


def addToNearestDemandedCustomerTour():
    # last node of all tours
    last_nodes = []
    for j in range(0, num_tours):
        last_nodes.append(getLastNodeOfATour(j))
    # find nearest node for each last node of all tours

    nearest_nodes = []
    for j in range(0, num_tours):
        # [1,2,4,5]
        unvisited_nodes = getAllVisitedNodes()
        unvisitedNodes = getAllUnvisitedNodesWithCoordinates(nodes, unvisited_nodes)
        nearest_nodes.append(findNearestNodes(nodes, unvisitedNodes, last_nodes[j], k=5))

    nearest_demanded_customer_node = []
    for j in range(0, num_tours):
        nearest_demanded_customer_node.append(getNearestDemandedCustomerNode(nearest_nodes[j], demands))

    lowest_cost_tour = nearest_demanded_customer_node.index(min(nearest_demanded_customer_node, key=lambda x: x[1]))

    addNodeToTour(lowest_cost_tour, nearest_demanded_customer_node[lowest_cost_tour][0])


def addToTourBasedOnHuristics(huristic):
    if huristic == 0:
        addToLowestCostTour()
    if huristic == 1:
        addToLowestCostTour()
    elif huristic == 2:
        addToMeanCostTour()
    elif huristic == 3:
        addToHighestCostTour()
    elif huristic == 4:
        addToRandomTour()
    elif huristic == 5:
        # addToNearestChargingStationTour()
        addToLowestCostTour()
    elif huristic == 6:
        # addToNearestDemandedCustomerTour()
        addToLowestCostTour()
    elif huristic == 7:
        # addToNearestDemandedCustomerTour()
        addToLowestCostTour()


def addSourceNodeToAllTours():
    for i in range(0, num_tours):
        addNodeToTour(i, depo_node)


def getTheMaxCostTour():
    max_cost = 0
    for i in range(0, num_tours):
        if getTourCost(i) > max_cost:
            max_cost = getTourCost(i)
    return max_cost
