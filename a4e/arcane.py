import math
import random
import string
from scipy.spatial import distance

from a4e.grid import Grid
from a4e.network import Graph
from a4e.node import Node
from a4e.node_attribute import NodeAttribute


# ! UNUSED: The method body is added into generate_vertices method
def limited_normal_dist(_attrwise_node_list: list[Node]):
    mu = 0.0
    sigma = 0.5
    affinity_list = []
    num_range = len(_attrwise_node_list)
    # ! TODO: create_comm_vertices_dict metodu ile bağlanmalı ve her attribute başına ayrı çalışmalı
    for i in range(num_range):
        result = random.normalvariate(mu, sigma)
        # liste.append(result)
        if -1.0 <= result <= 1.0:
            # print("In bound")
            r_res = round(result, 2)
            affinity_list.append(r_res)
        else:
            # print("Not in bound")
            # """
            while result < -1.0 or result > 1.0:
                result = random.normalvariate(mu, sigma)
                if -1.0 <= result <= 1.0:
                    r_res = round(result, 2)
                    affinity_list.append(r_res)
                    break

    return affinity_list


# ! UNUSED: Create an interval dict between every pairwise distance of nodes
def create_interval_dict(_distances_dict) -> dict:
    dict_keys = []

    max_value = max([value for dict_values in _distances_dict.values() for key, value in dict_values.items()])

    for index in range(int(max_value)):
        if index == int(max_value) - 1:
            dict_keys.append((index, int(max_value)))
            dict_keys.append((index + 1, int(max_value) + 1))
            break
        dict_keys.append((index, index + 1))

    interval_dict = dict.fromkeys(dict_keys)

    for keys, dict_values in _distances_dict.items():
        for key, distance in dict_values.items():
            distance_end = math.ceil(distance)
            distance_start = int(distance)
            if distance_start == distance_end:
                distance_end += 1
            # key_set = {keys, key}
            try:
                interval_dict[(distance_start, distance_end)].update({(keys, key): distance})
                # TODO: for controlling unique one-way edges
                # if key_set in interval_dict[(distance_start, distance_end)].keys():
                #    break
            except (KeyError, AttributeError):
                interval_dict[(distance_start, distance_end)] = {(keys, key): distance}

    return interval_dict


# ? Used to determine if the two nodes are near enough with themselves using "distance_threshold"
def is_node_in_inner_circle_of(_selected_node: Node, _to_be_checked_node: Node,
                               max_distance: float = None, distance_threshold: float = None) -> bool:
    proximity = 0.0
    # TODO: Need to determine "inner_circle_radius" with more solid calculation
    # ! A constant distance threshold can be given to inner circle radius
    if distance_threshold is not None:
        inner_circle_radius = distance_threshold
    elif max_distance is not None:
        # ! USE BELOW if you want normalized proximity approach
        # ? Using normalized physical proximity approach
        btw_distance = _selected_node.get_distance(_to_be_checked_node)
        proximity = btw_distance / max_distance
        inner_circle_radius = 0
    elif len(_selected_node.attributes) <= 1:
        # ? Create an edge until (sqrt(2)) one grid distance
        inner_circle_radius = 2.0 ** (1 / 2)
    else:
        # ! USE BELOW if you want earlier approach
        # ? Create an edge with the most distance of: attributes/2 * sqrt(2)
        # ? (half of the attribute count times one grid distance)
        inner_circle_radius = (1 / 2 * len(_selected_node.attributes)) * (2.0 ** (1 / 2))

    if max_distance is None and math.dist(_selected_node.get_xy(), _to_be_checked_node.get_xy()) <= inner_circle_radius:
        return True
    elif max_distance is not None:
        if proximity < 0.1:
            return True
        else:
            return False
    else:
        return False


# ! UNUSED: Get the inner circle neighbors of a particular node
def get_inner_circle_nodes_of(_node: Node, _distances_dict: dict):
    if len(_node.attributes) == 0:
        inner_circle_radius = 2.0 ** (1 / 2)
    else:
        inner_circle_radius = len(_node.attributes) * (2.0 ** (1 / 2))

    inner_circle_nodes_of_node = \
        [inner_node for inner_node, dist in _distances_dict[_node].items() if dist <= inner_circle_radius]

    return inner_circle_nodes_of_node


# ? Power law divisors of a node size
def find_divisors(node_size: int):
    list_of = []

    index = node_size
    while index > 1:
        divided = int(index / 2)
        list_of.append(divided)
        step = abs(index - divided)
        index = index - step

    if sum(list_of) != node_size:
        for index, element in enumerate(list_of):
            list_of[index] += 1
            if sum(list_of) == node_size:
                break
            # TODO: check if this holds when there is a big node list (maybe the sum cannot complete to the node size)

    return list_of[::-1]


# ? Select min. pairwise distanced node to add it into sorted node list
def select_node(nodes: list[Node], distances: dict, attr_param=None):
    distances_summ = dict.fromkeys(nodes)
    min_value = float('inf')
    min_nod = None
    for nod, el in distances.items():
        distances_summ[nod] = sum(el.values())
        if distances_summ[nod] <= min_value:
            min_nod = nod
            min_value = distances_summ[nod]

    return min_nod


def select_most_positive_attributed_node(nodes: list[Node]):
    selected_node = nodes[0]
    current_max = 0
    for node in nodes:
        # ? The node with most positive attitude will be the starting node (can change)
        temp_sum = sum(node.get_affinity_levels_of_all_attributes())
        if temp_sum > current_max:
            selected_node = node
            current_max = temp_sum

    return selected_node


# ! DEPRECATED
# ? Select the node with the most attribute count to use it as a starting node in edge generation
# TODO: Adding affinity levels ?
def select_most_attributed_node(nodes: list[Node]):
    selected_node = nodes[0]
    for node in nodes:
        if len(node.attributes) >= len(selected_node.attributes):
            selected_node = node

    return selected_node


# ! UNUSED: Get distance between two nodes in the network
def get_distance(first_node: Node, second_node: Node):
    return math.dist(first_node.get_xy(), second_node.get_xy())


# ? Creating a random graph for testing purposes
def create_random_graph(size: int, community_size: int):
    c = Arcane()
    vertex_list = []
    tuple_list = []
    attributes = c.generate_general_attributes(community_size)
    for i in range(size):
        random_attr_size = random.randint(0, community_size)
        temp_comm_list = []
        for comm in attributes:
            new_attr = NodeAttribute(comm.name, 0.0, i)
            temp_comm_list.append(new_attr)
        random_attributes = random.sample(temp_comm_list, random_attr_size)
        # random_attributes = random.sample(attributes, random_attr_size)

        random_x = random.randint(-500, 500)
        random_y = random.randint(-500, 500)
        random_pos = (random_x, random_y)
        if random_pos in tuple_list:
            while random_pos not in tuple_list:
                random_x = random.randint(-500, 500)
                random_y = random.randint(-500, 500)
                random_pos = (random_x, random_y)
            node = Node(i, "N" + str(i), random_attributes, random_pos)
            vertex_list.append(node)
        else:
            node = Node(i, "N" + str(i), random_attributes, random_pos)
            vertex_list.append(node)

    affinity_level_list = []
    mu = 0.0
    sigma = 0.5
    for node in vertex_list:
        for i in range(len(node.attributes)):
            result = random.normalvariate(mu, sigma)
            if -1.0 <= result <= 1.0:
                r_res = round(result, 2)
                affinity_level_list.append(r_res)
            else:
                while result < -1.0 or result > 1.0:
                    result = random.normalvariate(mu, sigma)
                    if -1.0 <= result <= 1.0:
                        r_res = round(result, 2)
                        affinity_level_list.append(r_res)
                        break

    print(affinity_level_list)
    print(max(affinity_level_list), min(affinity_level_list))

    for node in vertex_list:
        for attr in node.attributes:
            random_affinity = random.sample(affinity_level_list, 1)
            attr.set_affinity_level(random_affinity[0])
            affinity_level_list.remove(random_affinity[0])

    distances_dict = c.calculate_distances(vertex_list)
    similarity_threshold = 1
    edges_dict, revised_node_list = c.generate_edges_with_similar_distance(vertex_list, distances_dict,
                                                                           similarity_threshold)
    new_graph = Graph("random_G", revised_node_list, edges_dict)
    return new_graph


# ? Creating a random graph on a grid
# TODO: Add attribute affinity levels
def create_random_grid_graph(exponent: int, roughness: float, community_size: int):
    c = Arcane()
    g = Grid(exponent, roughness)
    print(list(g.grid_pos_list))
    print(list(g.placeable_grid_positions))
    temp_pos_list = g.placeable_grid_positions.copy()
    vertex_list = []
    attributes = c.generate_meta_communities(community_size)
    for i in range(len(g.placeable_grid_positions)):
        random_attr_size = random.randint(0, community_size)
        random_attributes = random.sample(attributes, random_attr_size)
        random_pos = random.choice(temp_pos_list)
        temp_pos_list.remove(random_pos)
        node = Node(i, "N" + str(i), random_attributes, random_pos)
        vertex_list.append(node)

    distances_dict = c.calculate_distances(vertex_list)
    similarity_threshold = 1
    edges_dict, revised_node_list = c.generate_edges_with_similar_distance(vertex_list, distances_dict,
                                                                           similarity_threshold)
    new_graph = Graph("random_grid_G", revised_node_list, edges_dict)
    return new_graph


# ? Compare the affinity levels of two similar nodes
# TODO: Add a weight list tuple for every pair of mutual node attributes :
#  (e.g. the nodes can be Nx and Ny and the mutual attr.s can be a1 and a3, the tuples then: a1 = (2, 3), a3 = (1, 2).)
def affinity_level_comparator(_selected_node: Node, _node_to_compare: Node,
                              weight_tuple_list: list[tuple[int, int]] = None) -> bool:
    summed_affinity_of_nodes = 0
    mutual_attributes = _selected_node.attributes
    # ! UNCOMMENT BELOW if you want to use mutual attribute approach
    # mutual_attributes = list(set(_selected_node.attributes).intersection(_node_to_compare.attributes))
    if len(mutual_attributes) == 0:
        print("There is no mutual attributes between "
              + _selected_node.name + " and " + _node_to_compare.name + ". No edge needed.")
        return False
    for i in range(len(mutual_attributes)):
        node_attr_index = _selected_node.get_attribute_index(mutual_attributes[i])
        node_to_compare_attr_index = _node_to_compare.get_attribute_index(mutual_attributes[i])
        if node_attr_index == -1 or node_to_compare_attr_index == -1:
            print("Error on attribute indexes!")
            return False
        summed_affinity_of_nodes += \
            _selected_node.attributes[node_attr_index].affinity_level \
            * _node_to_compare.attributes[node_to_compare_attr_index].affinity_level

    if summed_affinity_of_nodes > 0:
        return True
    else:
        return False


class Arcane:

    # ! UNUSED: We only create an instance to use the methods
    # TODO: Maybe all methods can be transformed to static methods
    def __init__(self, node_size: int = None, community_size: int = None):
        pass
        # self.community_list = self.generate_meta_communities(community_size)
        # self.node_list = self.generate_vertices(node_size)
        # self.edges_dict = self.generate_edges()
        # self.graph = Graph("graph", node_list, edges_dict)

    # ? Calculate pairwise distances between all the graph nodes and add it to a dictionary
    def calculate_distances(self, _node_list: list[Node]):
        temp_distances_dict = {node: {} for node in _node_list}
        for a_node in _node_list:
            for another_node in _node_list:
                if a_node.node_id >= another_node.node_id:  # a_node == another_node:
                    continue
                else:
                    distance_a_another = round(math.dist(a_node.get_xy(), another_node.get_xy()), 3)
                    temp_distances_dict[a_node][another_node] = distance_a_another
                    temp_distances_dict[another_node][a_node] = distance_a_another
                    """
                    try:
                        temp_distances_dict[a_node].update({another_node: distance_a_another})
                    except (KeyError, AttributeError):
                        temp_distances_dict[a_node] = {another_node: distance_a_another}
                    try:
                        temp_distances_dict[another_node].update({a_node: distance_a_another})
                    except (KeyError, AttributeError):
                        temp_distances_dict[another_node] = {a_node: distance_a_another}
                    """

        return temp_distances_dict

    def calculate_distances_improved(self, _node_list: list[Node]):
        positions_of_node_list = [node.get_xy() for node in _node_list]
        distances = distance.cdist(positions_of_node_list, positions_of_node_list, 'euclidean')

        d_dict = {node: {} for node in _node_list}
        for node in _node_list:
            for another_node in _node_list:
                if node.node_id >= another_node.node_id:  # a_node == another_node:
                    continue
                else:
                    d_dict[node][another_node] = distances[node.node_id][another_node.node_id]
                    d_dict[another_node][node] = distances[node.node_id][another_node.node_id]

        return d_dict

    # ! DEPRECATED: Old version of the (generate_general_attributes) method -without NodeAttribute class
    def generate_meta_communities(self, community_size: int) -> list[str]:
        temp_communities = []
        index = 0
        comm_name = "a" + str(index)
        for i in range(community_size):
            temp_communities.append(comm_name)
            index += 1
            comm_name = "a" + str(index)

        print("List of meta-communities: ", list(temp_communities))
        return temp_communities

    # ? Generate the general attributes (communities) that will be distributed to the created nodes
    def generate_general_attributes(self, community_size: int) -> list[NodeAttribute]:
        temp_communities = []
        index = 0
        comm_name = "label_" + str(index)
        for i in range(community_size):
            temp_attribute = NodeAttribute(comm_name, 0.0)
            temp_communities.append(temp_attribute)
            index += 1
            comm_name = "label_" + str(index)

        # ! Open this for debug purposes
        # print("List of meta-communities: ", list(temp_communities))
        return temp_communities

    # ? Generate the graph nodes, name them, add random attributes and normally distribute the levels of those attr.s
    def generate_vertices(self, node_size: int, community_list: list = None,
                          positions_grid: list = None) -> list[Node]:
        temp_nodes = []
        name_char = 65
        name_int = 0
        for i in range(node_size):  # ? TODO: was node_size, changed to len(positions_grid)
            # ! DEPRECATED: Belirgin bir community listesi verilmezse random attribute ve community oluşturur.
            if community_list is None:
                random_trait_size = random.randint(0, 10)
                random_traits = []
                for j in range(random_trait_size):
                    random_str = random.choice(string.ascii_letters)
                    random_num = random.randint(0, node_size)
                    random_traits.append(random_str + str(random_num))

                node_name = chr(name_char) + str(name_int)
                temp_nodes.append(Node(i, node_name, random_traits, positions_grid[i]))
            # ? Belirgin bir community listesi verildiğinde bunlardan random seçerek node içine atılır.
            else:
                # ? For adding 1-to-length-1 attributes
                # random_attr_size = random.randint(1, len(community_list) - 1)
                # ? For adding 0-to-length attributes
                # ? Samples random attributes with normally distributed affinity values
                # random_attr_size = random.randint(0, len(community_list))
                # random_attributes = distribute_affinity_and_return_attributes(community_list, random_attr_size)
                temp_comm_list = []
                for comm in community_list:
                    new_attr = NodeAttribute(comm.name, 0.0, i)
                    temp_comm_list.append(new_attr)
                # ! UNCOMMENT BELOW if you want to sample from communities as attributes
                # random_attributes = random.sample(temp_comm_list, random_attr_size)
                node_name = chr(name_char) + str(name_int)
                # * positions_grid, Grid class'ındaki diamond square algoritmasından geliyor.
                # ! Use "random_attributes" in Node(...) if you want to sample attributes from attributes list
                if positions_grid is not None:
                    random_pos = random.sample(positions_grid, 1)
                    temp_nodes.append(Node(i, node_name, temp_comm_list, random_pos[0]))  # positions_grid[i]))
                    positions_grid.remove(random_pos[0])
                # * For demonstration purposes
                # print(node_name, " -> [Attributes]: ", list(random_attributes))
            # ? Node isimlendirmesi yapmak için: e.g.: A1, A2, ... , A9, B1, B2, ... z9
            if 65 <= name_char < 90:
                name_char += 1
            elif 90 <= name_char < 97:
                name_char = 97
            else:
                name_char += 1
                if name_char > 122:
                    name_char = 65
            if i % 51 == 0 and i != 0:
                name_int += 1

        # ? For creating normally distributed attribute affinity levels in certain limits for the values
        affinity_level_list = []
        mu = 0.0
        sigma = 0.5
        for node in temp_nodes:
            for i in range(len(node.attributes)):
                result = random.normalvariate(mu, sigma)
                if -1.0 <= result <= 1.0:
                    r_res = round(result, 2)
                    affinity_level_list.append(r_res)
                else:
                    while result < -1.0 or result > 1.0:
                        result = random.normalvariate(mu, sigma)
                        if -1.0 <= result <= 1.0:
                            r_res = round(result, 2)
                            affinity_level_list.append(r_res)
                            break
        # * For demonstration purposes
        # print(affinity_level_list)
        # print(max(affinity_level_list))
        # print(min(affinity_level_list))

        # ? Adding generated affinity levels with random sampling to the attr.s of the nodes
        for node in temp_nodes:
            for attr in node.attributes:
                random_affinity = random.sample(affinity_level_list, 1)
                attr.set_affinity_level(random_affinity[0])
                affinity_level_list.remove(random_affinity[0])

        # * For demonstration purposes
        """
        print("Created nodes: ", list(temp_nodes))
        for node in temp_nodes:
            node.print_attributes()
        """

        return temp_nodes

    # ? Generate edges with distances between nodes and the similarity between nodes (using count of mutual attr.s)
    def generate_edges_with_similar_distance(self, _node_list: list[Node], _distances_dict: dict,
                                             _similarity_threshold: int = None):
        revised_node_list = _node_list.copy()
        max_x = 0.0
        max_y = 0.0
        for node in revised_node_list:
            temp_x = node.get_x()
            if temp_x > max_x:
                max_x = temp_x
            temp_y = node.get_y()
            if temp_y > max_y:
                max_y = temp_y
        max_distance = math.sqrt(max_x**2 + max_y**2)

        temp_edges_dict = {node: None for node in _node_list}
        starting_node = select_most_positive_attributed_node(_node_list)
        # ! For demonstration and debug purposes
        # print("Selected starting node: ", starting_node.name, end=" -> ")
        # starting_node.print_attributes()
        nodewise_sorted_keys_dict = {}
        for node, other_node_distances in _distances_dict.items():
            node_sorted_keys = sorted(other_node_distances, key=other_node_distances.get)
            nodewise_sorted_keys_dict[node] = node_sorted_keys

        node_list_copy = _node_list.copy()
        distance_dict_copy = _distances_dict.copy()
        sorted_node_list = []
        # ? Using inner circle nodes approach
        # inner_nodes_dict = {node: None for node in _node_list}
        for i in range(len(_node_list)):
            selected = select_node(node_list_copy, distance_dict_copy)
            # inner_nodes_of_selected = get_inner_circle_nodes_of(selected, _distances_dict)
            sorted_node_list.append(selected)
            # inner_nodes_dict[selected] = inner_nodes_of_selected
            node_list_copy.remove(selected)
            distance_dict_copy.pop(selected)

        powerlaw_node_list = find_divisors(len(_node_list))
        # ! For demonstration and debug purposes
        # print("Powerlaw node list: ", list(powerlaw_node_list))
        powerlaw_edge_list = powerlaw_node_list[::-1]
        iteration_count = 0
        while iteration_count < len(powerlaw_node_list):
            if iteration_count == 0 and powerlaw_node_list[iteration_count] == 1:
                edges_for_selected = nodewise_sorted_keys_dict[starting_node]
                # TODO: Distance threshold can be added if needed
                # ! Now it works for the distances of (node.attributes * sqrt(2))
                """
                    added_edges workflow:
                    - take nodes for adding edges using nodewise_sorted_keys_dict and power_law
                        * e.g. creates 10 edges for 1 node, 5 edges for 2 nodes, 2 edges for 5 nodes, 
                        1 edge for 10 nodes. 
                        (takes nodes from sorted distances between nodes sequentially to create edges)
                    - checks if mutual attributes of the nodes are greater than certain user-input threshold
                    - checks if the distance between two nodes for edge addition is close
                        * a distance_threshold can also be used
                """
                # TODO: Use a count for every node that the power-law of edges are met or not
                #   If the count is smaller than the current power-law value of node, then do not remove that node
                #   If the count is equal to power-law value, then remove that node from the sorted list
                added_edges = [edges_for_selected[i] for i in range(powerlaw_edge_list[iteration_count])
                               if
                               # ! UNCOMMENT BELOW if you need mutual attributes approach
                               # len(set(starting_node.attributes).intersection(edges_for_selected[i].attributes)) >=
                               # _similarity_threshold and
                               is_node_in_inner_circle_of(starting_node, edges_for_selected[i], max_distance)
                               and affinity_level_comparator(starting_node, edges_for_selected[i])]  # distance_thres
                temp_edges_dict[starting_node] = added_edges
                sorted_node_list.remove(starting_node)
            elif iteration_count == 0 and powerlaw_node_list[iteration_count] != 1:
                firstly_selected_node = starting_node
                edges_for_selected = nodewise_sorted_keys_dict[firstly_selected_node]
                added_edges = [edges_for_selected[i] for i in range(powerlaw_edge_list[iteration_count])
                               # TODO: check list index out of range for a few node and attributes
                               if
                               # ! UNCOMMENT BELOW if you need mutual attributes approach
                               # len(set(firstly_selected_node.attributes).intersection(edges_for_selected[i].attributes))
                               # >= _similarity_threshold and
                               is_node_in_inner_circle_of(firstly_selected_node, edges_for_selected[i], max_distance)
                               and affinity_level_comparator(firstly_selected_node, edges_for_selected[i])]
                # distance_thres
                temp_edges_dict[firstly_selected_node] = added_edges
                sorted_node_list.remove(firstly_selected_node)
                for i in range(1, powerlaw_node_list[iteration_count]):
                    if i >= len(sorted_node_list):
                        selected_node = random.choice(sorted_node_list)
                    else:
                        selected_node = sorted_node_list[i]
                    edges_for_selected = nodewise_sorted_keys_dict[selected_node]
                    added_edges = [edges_for_selected[i] for i in range(powerlaw_edge_list[iteration_count])
                                   if
                                   # ! UNCOMMENT BELOW if you need mutual attributes approach
                                   # len(set(selected_node.attributes).intersection(edges_for_selected[i].attributes))
                                   # >= _similarity_threshold and
                                   is_node_in_inner_circle_of(selected_node, edges_for_selected[i], max_distance)
                                   and affinity_level_comparator(selected_node, edges_for_selected[i])]
                    temp_edges_dict[selected_node] = added_edges
                    sorted_node_list.remove(selected_node)
            else:
                for i in range(powerlaw_node_list[iteration_count]):
                    if i >= len(sorted_node_list):
                        selected_node = random.choice(sorted_node_list)
                    else:
                        selected_node = sorted_node_list[i]
                    edges_for_selected = nodewise_sorted_keys_dict[selected_node]
                    added_edges = [edges_for_selected[i] for i in range(powerlaw_edge_list[iteration_count])
                                   if
                                   # ! UNCOMMENT BELOW if you need mutual attributes approach
                                   # len(set(selected_node.attributes).intersection(edges_for_selected[i].attributes))
                                   # >= _similarity_threshold and
                                   is_node_in_inner_circle_of(selected_node, edges_for_selected[i], max_distance)
                                   and affinity_level_comparator(selected_node, edges_for_selected[i])]
                    # ? Added another 'and' statement as 'affinity_level_calculator' method to check if the
                    # ? attributes of the selected node and the node to be compared evaluates to a positive number
                    """
                        mutual_attrs = set(selected_node.attributes).intersection(node_to_compare.attributes)
                        if for all sum(selected_node.mutual_attrs[i] * node_to_compare.mutual_attrs[i]) >= 0.0
                            return true
                        else
                            return false
                    """
                    temp_edges_dict[selected_node] = added_edges
                    sorted_node_list.remove(selected_node)

            iteration_count += 1

        for node in temp_edges_dict.keys():
            if temp_edges_dict[node] is None:
                temp_edges_dict[node] = []
        # ? Prune all the nodes without any attributes, edges
        # TODO: It doesn't do pruning correctly
        revised_edge_dict = temp_edges_dict.copy()
        # ! TRYING for deleting all nodes without any edges (DEPRECATED)
        """
        for node in _node_list:
            if len(node.attributes) == 0:
                try:
                    revised_node_list.remove(node)
                    revised_edge_dict.pop(node)
                except ValueError:
                    print(node.name + " already deleted.")
            if len(temp_edges_dict[node]) == 0:
                try:
                    revised_node_list.remove(node)
                    revised_edge_dict.pop(node)
                except ValueError:
                    print(node.name + " already deleted.")

        print(str(len(revised_node_list)) + " : " + str(len(revised_edge_dict.keys()))
              + " : " + str(sum(len(neighbor) for neighbor in revised_edge_dict.values())))

        print(list(revised_node_list))
        print(dict(revised_edge_dict))

        for node, neighbors in revised_edge_dict.items():
            for neighbor in neighbors:
                if neighbor not in revised_node_list:
                    print(node.name + " : " + neighbor.name)
                    revised_edge_dict[node].remove(neighbor)

        print(str(len(revised_node_list)) + " : " + str(len(revised_edge_dict.keys()))
              + " : " + str(sum(len(neighbor) for neighbor in revised_edge_dict.values())))

        print(list(revised_node_list))
        print(dict(revised_edge_dict))
        
        # Comment finished here
        print(str(len(revised_node_list)) + " : " + str(len(revised_edge_dict.keys())))

        for node, neighbors in temp_edges_dict.items():
            if (temp_edges_dict[node] is None) or (len(temp_edges_dict[node]) == 0):
                # ? If you want to prune all the nodes without any edge
                if node in revised_node_list:
                    revised_node_list.remove(node)
                if node in revised_edge_dict.keys():
                    revised_edge_dict.pop(node)
                # ? If you want to add nodes without any edge
                # temp_edges_dict[node] = []
        print(str(len(revised_node_list)) + " : " + str(len(revised_edge_dict.keys())))

        for node, neighbors in temp_edges_dict.items():
            for neighbor in neighbors:
                if neighbor not in revised_node_list:
                    revised_edge_dict[node].remove(neighbor)

        print(str(len(revised_node_list)) + " : " + str(len(revised_edge_dict.keys())))
        """
        # ! ==============================================================================

        return revised_edge_dict, revised_node_list

    # ! DEPRECATED: Generate edges by using only distances between nodes
    def generate_edges_with_distance(self, _node_list: list[Node], _distances_dict: dict):
        # global sorted_keys_dict
        temp_edges_dict = {node: None for node in _node_list}
        sorted_keys_dict = {}
        for node, other_node_distances in _distances_dict.items():
            node_sorted_keys = sorted(other_node_distances, key=other_node_distances.get)
            sorted_keys_dict[node] = node_sorted_keys

        node_list_copy = _node_list.copy()
        distance_dict_copy = _distances_dict.copy()
        totally_sorted_node_list = []
        for i in range(len(_node_list)):
            selected = select_node(node_list_copy, distance_dict_copy)
            totally_sorted_node_list.append(selected)
            node_list_copy.remove(selected)
            distance_dict_copy.pop(selected)

        # intervals = create_interval_dict(_distances_dict)
        iteration_count = 0
        powerlaw_node_list = find_divisors(len(_node_list))
        powerlaw_edge_list = powerlaw_node_list[::-1]
        while iteration_count < len(powerlaw_node_list):
            for i in range(powerlaw_node_list[iteration_count]):
                if i >= len(totally_sorted_node_list):
                    selected_node = random.choice(totally_sorted_node_list)
                else:
                    selected_node = totally_sorted_node_list[i]  # select_node(_node_list, _distances_dict)
                edges_for_selected = sorted_keys_dict[selected_node]
                added_edges = [edges_for_selected[i] for i in range(powerlaw_edge_list[iteration_count])]
                temp_edges_dict[selected_node] = added_edges
                # TODO: clear dict from added nodes
                totally_sorted_node_list.remove(selected_node)

            iteration_count += 1

        """
        dummy_index = 5
        while dummy_index >= 1:
            select_node(_node_list, _distances_dict)

        for node, nearest_nodes_keys in sorted_keys_dict.items():
            # TODO: range need to be changed according to power law for every node
            if dummy_index > len(nearest_nodes_keys):
                dummy_index = len(nearest_nodes_keys)
            for index in range(dummy_index):     # power_law(sorted_keys_dict)):
                try:
                    temp_edges_dict[node].append(nearest_nodes_keys[index])
                except (KeyError, AttributeError):
                    temp_edges_dict[node] = [nearest_nodes_keys[index]]
            dummy_index -= 1
            if dummy_index < 1:
                break

        # temp_distances_dict = {node: None for node in _node_list}
        # nearest_nodes_dict = {node: None for node in _node_list}
        for node, neighbors in temp_edges_dict.items():
            if temp_edges_dict[node] is None:
                temp_edges_dict[node] = []
        """
        return temp_edges_dict

        # TODO: This part needed to be done by using recursion
        # min_distance = float('inf')
        # for node, other_node_distances in _distances_dict.items():
        #     for other_node, distance in other_node_distances.items():
        #         print(other_node, distance)

        """
        for near_node_dict in temp_distances_dict[a_node]:
            for node_key, node_to_node_distance in near_node_dict.items():
                if node_to_node_distance < min_distance:
                    min_distance = node_to_node_distance
                    nearest_nodes_dict[a_node] = node_key
                # nearest_node_for_a_node = min([y for x in temp_distances_dict[a_node] for y in x])
                # nearest_nodes_dict[a_node] = nearest_node_for_a_node
        nearest_nodes_dict[a_node].append()
        return nearest_nodes_dict
        """

    # ! UNUSED: Generate edges by using only similarity between nodes
    def generate_edges(self, _node_list: list[Node], similarity_threshold=None) -> dict[Node: list[Node]]:
        # ? For generating edges between the nodes that have mutual node attributes
        temp_edges_dict = {node: None for node in _node_list}

        for a_node in _node_list:
            for another_node in _node_list:
                if a_node == another_node:
                    continue
                else:
                    similarity_ratio = len(set(a_node.attributes).intersection(another_node.attributes))
                    if similarity_ratio >= similarity_threshold:
                        try:
                            temp_edges_dict[a_node].append(another_node)
                        except (KeyError, AttributeError):
                            temp_edges_dict[a_node] = [another_node]

        for node, neighbors in temp_edges_dict.items():
            if temp_edges_dict[node] is None:
                temp_edges_dict[node] = []

        # For generating a complete graph with edges that cover every node
        # Earlier code example
        """
        temp_edges = {node: None for node in _node_list}
        for i, node in enumerate(_node_list):
            temp_edges[node] = _node_list[i+1::]
        return temp_edges
        """

        return temp_edges_dict


# Test usage
# graph = create_random_graph(100, 8)
# graph.print_graph()
