import math
import os
import sys
from datetime import datetime

import numpy as np

from anatolia.node import Node

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import random
from scipy.spatial import distance

from anatolia.node_attribute import NodeAttribute


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def combine_color(colorRGBA1, colorRGBA2):
    alpha = 255 - ((255 - colorRGBA1[3]) * (255 - colorRGBA2[3]) / 255)
    red = (colorRGBA1[0] * (255 - colorRGBA2[3]) + colorRGBA2[0] * colorRGBA2[3]) / 255
    green = (colorRGBA1[1] * (255 - colorRGBA2[3]) + colorRGBA2[1] * colorRGBA2[3]) / 255
    blue = (colorRGBA1[2] * (255 - colorRGBA2[3]) + colorRGBA2[2] * colorRGBA2[3]) / 255
    return int(red), int(green), int(blue), int(alpha)


def find_correct_filename(path: str, filename: str, count: int):
    if filename + '(' + str(count) + ').png' not in os.listdir(path):
        return filename + '(' + str(count) + ').png'
    else:
        return find_correct_filename(path, filename, count + 1)


class Graph:

    def __init__(self, name: str, vertices: list[Node] = None, adjacency_dict: dict[Node, list[Node]] = None,
                 meta_communities: list = None, distances_dict: dict[Node, dict[Node, float]] = None,
                 connected_components: list = None):
        self.name = name
        if vertices is None:
            self.vertices = list()
        else:
            self.vertices = vertices
        if adjacency_dict is None:
            self.adjacency_list = dict()
        else:
            self.adjacency_list = adjacency_dict
        if distances_dict is None:
            self.node_distances_dict = self.calculate_distances()
        else:
            self.node_distances_dict = distances_dict
        # TODO: her community'de hangi vertex'lerin olduğunu bilmek için bunun içine de eklemeli.
        self.community_vertices_dict: dict[NodeAttribute, list[Node]] = dict()
        self.vertex_communities_dict: dict[Node, list[NodeAttribute]] = dict()
        if meta_communities is None:
            self.meta_communities = list()
        else:
            self.meta_communities = meta_communities
        if connected_components is None:
            self.connected_components = list()
        else:
            nx_graph = self.convert_to_nx_graph_with_ids()
            nx_components = nx.connected_components(nx_graph)
            self.connected_components = [co for co in sorted(nx_components, key=len, reverse=True) if len(co) > 2]
        self.gt_community_vertices_dict: dict[int, set[int]] = \
            dict(zip(range(len(self.connected_components)), self.connected_components))

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def print_graph(self):
        print("List of nodes: ")
        print(list(self.vertices))
        for node in self.vertices:
            print("Node " + node.name + ": ", end="-> ")
            node.print_attributes()
        print("Adjacency matrix of nodes: ")
        for node, adjacent_nodes in self.adjacency_list.items():
            print("{" + node.name, end=": ")
            for i in range(len(adjacent_nodes)):
                if i < len(adjacent_nodes) - 1:
                    print(adjacent_nodes[i].name, end=", ")
                else:
                    print(adjacent_nodes[i].name, end="} ")
            print()
        # print({node.name: for node, adjacent_nodes in self.adjacency_list.items()})

    # Add a list of nodes
    def add_nodes(self, _vertices_to_add: list[Node]):
        self.vertices.extend(_vertices_to_add)

    def get_node_by_id(self, get_id):
        if get_id < len(self.vertices):
            return self.vertices[get_id]

    def get_undirected_edges_as_tuples_for_node(self, _node: Node):
        undirected_edge_tuple_list = []
        neighbors = self.get_neighbors(_node)
        for neighbor in neighbors:
            undirected_edge_tuple_list.append((_node, neighbor))

    def get_edges_as_set_of_tuples(self):
        edge_tuple_set = set()
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                if (neighbor, node) not in edge_tuple_set:
                    edge_tuple_set.add((node, neighbor))

        return edge_tuple_set

    # Get undirected edge count of a graph with sets
    def get_edge_count_improved(self) -> int:
        edge_set = self.get_edges_as_set_of_tuples()
        return len(edge_set)

    # ! DEPREATED: Get undirected edge count of a graph
    def get_edge_count(self) -> int:
        temp_undirected_edge_count = 0
        undirected_edge_tuple_list = []
        for node, connected_nodes in self.adjacency_list.items():
            for connected_node in connected_nodes:
                undirected_edge_tuple_list.append((node, connected_node))
                if (connected_node, node) not in undirected_edge_tuple_list:
                    temp_undirected_edge_count += 1

        return temp_undirected_edge_count

    def get_edge_count_acc_to_attribute(self, node: Node, attr: NodeAttribute):
        node_edge_count_acc_to_attr = 0
        node_edge_tuple_list = []
        node_neighbors = self.get_neighbors(node)

        for neighbor in node_neighbors:
            if node.attributes[node.get_attribute_index(attr)].affinity_level \
                    * neighbor.attributes[neighbor.get_attribute_index(attr)].affinity_level > 0:
                node_edge_tuple_list.append((node, neighbor))
                if (neighbor, node) not in node_edge_tuple_list:
                    node_edge_count_acc_to_attr += 1
                    # TODO: abs(node.attributes[node.get_attribute_index(attr)].affinity_level / len(node.attributes))

        return int(node_edge_count_acc_to_attr)

    def get_total_edge_count_acc_to_attr(self, attr: NodeAttribute):
        total_edge_of_node_acc_to_attr = 0
        for node in self.vertices:
            # list_of_node_edges = self.get_undirected_edges_as_tuples(node)
            total_edge_of_node_acc_to_attr += self.get_edge_count_acc_to_attribute(node, attr)

        return total_edge_of_node_acc_to_attr

    def get_count_of_nodes_less_than_attr(self, attr: NodeAttribute):
        count_less = 0
        for node in self.vertices:
            if node.attributes[node.get_attribute_index(attr)].affinity_level < 0:
                count_less += 1

        return count_less

    def get_count_of_nodes_greater_than_attr(self, attr: NodeAttribute):
        count_greater = 0
        for node in self.vertices:
            if node.attributes[node.get_attribute_index(attr)].affinity_level >= 0:  # ! For evaluation purposes
                # ? greater/node_size + less/node_size must be equal to 1 (p + q = 1) (if one of them have >=)
                count_greater += 1

        return count_greater

    def convert_to_nx_graph(self):
        nx_graph = nx.from_dict_of_lists(self.adjacency_list)
        return nx_graph

    def convert_to_nx_graph_with_ids(self) -> nx.Graph:
        adjacency_dict_with_ids = dict()
        for node, neighbors in self.adjacency_list.items():
            adjacency_dict_with_ids[node.node_id] = []
            for neighbor in neighbors:
                adjacency_dict_with_ids[node.node_id].append(neighbor.node_id)
        nx_graph = nx.from_dict_of_lists(adjacency_dict_with_ids)
        return nx_graph

    def get_affinity_levels_of_certain_attr(self, _attr: NodeAttribute) -> list[float]:
        affinity_level_list_for_attr = []
        for node in self.vertices:
            attr_affinity = node.attributes[node.get_attribute_index(_attr)].affinity_level
            affinity_level_list_for_attr.append(attr_affinity)
        return affinity_level_list_for_attr

    # Get neighbors of a node
    def get_neighbors(self, node: Node) -> list[Node]:
        return self.adjacency_list[node]

    def calculate_distances(self):
        temp_distances_dict = {node: None for node in self.vertices}
        for a_node in self.vertices:
            for another_node in self.vertices:
                if a_node.node_id >= another_node.node_id:  # a_node == another_node:
                    continue
                else:
                    distance_a_another = math.dist(a_node.get_xy(), another_node.get_xy())
                    try:
                        temp_distances_dict[a_node].update({another_node: distance_a_another})
                    except (KeyError, AttributeError):
                        temp_distances_dict[a_node] = {another_node: distance_a_another}
                    try:
                        temp_distances_dict[another_node].update({a_node: distance_a_another})
                    except (KeyError, AttributeError):
                        temp_distances_dict[another_node] = {a_node: distance_a_another}

        return temp_distances_dict

    def calculate_distances_improved(self):
        positions_of_node_list = [node.get_xy() for node in self.vertices]
        distances = distance.cdist(positions_of_node_list, positions_of_node_list, 'euclidean')

        d_dict = {node: {} for node in self.vertices}
        for node in self.vertices:
            for another_node in self.vertices:
                if node.node_id >= another_node.node_id:  # a_node == another_node:
                    continue

                d_dict[node][another_node] = distances[node.node_id][another_node.node_id]
                d_dict[another_node][node] = distances[another_node.node_id][node.node_id]

        return d_dict

    def gather_metacomms(self):
        _meta_communities = []
        if self.meta_communities is None or len(self.meta_communities) == 0:
            for vertex in self.vertices:
                for attr in vertex.attributes:
                    if attr not in _meta_communities:
                        _meta_communities.append(attr)
            self.meta_communities = _meta_communities
        else:
            _meta_communities = self.meta_communities

        return _meta_communities

    def create_metacomm_vertices_dict(self):
        # Determine meta-communities for vertices
        self.community_vertices_dict = dict.fromkeys(self.meta_communities)
        for vertex in self.vertices:
            for key in self.community_vertices_dict.keys():
                if key in vertex.attributes:
                    try:
                        self.community_vertices_dict[key].append(vertex)
                    except (KeyError, AttributeError):
                        self.community_vertices_dict[key] = [vertex]
        # print(_meta_communities)

    def reverse_comm_vertex_map(self):
        # Reverse community-vertex map to understand comm.s from vertex standpoint
        self.vertex_communities_dict = dict.fromkeys(self.vertices)
        for comm, vertices in self.community_vertices_dict.items():
            for node in vertices:
                try:
                    self.vertex_communities_dict[node].append(comm)
                except (KeyError, AttributeError):
                    self.vertex_communities_dict[node] = [comm]

    def plot_node_attribute_distribution(self):
        attribute_list_for_plotting = []
        mu = 0.0
        sigma = 0.5

        for node, comms in self.vertex_communities_dict.items():
            if comms is not None:
                for attr in node.attributes:
                    attribute_list_for_plotting.append(attr.affinity_level)

        count, bins, ignored = plt.hist(attribute_list_for_plotting, len(attribute_list_for_plotting), density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=1,
                 color='r')
        # plt.hist(liste, bins=50)
        plt.xlabel("Affinity level of the node attributes")
        plt.ylabel("Frequency")
        plt.show()

    def create_random_color_map_for_metacomms(self):
        # Create random colors for all meta-communities
        # TODO: If the colors for communities are important, we need to add this into graph in main method
        metacommunity_colors = {metacomm: None for metacomm in self.meta_communities}
        for meta_community in self.meta_communities:  # for i in range(len(_meta_communities)):
            hexadecimal = "#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])
            metacommunity_colors[meta_community] = hexadecimal

        return metacommunity_colors

    def create_random_color_map_for_comms(self):
        # Create random colors for all communities
        community_colors = {comm: None for comm in range(len(self.connected_components))}
        for community in self.gt_community_vertices_dict.keys():  # for i in range(len(_meta_communities)):
            hexadecimal = "#" + ''.join([random.choice('ABCDEF0123456789') for j in range(6)])
            community_colors[community] = hexadecimal

        return community_colors

    def create_color_map_for_vertices(self, metacommunity_colors: dict):
        # Create a color map to map it with the nodes in the drawn graph
        # TODO: If the colors for communities are important, we need to add this into graph in main method
        color_map = []
        for vertex in self.vertices:
            for community in self.meta_communities:
                if community in vertex.attributes:
                    if len(vertex.attributes) <= 1:
                        # print(vertex)
                        color_map.append(metacommunity_colors[community])
                    else:
                        # blend colors
                        # print(vertex)
                        # TODO: blend colors or draw circular community limits
                        color_map.append(metacommunity_colors[community])
                        break

        # TODO: bounding circle eklenmesi yapıldığında bu kısıma gerek kalmayacak
        if len(color_map) != len(self.vertices):
            for i in range(len(color_map), len(self.vertices)):
                color_map.append('#000000')

        return color_map

    def create_color_map_for_vertices_with_gt_communities(self, community_colors: dict):
        color_map = ['' for index in range(len(self.vertices))]
        for comm_index, vertices_set in self.gt_community_vertices_dict.items():
            for vertex in self.vertices:
                if vertex.node_id in vertices_set:
                    color_map[vertex.node_id] = community_colors[comm_index]
        for ind, color in enumerate(color_map):
            if color_map[ind] == '':
                color_map[ind] = '#000000'
        return color_map

    # Create meta-communities by using attributes of the nodes in the graph
    def create_communities(self, overwritten_filename: str = None, _placement_ratio: float = None):
        # Adding all attributes under nodes as meta-communities
        self.gather_metacomms()

        # Determine meta-communities for vertices
        self.create_metacomm_vertices_dict()

        # Reverse community-vertex map to understand comm.s from vertex standpoint
        self.reverse_comm_vertex_map()

        # self.create_component_vertices_dict()

        # ? Uses vertex_communities_dict to plot a normal distribution histogram of node attributes levels
        # self.plot_node_attribute_distribution()

        # Create random colors for all meta-communities
        metacommunity_colors = self.create_random_color_map_for_metacomms()

        # Create random colors for all connected components
        community_colors = self.create_random_color_map_for_comms()

        # Create a color map to map it with the nodes in the drawn graph
        color_map = self.create_color_map_for_vertices(metacommunity_colors)

        comm_color_map = self.create_color_map_for_vertices_with_gt_communities(community_colors)

        # print("Meta-Community Colors: ", metacommunity_colors)
        # print("Meta-Communities: ", self.community_vertices_dict.items())

        print("Community Colors: ", community_colors)
        print("Communities: ", self.gt_community_vertices_dict.items())

        if overwritten_filename is not None:
            # TODO: bugfix gerekebilir
            self.draw_xy_circle(comm_color_map, community_colors, 120, overwritten_filename, _placement_ratio)
        else:
            # TODO: "self.draw" community'leri birbiri arasında edge varmış gibi gösteriyor
            # self.draw(color_map, metacommunity_colors)
            # TODO: "self.draw_circle" her meta-community için node'lar etrafına iç içe circle'lar
            #  çiziyor (spring_layout)
            # self.draw_circle(color_map, metacommunity_colors)
            # TODO: "self.draw_circle" her meta-community için belli koordinatta node'lar etrafına
            #  iç içe circle'lar çiziyor
            # self.draw_xy_circle(color_map, metacommunity_colors, _placement_ratio)
            self.draw_xy_circle(comm_color_map, community_colors, 120, None, _placement_ratio)

    def grid_layout(self, _scale=None) -> dict:
        pos_dict = dict.fromkeys(self.vertices)
        for vertex in self.vertices:
            if _scale is None:
                x_val = vertex.get_x() * 20
                y_val = vertex.get_y() * 20
            else:
                x_val = vertex.get_x() * 20 * _scale
                y_val = vertex.get_y() * 20 * _scale
            pos_dict[vertex] = (x_val, y_val)  # vertex.get_xy()
        return pos_dict

    def total_draw(self, _placement_fill_ratio):
        self.create_communities(_placement_fill_ratio)

    # Drawing the graph by using networkx and matplotlib in xy coordinates with including circles
    def draw_xy_circle(self, _color_map: list = None, _community_colors: dict = None, dpi_value: int = None,
                       _overwritten_filename: str = None, _placement_ratio: float = None):
        # * Default variables
        default_dpi = None
        default_node_size = 300
        default_patch_radius = 0.5
        node_size = default_node_size * 5
        node_size_factor = node_size / default_node_size
        vertex_radius = default_patch_radius + (node_size_factor * 0.1)

        if dpi_value is None:
            default_dpi = 120  # ? = (16x9) HD resolution (1920x1080) (multiply it with x2 for 4K)
            my_fig = plt.figure(figsize=(32, 18), dpi=default_dpi)
        else:
            my_fig = plt.figure(figsize=(32, 18), dpi=dpi_value)

        # ? Figure size components
        fig_width, fig_height = my_fig.get_size_inches() * my_fig.dpi
        pixel_size = fig_width * fig_height
        print(fig_width, fig_height)
        # ? -----------------------
        # ! UNCOMMENT BELOW if you want to scale the figure as tight layout
        """
        print("Do you want the figure to be drawn with tight layout for a square drawing space?")
        layout_selection = int(input("Enter 0 for YES.\nEnter any other integer for NO: \n"))
        if layout_selection == 0:
            my_fig.tight_layout()
        """

        nx_graph = nx.from_dict_of_lists(self.adjacency_list)

        print("Do you want the figure to be drawn with the nodes at their grid positions?")
        placing_selection = int(input("Enter 0 for YES.\n"
                                      "Enter any other integer for selecting a different placing option: \n"))
        if placing_selection == 0:
            pos = self.grid_layout()
        else:
            print("Please select one option: ")
            other_placing = int(input("Enter 0 for spring layout.\n"
                                      "Enter 1 for circular layout.\n"
                                      "Enter 2 for shell layout.\n"
                                      # "Enter 3 for spectral layout.\n"
                                      "Enter 3 for spiral layout.\n"
                                      "Enter 4 for Kamada-Kawai layout.\n"
                                      "Any other value will draw the grid layout as default: \n"))
            if other_placing == 0:
                pos = nx.spring_layout(nx_graph)
            elif other_placing == 1:
                pos = nx.circular_layout(nx_graph)
            elif other_placing == 2:
                # ? For adding color in shell layout according to connected components
                # S = [c for c in nx.connected_components(nx_graph)]
                # l = [list(li) for li in S]
                pos = nx.shell_layout(nx_graph)  # , l)
            # elif other_placing == 3:
            #     pos = nx.spectral_layout(nx_graph)
            elif other_placing == 3:
                pos = nx.spiral_layout(nx_graph)
            elif other_placing == 4:
                pos = nx.kamada_kawai_layout(nx_graph)
            else:
                # * Default layout is grid layout
                pos = self.grid_layout()
        node_size = 200  # * DEFAULT: node_size = 300
        if _color_map is None:
            nx.draw(nx_graph, pos, edge_color='black', node_size=node_size, with_labels=True, font_size=8)
        else:
            nx.draw(nx_graph, pos, node_color=_color_map, edge_color='black', node_size=node_size, with_labels=True,
                    font_size=8)

        curr_axis = plt.gca()
        print("Do you want the figure to be drawn with square aspect ratio for same scaling of x and y values?")
        aspect_selection = int(input("Enter 0 for YES.\nEnter any other integer for NO: \n"))
        if aspect_selection == 0:
            curr_axis.set_aspect('equal')
        else:
            curr_axis.set_aspect('auto')
        # curr_axis.set_aspect('auto')  # ! or 'equal' if you need same scaling for x and y

        max_x = max([pos[node][0] for node in self.vertices])
        max_y = max([pos[node][1] for node in self.vertices])

        max_radius = min(plt.gcf().get_size_inches()) * min(max_x, max_y) / 2 * (1 / np.sqrt(len(self.vertices)))

        scaling_factor = node_size / max_radius

        if placing_selection == 0:
            vertex_circle_data_dict = dict.fromkeys(self.vertex_communities_dict.keys())
            # vertex_radius = 52  # node_size / len(self.vertices) ** 3
            # * Uncomment block comment for drawing circle patches around the nodes for
            # * meta-communities they belong TODO: correct patch size could not be determined
            """
            for vertex, communities in self.vertex_communities_dict.items():
                if communities is not None:
                    curr_i = len(communities)
                    for curr_com_idx, community in enumerate(communities):
                        # TODO: get node size for comparison  # radius = 0.1 + (curr_i / 100)
                        # TODO: radius value is needed to be arranged according to the vertex area size
                        radius = curr_com_idx * max_radius * scaling_factor
                        circle = plt.Circle((pos[vertex][0], pos[vertex][1]), radius, fill=True,
                                            color=_community_colors[community])
                        if vertex_circle_data_dict[vertex] is None:
                            vertex_circle_data_dict[vertex] = [circle]
                        else:
                            vertex_circle_data_dict[vertex].append(circle)
                        curr_axis.add_patch(circle)
                        curr_i -= 1
            """
            # Adding text data TODO: change this to better version
            # plt.annotate(str(list(self.community_vertices_dict.items())), xy=(-1.15, 1.1), fontsize=6)

        if default_dpi is not None:
            plt.show()
        else:
            if _overwritten_filename is not None:
                if os.path.isfile("..\\resulting_graph_plots\\" + _overwritten_filename + ".png"):
                    overwritten_name = find_correct_filename("..\\resulting_graph_plots\\", _overwritten_filename, 1)
                    # os.remove("..\\Results\\nw_plot_" + _overwritten_filename + ".png")
                    try:
                        plt.show()
                        plt.savefig("..\\resulting_graph_plots\\" + overwritten_name)
                        print("Figure is overwritten as a png file to the results folder.\n")
                    except FileNotFoundError:
                        print("Save error on figure.")
                    except (Exception, ) as e:
                        print("Unexpected error:", sys.exc_info()[0], "\n", str(e))
                        raise
                else:
                    try:
                        plt.show()
                        plt.savefig("..\\resulting_graph_plots\\" + _overwritten_filename + ".png")
                        print("Figure is saved as a png file to the results folder.\n")
                    except FileNotFoundError:
                        print("Save error on figure.")
                    except (Exception, ) as e:
                        print("Unexpected error:", sys.exc_info()[0], "\n", str(e))
                        raise
            else:
                date_padding = (str(datetime.now().year)[2:] + f"{datetime.now().month:02d}"
                                + f"{datetime.now().day:02d}" + "_" + f"{datetime.now().hour:02d}"
                                + f"{datetime.now().minute:02d}" + f"{datetime.now().second:02d}")
                plt.show()
                plt.savefig("..\\resulting_graph_plots\\nw_plot_" + date_padding + ".png")
                print("Figure is saved as a png file to the results folder.\n")

    # Drawing the graph by using networkx and matplotlib and with labels with including circles
    def draw_circle(self, _color_map: list = None, _community_colors: dict = None):
        plt.figure(dpi=800)
        nx_graph = nx.from_dict_of_lists(self.adjacency_list)
        pos = nx.spring_layout(nx_graph)
        if _color_map is None:
            nx.draw(nx_graph, pos, edge_color='black', with_labels=True, font_size=8)
        else:
            # TODO: temporary solution (edge_color'ı kaldır) (node_color da kalkmalı, comm çizgiler ile belli olacak)
            # nx.draw(nx_graph, pos, node_color=_color_map, edge_color='white', with_labels=True)
            nx.draw(nx_graph, pos, edge_color='black', with_labels=True, font_size=8)

        curr_axis = plt.gca()
        curr_axis.set_aspect('equal')
        vertex_circle_data_dict = dict.fromkeys(self.vertex_communities_dict.keys())
        for vertex, communities in self.vertex_communities_dict.items():
            curr_i = len(communities)
            # TODO: if curr_i is very large, try to use a number greater than 100 as divisor for circle size
            # TODO: 0.05 is used as node size but it can be parametrized
            for curr_com_idx, community in enumerate(communities):
                circle = plt.Circle((pos[vertex].item(0), pos[vertex].item(1)), 0.055 + (curr_i / 100), fill=True,
                                    color=_community_colors[community])
                if vertex_circle_data_dict[vertex] is None:
                    vertex_circle_data_dict[vertex] = [circle]
                else:
                    vertex_circle_data_dict[vertex].append(circle)
                curr_axis.add_patch(circle)
                curr_i -= 1

        # Adding text data
        plt.annotate(str(list(self.community_vertices_dict.items())), xy=(-1.15, 1.1), fontsize=6)

        # patch = mpatches.PathPatch(path, fill=False, color=_community_colors[community], linewidth=1.2)

        plt.show()

    # Drawing the graph by using networkx and matplotlib and with labels with edges as communities
    def draw(self, _color_map: list = None, _community_colors: dict = None):
        plt.figure(dpi=800)
        nx_graph = nx.from_dict_of_lists(self.adjacency_list)
        # pos = nx.spring_layout(nx_graph)
        pos = self.grid_layout()
        if _color_map is None:
            nx.draw(nx_graph, pos, with_labels=True)
        else:
            # TODO: temporary solution (edge_color'ı kaldır) (node_color da kalkmalı, comm çizgiler ile belli olacak)
            # nx.draw(nx_graph, pos, node_color=_color_map, edge_color='white', with_labels=True)
            nx.draw(nx_graph, pos, edge_color='black', with_labels=True)

        m_path = mpath.Path
        pathdata_dict = dict.fromkeys(self.community_vertices_dict.keys())
        for community, vertices in self.community_vertices_dict.items():
            pathdata_dict[community] = [(m_path.MOVETO, (pos[vertices[0]][0], pos[vertices[0]][1]))]
            for i, vertex in enumerate(vertices):
                pathdata_dict[community].append((m_path.LINETO, (pos[vertex][0], pos[vertex][1])))

        # Adding text data
        # plt.annotate(str(list(self.community_vertices_dict.keys())), xy=(-1.15, 1.1), fontsize=10)
        plt.annotate(str(list(self.community_vertices_dict.items())), xy=(-1.15, 1.1), fontsize=6)

        # Adding community lines; TODO: change it to cover the nodes with bounding areas

        patch_list = []
        for community, comm_path_data in pathdata_dict.items():
            codes, verts = zip(*comm_path_data)
            path = m_path(verts, codes)
            patch = mpatches.PathPatch(path, fill=False, color=_community_colors[community], linewidth=1.2)
            patch_list.append(patch)
            plt.gca().add_patch(patch)

        """
        Path = mpath.Path
        pathdata = [
            (Path.MOVETO, (pos[self.vertices[0]].item(0), pos[self.vertices[0]].item(1))),
            (Path.LINETO, (pos[self.vertices[1]].item(0), pos[self.vertices[1]].item(1)))
        ]

        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        patch = mpatches.PathPatch(
            path, fill=False, color='yellow', alpha=0.5)
        plt.gca().add_patch(patch)

        # circle = Circle((0, 0), 0.15, fill=False, color='red')
        # plt.gca().add_patch(circle)
        """

        plt.show()
