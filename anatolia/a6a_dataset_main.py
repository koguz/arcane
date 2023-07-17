import time
from datetime import datetime
import pickle

import networkx as nx

from anatolia.network import Graph
from anatolia.a6a import AnatoliA
from anatolia.grid import Grid
import os.path
from PIL import Image


# Comment styles
# * Info
# ? Error
# ! Warning
# + Custom

# * Greek letters for parameters
# Grid Exponent - Gamma - \u0393
# Roughness - rho - \u03C1
# Node placement threshold - tau_n - \u03C4 \u2099
# Number of general attributes - C_mc - \u2098
# Similarity threshold - tau_s - \u03C4 \u209B

def add_to_pickle(file_name, data_to_dump):
    with open(file_name, 'wb') as pickle_data:
        pickle.dump(data_to_dump, pickle_data, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(file_name):
    with open(file_name, 'rb') as pickle_data:
        dumped_data = pickle.load(pickle_data)
    return dumped_data


if __name__ == '__main__':
    pickle_dir = "..\\pickle_files"
    results_dir = "..\\resulting_graph_plots\\"

    print("Welcome to Community Graph Generation Algorithm.\n______________________________________________")
    try:
        while True:
            print("=========================================")
            print("TO LOAD EARLIER GRAPHS, PRESS 0 !")
            print("TO CREATE AND SAVE A NEW GRAPH, PRESS 1 !")
            while True:
                try:
                    selection = int(input('ENTER SELECTION: '))
                    assert 0 <= selection <= 1
                except ValueError:
                    print("Not an integer! Please enter an integer.")
                except AssertionError:
                    print("Please enter 0 or 1")
                else:
                    break

            if selection == 0:
                if os.path.isdir(pickle_dir) and os.listdir(pickle_dir):
                    current_nodes_list = []
                    current_edgedicts_list = []
                    current_metadata_list = []
                    initial_node_files_count = 0
                    initial_edge_files_count = 0
                    for filename in os.listdir(pickle_dir):
                        if filename.startswith("nodes"):
                            current_nodes_list.append(read_from_pickle(pickle_dir + "\\" + filename))
                            initial_node_files_count += 1
                        elif filename.startswith("edges"):
                            current_edgedicts_list.append(read_from_pickle(pickle_dir + "\\" + filename))
                            initial_edge_files_count += 1
                        elif filename.startswith("metadata"):
                            current_metadata_list.append(read_from_pickle(pickle_dir + "\\" + filename))
                    print("====================================")
                    print("Select a stored graph data to load: ")
                    print(''.join("{0} : {1}\n"
                                  .format(idx, graph["to_string"]) for idx, graph in enumerate(current_metadata_list)))
                    graph_index = int(input("ENTER INDEX OF THE SELECTED GRAPH: "))
                    selected_node_list = current_nodes_list[graph_index]
                    selected_edges_dict = current_edgedicts_list[graph_index]
                    selected_graph = Graph(current_metadata_list[graph_index]["name"], selected_node_list,
                                           selected_edges_dict)
                    print("=================================================================================")
                    print("Do you want to open the drawn image (ENTER 0) OR want to draw it again (ENTER 1)?")
                    image_selection = int(input("ENTER SELECTION: "))
                    if image_selection == 0:
                        if os.path.isfile(
                                results_dir + "nw_plot_" + current_metadata_list[graph_index]["timestamp"] + ".png"):
                            image = Image.open(
                                results_dir + "nw_plot_" + current_metadata_list[graph_index]["timestamp"] + ".png")
                            image.show()
                        else:
                            selected_graph.create_communities(
                                "nw_plot_" + current_metadata_list[graph_index]["timestamp"])
                            time.sleep(2.0)
                            image = Image.open(
                                results_dir + "nw_plot_" + current_metadata_list[graph_index]["timestamp"] + ".png")
                            image.show()
                    elif image_selection == 1:
                        selected_graph.create_communities("nw_plot_" + current_metadata_list[graph_index]["timestamp"])
                else:
                    print("There are not any stored files.")

            elif selection == 1:
                method_start_time = datetime.now()

                parameters = []
                anatolia = AnatoliA()
                x = int(input("Give (2 power of x) value for creating the grid size of (2^x)+1: \n"))
                parameters.append(x)
                roughness = float(
                    input("Give roughness value for grid creation (Bigger value gives more steepness): \n"))
                parameters.append(roughness)
                grid = Grid(x, roughness)  # (3, 0.4)
                parameters.append(grid.place_threshold)
                # grid.print_grid()
                print("You can place ", grid.placeable_node_size, " nodes into grid.")
                vertex_count = grid.placeable_node_size
                metacomm_count = int(input("Please enter the meta-community count (int): \n"))
                parameters.append(metacomm_count)
                # comm_list = anatolia.generate_meta_communities(metacomm_count)
                comm_list = anatolia.generate_general_attributes(metacomm_count)
                node_list = anatolia.generate_vertices(vertex_count, comm_list, grid.placeable_grid_positions)
                # ? Use node list to create community_vertices dictionary
                community_nodes_dict = dict.fromkeys(comm_list)
                for node in node_list:
                    for key in community_nodes_dict.keys():
                        if key in node.attributes:
                            try:
                                community_nodes_dict[key].append(node)
                            except (KeyError, AttributeError):
                                community_nodes_dict[key] = [node]
                print(len(community_nodes_dict))
                # ? Create pairwise distance dictionary between all vertices of the graph
                distances_dict = anatolia.calculate_distances(node_list)
                # ? (Interval of distances dict) between vertices if needed (uncomment if needed)
                # interval_dict = anatolia.create_interval_dict(distances_dict)
                # ? Generate edges by using both the coordinate distances and attribute similarity
                similarity_threshold = int(input("Please give similarity threshold: (int) "
                                                 "\n (This value is actually the count of mutual attributes needed "
                                                 "for forming an edge between the nodes.) \n"))
                parameters.append(similarity_threshold)
                edges_dict, revised_node_list = anatolia.generate_edges_with_similar_distance(node_list, distances_dict,
                                                                                              similarity_threshold)
                nx_id_graph = nx.from_dict_of_lists(edges_dict)
                nx_connected = nx.connected_components(nx_id_graph)
                connected_components = [co for co in sorted(nx_connected, key=len, reverse=True) if len(co) > 2]
                date_padding = (str(datetime.now().year)[2:] + f"{datetime.now().month:02d}"
                                + f"{datetime.now().day:02d}" + "_" + f"{datetime.now().hour:02d}"
                                + f"{datetime.now().minute:02d}" + f"{datetime.now().second:02d}")
                new_graph_name = ("G_" + date_padding)
                new_graph = Graph(new_graph_name,
                                  revised_node_list, edges_dict, comm_list,
                                  distances_dict, connected_components)
                new_graph.create_communities("nw_plot_" + date_padding)
                graph_metadata = {"name": new_graph.name,
                                  "node_count": len(new_graph.vertices),
                                  "edge_count": new_graph.get_edge_count(),
                                  "comm_count": len(new_graph.meta_communities),
                                  "parameters": parameters,
                                  "to_string":
                                      "Graph " + new_graph.name + " with " + str(
                                          len(new_graph.vertices)) + " vertices, " +
                                      str(int(
                                          sum(len(new_graph.adjacency_list[edge]) for edge in
                                              new_graph.adjacency_list))) +
                                      " edges and " + str(len(new_graph.meta_communities)) + " meta-communities. " +
                                      "(\u0393: " + str(parameters[0]) + ", \u03C1: " + str(parameters[1]) +
                                      ", \u03C4\u2099: " + str(parameters[2]) + ", C\u2098: " + str(parameters[3]) +
                                      ", \u03C4\u209B: " + str(parameters[4]) + " )",
                                  "timestamp": date_padding}
                # TODO: Check if dates are same for figure and graph name (it needed to wait for a minute to load image)
                # ! image = Image.open('..\\Results\\nw_plot_' + date_padding)
                # ! image.show()

                method_end_time = datetime.now()
                print('Method Duration: {}'.format(method_end_time - method_start_time))

                # * See the visualized graph and decide to keep it or not.
                print("DO YOU WANT TO STORE THE CURRENT GRAPH?")
                print("TO FORGET, PRESS 0 !")
                print("TO STORE, PRESS 1 !")
                while True:
                    try:
                        store_selection = int(input('ENTER: '))
                        assert 0 <= store_selection <= 1
                    except ValueError:
                        print("Not an integer! Please enter an integer.")
                    except AssertionError:
                        print("Please enter 0 or 1")
                    else:
                        break

                if store_selection == 1:
                    try:
                        add_to_pickle(pickle_dir + '\\nodes_' + date_padding + '.pkl', node_list)
                        add_to_pickle(pickle_dir + '\\edges_' + date_padding + '.pkl', edges_dict)
                        add_to_pickle(pickle_dir + '\\metadata_' + date_padding + '.pkl', graph_metadata)
                        print("The current graph successfully stored into pickle file.")
                    except FileNotFoundError:
                        print("There are not a folder to store the graph data.")
                else:
                    print("! Forgetting the current graph.")
    except KeyboardInterrupt:
        # ? Use Ctrl + F2 to exit the program
        exit(0)
