# ARCANE: Attribute-based Realistic Community and Associate NEtwork algorithm

An Object-Oriented Algorithm in _Python_ for our article "From attributes to communities: A novel approach in social network generation".

## Citation:

If you use ARCANE in your research studies, please cite:

> Uludağlı MC, Oğuz K. 2024. From attributes to communities: a novel approach in social network generation. _PeerJ Comput. Sci._ 10:e2483 http://doi.org/10.7717/peerj-cs.2483

As ''bibtex'' citation:

> @article{10.7717/peerj-cs.2483,  
> title = {From attributes to communities: a novel approach in social network generation},  
> author = {Uludağlı, Muhtar Çağkan and Oğuz, Kaya},  
> year = 2024,  
> month = nov,  
> keywords = {Graph generation, Node attributes, Social networks, Community},  
> volume = 10,  
> pages = {e2483},  
> journal = {PeerJ Computer Science},  
> issn = {2376-5992},  
> url = {https://doi.org/10.7717/peerj-cs.2483},  
> doi = {10.7717/peerj-cs.2483},  
> }  


## How to Use ARCANE:

We have 5 different classes to use in 5 different source code files:
```python
Grid()              # in grid.py, 
NodeAttribute()     # in node_attribute.py, 
Node()              # in node.py, 
Graph()             # in network.py, 
Arcane()            # in arcane.py  
```

The essential parameters you need to get from the user for ARCANE to work are:
```python
x                       #: Grid size exponent as in (2^x)*(2^x) size matrix   
roughness               #: A float value between 0-1 for the steepness/the flatness of the change of values for grid elements  
grid.placing_threshold  #: A value to determine the node-placeable positions    
```
> If grid.placing_threshold is less than the value at the grid position, then a node will be placed at that position. Not placed, if otherwise.  
> Its default value is 'None', if you want to place the nodes with a density-based distribution (which is the employed approach in our evaluations).
```python
attribute_count         #: The number of general attributes to put inside every node    
similarity_threshold    #: the number of mutual attributes needed for forming an edge between the nodes    
```

### To Generate a New Graph:
1.  Firstly, you need to create a new Arcane() instance to use its methods with:
```python
arcane = Arcane()
```

2.  You can create the grid instance after getting the necessary parameters:
```python
grid = Grid(x, roughness)
```

3.  The vertex count is actually defined by the grid you have created:
```python
vertex_count = grid.placeable_node_size
```

4.  After that, you need to generate general attributes:
```python
attr_list = arcane.generate_general_attributes(attribute_count)
```

5.  Because you have all the necessary parameters at this point, you can generate the vertices of the graph:
```python
node_list = arcane.generate_vertices(vertex_count, attr_list, grid.placeable_grid_positions)
```

6. You need to create pairwise conceptual distances dictionary between the vertices for edge generation with:
```python
distances_dict = arcane.calculate_distances(node_list)
```

7. To generate the edges of the graph after this point, you need to use:
```python
edges_dict, revised_node_list = arcane.generate_edges_with_similar_distance(node_list, distances_dict, similarity_threshold)
```

8. You can create the overall graph at the end by using:
```python
new_graph_name = "Graph_0" 
new_graph = Graph(new_graph_name, revised_node_list, edges_dict, attr_list, distances_dict, connected_components)
# connected_component parameter is used for drawing the graph with a different color for every connected component.
# Its default value is 'None', if you do not want to draw with colors according to it.
```

9. To visualize your newly constructed graph, you need to use:
```python
new_graph.create_communities("a_string_name_for_graph_drawing.png")
```

### Using Our Test Code to Generate New Graphs:
All the essential things we have given in here is employed in our __"arcane_dataset_main.py"__ source code file 
in an ordered fashion. All the needed parameters, creating new instances, generating and drawing the resulting graphs, 
naming these graphs and saving them as .pickle files to use them afterward are managed with the text prompts from our test code.

### External Dataset Used for Evaluations
We used Sinanet dataset from https://github.com/smileyan448/Sinanet of the article "Jia, C., Li, Y., Carson, M. B., Wang, X., & Yu, J. (2017). Node attribute-enhanced community detection in complex networks. Scientific reports, 7(1), 2626.". If you also use this dataset, please cite it accordingly.
