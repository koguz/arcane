import json
import math

from a4e.node_attribute import NodeAttribute


class Node:

    def __init__(self, node_id: int, name: str, attributes: list[NodeAttribute] = None, xy: tuple[float, float] = None):
        self.node_id = node_id
        self.name = name
        if xy is None:
            self.xy = (0.0, 0.0)
        else:
            self.xy = xy
        if attributes is None:
            self.attributes = list()
        else:
            temp_attr_list = list()
            for i in range(len(attributes)):
                temp_attr_list.append(attributes[i])
            self.attributes = temp_attr_list
        self.attributes_values_dict = dict()
        for attr in self.attributes:
            self.attributes_values_dict[attr.name] = attr.affinity_level

    def __repr__(self):
        # return f'Node(node_id={self.node_id}, name={self.name}, attributes={self.attributes}'
        return self.name

    # def __str__(self):
    #     return self.node_id

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.name < other.name

    # def __get__(self, instance, owner):
        # pass

    def set_xy(self, x, y):
        self.xy = (x, y)

    def get_xy(self):
        return self.xy

    def get_x(self):
        return self.xy[0]

    def get_y(self):
        return self.xy[1]

    def get_distance(self, another: "Node"):
        return math.dist(self.get_xy(), another.get_xy())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def get_attribute(self, attribute: NodeAttribute):
        if attribute in self.attributes:
            return attribute
        else:
            return None

    def get_attribute_index(self, attribute: NodeAttribute):
        if attribute in self.attributes:
            ind_of_attr = self.attributes.index(attribute)
            return ind_of_attr
        else:
            return -1

    def add_attributes(self, attribute_list):
        self.attributes.extend(attribute_list)

    def get_affinity_levels_of_all_attributes(self) -> list[float]:
        affinity_level_list = []
        for attr in self.attributes:
            affinity_level_list.append(attr.affinity_level)
        return affinity_level_list

    def get_attribute_value_dict(self) -> dict:
        attr_value_dict = dict()
        for attr in self.attributes:
            attr_value_dict[attr.name] = attr.affinity_level

        return attr_value_dict

    def get_attribute_value_list(self) -> list:
        attr_value_list = list()
        for attr in self.attributes:
            attr_value_list.append(attr.affinity_level)

        return attr_value_list

    def print_attributes(self):
        print(self.name, "->", end=" ")
        for i in range(len(self.attributes)):
            if i < len(self.attributes) - 1:
                print(self.attributes[i].name, ": (", self.attributes[i].affinity_level, ")", end=", ")
            else:
                print(self.attributes[i].name, ": (", self.attributes[i].affinity_level, ")")
        # print()