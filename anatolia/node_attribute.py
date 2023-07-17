
class NodeAttribute:

    def __init__(self, name: str, affinity_level: float = None, ver: int = None):
        self.name = name
        self.affinity_level = affinity_level
        self.ver = ver

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, NodeAttribute):
            return self.name == other.name

    def set_affinity_level(self, level_to_set: float):
        self.affinity_level = level_to_set
