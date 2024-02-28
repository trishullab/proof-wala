import typing
import random
import os
import tracemalloc
from datetime import datetime
try:
    from .best_first_search import BestFirstSearch
    from .beam_search import BeamSearch
    from .search import Node, Edge, SearchAlgorithm
except ImportError:
    from best_first_search import BestFirstSearch
    from beam_search import BeamSearch
    from search import Node, Edge, SearchAlgorithm

class Heuristic:
    def __init__(self, goal_value: int):
        self.goal_value = goal_value

    def __call__(self, node: Node) -> float:
        return abs(int(node.name) - self.goal_value)

class GenerateChildren:
    def __init__(self, start_value: int, goal_value: int, goal_level: int):
        self.goal_value = goal_value
        self.goal_level = goal_level
        self._start_node = Node(str(start_value))
        self._node_map = {}
        self._parent_map = {self._start_node: []}
        self._level = 0

    def __call__(self, node: Node) -> typing.Tuple[typing.List[Node], typing.List[Edge]]:
        """
        Generates a random number of child nodes for a given node. Each child node's value is
        either +1, +2, +3, +4, +5, .., +50 from the parent node's value, chosen randomly.
        """
        if node.name in self._node_map:
            new_node, _ = self._node_map[node.name]
            assert isinstance(new_node.children, list), "Children must be a list"
            return new_node.children, new_node.edges
        else:
            parent_value = int(node.name)
            num_children = random.randint(0, 30)  # Randomly choose how many children to generate (0 to 3 for this example)
            children = []
            edges = []
            unique_values = set()  # Keep track of unique child values
            for _ in range(num_children):
                increment = random.choice([i for i in range(51)])  # Choose either +0, +1, +2, +3, +4, .., +49
                child_value = parent_value + increment
                if child_value not in unique_values and child_value != self.goal_value:
                    unique_values.add(child_value)
                    children.append(Node(str(child_value)))
                    edges.append(Edge(str(increment), 0, None))
            parent_map = self._parent_map[node]
            self._level = 0 if len(parent_map) == 0 else max([self._node_map[parent.name][1] for parent in parent_map]) + 1
            if self._level == self.goal_level:
                children.append(Node(str(self.goal_value)))
                edges.append(Edge(str(self.goal_value - parent_value), 0, None))
            if self._level > self.goal_level:
                children = []
            if self._level != self.goal_level:
                assert all([int(child.name) != self.goal_value for child in children]), "Goal value should not be in children"
            new_node = Node(str(parent_value))
            cloned_children = [Node(child.name) for child in children]
            for child in cloned_children:
                new_node.add_child(child)
                if child not in self._parent_map:
                    self._parent_map[child] = [new_node]
                else:
                    self._parent_map[child].append(new_node)
            self._node_map[node.name] = (new_node, self._level)
            assert isinstance(children, list), "Children must be a list"
            return children, edges

def test_search_algorithm(algo: SearchAlgorithm, start_value: int = 0, goal_value: int = 10, goal_level: int = 3, timeout_in_secs: float = 60, generate_children: typing.Callable[[Node], typing.List[Node]] = None) -> typing.Callable[[Node], typing.List[Node]]:
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    # print("Current memory usage:", current / 10**6, "MB")
    # print("Peak memory usage:", peak / 10**6, "MB")
    start_memory = current
    start_peak = peak
    start_node = Node(str(start_value))
    goal_node = Node(str(goal_value))

    heuristic = Heuristic(goal_value)  # Generate the heuristic function based on the goal value
    # Generate the dynamic child generation function
    generate_children = GenerateChildren(start_value, goal_value, goal_level) if generate_children is None else generate_children 

    print("Testing {}...".format(algo.__class__.__name__))
    # Run the search algorithm with the generated heuristic and dynamic child generation
    goal_node_in_tree, found, time = algo.search(start_node, goal_node, heuristic, generate_children, 4, True, timeout_in_secs=timeout_in_secs)
    print("Search complete")
    print("Time elapsed:", time, "seconds")
    # print("Memory usage:", current / 10**6, "MB")
    # print("Peak memory usage:", peak / 10**6, "MB")
    # print("Memory usage change:", (current - start_memory) / 10**6, "MB")
    current, peak = tracemalloc.get_traced_memory()
    print("Peak memory usage change:", (peak - start_peak) / 10**6, "MB")
    tracemalloc.stop()
    if found:
        path = algo.reconstruct_path(start_node, goal_node_in_tree)
        # print("Path to goal:", [node.name for node in path])
        all_paths = algo.reconstruct_all_paths(start_node, goal_node_in_tree)
        if len(all_paths) > 1:
            print("Found multiple paths to goal:")
            # for path in all_paths:
            #     print([node.name for node in path])
        else:
            one_path = all_paths[0]
            assert len(path) == len(one_path), "Reconstructed path length does not match"
            assert all([node.name == path_node.name for node, path_node in zip(path, one_path)]), "Reconstructed path does not match"
            print("Only one path found to goal")
    else:
        all_paths = []
        print("No path found to goal")
    dot = algo.visualize_search(start_node, show=False, mark_paths=all_paths)
    print("Saving visualization...")
    os.makedirs(".log/search/test", exist_ok=True)
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = ".log/search/test/{}_{}".format(algo.__class__.__name__, time_now)
    dot.render(file_name, format='png', quiet=True)
    return generate_children # Return the dynamic child generation function for testing testing other algorithms

# Example usage
if __name__ == "__main__":
    random.seed(0xfacade)
    # See the memory usage of the search algorithms
    children_map = test_search_algorithm(BestFirstSearch(), 0, 1000, 12, timeout_in_secs=100)
    random.seed(0xfacade)
    test_search_algorithm(BeamSearch(2), 0, 1000, 12, timeout_in_secs=100, generate_children=children_map)