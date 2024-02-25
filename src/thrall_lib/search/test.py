import typing
import random
import os
from datetime import datetime
try:
    from .best_first_search import BestFirstSearch
    from .search import Node
except ImportError:
    from best_first_search import BestFirstSearch
    from search import Node

class Heuristic:
    def __init__(self, goal_value: int):
        self.goal_value = goal_value

    def __call__(self, node: Node) -> float:
        return abs(int(node.name) - self.goal_value)

def generate_children(node: Node) -> typing.List[Node]:
    """
    Generates a random number of child nodes for a given node. Each child node's value is
    either +1, +2, +3 from the parent node's value, chosen randomly. The number of children
    can be 0, 1, 2, 3 or more, also chosen randomly.
    """
    parent_value = int(node.name)
    num_children = random.randint(0, 15)  # Randomly choose how many children to generate (0 to 3 for this example)
    children = []
    unique_values = set()  # Keep track of unique child values
    for _ in range(num_children):
        increment = random.choice([0, 1, 2, 3])  # Choose either +0, +1, +2 or +3
        child_value = parent_value + increment
        if child_value not in unique_values:
            unique_values.add(child_value)
            children.append(Node(str(child_value)))

    return children

# Example usage
if __name__ == "__main__":
    start_value = 0  # Starting node value
    goal_value = 10  # Goal node value (the integer to reach)

    start_node = Node(str(start_value))
    goal_node = Node(str(goal_value))
    algorithm = BestFirstSearch()

    heuristic = Heuristic(goal_value)  # Generate the heuristic function based on the goal value

    # Run best first search with the generated heuristic and dynamic child generation
    goal_node_in_tree, found, time = algorithm.search(start_node, goal_node, heuristic, generate_children, 4, True, timeout_in_secs=60)
    print("Search complete")
    print("Time elapsed:", time)
    dot = algorithm.visualize_search(start_node, show=False)
    if found:
        path = algorithm.reconstruct_path(start_node, goal_node_in_tree)
        print("Path to goal:", [node.name for node in path])
        all_paths = algorithm.reconstruct_all_paths(start_node, goal_node_in_tree)
        algorithm.mark_paths_in_visualization(dot, all_paths)
        if len(all_paths) > 1:
            print("Found multiple paths to goal:")
            for path in all_paths:
                print([node.name for node in path])
        else:
            one_path = all_paths[0]
            assert len(path) == len(one_path), "Reconstructed path length does not match"
            assert all([node.name == path_node.name for node, path_node in zip(path, one_path)]), "Reconstructed path does not match"
            print("Only one path found to goal")
    else:
        print("No path found to goal")
    print("Showing visualization...")
    os.makedirs(".log/search/test", exist_ok=True)
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = ".log/search/test/best_first_search_{}".format(time_now)
    dot.render(file_name, format='png', quiet=True)