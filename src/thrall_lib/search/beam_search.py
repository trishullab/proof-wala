import time
from multiprocessing import Pool
import os
import typing
try:
    from .search import Node, SearchAlgorithm
except ImportError:
    from search import Node, SearchAlgorithm

class BeamSearch(SearchAlgorithm):
    def __init__(self, beam_width: int = 1):
        assert beam_width > 0, "Beam width must be greater than 0"
        super().__init__()
        self.beam_width = beam_width  # The maximum number of nodes to consider at each level

    def search(
            self,
            start: Node, 
            goal: Node,
            heuristic: typing.Callable[[Node], float], 
            generate_children: typing.Callable[[Node], typing.List[Node]] = None,
            parallel_count: int = None,
            build_tree: bool = True,
            timeout_in_secs: float = None) -> typing.Tuple[Node, bool, float]:

        assert (generate_children is not None and build_tree) or not build_tree, "Must provide generate_children function"
        time_elapsed = 0
        start_time = time.time()
        timeout_in_secs = timeout_in_secs if timeout_in_secs else float('inf')

        # Instead of a min heap, use a list to maintain the current level's nodes
        current_level = [(heuristic(start), start)]
        next_level = []
        explored = {start: start}  # Keep track of explored nodes to avoid cycles

        # Initialize multiprocessing pool
        pool = Pool(processes=parallel_count if parallel_count else max(os.cpu_count() - 1, 1))

        while current_level:
            time_elapsed = time.time() - start_time
            if time_elapsed > timeout_in_secs:
                pool.close()
                pool.join()
                return (start, False, time_elapsed)
            tree_map = {n: [] for _, n in current_level}
            level_explored = set()
            for _, current_node in current_level:
                if current_node == goal:
                    pool.close()
                    pool.join()
                    time_elapsed = time.time() - start_time
                    return (current_node, True, time_elapsed)

                if build_tree:
                    children, edges = generate_children(current_node)
                else:
                    children = current_node.children
                children_to_explore = set()
                for idx, child in enumerate(children):
                    if child not in explored and child not in level_explored:
                        level_explored.add(child)
                        children_to_explore.add(child)
                    if build_tree:
                        tree_map[current_node].append((child, edges[idx]))

                child_costs = pool.starmap(heuristic, [[child] for child in children])
                for child, cost in zip(children, child_costs):
                    if child in children_to_explore:
                        next_level.append((cost, child))
            
            current_level = next_level
            next_level = []
            current_level.sort(key=lambda x: x[0])  # Sort by heuristic value
            current_level = current_level[:self.beam_width]  # Keep only the top beam_width nodes
            beam_nodes = set([n for _, n in current_level])
            
            if build_tree:
                # Go over the tree_map and add the children to the current node
                for parent, children in tree_map.items():
                    for child, edge in children:
                        if child in beam_nodes:
                            if child not in explored:
                                parent.add_child(child, edge)
                                explored[child] = child
                            else:
                                parent.add_child(explored[child], edge)


        pool.close()
        pool.join()
        return (start, False, time_elapsed)
