import time
from multiprocessing import Pool
import os
import typing
import logging
try:
    from .search import Node, Edge, SearchAlgorithm
except ImportError:
    from search import Node, Edge, SearchAlgorithm

class BeamSearch(SearchAlgorithm):
    def __init__(self, beam_width: int = 1):
        assert beam_width > 0, "Beam width must be greater than 0"
        super().__init__()
        self.beam_width = beam_width  # The maximum number of nodes to consider at each level

    def search(
            self,
            start: Node, 
            goal: Node,
            heuristic: typing.Callable[[Node, Edge, Node], float], 
            generate_children: typing.Callable[[Node, float], typing.Tuple[typing.List[Node], typing.List[Edge]]] = None,
            parallel_count: int = None,
            build_tree: bool = True,
            timeout_in_secs: float = None,
            logger: logging.Logger = None) -> typing.Tuple[Node, bool, float]:

        assert (generate_children is not None and build_tree) or not build_tree, "Must provide generate_children function"
        time_elapsed = 0
        start_time = time.time()
        timeout_in_secs = timeout_in_secs if timeout_in_secs else float('inf')
        logger = logger if logger else logging.getLogger(__name__)

        # Instead of a min heap, use a list to maintain the current level's nodes
        current_level = [(heuristic(None, None, start), start)]
        next_level = []
        explored = set([start])  # Keep track of explored nodes to avoid cycles

        # Initialize multiprocessing pool
        pool = Pool(processes=parallel_count if parallel_count else max(os.cpu_count() - 1, 1))
        start.cummulative_score = 0
        start.distance_from_root = 0
        tree_nodes : typing.Dict[Node, Node] = {}  # Map of all nodes in the tree
        tree_nodes[start] = start
        should_stop = False
        while current_level:
            time_elapsed = time.time() - start_time
            if time_elapsed > timeout_in_secs:
                pool.close()
                pool.join()
                return (start, False, time_elapsed)
            new_tree_nodes : typing.List[typing.Tuple[Node, Node, Edge]] = []

            # For early stopping, check if the goal node is in the current level
            logger.info(f"Frontier size: {len(current_level)}")
            # logger.info("Dumping the frontier nodes")
            # logger.info("-" * 50)
            for _, current_node in current_level:
                # logger.info(f"Frontier node distance from root: {current_node.distance_from_root}, frontier node:\n {current_node}")
                if current_node == goal:
                    pool.close()
                    pool.join()
                    time_elapsed = time.time() - start_time
                    assert len(current_node.parents) > 0, f"Goal node must have at least one parent"
                    return (current_node, True, time_elapsed)
            # logger.info("Dumped the frontier")
            # logger.info("-" * 50)

            for _, current_node in current_level:
                if build_tree:
                    remaining_timeout = timeout_in_secs - time_elapsed
                    children, edges = generate_children(current_node, remaining_timeout)
                    time_elapsed = time.time() - start_time
                else:
                    children = current_node.children
                    edges = current_node.edges
                children_to_explore : typing.Set[typing.Tuple[Node, Edge]] = set()
                for child, edge in zip(children, edges):
                    if child not in explored:
                        children_to_explore.add((child, edge))
                    if build_tree:
                        new_tree_nodes.append((current_node, child, edge))

                unique_children_edge = list(children_to_explore)
                child_costs = pool.starmap(heuristic, [[current_node, edge, child] for child, edge in unique_children_edge])
                for (child, _), cost in zip(unique_children_edge, child_costs):
                    if child not in tree_nodes:
                        next_level.append((cost, child))
                # Iterate over the children to check if we found the goal node for early stopping
                for child, edge in zip(children, edges):
                    if child == goal:
                        should_stop = True
                if should_stop:
                    break
            
            current_level = next_level
            next_level = []
            current_level.sort(key=lambda x: x[0])  # Sort by heuristic value
            current_level = current_level[:self.beam_width]  # Keep only the top beam_width nodes
            beam_nodes = set([n for _, n in current_level])
            nodes_part_of_tree = []
            if build_tree:
                # Go over the tree_map and add the children to the current node
                for parent, child, edge in new_tree_nodes:
                    parent_distance = parent.distance_from_root
                    parent_cummulative = parent.cummulative_score
                    if child in beam_nodes:
                        if child not in tree_nodes:
                            child.distance_from_root = parent_distance + 1
                            child.cummulative_score = parent_cummulative + edge.score
                            parent.add_child(child, edge)
                            tree_nodes[child] = child
                            explored.add(child)
                            nodes_part_of_tree.append(child)
                        else:
                            child_idx = None
                            try:
                                child_idx = parent.children.index(child)
                            except ValueError:
                                pass
                            if child_idx is None:
                                existing_child = tree_nodes[child]
                                existing_child.distance_from_root = min(existing_child.distance_from_root, parent_distance + 1)
                                existing_child.score = min(existing_child.score, child.score)
                                existing_child.cummulative_score = min(existing_child.cummulative_score, parent_cummulative + edge.score)
                                parent.add_child(existing_child, edge)
                                nodes_part_of_tree.append(existing_child)
                            elif edge != parent.edges[child_idx]:
                                if edge not in parent.edges[child_idx].equivalent_edges:
                                    parent.edges[child_idx].add_equivalent_edge(edge)
                                    existing_child = parent.children[child_idx]
                                    existing_child.distance_from_root = min(existing_child.distance_from_root, parent_distance + 1)
                                    existing_child.score = min(existing_child.score, child.score)
                                    existing_child.cummulative_score = min(existing_child.cummulative_score, parent_cummulative + edge.score)
                current_level = [(node.cummulative_score, node) for node in nodes_part_of_tree]
                current_level.sort(key=lambda x: x[0])
    
        pool.close()
        pool.join()
        return (start, False, time_elapsed)
