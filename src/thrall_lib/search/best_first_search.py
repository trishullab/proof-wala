import heapq
import os
import typing
import time

from multiprocessing import Pool
try:
    from .search import Node, Edge, SearchAlgorithm
except ImportError:
    from search import Node, Edge, SearchAlgorithm


class BestFirstSearch(SearchAlgorithm):
    def __init__(self):
        super().__init__()

    def search(
            self,
            start: Node, 
            goal: Node,
            heuristic: typing.Callable[[Node, Edge, Node], float], 
            generate_children: typing.Callable[[Node], typing.Tuple[typing.List[Node], typing.List[Edge]]] = None,
            parallel_count: int = None,
            build_tree: bool = True,
            timeout_in_secs: float = None) -> typing.Tuple[Node, bool, float]:

        assert (generate_children is not None and build_tree) or not build_tree, "Must provide generate_children function"
        time_elapsed = 0
        start_time = time.time()
        timeout_in_secs = timeout_in_secs if timeout_in_secs else float('inf')
        start.cummulative_score = 0
        start.distance_from_root = 0
        frontier : typing.List[typing.Tuple[float, Node]] = [(heuristic(None, None, start), start)]
        explored = {start: start}  # Now keeping track of explored nodes directly
        heapq.heapify(frontier) # this is a min heap

        # Initialize multiprocessing pool
        pool = Pool(processes=parallel_count if parallel_count else max(os.cpu_count() - 1, 1))
        while frontier:
            _, current_node = heapq.heappop(frontier) # Pop the node with the lowest score

            if current_node == goal:
                pool.close()
                pool.join()
                assert len(current_node.parents) > 0, f"Goal node must have at least one parent"
                return (current_node, True, time_elapsed)

            if time_elapsed > timeout_in_secs:
                pool.close()
                pool.join()
                return (start, False, time_elapsed)

            child_edge_set : typing.Set[typing.Tuple[Node, Edge]] = set()
            if build_tree:
                children, edges = generate_children(current_node)
                for child, edge in zip(children, edges):
                    if child not in explored:
                        child.distance_from_root = min(child.distance_from_root, current_node.distance_from_root + 1)
                        child.cummulative_score = min(child.cummulative_score, current_node.cummulative_score + edge.score)
                        current_node.add_child(child, edge)  # Add child to current node's children list
                        explored[child] = child  # Add child to explored set
                        child_edge_set.add((child, edge))
                    else:
                        child_idx = None
                        try:
                            child_idx = current_node.children.index(child)
                        except ValueError:
                            pass
                        if child_idx is None:
                            explored_child = explored[child]
                            explored_child.distance_from_root = min(explored_child.distance_from_root, current_node.distance_from_root + 1)
                            explored_child.cummulative_score = min(explored_child.cummulative_score, current_node.cummulative_score + edge.score)
                            current_node.add_child(explored_child, edge)
                            child_edge_set.add((explored_child, edge))
                        elif edge != current_node.edges[child_idx]:
                            if edge not in current_node.edges[child_idx].equivalent_edges:
                                current_node.edges[child_idx].add_equivalent_edge(edge)
                                explored_child = current_node.children[child_idx]
                                if explored_child.cummulative_score > current_node.cummulative_score + edge.score:
                                    if (explored_child, current_node.edges[child_idx]) in child_edge_set:
                                        child_edge_set.remove((explored_child, current_node.edges[child_idx]))
                                    child_edge_set.add((explored_child, edge))
                                explored_child.distance_from_root = min(explored_child.distance_from_root, current_node.distance_from_root + 1)
                                explored_child.cummulative_score = min(explored_child.cummulative_score, current_node.cummulative_score + edge.score)
                                
            else:
                children = [(child, edge) for child, edge in zip(current_node.children, current_node.edges)]
                for child, edge in children:
                    if child not in explored:
                        explored[child] = child
                        child_edge_set.add((child, edge))
            unique_children_edges = list(child_edge_set)
            child_costs = pool.starmap(heuristic, [[current_node, edge, node] for node, edge in unique_children_edges])

            for (child, _), cost in zip(unique_children_edges, child_costs):
                child.score = cost
                heapq.heappush(frontier, (cost, child))
            
            time_elapsed = time.time() - start_time

        pool.close()
        pool.join()
        return (start, False, time_elapsed)