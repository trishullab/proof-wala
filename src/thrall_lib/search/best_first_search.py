import heapq
import os
import typing
import time

from multiprocessing import Pool
try:
    from .search import Node, SearchAlgorithm
except ImportError:
    from search import Node, SearchAlgorithm


class BestFirstSearch(SearchAlgorithm):
    def __init__(self):
        super().__init__()

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
        frontier : typing.List[typing.Tuple[float, Node]] = [(heuristic(start), start)]
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

            children_to_explore = set()
            if build_tree:
                children, edges = generate_children(current_node)
                for idx, child in enumerate(children):
                    if child not in explored:
                        current_node.add_child(child, edges[idx])  # Add child to current node's children list
                        explored[child] = child  # Add child to explored set
                        children_to_explore.add(child)
                    else:
                        child_idx = None
                        try:
                            child_idx = current_node.children.index(child)
                        except ValueError:
                            pass
                        if child_idx is None:
                            current_node.add_child(explored[child], edges[idx])
                        elif edges[idx] != current_node.edges[child_idx]:
                            if edges[idx] not in current_node.edges[child_idx].equivalent_edges:
                                current_node.edges[child_idx].add_equivalent_edge(edges[idx])
            else:
                children = [child for child in current_node.children]
                for child in children:
                    if child not in explored:
                        explored[child] = child
                        children_to_explore.add(child)
            unique_children = list(children_to_explore)
            child_costs = pool.starmap(heuristic, [[child] for child in unique_children])

            for child, cost in zip(unique_children, child_costs):
                child.score = cost
                heapq.heappush(frontier, (cost, child))
            
            time_elapsed = time.time() - start_time

        pool.close()
        pool.join()
        return (start, False, time_elapsed)