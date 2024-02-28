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
                        # If the child is already explored, we need to update the parent to the current node
                        # This is necessary for keeping track of the tree structure
                        current_node.add_child(explored[child], edges[idx])
            else:
                children = [child for child in current_node.children]
                for child in children:
                    if child not in explored:
                        explored[child] = child
                        children_to_explore.add(child)

            child_costs = pool.starmap(heuristic, [[child] for child in children])

            children_added_to_frontier = set()
            for child, cost in zip(children, child_costs):
                if child in children_to_explore and child not in children_added_to_frontier:
                    child.score = cost
                    heapq.heappush(frontier, (heuristic(child), child))
                    children_added_to_frontier.add(child)
            
            time_elapsed = time.time() - start_time

        pool.close()
        pool.join()
        return (start, False, time_elapsed)