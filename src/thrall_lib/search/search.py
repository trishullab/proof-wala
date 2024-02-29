import typing
from graphviz import Digraph
from abc import ABC, abstractmethod

def escape_string(string: str):
    escaped_string = string.replace('\n', '\\n')
    return escaped_string

class Edge:
    def __init__(self, label: str, score: float = 0, other_data: typing.Any = None):
        assert isinstance(label, str), "Edge label must be a string"
        assert isinstance(score, (int, float)), "Edge score must be a number"
        self.label = label
        self.score = float(score)
        self.other_data = other_data
        self.equivalent_edges : typing.List['Edge'] = []
    
    def add_equivalent_edge(self, edge: 'Edge'):
        self.equivalent_edges.append(edge)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.label == other.label
    
    def __hash__(self):
        return hash(self.label)
    
    def __str__(self) -> str:
        return f"Edge({self.label}, {self.score})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Node:
    def __init__(self, name: str, score: float = 0, other_data: typing.Any = None):
        assert isinstance(name, str), "Node name must be a string"
        assert isinstance(score, (int, float)), "Node score must be a number"
        self.name = name
        self.score = float(score)
        self.other_data = other_data
        # It is very important to make sure that the members below are lists
        # and not sets, because hash will fail when this objects gets pickled
        # and sent to another process while using multiprocessing
        self.parents : typing.List['Node'] = []  # Keep track of parents for tree representation
        self.children : typing.List['Node'] = []  # Keep track of children for tree representation
        self.edges : typing.List[Edge] = []  # Keep track of edges for graph representation

    def add_child(self, child: 'Node', edge: Edge = None):
        """
        Note: Do not add the same child node to the children list more than once.
        We cannot check for duplicates here because it would be too slow.
        """
        self.children.append(child)
        child.parents.append(self)
        self.edges.append(edge)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self) -> str:
        return f"Node({self.name}, {self.score})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __lt__(self, other):
        assert isinstance(other, Node), "Can only compare Node objects"
        return self.score < other.score
    
    def __le__(self, other):
        assert isinstance(other, Node), "Can only compare Node objects"
        return self.score <= other.score
    
    def __gt__(self, other):
        assert isinstance(other, Node), "Can only compare Node objects"
        return self.score > other.score
    
    def __ge__(self, other):
        assert isinstance(other, Node), "Can only compare Node objects"
        return self.score >= other.score

class SearchAlgorithm(ABC):
    def __init__(self):
        pass

    def reconstruct_path(self, start_node: Node, goal_node: Node) -> typing.List[Node]:
        """
        Iteratively reconstructs a path from start_node to goal_node, avoiding cycles.
        
        :param start_node: The starting node.
        :param goal_node: The goal node.
        :return: A list of nodes representing the path, or an empty list if no path exists.
        """
        if start_node == goal_node:
            return [start_node]

        visited = set()
        stack = [(start_node, [start_node])]  # Stack holds tuples of (current_node, path_to_node)

        while stack:
            current_node, path = stack.pop()
            visited.add(current_node)

            for child in current_node.children:
                if child in visited:
                    continue  # Skip visited nodes to avoid cycles
                
                if child == goal_node:
                    return path + [child]  # Return the path including the goal node
                
                stack.append((child, path + [child]))

        return []  # No path found if we reach this point
    
    def reconstruct_all_paths(self, start_node: Node, goal_node: Node) -> typing.List[typing.List[Node]]:
        """
        Iteratively reconstructs all paths from start_node to goal_node, handling cycles.
        
        :param start_node: The starting node for paths.
        :param goal_node: The goal node to reach.
        :return: A list of lists, where each inner list represents a path of nodes from start to goal.
        """
        all_paths = []
        queue = [(start_node, [start_node])]  # Queue holds tuples of (current_node, path_to_node)
        
        while queue:
            current_node, path = queue.pop(0)
            
            # If the current node is the goal, add the current path to the list of all paths
            if current_node == goal_node:
                all_paths.append(path)
                continue
            
            # Explore children, avoiding cycles by checking if the child is already in the current path
            for child in current_node.children:
                if child not in path:  # Check to avoid revisiting nodes in the current path (simple cycle avoidance)
                    new_path = path + [child]
                    queue.append((child, new_path))

        return all_paths

    def visualize_search(self, 
        root: Node, 
        save_to_file: str = None, 
        show: bool = True,
        mark_paths: typing.List[typing.List[Node]] = []) -> Digraph:
        """
        Visualizes the tree structure starting from the given root node using Graphviz.
        
        :param root: The root node of the tree to visualize.
        """
        dot = Digraph()
        node_queue = [root]
        visited = set()
        edges = set()
        text_width = 75
        node_num = 0
        unique_node_names = {}
        nodes_in_paths = set()
        edges_in_paths = set()
        for path in mark_paths:
            for i, node in enumerate(path):
                nodes_in_paths.add(node.name)
                if i < len(path) - 1:
                    edges_in_paths.add((node.name, path[i + 1].name))
        while node_queue:
            current_node = node_queue.pop(0)
            full_node_name = current_node.name
            node_name = (current_node.name[:text_width] + "...") if len(current_node.name) > text_width else current_node.name
            node_name = escape_string(node_name)
            if full_node_name not in unique_node_names:
                unique_node_names[full_node_name] = node_num
                node_num += 1
            full_node_num = unique_node_names[full_node_name]
            if full_node_name in nodes_in_paths:
                dot.node(str(full_node_num), label=node_name, style='filled', fillcolor='lightblue')
            else:
                dot.node(str(full_node_num), label=node_name)
            
            for child_idx, child in enumerate(current_node.children):
                # Add edge from parent to child
                if (current_node, child) not in edges:
                    edges.add((current_node, child))
                    child_name = child.name
                    if child_name not in unique_node_names:
                        unique_node_names[child_name] = node_num
                        node_num += 1
                    child_num = unique_node_names[child_name]
                    edg = current_node.edges[child_idx]
                    edge_labels = []
                    if edg is not None:
                        eqv_edges = edg.equivalent_edges + [edg]
                        for eq_edge in eqv_edges:
                            edge_label = eq_edge.label
                            edge_label = edge_label[:text_width] + "..." if len(edge_label) > text_width else edge_label
                            edge_label = escape_string(edge_label)
                            edge_labels.append(edge_label)
                    else:
                        edge_labels = [None]
                    for edge_label in edge_labels:
                        if (current_node.name, child.name) in edges_in_paths:
                            dot.edge(str(full_node_num), str(child_num), label=edge_label, color='red', penwidth='2.0')
                        else:
                            dot.edge(str(full_node_num), str(child_num), label=edge_label)
                if child not in visited:
                    # Add child to the queue to process its children later
                    node_queue.append(child)
            visited.add(current_node)
        
        # Render the graph to a file (e.g., 'tree.png')
        if save_to_file:
            dot.render(save_to_file, format='png', quiet=True)
        if show:
            dot.view()
        return dot

    @abstractmethod
    def search(
            start: Node, 
            goal: Node,
            heuristic: typing.Callable[[Node], float], 
            generate_children: typing.Callable[[Node], typing.Tuple[typing.List[Node], typing.List[Edge]]] = None,
            parallel_count: int = None,
            build_tree: bool = True,
            timeout_in_secs: float = None) -> typing.Tuple[Node, bool, float]:
        """
        Abstract method for implementing search algorithms.
        :param start: The start node for the search.
        :param goal: The goal node for the search.
        :param heuristic: A function that takes a Node and returns a float heuristic value for that node.
        It should return min value for the node which should be explored first.
        :param generate_children: A function that takes a Node and returns a list of child nodes, or None if build_tree is False.
        :param parallel_count: An optional integer for the number of parallel processes to use.
        :param build_tree: A boolean indicating whether to build a tree while searching, default True. 
        Set to False only if the tree is already built and children are already added to nodes.
        :param timeout_in_secs: An optional float for the maximum time to run the search.


        :return: Returns a tuple of the form (Node, bool, float), 
        where the first element is the goal node found in the tree (or the start node if not found),
        the second element is a boolean indicating whether the goal was found, and the third element is the time elapsed.
        """
        pass