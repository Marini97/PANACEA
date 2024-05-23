import networkx as nx
import pandas as pd

# define the structure of a node of the tree
class Node:
    def __init__(self, label, refinement="disjunctive", comment=""):
        self.label = label
        self.refinement = refinement
        self.type, self.action, self.cost, self.time, self.role = self.comment_to_data(comment)
        
    def comment_to_data(self, comment):
        type, action, cost, role, time = "", "", "", "", ""
        
        for line in comment.split('\n'):
            if line.startswith('Type:'):
                type = line.split(': ')[1]
            elif line.startswith('Action:'):
                action = line.split(': ')[1]
            elif line.startswith('Cost:'):
                cost = line.split(': ')[1]
            elif line.startswith('Time:'):
                time = line.split(': ')[1]
            elif line.startswith('Role:'):
                role = line.split(': ')[1]
                
        return type, action, cost, time, role
    
    def to_string(self):
        return "Label: " + self.label + "\nRefinement: " + self.refinement + "\nType: " + self.type + "\nAction: " + self.action + "\nCost: " + self.cost + "\nRole: " + self.role
            
# define the structure of the tree
class Tree:
    def __init__(self):
        self.root = None
        self.nodes = []
        self.edges = []
        
    def add_node(self, node):
        if self.nodes == []:
            self.root = node
        self.nodes.append(node)
        
    def add_edge(self, parent, child):
        self.edges.append(((parent.label, child.label), child.action))
        
    def get_node(self, label):
        for node in self.nodes:
            if node.label == label:
                return node
        return None
    
    def get_parent(self, node):
        for edge, action in self.edges:
            if edge[1] == node.label and action == node.action:
                return edge[0]
        return None
    
    def get_children(self, node):
        children = set()
        for edge, _ in self.edges:
            if edge[0] == node.label:
                children.add(edge[1])
        return children
    
    def to_string(self):
        string = ""
        for node in self.nodes:
            string += node.to_string() + "\n\n"
        return string
    
    def to_graph(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.label, color="Red" if node.role == "Attacker" else "Green")
        for edge, action in self.edges:
            G.add_edge(edge[0], edge[1], action=action)
        return G
    
    def to_dataframe(self):
        data = []
        for node in self.nodes:
            children = self.get_children(node)
            parent = self.get_parent(node)
            data.append([node.label, node.refinement, node.type, node.action, node.cost, node.role, node.time, parent, children])
        return pd.DataFrame(data, columns=["Label", "Refinement", "Type", "Action", "Cost", "Role", "Time", "Parent", "Children"])
    
    def hierarchy_pos(self, G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = self.hierarchy_pos(G, child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
    
    def prune(self, label):
        """
        Prunes the tree but keeps the path to the root.
        If the parents has the refinement "conjunctive" then it keeps the subtree.

        Args:
            label (str): The label of the subtree to keep.
            
        Returns:
            Tree: a new pruned tree object.
        """    
        path = self.get_path_to_node(label)
        tree = Tree()
        for parent in path:
            parent_node = self.get_node(parent)
            if parent_node.refinement == "conjunctive" or parent == label:
                subtree = self.get_subtree(parent)
                tree.nodes += subtree.nodes
                tree.edges += subtree.edges
                break
            else:
                tree.add_node(parent_node)
                children = [c for c in self.get_children(parent_node) if c in path or self.get_node(c).role == "Defender"]
                for child in children:
                    tree.add_node(self.get_node(child))
                    tree.add_edge(parent_node, self.get_node(child))
        return tree
                
            
            
    def get_path_to_node(self, label):
        """
        Returns the path to the root of the node with the given label.
        
        Args:
            label (str): The label of the node.
            
        Returns:
            list: The path to the root.
        """
        path = []
        node = self.get_node(label)
        while node is not None:
            path.append(node.label)
            node = self.get_node(self.get_parent(node))
        return path[::-1]
    
    def get_subtree(self, label):
        """
        Returns the subtree with the root at the node with the given label.

        Args:
            label (str): The label of the root node of the subtree.

        Returns:
            Tree: The subtree.
        """
        tree = Tree()
        queue = [label]
        while queue:
            parent = queue.pop(0)
            parent_node = self.get_node(parent)
            tree.add_node(parent_node)
            children = self.get_children(parent_node)
            for child in children:
                child_node = self.get_node(child)
                tree.add_edge(parent_node, child_node)
                queue.append(child)
                
        return tree
        