import networkx as nx
import pandas as pd

# define the structure of a node of the tree
class Node:
    def __init__(self, label, refinement="disjunctive", comment=""):
        self.label = label
        self.refinement = refinement
        self.type, self.action, self.cost, self.role = self.comment_to_data(comment)
        
    def comment_to_data(self, comment):
        type, action, cost, role = "", "", "", ""
        
        for line in comment.split('\n'):
            if line.startswith('Type:'):
                type = line.split(': ')[1]
            elif line.startswith('Action:'):
                action = line.split(': ')[1]
            elif line.startswith('Cost:'):
                cost = line.split(': ')[1]
            elif line.startswith('Role:'):
                role = line.split(': ')[1]
                
        return type, action, cost, role
    
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
            data.append([node.label, node.refinement, node.type, node.action, node.cost, node.role, parent, children])
        return pd.DataFrame(data, columns=["Label", "Refinement", "Type", "Action", "Cost", "Role", "Parent", "Children"])
    