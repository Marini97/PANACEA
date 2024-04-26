import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tree import Node, Tree

# Get the children of a node
def parse_children(node):
    children = []
    for child in node:
        if child.tag == 'node':
            children.append(child)
    return children

# Parse a node
def parse_node(node):
    refinement = node.attrib['refinement']
    label = node.find('label').text.replace(" ", "")
    comment = node.find('comment').text
    return Node(label, refinement, comment)

# Parse xml file to tree
def parse_file(file):
    xml = ET.parse(file) # 'tree/Data_Exfiltration.xml'
    r = xml.find('node')

    root = parse_node(r)
    
    tree = Tree()
    queue = [(root, r)]

    while queue:
        parent_node, parent = queue.pop()
        tree.add_node(parent_node)
        
        children = parse_children(parent)
        for child in children:
            child_node = parse_node(child)
            queue = [(child_node, child)] + queue
            tree.add_edge(parent_node, child_node)
            
    return tree


# Get info from the dataframe
def get_info(df):
    goal = df.loc[df["Type"] == "Goal"]["Label"].values[0]
    actions_to_goal = set(df.loc[df["Parent"] == goal]["Action"].values)

    df_attacker = df.loc[df["Role"] == "Attacker"]
    df_attacker = df_attacker.loc[df_attacker["Type"] != "Goal"]

    df_defender = df.loc[df["Role"] == "Defender"]

    list_initial = df_attacker.loc[df_attacker["Action"] == ""]["Label"].to_list()
    df_attacker = df_attacker.loc[df_attacker["Action"] != ""]

    # df actions with preconditions, effect and costs
    attacker_actions = {}
    defender_actions = {}

    for _, row in df.iterrows():
        action = row["Action"]
        
        if action == "":
            continue
        
        effect = row["Parent"]
        cost = row["Cost"]
        refinement = df.loc[df['Label'] == effect]["Refinement"].values[0]
        
        if refinement == "conjunctive" and row["Type"] == "Attribute":
            preconditions = df.loc[row["Parent"] == df["Label"]]["Children"].values[0]
        elif row["Type"] == "Attribute":
            preconditions = [row["Label"]]
        else:
            preconditions = []

        if row["Role"] == "Attacker" and action not in attacker_actions:
            preconditions = [p for p in preconditions if p not in df_defender["Label"].values]
            attacker_actions[action] = {
                "preconditions" : preconditions, 
                "effect" : effect, 
                "cost" : cost,
                "refinement" : refinement}
        elif action in attacker_actions.keys():
            attacker_actions[action]["preconditions"] += preconditions
        elif row["Role"] == "Defender" and action not in defender_actions:
                defender_actions[action] = {
                    "preconditions" : preconditions, 
                    "effect" : effect, 
                    "cost" : cost,
                    "refinement" : refinement}
                
    return goal, actions_to_goal, list_initial, attacker_actions, defender_actions, df_attacker, df_defender


# Get the string of the prism model
def get_prism_model(tree):
    df = tree.to_dataframe()
    goal, actions_to_goal, list_initial, attacker_actions, defender_actions, df_attacker, df_defender = get_info(df)
    text = "smg\n\nplayer attacker\n\tattacker,\n\t"

    for a in attacker_actions.keys():
        text += f"[{a}], "
        
    text = text[:-2]
    text += "\nendplayer\nplayer defender\n\tdefender,\n\t"

    for a in defender_actions.keys():
        text += f"[{a}], "
    
    text = text[:-2]   
    text += "\nendplayer\n\nglobal sched : [1..2];\n\n"

    text += f'global {goal} : bool;\nlabel "terminate" = {goal}=true;\n\n'

    for a in set(df_attacker.loc[df_attacker["Type"] == "Attribute"]["Label"].values):
        text += "global " + a + " : bool;\n"
        
    for a in set(list_initial):
        text += "global " + a + " : bool init true;\n"

    text += "\nmodule attacker\n\n"

    for a in set(df_attacker["Action"].values):
        text += f"\t{a} : bool;\n"
        
    text += "\n"

    for a in attacker_actions.keys():
        preconditions = attacker_actions[a]["preconditions"]
        effect = attacker_actions[a]["effect"]
        if attacker_actions[a]["refinement"] == "disjunctive":
            refinement = "|"
        else:
            refinement = "&"
            
        text += f"\t[{a}] sched=1 & !{goal} & !{effect} & !{a}"
        # check if the node is a leaf
        if preconditions != []:
            text += " & ("
            for p in set(preconditions):
                text += f"{p} {refinement} "
            text = f"{text[:-3]})"
        
        text += f" -> ({effect}'=true) & ({a}'=true) & (sched'=2);\n"
        
    text += "\nendmodule\n\nmodule defender\n\n"

    defender_attributes = set(df_defender.loc[df_defender["Type"] == "Attribute"]["Label"].values)
    for a in defender_attributes:
        text += f"\t{a} : bool;\n"
        
    text += "\n"
        
    for a in defender_actions.keys():
        preconditions = defender_actions[a]["preconditions"]
        effect = defender_actions[a]["effect"]
        if defender_actions[a]["refinement"] == "disjunctive":
            refinement = "|"
        else:
            refinement = "&"
            
        if effect in defender_attributes:
            text += f"\t[{a}] sched=2 & !{goal} & !{effect}"
            if preconditions != []:
                text += " & ("
                for p in set(preconditions):
                    text += f"{p} {refinement} "
                text = f"{text[:-3]})"
            text += f" -> ({effect}'=true) & (sched'=1);\n"
        else:
            text += f"\t[{a}] sched=2 & !{goal} & {effect}"
            if preconditions != []:
                text += " & ("
                for p in set(preconditions):
                    text += f"{p} {refinement} "
                text = f"{text[:-3]})"
            text += f" -> ({effect}'=false) & (sched'=1);\n"
        
    text += '\nendmodule\n\nrewards "attacker"\n\n'

    for a in attacker_actions.keys():
        text += f"\t[{a}] true : {attacker_actions[a]['cost']};\n"
        
    text += '\nendrewards\n\nrewards "defender"\n\n'

    for a in actions_to_goal:
        text += f"\t[{a}] true : 500;\n"
    for a in defender_actions.keys():
        text += f"\t[{a}] true : {defender_actions[a]['cost']};\n"
          
    text += "\nendrewards"

    return text

def save_prism_model(prism_model, file):
    with open(file, 'w') as f:
        f.write(prism_model)
        f.close()
    save_prism_properties(file.replace('.prism', '.props'))
    
def save_prism_properties(file):
    with open(file, 'w') as f:
        f.write('// The minimum probability to reach a state labeled "terminate"\n')
        f.write('Pmin=? [ F "terminate" ]\n')
        f.write('// The minimum expected cost that attacker and defender can guarantee to reach a state labeled "terminate"\n')
        f.write('<<attacker,defender>>R{"attacker"}min=? [ F "terminate" ]\n')
        f.write('// Each agent tries to get the minimum expected cost to reach a terminate state\n')
        f.write('<<attacker,defender>>R{"attacker"}min=? [ F "terminate" ] + R{"defender"}min=? [ F "deadlock" ]\n')
        f.close()
    
