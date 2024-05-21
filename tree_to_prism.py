import xml.etree.ElementTree as ET
from tree import Node, Tree

def parse_children(node):
    """
    Parses the children of a given node and returns a list of child nodes.

    Args:
        node (Element): The parent node whose children need to be parsed.

    Returns:
        list: A list of child nodes.

    """
    children = []
    for child in node:
        if child.tag == 'node':
            children.append(child)
    return children

def parse_node(node):
    """
    Parses a node element and returns a Node object.

    Args:
        node (Element): The XML element representing a node.

    Returns:
        Node: The parsed Node object.

    """
    refinement = node.attrib['refinement']
    label = node.find('label').text.replace(" ", "")
    try:
        comment = node.find('comment').text
    except AttributeError:
        comment = ""
    return Node(label, refinement, comment)

def parse_file(file):
    """
    Parses an XML file and constructs a tree representation.

    Args:
        file (str): The path to the XML file.

    Returns:
        Tree: The constructed tree object.
    """
    xml = ET.parse(file)
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

def get_info(df):
    """
    Extracts information from a DataFrame and returns relevant data.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the information.

    Returns:
        tuple: A tuple containing the following elements:
            - goal (str): The goal extracted from the DataFrame.
            - actions_to_goal (set): A set of actions leading to the goal.
            - list_initial (list): A list of initial labels.
            - attacker_actions (dict): A dictionary of attacker actions with their properties.
            - defender_actions (dict): A dictionary of defender actions with their properties.
            - df_attacker (pandas.DataFrame): The filtered DataFrame for attacker actions.
            - df_defender (pandas.DataFrame): The filtered DataFrame for defender actions.
    """
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
    print(goal)
    for _, row in df.iterrows():
        action = row["Action"]
        
        if action == "":
            continue
        
        effect = row["Parent"]
        cost = row["Cost"]
        refinement = df.loc[df['Label'] == effect]["Refinement"].values[0]
        time = row["Time"]
        
        if refinement == "conjunctive" and row["Type"] == "Attribute":
            preconditions = df.loc[row["Parent"] == df["Label"]]["Children"].values[0]
            preconditions = [p for p in preconditions if row["Role"]==df.loc[df["Label"] == p]["Role"].values[0]]
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
                "time" : time,
                "refinement" : refinement}
        elif action in attacker_actions.keys():
            attacker_actions[action]["preconditions"] += preconditions
        elif row["Role"] == "Defender" and action not in defender_actions:
                defender_actions[action] = {
                    "preconditions" : preconditions, 
                    "effect" : effect, 
                    "cost" : cost,
                    "time" : time,
                    "refinement" : refinement}
                
    return goal, actions_to_goal, list_initial, attacker_actions, defender_actions, df_attacker, df_defender

def get_prism_model(tree):
    """
    Converts a tree object into a PRISM model.

    Args:
        tree: The tree object to be converted.

    Returns:
        A string representing the PRISM model.
    """
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

def get_prism_model_time(tree):
    """
    Converts a tree object into a PRISM model with time.

    Args:
        tree: The tree object to be converted.

    Returns:
        A string representing the PRISM model.
    """
    df = tree.to_dataframe()
    goal, actions_to_goal, list_initial, attacker_actions, defender_actions, df_attacker, df_defender = get_info(df)
    attacker_max_time = max(df_attacker["Time"].values)
    defender_max_time = max(df_defender["Time"].values)
    
    text = "smg\n\nplayer attacker\n\tattacker, [wait1],\n\t"

    for a in attacker_actions.keys():
        text += f"[start{a}], [end{a}], "
        
    text = text[:-2]
    text += "\nendplayer\nplayer defender\n\tdefender, [wait2],\n\t"

    for a in defender_actions.keys():
        text += f"[start{a}], [end{a}], "
    
    text = text[:-2]   
    text += "\nendplayer\n\nglobal sched : [1..2];\n\n"

    text += f'global {goal} : bool;\nlabel "terminate" = {goal}=true;\n\n'

    for a in set(df_attacker.loc[df_attacker["Type"] == "Attribute"]["Label"].values):
        text += "global " + a + " : bool;\n"
        
    for a in set(list_initial):
        text += "global " + a + " : bool init true;\n"

    text += "\nmodule attacker\n\n"

    for a in set(df_attacker["Action"].values):
        text += f"\tprogress{a} : bool;\n"
        
    text += "\n"
    text += f"\ttime1 : [-1..{attacker_max_time}];\n"
    text += f"\t[wait1] sched=1 & time1>0 -> (sched'=2) & (time1'=time1-1);\n\n"

    for a in attacker_actions.keys():
        preconditions = attacker_actions[a]["preconditions"]
        effect = attacker_actions[a]["effect"]
        time = attacker_actions[a]["time"]
        
        if attacker_actions[a]["refinement"] == "disjunctive":
            refinement = "|"
            fail_refinement = "&"
        else:
            refinement = "&"
            fail_refinement = "|"
        
        # check if the node is a leaf
        precon = ""
        fail = ""
        if preconditions != []:
            precon += " & ("
            fail += " & ("
            for p in set(preconditions):
                precon += f"{p} {refinement} "
                fail += f"!{p} {fail_refinement} "
            precon = f"{precon[:-3]})"
            fail = f"{fail[:-3]})"
            
        text += f"\t[start{a}] sched=1 & time1<0 & !progress{a} & !{goal} & !{effect}{precon} -> (sched'=2) & (time1'={time}) & (progress{a}'=true);\n"
        text += f"\t[end{a}] sched=1 & time1=0 & progress{a} & !{goal} & !{effect}{precon} -> (time1'=time1-1) & (progress{a}'=false) & ({effect}'=true);\n"
        if preconditions != []:
            text += f"\t[fail{a}] sched=1 & time1=0 & progress{a} & !{goal} & !{effect}{fail} -> (time1'=time1-1) & (progress{a}'=false);\n\n"
        
    text += "\nendmodule\n\nmodule defender\n\n"

    defender_attributes = set(df_defender.loc[df_defender["Type"] == "Attribute"]["Label"].values)
    for a in defender_attributes:
        text += f"\t{a} : bool;\n"
    
    text += "\n"
    for a in set(df_defender["Action"].values):
        text += f"\tprogress{a} : bool;\n"
        
    text += f"\n\ttime2 : [-1..{defender_max_time}];\n"
    text += f"\t[wait2] sched=2 & time2>0 -> (sched'=1) & (time2'=time2-1);\n\n"
        
    for a in defender_actions.keys():
        preconditions = defender_actions[a]["preconditions"]
        effect = defender_actions[a]["effect"]
        time = defender_actions[a]["time"]
        
        if defender_actions[a]["refinement"] == "disjunctive":
            refinement = "|"
        else:
            refinement = "&"
            
        if effect in defender_attributes:
            precon = ""
            if preconditions != []:
                precon += " & ("
                for p in set(preconditions):
                    precon += f"{p} {refinement} "
                precon = f"{precon[:-3]})"
            text += f"\t[start{a}] sched=2 & time2<0 & !progress{a} & !{goal} & !{effect}{precon} -> (sched'=1) & (time2'={time}) & (progress{a}'=true);\n"
            text += f"\t[end{a}] sched=2 & time2=0 & progress{a} & !{goal} & !{effect}{precon} -> (time2'=time2-1) & (progress{a}'=false) & ({effect}'=true);\n"
        else:
            precon = ""
            if preconditions != []:
                precon += " & ("
                for p in set(preconditions):
                    precon += f"{p} {refinement} "
                precon = f"{precon[:-3]})"
            text += f"\t[start{a}] sched=2 & time2<0 & !progress{a} & !{goal} & {effect}{precon} -> (sched'=1) & (time2'={time}) & (progress{a}'=true);\n"
            text += f"\t[end{a}] sched=2 & time2=0 & progress{a} & !{goal} & {effect}{precon} -> (time2'=time2-1) & (progress{a}'=false) & ({effect}'=false);\n"
        
    text += '\nendmodule\n\nrewards "attacker"\n\n'

    for a in attacker_actions.keys():
        text += f"\t[start{a}] true : {attacker_actions[a]['cost']};\n"
        
    text += '\nendrewards\n\nrewards "defender"\n\n'

    for a in actions_to_goal:
        text += f"\t[end{a}] true : {int(attacker_actions[a]['cost'])*10};\n"
    for a in defender_actions.keys():
        text += f"\t[start{a}] true : {defender_actions[a]['cost']};\n"
          
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
    
