import argparse
import tree_to_prism as tp
import os

def get_prism_solo_models(df):
    def get_prism_model_solo_defender(goal, list_initial, defender_actions, df_attacker, df_defender):
        attacker_attributes = set(df_attacker.loc[df_attacker["Type"] == "Attribute"]["Label"].values)
        defender_attributes = set(df_defender.loc[df_defender["Type"] == "Attribute"]["Label"].values)
        
        text = "smg\n\nplayer attacker\n\tattacker,\n\t"
        text += "[skip]"
        text += "\nendplayer\nplayer defender\n\tdefender,\n\t"

        for a in defender_actions.keys():
            text += f"[{a}], "
        
        text = text[:-2]   
        text += "\nendplayer\n\nglobal sched : [1..2];\n\n"
        
        text += f'global {goal} : [0..1];\n'
        text += f'label "terminate" ='
        for a in attacker_attributes:
            text += f" {a}!=1 &"
        text = text[:-2]
        text += ';\n\n'
        
        for a in attacker_attributes:
            text += f"const {a.lower().replace('_', '')};\n"
            
        for a in attacker_attributes:
            text += f"global {a} : [0..2];\n"
            
        for a in set(list_initial):
            text += "global " + a + " : [1..2];\n"

        text += "\nmodule attacker\n\n"
                
        text += f"\t[skip] sched=1 -> (sched'=2);\n"
            
        text += "\nendmodule\n\nmodule defender\n\n"

        for a in defender_attributes:
            text += f"\t{a} : [0..1];\n"
            
        text += "\n"
        
        for a in defender_actions.keys():
            preconditions = defender_actions[a]["preconditions"]
            effect = defender_actions[a]["effect"]
            if defender_actions[a]["refinement"] == "disjunctive":
                refinement = "|"
            else:
                refinement = "&"
                
            if effect in defender_attributes:
                text += f"\t[{a}] sched=2 & !{goal}=1 & {effect}=0"
                if preconditions != []:
                    text += " & ("
                    for p in set(preconditions):
                        text += f"{p}=1 {refinement} "
                    text = f"{text[:-3]})"
                text += f" -> ({effect}'=1) & (sched'=1);\n"
            else:
                text += f"\t[{a}] sched=2 & !{goal}=1 & !{effect}=2"
                if preconditions != []:
                    text += " & ("
                    for p in set(preconditions):
                        text += f"{p}=1 {refinement} "
                    text = f"{text[:-3]})"
                text += f" -> ({effect}'=2) & (sched'=1);\n"
            
        text += '\nendmodule\n\nrewards "attacker"\n\n'
        text += '\nendrewards\n\nrewards "defender"\n\n'
        for a in defender_actions.keys():
            text += f"\t[{a}] true : {defender_actions[a]['cost']};\n"
            
        text += "\nendrewards\n\n"
        text += f"init sched=1 & {goal}=0 & "
        for a in attacker_attributes:
            text += f"{a}={a.lower().replace('_','')} & "
        if list_initial != []:
            for a in set(list_initial):
                text += f"{a}=1 & "
        text = text[:-3]
        text += " endinit\n\n"
        
        return text
    
    def get_prism_model_attacker(goal, list_initial, attacker_actions, df_attacker):
        attacker_attributes = set(df_attacker.loc[df_attacker["Type"] == "Attribute"]["Label"].values)
        text = "smg\n\nplayer attacker\n\tattacker,\n\t"

        for a in attacker_actions.keys():
            text += f"[{a}], "
            
        text = text[:-2]
        text += "\nendplayer\nplayer defender\n\tdefender,\n\t"
        text += "[skip]" 
        text += "\nendplayer\n\nglobal sched : [1..2];\n\n"

        text += f'global {goal} : [0..1];\nlabel "terminate" = {goal}=1;\n\n'

        for a in attacker_attributes:
            text += "global " + a + " : [0..2];\n"
            
        for a in set(list_initial):
            text += "global " + a + " : [1..2];\n"

        text += "\nmodule attacker\n\n"

        for a in attacker_actions.keys():
            text += f"\t{a} : bool;\n"
            
        text += "\n"

        for a in attacker_actions.keys():
            preconditions = attacker_actions[a]["preconditions"]
            effect = attacker_actions[a]["effect"]
            effects = f"({effect}'=1)"
            if attacker_actions[a]["refinement"] == "disjunctive":
                refinement = "|"
            else:
                refinement = "&"
                
            # check if the node is a leaf
            precon = ""
            if preconditions != []:
                precon += " & ("
                for p in set(preconditions):
                    precon += f"{p}=1 {refinement} "
                precon = f"{precon[:-3]})"
                
            text += f"\t[{a}] sched=1 & !{goal}=1 & {effect}=0 & !{a}{precon} -> {effects} & ({a}'=true) & (sched'=2);\n"
            
        text += "\nendmodule\n\nmodule defender\n\n"

        text += f"\t[skip] sched=2 -> (sched'=1);\n"
            
        text += '\nendmodule\n\nrewards "attacker"\n\n'

        for a in attacker_actions.keys():
            text += f"\t[{a}] true : {attacker_actions[a]['cost']};\n"
            
        text += '\nendrewards\n\nrewards "defender"\n\n'
        
        text += "\nendrewards"

        return text
                
    goal, _, list_initial, attacker_actions, defender_actions, df_attacker, df_defender = tp.get_info(df)
    prism_model_defender = get_prism_model_solo_defender(goal, list_initial, defender_actions, df_attacker, df_defender)
    prism_model_attacker = get_prism_model_attacker(goal, list_initial, attacker_actions, df_attacker)
    return prism_model_defender, prism_model_attacker
    
def save_prism_defender_properties(file):
    with open(file, 'w') as f:
        f.write('// Each agent tries to get the minimum expected cost to reach a terminate state\n')
        f.write('<<attacker,defender>>R{"attacker"}min=? [ F "terminate" ] + R{"defender"}min=? [ F "terminate" ]\n')
        f.close()
        
def main():
    parser = argparse.ArgumentParser(description='Process XML file from ADTool and generate PRISM model')
    parser.add_argument('--input', '-i', type=str, help='Path to the XML file from ADTool')
    args = parser.parse_args()

    tree = tp.parse_file(args.input)
    df = tree.to_dataframe()
    prism_model_defender, prism_model_attacker = get_prism_solo_models(df)
    path_output = os.path.join(os.path.dirname(args.input),"../prism")
    tp.save_prism_model(prism_model_defender, os.path.join(path_output, "Defender.prism"))
    tp.save_prism_model(prism_model_attacker, os.path.join(path_output, "Attacker.prism"))
    prism_model = tp.get_prism_model(tree)
    tp.save_prism_model(prism_model, os.path.join(path_output, "PANACEA.prism"))

    # save the properties file in the same directory as the output file
    tp.save_prism_properties(os.path.join(path_output, "properties.props"))
    save_prism_defender_properties(os.path.join(path_output, "defender.props"))

    
if __name__ == '__main__':
    main()