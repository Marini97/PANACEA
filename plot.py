import csv
import os

import matplotlib.pyplot as plt

name_to_nodes = {
    "A_1": 29,
    "A_4": 25,
    "A_6": 10,
    "root": 34
}


def parse_prism_results(exp_type, parsed_results):
    if exp_type == "time":
        results = list(sorted(list(filter(lambda x: "time" in x[0], parsed_results))))
    else:
        results = list(sorted(list(filter(lambda x: "time" not in x[0], parsed_results))))
    return results


def plot_time_size_figure(result_path):
    def plot_time_size_line(exp_type, parsed_results, label, color, marker):
        to_plot = {'x': [], 'y': []}
        results = parse_prism_results(exp_type, parsed_results)
        for row in results:
            to_plot['x'].append(int(row[0].replace("_time", "")))
            to_plot['y'].append(float(row[-2]))
        plt.plot(to_plot['x'], to_plot['y'], label=label, linestyle="dashed", fillstyle='none', color=color,
                 marker=marker)

    with open(result_path, "r") as result_file:
        parsed_result = [row for row in csv.reader(result_file, delimiter=",") if row]

    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)
    plot_time_size_line("time", parsed_result, "Time", "red", "o")
    plot_time_size_line("no-time", parsed_result, "No-Time", "orange", "^")
    # plt.xticks(range(0, 13))
    # plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    # plt.xlim([0, 13])
    # plt.ylim([0, 30000])

    plt.yscale('log', base=10)
    plt.xlabel('ADT Size [N. Nodes]')
    plt.ylabel('Planning Time [s]')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=3, prop={'size': 6})

    plt.savefig(
        os.path.join(experiment3_figures_path, f"time_size_figure.pdf"), format="pdf", bbox_inches='tight'
    )


def plot_mdp_size_tree_size_figure(result_path):
    def plot_mdp_size_tree_size_line(exp_type, parsed_results, label, color, marker):
        to_plot = {'x': [], 'y': []}
        results = parse_prism_results(exp_type, parsed_results)
        for row in results:
            to_plot['x'].append(int(row[0].replace("_time", "")))
            to_plot['y'].append(float(row[1]))
        plt.plot(to_plot['x'], to_plot['y'], label=label, linestyle="dashed", fillstyle='none', color=color,
                 marker=marker)

    with open(result_path, "r") as result_file:
        parsed_result = [row for row in csv.reader(result_file, delimiter=",") if row]

    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)
    plot_mdp_size_tree_size_line("time", parsed_result, "Time", "red", "o")
    plot_mdp_size_tree_size_line("no-time", parsed_result, "No-Time", "orange", "^")
    # plt.xticks(range(0, 13))
    plt.yticks([0, 50, 100, 150, 200], ["0", "50M", "100M", "150M", "200M"])
    # plt.xlim([0, 13])
    # plt.ylim([0, 30000])

    plt.yscale('log', base=10)
    plt.xlabel('ADT Size [N. Nodes]')
    plt.ylabel('MDP States')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=3, prop={'size': 6})

    plt.savefig(
        os.path.join(experiment3_figures_path, f"mdp_adt_size_figure.pdf"), format="pdf", bbox_inches='tight'
    )


def plot_mdp_size_time_figure(result_path):
    def plot_mdp_size_time_line(exp_type, parsed_results, label, color, marker):
        to_plot = {'x': [], 'y': []}
        results = parse_prism_results(exp_type, parsed_results)
        for row in results:
            to_plot['x'].append(float(row[1]) / 1000000)
            to_plot['y'].append(float(row[-2]))
        plt.plot(to_plot['x'], to_plot['y'], label=label, linestyle="dashed", fillstyle='none', color=color,
                 marker=marker)

    with open(result_path, "r") as result_file:
        parsed_result = [row for row in csv.reader(result_file, delimiter=",") if row]

    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)
    plot_mdp_size_time_line("time", parsed_result, "Time", "red", "o")
    # plot_mdp_size_time_line("no-time", parsed_result, "No-Time", "orange", "^")
    # plt.xticks(range(0, 13))
    plt.xticks([0, 50, 100, 150, 200], ["0", "50M", "100M", "150M", "200M"])
    # plt.xlim([0, 13])
    # plt.ylim([0, 30000])

    plt.xlabel('MDP States')
    plt.ylabel('Planning Time [s]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=3, prop={'size': 6})

    plt.savefig(
        os.path.join(experiment3_figures_path, f"mdp_size_time_figure.pdf"), format="pdf", bbox_inches='tight'
    )


import re
import numpy as np


def plot_reward_figure(result_path):
    def parse_dot_file(file_path):
        results = []
        with open(file_path, "r") as dot_file:
            lines = dot_file.readlines()
        for line in lines:
            if ":" not in line:
                continue
            results.append(line.split(":")[-1].split("\"")[0])
        return results

    def get_actions_rewards(prism_path, agent):
        results = {}
        reward_pattern = re.compile(rf'rewards "{agent}"(.*?)endrewards', re.DOTALL)
        with open(prism_path, "r") as dot_file:
            lines = dot_file.read()
        match = reward_pattern.search(lines)

        actions_rewards = match.group(1).strip().split(";")
        for line in actions_rewards:
            match = re.search(r'\[(.*?)\].*:\s*(\d+)', line)
            if match:
                action = match.group(1)
                cost = match.group(2)
                results[action] = int(cost)
        return results

    def compute_reward(actions, actions_rewards):
        reward = 0
        for action in actions:
            if action in actions_rewards:
                reward += int(actions_rewards[action])
        return reward

    def compute_rewards(experiment_name):
        experiment_path = os.path.join(result_path, experiment_name)
        dot_path = os.path.join(experiment_path, f"{experiment_name}.dot")
        prism_path = os.path.join(experiment3_prism_path, f"{experiment_name}.prism")
        actions = parse_dot_file(dot_path)
        attacker_rewards = get_actions_rewards(prism_path, "attacker")
        defender_rewards = get_actions_rewards(prism_path, "defender")
        attacker_reward = compute_reward(actions, attacker_rewards)
        defender_reward = compute_reward(actions, defender_rewards)
        return attacker_reward, defender_reward

    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)

    to_plot = {'x': []}
    time_rewards = {'Att': [], 'Def': []}
    no_time_rewards = {'Att': [], 'Def': []}
    offset = 5
    for experiment_name in filter(lambda x: "time" not in x, sorted(os.listdir(result_path))):
        ar_no_time, dr_no_time = compute_rewards(experiment_name)
        ar_time, dr_time = compute_rewards(f"{experiment_name}_time")
        to_plot['x'].append(int(experiment_name))
        time_rewards['Att'].append(ar_time)
        time_rewards['Def'].append(dr_time)
        no_time_rewards['Att'].append(ar_no_time)
        no_time_rewards['Def'].append(dr_no_time)
        offset += 5

    plt.bar(list(map(lambda x: x - 0.5, to_plot['x'])), np.array(time_rewards['Att']), label="Att (TIME)", fill=None,
            hatch="////", edgecolor="blue")
    plt.bar(list(map(lambda x: x - 0.5, to_plot['x'])), np.array(time_rewards['Def']), label="Def (TIME)",
            bottom=np.array(time_rewards['Att']), fill=None, hatch="////", edgecolor="green")

    plt.bar(list(map(lambda x: x + 0.5, to_plot['x'])), np.array(no_time_rewards['Att']), label="Att (NO-TIME)",
            fill=None, hatch="oooo", edgecolor="red")
    plt.bar(list(map(lambda x: x + 0.5, to_plot['x'])), np.array(no_time_rewards['Def']), label="Def (NO-TIME)",
            bottom=np.array(no_time_rewards['Att']), fill=None, hatch="oooo", edgecolor="orange")

    # plot_mdp_size_time_line("no-time", parsed_result, "No-Time", "orange", "^")
    # plt.xticks(range(0, 13))
    xticks = [0]
    xticks.extend(sorted(name_to_nodes.values()))
    xticks.extend([40])
    plt.xticks(xticks)
    # plt.xlim([0, 13])
    # plt.ylim([0, 30000])

    plt.xlabel('N. Nodes')
    plt.ylabel('Cost')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=2, prop={'size': 6})

    plt.savefig(
        os.path.join(experiment3_figures_path, f"reward_figure.pdf"), format="pdf", bbox_inches='tight'
    )

def parse_dot_file(file_path):
    results = []
    with open(file_path, "r") as dot_file:
        lines = dot_file.readlines()
    for line in lines:
        if ":" not in line:
            continue
        results.append(line.split(":")[-1].split("\"")[0])
    return results

def parse_csv_file(file_path):
    results = []
    with open(file_path, "r") as csv_file:
        lines = csv_file.readlines()
    for line in lines:
        # [action]
        if "[" not in line:
            continue
        results.append(line.split("[")[1].split("]")[0])
    return results

def parse_defender_file(file_path):
    # get last element of each row if it's an integer
    parsed_result = [int(row[-1]) for row in csv.reader(open(file_path), delimiter=",") if
                     row and row[-1].isdigit()]
    reward = sum(parsed_result) / len(parsed_result)
    return reward

def get_actions_rewards(prism_path, agent):
    results = {}
    reward_pattern = re.compile(rf'rewards "{agent}"(.*?)endrewards', re.DOTALL)
    with open(prism_path, "r") as prism_file:
        lines = prism_file.read()
    match = reward_pattern.search(lines)

    actions_rewards = match.group(1).strip().split(";")
    for line in actions_rewards:
        match = re.search(r'\[(.*?)\].*:\s*(\d+)', line)
        if match:
            action = match.group(1)
            cost = match.group(2)
            results[action] = int(cost)
    return results

def compute_reward(actions, actions_rewards, attacker_goal=None):
    reward = 0
    att_winner = False
    for action in actions:
        if action in actions_rewards:
            if attacker_goal and action == attacker_goal:
                att_winner = True
                continue
            reward += int(actions_rewards[action])
    return reward, att_winner

def compute_rewards(experiment_name, result_path, prism_path):
    experiment_path = os.path.join(result_path, experiment_name)
    if "defender" in experiment_name:
        defender_reward = parse_defender_file(os.path.join(experiment_path, f"{experiment_name}.csv"))
        return 0, defender_reward
    if os.path.exists(os.path.join(experiment_path, f"{experiment_name}.dot")):
        dot_path = os.path.join(experiment_path, f"{experiment_name}.dot")
        actions = parse_dot_file(dot_path)
    elif os.path.exists(os.path.join(experiment_path, f"{experiment_name}.csv")):
        csv_file = os.path.join(experiment_path, f"{experiment_name}.csv")
        actions = parse_csv_file(csv_file)

    prism_path = os.path.join(prism_path, f"{experiment_name}.prism")
    attacker_rewards = get_actions_rewards(prism_path, "attacker")
    defender_rewards = get_actions_rewards(prism_path, "defender")
    attacker_reward, _ = compute_reward(actions, attacker_rewards)
    defender_reward, attacker_winner = compute_reward(actions, defender_rewards, "exfiltrateData")
    return attacker_reward, defender_reward, attacker_winner

def plot_reward_adt_mdp(prism_path, result_path, figures_path, experiments_order = None):
    """
    histogram with rewards from experiment
    """
    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)

    to_plot = {'x': []}
    rewards = {'Att': [], 'Def': []}

    x_values = []
    offset = 0
    for experiment_name in sorted(os.listdir(result_path)) if not experiments_order else experiments_order:
        ar, dr, att_win = compute_rewards(experiment_name, result_path, prism_path)
        to_plot['x'].append(experiment_name.replace("_", " "))
        rewards['Att'].append(ar)
        rewards['Def'].append(dr)

        x_values.append(offset)
        if att_win:
            plt.text(offset - 0.5, ar + 10, "W")
            plt.text(offset + 3.5, dr + 10, "L")
        else:
            plt.text(offset - 0.5, ar + 10, "L")
            plt.text(offset + 3.5, dr + 10, "W")

        offset += 10

    plt.bar(x_values, np.array(rewards['Att']), label="Attacker", fill=None,
            hatch="////", edgecolor="blue", width=4)
    plt.bar(list(map(lambda x: x[1] + 4 if rewards["Att"][x[0]] and rewards["Def"][x[0]] else x[1], enumerate(x_values))), np.array(rewards['Def']), label="Defender", fill=None, hatch="////", edgecolor="green", width=4)
    
    plt.xticks(list(map(lambda x: x[1] + 2 if rewards["Att"][x[0]] and rewards["Def"][x[0]] else x[1], enumerate(x_values))), to_plot['x'], fontsize=7)
    plt.yticks(range(0, 800, 200))
    
    #plt.xlabel('Experiment')
    plt.ylabel('Cost')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=2, prop={'size': 6})

    plt.savefig(
        os.path.join(figures_path, f"reward_solo_multi_figure.pdf"), format="pdf", bbox_inches='tight'
    )


def plot_reward_single_multi(prism_path, result_path, figures_path, experiments_order=None):
    """
    histogram with rewards from experiment
    """
    plt.clf()
    plt.grid(linestyle='--', linewidth=0.5)

    to_plot = {'x': []}
    rewards = {'Att': [], 'Def': []}

    x_values = []
    offset = 0
    for experiment_name in sorted(os.listdir(result_path)) if not experiments_order else experiments_order:
        ar, dr, att_win = compute_rewards(experiment_name, result_path, prism_path)
        to_plot['x'].append(experiment_name.replace("_", " "))
        rewards['Att'].append(ar)
        rewards['Def'].append(dr)
        print(experiment_name, ar, dr)
        x_values.append(offset)
        if ar > 0 and dr > 0:
            if att_win:
                plt.text(offset - 0.5, ar + 10, "W")
                plt.text(offset + 3.5, dr + 10, "L")
            else:
                plt.text(offset - 0.5, ar + 10, "L")
                plt.text(offset + 3.5, dr + 10, "W")

        offset += 10

    plt.bar(x_values, np.array(rewards['Att']), label="Attacker", fill=None,
            hatch="////", edgecolor="blue", width=4)
    plt.bar(
        list(map(lambda x: x[1] + 4 if rewards["Att"][x[0]] and rewards["Def"][x[0]] else x[1], enumerate(x_values))),
        np.array(rewards['Def']), label="Defender", fill=None, hatch="////", edgecolor="green", width=4)

    plt.xticks(
        list(map(lambda x: x[1] + 2 if rewards["Att"][x[0]] and rewards["Def"][x[0]] else x[1], enumerate(x_values))),
        to_plot['x'], fontsize=7)
    plt.yticks(range(0, 800, 200))

    # plt.xlabel('Experiment')
    plt.ylabel('Cost')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), labelspacing=0.2, ncols=2, prop={'size': 6})

    plt.savefig(
        os.path.join(figures_path, f"reward_solo_multi_figure.pdf"), format="pdf", bbox_inches='tight'
    )
    
    
    
if __name__ == '__main__':
    plt.figure(figsize=(4, 2))
    
    experiment1_path = os.path.join("experiments", "experiment1")
    experiment1_prism_path = os.path.join(experiment1_path, "prism")
    experiment1_results_path = os.path.join(experiment1_path, "results")
    experiment1_figures_path = os.path.join(experiment1_path, "figures")

    experiment2_path = os.path.join("experiments", "experiment2")
    experiment2_prism_path = os.path.join(experiment2_path, "prism")
    experiment2_results_path = os.path.join(experiment2_path, "results")
    experiment2_figures_path = os.path.join(experiment2_path, "figures")

    experiment3_path = os.path.join("experiments", "experiment3")
    experiment3_prism_path = os.path.join(experiment3_path, "prism")
    experiment3_results_path = os.path.join(experiment3_path, "results")
    experiment3_figures_path = os.path.join(experiment3_path, "figures")

    experiment4_path = os.path.join("experiments", "experiment4")
    experiment4_prism_path = os.path.join(experiment4_path, "prism")
    experiment4_results_path = os.path.join(experiment4_path, "results")
    experiment4_figures_path = os.path.join(experiment4_path, "figures")

    
    os.makedirs(experiment1_figures_path, exist_ok=True)
    os.makedirs(experiment2_figures_path, exist_ok=True)
    os.makedirs(experiment3_figures_path, exist_ok=True)
    plot_reward_figure(experiment3_results_path)
    plot_reward_adt_mdp(experiment1_prism_path, experiment1_results_path, experiment1_figures_path,
                        ["R-ADT_LC", "R-ADT_LD", "PANACEA"])
    plot_reward_adt_mdp(experiment2_prism_path, experiment2_results_path, experiment2_figures_path,
                        ["R-ADT_LC", "R-ADT_LD", "PANACEA"])
    plot_reward_single_multi(experiment4_prism_path, experiment4_results_path, experiment4_figures_path)
    plot_time_size_figure(os.path.join(experiment3_path, "result.csv"))
    plot_mdp_size_tree_size_figure(os.path.join(experiment3_path, "result.csv"))
    plot_mdp_size_time_figure(os.path.join(experiment3_path, "result.csv"))
    
