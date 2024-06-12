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
        os.path.join(figures_path, f"time_size_figure.pdf"), format="pdf", bbox_inches='tight'
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
        os.path.join(figures_path, f"mdp_adt_size_figure.pdf"), format="pdf", bbox_inches='tight'
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
        os.path.join(figures_path, f"mdp_size_time_figure.pdf"), format="pdf", bbox_inches='tight'
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
        prism_path = os.path.join(experiments_prism_path, f"{experiment_name}.prism")
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
        os.path.join(figures_path, f"reward_figure.pdf"), format="pdf", bbox_inches='tight'
    )


if __name__ == '__main__':
    plt.figure(figsize=(4, 2))
    experiments_path = os.path.join("experiments", "experiment3")
    experiments_prism_path = os.path.join(experiments_path, "prism")
    experiments_results_path = os.path.join(experiments_path, "results")

    figures_path = os.path.join(experiments_path, "figures")

    os.makedirs(figures_path, exist_ok=True)
    plot_reward_figure(experiments_results_path)
    plot_time_size_figure(os.path.join(experiments_path, "result.csv"))
    plot_mdp_size_tree_size_figure(os.path.join(experiments_path, "result.csv"))
    plot_mdp_size_time_figure(os.path.join(experiments_path, "result.csv"))
