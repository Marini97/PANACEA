import argparse
import os

import tree_to_prism as tp


def main():
    parser = argparse.ArgumentParser(description='Process XML file from ADTool and generate PRISM model')
    parser.add_argument('--input', '-i', type=str, help='Path to the XML file from ADTool')
    parser.add_argument('--output', '-o', type=str, help='Path to the output file for the PRISM model')
    parser.add_argument('--props', action='store_true', help='Generate the properties file')
    parser.add_argument('--prune', '-p', type=str, help='Name of the subtree root to keep')
    parser.add_argument('--time', '-t', action='store_true', help='Generate a time-based PRISM model')
    args = parser.parse_args()

    tree = tp.parse_file(args.input)
    if args.prune:
        tree = tree.prune(args.prune)
    if args.time:
        prism_model = tp.get_prism_model_time(tree)
    else:
        prism_model = tp.get_prism_model(tree)
    file = args.output
    tp.save_prism_model(prism_model, file)
    if args.props:
        # save the properties file in the same directory as the output file
        path_output = "/".join(file.split("/")[:-1])
        tp.save_prism_properties(os.path.join(path_output, "properties.props"))


if __name__ == '__main__':
    main()
