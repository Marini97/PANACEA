import argparse
import tree_to_prism as tp

def main():
    parser = argparse.ArgumentParser(description='Process XML file from ADTool and generate PRISM model')
    parser.add_argument('--input', '-i', type=str, help='Path to the XML file from ADTool')
    parser.add_argument('--output', '-o', type=str, help='Path to the output file for the PRISM model')
    args = parser.parse_args()

    tree = tp.parse_file(args.input)
    tree.prune("AccesstoReverseShell")
    prism_model = tp.get_prism_model(tree)
    tp.save_prism_model(prism_model, args.output)
    
if __name__ == '__main__':
    main()