import argparse
import tree_to_prism as tp
import os
        
def main():
    parser = argparse.ArgumentParser(description='Process XML file from ADTool and generate PRISM model')
    parser.add_argument('--input', '-i', type=str, help='Path to the XML file from ADTool')
    args = parser.parse_args()

    tree = tp.parse_file(args.input)
    df = tree.to_dataframe()
    path_output = os.path.join(os.path.dirname(args.input),"../prism")
    prism_model = tp.get_prism_model_time(tree)
    tp.save_prism_model(prism_model, os.path.join(path_output, "PANACEA.prism"))

    # save the properties file in the same directory as the output file
    tp.save_prism_properties(os.path.join(path_output, "properties.props"))

    
if __name__ == '__main__':
    main()