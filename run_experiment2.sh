DIRECTORY=experiments/experiment2
PRISM_PATH=prism-games-3.2.1-linux64-x86
JVM_MEMORY=$1
CUDD_MEMORY=$2

rm -rf "$DIRECTORY"/prism/
rm -rf "$DIRECTORY"/results/
mkdir "$DIRECTORY"/prism/
mkdir "$DIRECTORY"/results/

for TREE_FILE in "$DIRECTORY"/trees/*.xml; do
    BASENAME=$(basename "$TREE_FILE" .xml)
    
    python3 exp2.py -i "$TREE_FILE"
done

# Loop over each .prism file in the directory
for MODEL_FILE in "$DIRECTORY"/prism/*.prism; do
  # Extract the base name of the file (without the directory and extension)
  BASENAME=$(basename "$MODEL_FILE" .prism)
  EXPERIMENT_DIRECTORY="$DIRECTORY"/results/"$BASENAME"
  mkdir -p $EXPERIMENT_DIRECTORY

  # if basename contains "defender" then it does the experiment with all possible values
  if [[ $BASENAME == *"defender"* ]]; then
    # get constants needed to run the experiment
    CONSTANTS=$(grep "const" "$MODEL_FILE" | awk '{print $2}')
    # replace ";" with "=0:2," at the end of each constant
    CONSTANTS=$(echo $CONSTANTS | sed 's/; /=0:2,/g')
    CONSTANTS=$(echo $CONSTANTS | sed 's/;/=0:2/g')
     # Run the command with /usr/bin/time and redirect the output to the respective file
    /usr/bin/time -v "$PRISM_PATH"/bin/prism -javamaxmem "$JVM_MEMORY"g -cuddmaxmem "$CUDD_MEMORY"g "$MODEL_FILE" "$DIRECTORY"/prism/properties.props -prop 1 -const "$CONSTANTS" -exportresults $EXPERIMENT_DIRECTORY/result.csv:csv  -exportstrat $EXPERIMENT_DIRECTORY/"$BASENAME".dot 2>&1 | tee $EXPERIMENT_DIRECTORY/"$BASENAME".log
  else
    # Run the command with /usr/bin/time and redirect the output to the respective file
    /usr/bin/time -v "$PRISM_PATH"/bin/prism -javamaxmem "$JVM_MEMORY"g -cuddmaxmem "$CUDD_MEMORY"g "$MODEL_FILE" "$DIRECTORY"/prism/properties.props -prop 1 -exportstrat $EXPERIMENT_DIRECTORY/"$BASENAME".dot 2>&1 | tee $EXPERIMENT_DIRECTORY/"$BASENAME".log
  fi

  done