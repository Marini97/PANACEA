#!\bin\bash

RESULTS_DIRECTORY=$1

echo "" > $RESULTS_DIRECTORY/../result.csv

for experiment_directory in "$RESULTS_DIRECTORY"*
do
        echo "$experiment_directory"
        RESULT_NAME=$(basename $experiment_directory .log)
        echo "$RESULT_NAME"
        RESULT_PATH=$experiment_directory/$RESULT_NAME.log
        echo "$RESULT_PATH"
        STATES=$(cat $RESULT_PATH | grep States | tr -s ' ' | cut -d " " -f 2)
        TRANSITIONS=$(cat $RESULT_PATH | grep Transitions | tr -s ' ' | cut -d " " -f 2)
        MODEL_CONSTRUCTION=$(cat $RESULT_PATH | grep "Time for model construction" | tr -s ' ' | cut -d " " -f 5)
        MODEL_CHECKING=$(cat $RESULT_PATH | grep "Time for model checking" | tr -s ' ' | cut -d " " -f 5)
        REWARD=$(cat $RESULT_PATH | grep "Result" | tr -s ' ' | cut -d " " -f 2)
        echo "$RESULT_NAME,$STATES,$TRANSITIONS,$MODEL_CONSTRUCTION,$MODEL_CHECKING,$REWARD" >> $RESULTS_DIRECTORY/../result.csv
done