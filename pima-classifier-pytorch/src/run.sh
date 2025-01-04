#! /bin/bash

source ../bin/activate

accuarcy=0

for i in {1..5}; do

  acc=$(python3 nn.py)
  accuracy=$( echo "$accuracy" + "$acc" | bc)

  echo "accuracy in iter $i : $accuracy"
done

avg_accuracy=$( echo " scale=10 $accuracy / 5" | bc)

echo "avg accuracy: $avg_accuracy"

deactivate
