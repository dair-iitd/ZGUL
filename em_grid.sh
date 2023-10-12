#!/bin/bash
for lr in 1.0 0.5 0.1 0.05 0.01;
do
for step in 1 5 10; 
do
for LANG in de is;
do
#bash test_pos_emea.sh $step $lr
#bash test_pos_emea.sh $1 $step $lr 3
bash test_ner_emea.sh $LANG $step $lr 3
bash test_pos_emea1.sh $LANG $step $lr 3
done
done
done
