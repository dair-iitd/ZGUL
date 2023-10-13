#!/bin/bash
for lr in 1.0 0.5 0.1 0.05 0.01;
do
for step in 1 5 10; 
do
bash test_masa_emea.sh hau $step $lr 3
bash test_masa_emea.sh ibo $step $lr 4
bash test_masa_emea.sh kin $step $lr 3
bash test_masa_emea.sh lug $step $lr 4
bash test_masa_emea.sh luo $step $lr 3
bash test_masa_emea.sh pcm $step $lr 4
done
done
