#!/bin/bash
./train -t 8 /home/jin/mpiranksvm/linearranksvm/MSLR-WEB10K/Fold1/train.txt 
./predict /home/jin/mpiranksvm/linearranksvm/MSLR-WEB10K/Fold1/test.txt train.txt.model MQ2008.txt
