#!/bin/bash
perl ts.pl < tshekid_office2003.csv > input2.txt
./apriori -ts input2.txt output-s.txt
./apriori -tr -s3.5 input2.txt output-r-s3.txt
./apriori -tr -c60 input2.txt output-r-c60.txt
