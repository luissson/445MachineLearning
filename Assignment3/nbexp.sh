#!/bin/bash

name=/anaconda3/bin/python

$name naive_bayes.py yeast_training.txt yeast_test.txt > yeastout.txt

$name naive_bayes.py satellite_training.txt satellite_test.txt > satelitteout.txt

$name naive_bayes.py pendigits_training.txt pendigits_test.txt > pendigitsout.txt
