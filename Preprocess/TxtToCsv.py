#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: TxtToCsv.py
@time: 10/1/18 4:33 PM
@desc: This file is to convert the dataset.txt into dataset.csv; It should be run before running preprocess.py
'''

import csv

def TxtToCsv(file_txt,file_csv):
    '''
    This is a file format converting function; convert file.txt into file.csv
    :param file_txt: file path of file.txt
    :param file_csv: file path of file.csv
    :return: no return
    '''
    with open(file_txt, 'r') as textfile:
        lines=(line.strip() for line in textfile)
        lines=(line.split(",") for line in lines if line)
        with open(file_csv, 'w') as filecsv:
            file=csv.writer(filecsv)
            file.writerows(lines)

TxtToCsv('dataset.txt','dataset.csv') # convert the dataset from .txt format into .csv format