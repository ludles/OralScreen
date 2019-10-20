# -*- coding: utf-8 -*-
# split chunks for parallelisation

import math
import sys
import csv
from glob import glob


def split_chunks(arr, m):
    ''' split the arr into m chunks as average as possible '''
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def main():
    # number of processes
    n = int(sys.argv[1])
    
    csv_paths = glob('./CSVResults/*.csv')
    csv_paths = split_chunks(csv_paths, n)
    
    with open('./debugging/temp.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for line in csv_paths:
            writer.writerow(line)

if __name__ == '__main__':
    main()