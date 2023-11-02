import os
import numpy as np
import pysam
import time
from multiprocessing.dummy import Pool as ThreadPool

def readname():
    temp = []
    filePath = r'./simu/0.75.txt'
    open_file = open( filePath)
    information = [[int(i.strip().split('\t')[0]), int(i.strip().split('\t')[1])] for i in open_file.readlines()]
    open_file.close()
    information = sorted(information, key=lambda x: x[0])
    for j1 in range(len(information)):
        temp.append([information[j1][0],information[j1][0]])
    return temp

def text_save(filename, data):
    file = open(filename, 'a+')
    for i in range(len(data)):
        if i <= (len(data) - 2):
            s = str(data[i]).replace('.0', ',')
            file.write(s)
        else:
            s = str(data[i]).replace('.0', '')
            file.write(s)
    s = '\n'
    file.write(s)
    file.close()

def process(i):
    k = int(i)
    print(k)
    chr = 'chr18'
    hg = "./data/hg19.fa"
    samfile = pysam.Samfile("./simu/sort-0.75.bam","rb")
    fastafile = pysam.Fastafile(hg)
    for pos in range(k, k + 100):
        cov = 0
        refbase = fastafile.fetch(chr, pos - 1, pos)
        true = np.zeros(100)
        for pc in samfile.pileup(chr, pos - 1, pos, stepper="all", fastafile=None):
            if pc.pos != pos - 1: continue
            depth = pc.n
            for pr in pc.pileups:
                if cov == 100: break
                if (pr.query_position == None):
                    true[cov] = 1
                    # print(pr.query_position)
                else:
                    if pr.alignment.seq[pr.query_position] == 'A' or pr.alignment.seq[pr.query_position] == 'a':
                        true[cov] = 5
                        # print(pr.alignment.seq[pr.query_position])
                    if pr.alignment.seq[pr.query_position] == 'C' or pr.alignment.seq[pr.query_position] == 'c':
                        true[cov] = 6
                        # print(pr.alignment.seq[pr.query_position])
                    if pr.alignment.seq[pr.query_position] == 'T' or pr.alignment.seq[pr.query_position] == 't':
                        true[cov] = 7
                        # print(pr.alignment.seq[pr.query_position])
                    if pr.alignment.seq[pr.query_position] == 'G' or pr.alignment.seq[pr.query_position] == 'g':
                        true[cov] = 8
                        # print(pr.alignment.seq[pr.query_position])
                cov += 1
        # print(refbase)
        # print(true)
        src = './simu/seq0.75/' +  str(k) + '.txt'
        text_save(src, true)

if  __name__ == '__main__':
    name = readname()
    items = []
    for i in name:
        pos = i[0]
        items.append(pos)
    items.sort()
    start = time.time()
    pool_size = 20
    pool = ThreadPool(pool_size)
    pool.map_async(process, items)
    pool.close()
    pool.join()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
