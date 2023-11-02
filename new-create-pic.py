import pysam
import numpy as np
from PIL import Image
import time
from multiprocessing.dummy import Pool as ThreadPool

def binarySearch(arr, l, r, x):
    if r >= l:
        mid = int(l + (r - l) / 2)
        if int(arr[mid].strip().split('\t')[1]) == x:
            return mid
        elif int(arr[mid].strip().split('\t')[1]) > x:
            return binarySearch(arr, l, mid - 1, x)
        else:
            return binarySearch(arr, mid + 1, r, x)
    else:
        return -1

def read_cigartuples_remove_redundancy(read_cigartuples):
    read_cigartuples_1 = [read_cigartuples[0]]
    for i in range(1, len(read_cigartuples)):
        if read_cigartuples[i][0] == read_cigartuples_1[-1][0]:
            read_cigartuples_1[-1][1] += read_cigartuples[i][1]
        else:
            read_cigartuples_1.append(read_cigartuples[i])
    return np.array(read_cigartuples_1)

def get_area_read_seq_inf(bam_file_path,chid, start, stop):
    read_temp, cigartuples_temp = [], []
    open_bam_file = pysam.AlignmentFile(bam_file_path, 'rb')
    for read in open_bam_file.fetch(contig=chid, start=max(0, start ), stop=stop ):  #1000
        if read.is_secondary == False:
            read_cigartuples = np.array(read.cigartuples)
            maped_type = read_cigartuples[:, 0]
            read_name, read_start, read_end = '', 0, 0
            if (3 not in maped_type[1:-1]) and (4 not in maped_type[1:-1]) and (5 not in maped_type[1:-1]) and (
                    6 not in maped_type[1:-1]) and (7 not in maped_type[1:-1]) and (8 not in maped_type[1:-1]):
                if (read_cigartuples[0][0] in [0, 2, 4]) and (read_cigartuples[-1][0] in [0, 2, 4]):
                    read_name = read.query_name
                    if int(read_cigartuples[0][0]) != 4:
                        read_start = read.reference_start + 1
                    else:
                        read_start = read.reference_start - read_cigartuples[0][1] + 1
                    if int(read_cigartuples[-1][0]) != 4:
                        read_end = read.reference_end
                    else:
                        read_end = read.reference_end + read_cigartuples[-1][1]
            if (read_end < start) or (read_start > stop):
                continue
            else:
                read_temp.append([read_name, read_start, read_end])
                del_mini_area_index = []
                for j in range(len(read_cigartuples)):
                    if (read_cigartuples[j][1] <= 3) and (read_cigartuples[j][0] == 1):
                        del_mini_area_index.append(j)
                read_cigartuples = np.delete(read_cigartuples, del_mini_area_index, axis=0)
                for k in range(len(read_cigartuples)):
                    if read_cigartuples[k][1] <= 3:
                        read_cigartuples[k][0] = 0
                for k2 in range(len(read_cigartuples)):
                    if  (read_cigartuples[k2][1] < 36) and (read_cigartuples[k2][0] == 2) :
                        read_cigartuples[k2][0] = 0
                read_cigartuples = read_cigartuples_remove_redundancy(read_cigartuples)
                for i in range(1, len(read_cigartuples) - 1):
                    if read_cigartuples[i][0] == 1:
                        if read_cigartuples[i - 1][1] >= read_cigartuples[i + 1][1]:
                            read_cigartuples[i - 1][1] -= 1
                        else:
                            read_cigartuples[i + 1][1] -= 1
                        read_cigartuples[i][1] = 1
                del_model_index = []
                for u in range(len(read_cigartuples)):
                    if read_cigartuples[u][1] == 0:
                        del_model_index.append(u)
                read_cigartuples = np.delete(read_cigartuples, del_model_index, axis=0)
                read_cigartuples = read_cigartuples_remove_redundancy(read_cigartuples)
                cigartuples_temp.append(read_cigartuples)
    open_bam_file.close()
    return read_temp, cigartuples_temp

def cigar_num_transform(num):
    if num == 0:  # M
        return [1]
    elif num == 1:  # I
        return [2]
    elif num == 2:  # D
        return [3]
    elif num == 4:  # S
        return [4]

def rearrange_read(read_temp, cigartuples_temp, start, stop):
    zero_threshold = np.zeros(shape=(len(read_temp), stop - start + 1))
    for i in range(len(read_temp)):
        read_start = read_temp[i][1]
        if read_start >= start:
            for j3 in range(len(cigartuples_temp[i])):
                zero_threshold[i, (read_start - start):min((stop - start + 1), (
                            read_start + cigartuples_temp[i][j3][1] - start))] = np.array((min(stop - start + 1, read_start +cigartuples_temp[i][j3][1] - start) - (read_start - start)) * cigar_num_transform(cigartuples_temp[i][j3][0]))
                read_start += cigartuples_temp[i][j3][1]
        elif read_start < start:
            read_start_index = 0
            cigartuples_index = 0
            for k2 in range(len(cigartuples_temp[i])):
                if (read_start + cigartuples_temp[i][k2][1]) >= start:
                    read_start_index = read_start
                    cigartuples_index = k2
                    break
                else:
                    read_start += cigartuples_temp[i][k2][1]
            for k4 in range(cigartuples_index, len(cigartuples_temp[i])):
                zero_threshold[i,
                max(0, read_start_index - start):min(read_start_index + cigartuples_temp[i][k4][1] - start,stop - start + 1)] = np.array((min(read_start_index +cigartuples_temp[i][k4][1] - start,stop - start + 1) - max(0,read_start_index - start)) * cigar_num_transform(cigartuples_temp[i][k4][0]))
                read_start_index += cigartuples_temp[i][k4][1]
    return zero_threshold

def get_array_nonzero_area(array):
    temp = []
    output_temp = []
    for i in range(len(array)):
        if array[i] != 0:
            temp.append(i)
    output_temp.append(temp[0])
    output_temp.append(temp[-1])
    return output_temp

def remove_redundancy_blank_space(zero_threshold):
    nonzero_area_temp = []
    max_read_index_temp = []
    none_max_read_index_temp = []
    already_moved_read_index = []
    height = zero_threshold.shape[0]
    length = zero_threshold.shape[1]
    for i in range(len(zero_threshold)):
        output_temp = get_array_nonzero_area(zero_threshold[i])
        nonzero_area_temp.append(output_temp)
    for j in range(len(nonzero_area_temp)):
        if (nonzero_area_temp[j][-1] - nonzero_area_temp[j][0] + 1) == length:
            continue
        else:
            none_max_read_index_temp.append(j)
    if len(none_max_read_index_temp) < 2:
        return zero_threshold
    else:
        for k in range(len(none_max_read_index_temp) - 1):
            for l in range(1, len(none_max_read_index_temp)):
                target_read_index = none_max_read_index_temp[k]
                need_move_read_index = none_max_read_index_temp[l]
                if (target_read_index not in already_moved_read_index) and (
                        need_move_read_index not in already_moved_read_index):
                    left = nonzero_area_temp[need_move_read_index][0]
                    right = nonzero_area_temp[need_move_read_index][1]
                    a = np.array(
                        zero_threshold[need_move_read_index, left:right + 1] + zero_threshold[target_read_index,left:right + 1])
                    b = np.array(zero_threshold[need_move_read_index, left:right + 1])
                    if sum(a == b) == (right - left + 1):
                        already_moved_read_index.append(need_move_read_index)
                        zero_threshold[target_read_index] += zero_threshold[need_move_read_index]
        return np.delete(zero_threshold, already_moved_read_index, axis=0)

def get_rgb(save_path, zero_threshold, start, stop, chid, pic_range):
    im = Image.new("RGB", ((stop - start + 1), len(zero_threshold)))
    for i in range(len(zero_threshold)):
        for j in range(stop - start + 1):
            if zero_threshold[i][j] == 0:
                im.putpixel((j, i), (255, 255, 255))
            elif zero_threshold[i][j] == 1:
                im.putpixel((j, i), (0, 255, 0))
            elif zero_threshold[i][j] == 2:
                im.putpixel((j, i), (0, 0, 0))
            elif zero_threshold[i][j] == 3:
                im.putpixel((j, i), (255, 0, 0))
            elif zero_threshold[i][j] == 4:
                im.putpixel((j, i), (0, 0, 255))
    out = im.resize((100, 100), Image.ANTIALIAS)
    out.save(save_path + pic_range + '.png')

def save_pics(bam_file_path, save_path, chid, start, stop, pic_range):
    read_temp, cigartuples_temp = get_area_read_seq_inf(bam_file_path, chid, start, stop)
    zero_threshold_1 = rearrange_read(read_temp, cigartuples_temp, start, stop)
    zero_threshold = remove_redundancy_blank_space(zero_threshold_1)
    if zero_threshold.shape[0] < 100:
        zeros = np.zeros(shape=(100 - zero_threshold.shape[0], zero_threshold.shape[1]))
        c = np.vstack((zero_threshold, zeros))
        vv = get_rgb(save_path, c, start, stop, chid, pic_range)
    else:
        vv = get_rgb(save_path, zero_threshold, start, stop, chid, pic_range)

def process(i):
    bam_file_path = r'./data/new-sort-HG003_PB_30x_RG_HP10XtrioRTG.bam'
    save_path = r'./hg003-pic/2.0pic/'
    chid = 'chr21'
    start = int(i.strip().split('\t')[0])
    print(start)
    stop = int(i.strip().split('\t')[1])
    range = str(i.strip().split('\t')[0])
    ccc = save_pics(bam_file_path, save_path, chid, start, stop, range)

if  __name__ == '__main__':
    chr_single_area_path =r'./hg003-pic/hg003.txt'
    open_single_del_area_file = open(chr_single_area_path)
    single_del_area_file = open_single_del_area_file.readlines()
    open_single_del_area_file.close()
    start = time.time()
    pool_size = 10
    pool = ThreadPool(pool_size)
    pool.map_async(process, single_del_area_file)
    pool.close()
    pool.join()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

