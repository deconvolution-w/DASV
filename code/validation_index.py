import numpy as np
import os
def sort_and_merge(ending):
    sorted_sort_and_merge_list = sorted(ending, key = lambda x:x[0])
    final_sort_and_merge_list = [sorted_sort_and_merge_list[0]]
    for j in range(1,len(sorted_sort_and_merge_list)):
        if sorted_sort_and_merge_list[j][0] > final_sort_and_merge_list[-1][-1]:
            final_sort_and_merge_list.append(sorted_sort_and_merge_list[j])
        else:
            final_sort_and_merge_list[-1][-1] = sorted_sort_and_merge_list[j][1]
    return final_sort_and_merge_list

def compare_with_vcf_and_tools_result(net, tools):
    open_file = open(net)
    net_list = [[int(i.strip().split('\t')[0]), int(i.strip().split('\t')[1])] for i in open_file.readlines()]
    open_file.close()
    net_list = sort_and_merge(net_list)
    temp = []
    for i in net_list:
        temp.append(i)
    net_shape_like_zero_array = np.zeros((len(temp), 1))
    open_file = open(tools)
    tools_list = [[int(i.strip().split('\t')[0]), int(i.strip().split('\t')[1])] for i in open_file.readlines()]
    open_file.close()
    tools_list = sort_and_merge(tools_list)
    temp1 = []
    for i in tools_list:
        temp1.append(i)
    tools_result_shape_like_zero_array = np.zeros((len(temp1), 1))

    net_array = np.hstack((np.array(temp), net_shape_like_zero_array)).astype(int)
    tools_result_array = np.hstack((np.array(temp1), tools_result_shape_like_zero_array)).astype(int)

    tp, fp, tn, fn = 0, 0, 0, 0
    for i in net_array:
        for j in tools_result_array:
            if ((j[1] < i[0]) or (j[0] > i[1])) == False:
                i[2] += 1
                j[2] += 1
    for i1 in tools_result_array :
        if i1[2] == 0:
            fn += 1
    for j1 in net_array:
        if j1[2] == 0:
            fp += 1
        else:
            tp += 1
    return fn, fp, tp

if __name__ == '__main__':
    net_area = r'./ending/area_predict/hg004-chr21.txt'
    tool_area = r'./code/hg004-benchmark/hg004/sniffles+smrtsv+svim+cutesv.txt'
    FN, FP, TP = 0, 0, 0
    fn, fp, tp = compare_with_vcf_and_tools_result(net_area, tool_area)
    FN = FN + fn
    FP = FP + fp
    TP = TP + tp
    precesion = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precesion * recall / (recall + precesion)
    print('precesion: ' + str(precesion) + '\nrecall: ' + str(recall) + '\nf1: ' + str(F1))
