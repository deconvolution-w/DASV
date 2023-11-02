import os

def get_single_area_result(single_area_result):
    k = 0
    temp = []
    open_file = open(single_area_result)
    single_area_result_list = [[int(i.strip().split('\t')[0]), int(i.strip().split('\t')[1])] for i in
                               open_file.readlines()]
    open_file.close()
    sorted_single_area_result_list = sorted(single_area_result_list, key=lambda x: x[0])
    for j1 in range(len(sorted_single_area_result_list)):
        if sorted_single_area_result_list[j1][1] == 1:
            temp.append([sorted_single_area_result_list[j1][0], sorted_single_area_result_list[j1][0] + 99])
            k = j1 + 1
            break
    for j2 in range(k, len(sorted_single_area_result_list)):
        if sorted_single_area_result_list[j2][1] == 1:
            if sorted_single_area_result_list[j2][0] <= temp[-1][1] + 1:
                temp[-1][1] = sorted_single_area_result_list[j2][0] + 99
            else:
                temp.append([sorted_single_area_result_list[j2][0], sorted_single_area_result_list[j2][0] + 99])
    return temp

def sort_and_merge__result(area_result):
    sorted_tools_result_list = sorted(area_result, key = lambda x:x[0])
    final_tools_result_list = [sorted_tools_result_list[0]]
    for j in range(1,len(sorted_tools_result_list)):
        if sorted_tools_result_list[j][0] > final_tools_result_list[-1][-1]:
            final_tools_result_list.append(sorted_tools_result_list[j])
        else:
            final_tools_result_list[-1][-1] = sorted_tools_result_list[j][1]
    return final_tools_result_list

if __name__ == '__main__':
    single_result_path = r'./point_predict/chr4.txt'
    single_area_result_path = r'./area_predict/chr4.txt'
    temp = get_single_area_result(single_result_path)
    temp = sort_and_merge__result(temp)
    f = open(single_area_result_path, 'w+')
    for i in temp:
        print(str(i[0]) + '\t' + str(i[1]), file=f)
    print("end\n")