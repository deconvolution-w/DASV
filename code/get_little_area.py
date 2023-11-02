save_chr_single_area_path = r'.\nn\TAIR.txt'

def get_del_point_inf(start,stop): #获得左或右的范围信息
    final_del_area = []
    for i in range(start,stop,100):
        final_del_area.append([i-50,i+49])
        # final_del_area.append([i, i + 99])
    return final_del_area

start = 10000000
stop  = 19000000
a = get_del_point_inf(start,stop)
f = open(save_chr_single_area_path, 'w+')
k=0
for i in a:
    print(str(i[0]) + "\t" + str(i[1]),file=f)
    k = k + 1
    print(k)
