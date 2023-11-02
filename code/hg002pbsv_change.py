cutesv_path = r'./hg003combisv/'
def get_cutesv_area(cutesv_path):
    open_cutesv = open(cutesv_path + 'hg003.pbsv.vcf')
    cutesv_list = [i.strip().split('\t') for i in open_cutesv.readlines()]
    open_cutesv.close()
    write_file = open(r'./hg003combisv/pbsv.vcf', "w+")
    for i in cutesv_list:
        if (len(i)<7):
            for j in range(0, len(i)):
                if j != (len(i) - 1):
                    write_file.write(str(i[j]) + '\t')
                else:
                    write_file.write(str(i[j]) + '\n')
        if (len(i) >= 7) and i[7] == 'INFO':
            for j in range(0, len(i)):
                if j != (len(i) - 1):
                    write_file.write(str(i[j]) + '\t')
                else:
                    write_file.write(str(i[j]) + '\n')
        if (len(i) >= 7) and i[7] != 'INFO':
            i[0] = 'chr21'
            i[6] = 'PASS'
            inf  = i[7].split(';')
            inf0 = inf[1]
            inf[1] = inf[2]
            inf[2] = inf0
            string = inf[0] +';'+ inf[1] +';'+ inf[2] +';'+ inf[3]
            i[7] = string
            print(i[7])
            for j in range(0, len(i)):
                if j != (len(i) - 1):
                    write_file.write(str(i[j]) + '\t')
                else:
                    write_file.write(str(i[j]) + '\n')
    write_file.close()

cutesv_result = get_cutesv_area(cutesv_path)