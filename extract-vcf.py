# svim.vcf的提取
svim_path=r'.\real\hg003\svim\chr21/'
def get_svim_area(svim_path):
    open_svim_result = open(svim_path + 'variants.vcf')
    svim_list = [i.strip().split('\t') for i in open_svim_result.readlines()]
    open_svim_result.close()
    write_file = open(r'.\hg003\benchmark\svim.txt', "w+")
    for j in svim_list:
        if(len(j)>8):
            if ('DEL' in j[7] and j[6] == 'PASS' and int(j[5])>=5):
                start = j[1]
                end = j[7].split(';')[1].split('=')[-1]
                write_file.write(start + '\t' + end + '\t' + str(int(end) - int(start) + 1) + '\n')
    write_file.close()
svim_result = get_svim_area(svim_path)

# sniffles.vcf的提取
sniffles_path = r'.\real\hg003\sniffles\\'
#sniffles_path = r'E:\PyCharmDocument\yxy\road_2\all-vcf\\'
def get_sniffles_area(sniffles_path):
    open_sniffles = open(sniffles_path + 'hg003_hg19_chr21.vcf')
    sniffles_list = [i.strip().split('\t') for i in open_sniffles.readlines()]
    open_sniffles.close()
    write_file = open(r'.\road_2\hg003\benchmark\sniffles.txt', "w+")
    for j in sniffles_list:
        if (len(j) >= 7):
            if 'SVTYPE=DEL' in j[7] and j[0]=='chr21':
                if j[6] in ['PASS']:
                    start = j[1]
                    end = j[7].split(';')[3].split('=')[-1]
                    write_file.write(start + '\t' + end + '\t' + str(int(end) - int(start) + 1) + '\n')
    write_file.close()
sniffles_result = get_sniffles_area(sniffles_path)

# cutesv.vcf的提取
cutesv_path = r'.\road_2\all-vcf\\'
def get_cutesv_area(cutesv_path):
    open_cutesv = open(cutesv_path + 'HG003_PB_30x_origin.vcf')
    cutesv_list = [i.strip().split('\t') for i in open_cutesv.readlines()]
    open_cutesv.close()
    write_file = open(r'.\road_2\hg003\benchmark\cutesv.txt', "w+")
    for j in cutesv_list:
        if (len(j) >= 7):
            if  j[0] == 'chr21' and 'SVTYPE=DEL' in j[7] :
                if j[6] in ['PASS']:
                    start = j[1]
                    end = j[7].split(';')[3].split('=')[-1]
                    write_file.write(start + '\t' + end + '\t' + str(int(end) - int(start) + 1) + '\n')
    write_file.close()
cutesv_result = get_cutesv_area(cutesv_path)

# pbsv.vcf的提取
pbsv_path = r'.\\'
def get_pbsv_area(pbsv_path):
    open_pbsv = open(pbsv_path + 'hg19.HG002.pbsv.vcf')
    pbsv_list = [i.strip().split('\t') for i in open_pbsv.readlines()]
    open_pbsv.close()
    write_file = open(r'.\pbsv\pbsv1.txt', "w+")
    for j in pbsv_list:
        if (len(j) >= 7):
            if  j[0] == '21' and 'SVTYPE=DEL' in j[7] :
                start = j[1]
                end = j[7].split(';')[2].split('=')[1]
                if int(end) - int(start) + 1 > 35:
                    write_file.write(start + '\t' + end + '\t' + str(int(end) - int(start) + 1) + '\n')
    write_file.close()
pbsv_result = get_pbsv_area(pbsv_path)

# Tier1_v0.6.vcf的提取
tier_path = r'.\real\hg002\\'
def get_tier_area(tier_path):
    open_tier = open(tier_path + 'HG002_SVs_Tier1_v0.6.vcf')
    tier_list = [i.strip().split('\t') for i in open_tier.readlines()]
    open_tier.close()
    write_file = open(r'.\real\hg002\tier.txt', "w+")
    for j in tier_list:
        if (len(j) >= 7):
            if  j[0] == '21' and 'SVTYPE=DEL' in j[7] and int(j[5]) >=10 :
                start = j[1]
                end = j[7].split('END=')[-1].split(';')[0]
                if int(end) - int(start) + 1 > 35:
                    write_file.write(start + '\t' + end + '\t' + str(int(end) - int(start) + 1) + '\n')
    write_file.close()
tier_result = get_tier_area(tier_path)
