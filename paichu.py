import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    picpath = r'./code/one'
    pic_list = os.listdir(picpath)
    pic_list.sort(key=lambda x: int(x[3:-4]))
    pic1=[]
    for pic in pic_list:
        pic2=pic[0:-4]
        pic1.append(pic2)
    print(len(pic1))
    seqpath = r'./code/information/one/'
    seq_list = os.listdir(seqpath)
    seq_list.sort(key=lambda x: int(x[3:-4]))
    seq1 = []
    for seq in seq_list:
        seq2 = seq[0:-4]
        seq1.append(seq2)
    print(len(seq1))
    paichu = []
    i = 0
    for x in seq1:
        if x not in pic1:
            x=x+str('.txt')
            paichu.append(x)
            # s = r'E:\PyCharmDocument\yxy\sequence' + '/' + str(x)
            # print('sum:%d,number:rm -rf %s'%(i,s))
            i = i+1
            os.remove(seqpath + '/' + str(x))
    print(i)
    print(len(seq1) - i)


