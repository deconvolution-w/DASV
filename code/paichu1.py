import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    picpath = r'./hg002-pic/one'
    filename = r'./210811.txt'
    file = open(filename, 'a+')
    pic_list = os.listdir(picpath)
    pic1=[]
    for pic in pic_list:
        pic2=pic[3:-4]
        print(pic2)
        pic1.append(pic2)
        file.write(pic2)
        s = '\n'
    file.write(s)
    file.close()