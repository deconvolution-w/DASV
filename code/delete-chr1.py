import os
import sys
path0=r'./hg002-pic/chr1'
path1=r'./hg002-pic/chr1'+'/'
sys.path.append(path1)
files = os.listdir(path0)
files.sort(key= lambda x:int(x[3:-4]))
for filename in files:
    portion = os.path.splitext(filename)
    newname = portion[0][4:]+'.png'
    filenamedir=path1 +filename
    newnamedir=path1+newname
    os.rename(filenamedir,newnamedir)