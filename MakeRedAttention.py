from PIL import Image
import numpy as np
import os
import time
from multiprocessing.dummy import Pool as ThreadPool
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def process(image,image_src,new_src):
    print(image)
    im1 = Image.open(image_src + '/' + image)
    rgb_im1 = im1.convert('RGB')
    rgb_im1 = np.array(rgb_im1)
    rgb_im2 = np.ones(30000).reshape(100, 100, 3)
    rgb_im2 = rgb_im2 * 255
    for i in range(100):
        for j in range(100):
            if rgb_im1[i, j][0] == 255 and rgb_im1[i, j][1] == 0 and rgb_im1[i, j][2] == 0 :
                rgb_im2[i, j] = (255, 255, 255)
            else:
                rgb_im2[i, j] = (0, 0, 0)
    img3 = np.concatenate([rgb_im1, rgb_im2], axis=1)
    img3 = Image.fromarray(np.uint8(img3))
    newpath = new_src + '/' + image
    img3.save(newpath)

if  __name__ == '__main__':
    image_src = r'./assemblies/test/picture/'
    new_src = r'./assemblies/test/picture-attention/'
    image_file = os.listdir(image_src)
    image_file.sort(key=lambda x: int(x[:-4]))
    start = time.time()
    pool_size = 10
    pool = ThreadPool(pool_size)
    for image in image_file:
        pool.apply_async(process, args=(image,image_src,new_src))
    pool.close()
    pool.join()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

