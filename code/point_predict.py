import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.layers import Conv2D,Dropout,LeakyReLU,MaxPooling2D,Reshape,Conv1D,MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Flatten
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def read_image(img_name):
    im=Image.open(img_name).convert('RGB')
    data=np.array(im)
    return data
def read_seq(filename):
    tracks = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
    tracks = tracks.T
    return tracks

def load(PicPath,SeqPath):
    pic_list = os.listdir(PicPath)
    pic_list.sort(key=lambda x: int(x[:-4]))
    print(pic_list)
    list=[]
    for point in pic_list:
        list.append(point[:-4])
    length0 = len(pic_list)
    X1 = []
    for i in range(0, length0):
        if pic_list[i].endswith('.png'):
                fd = os.path.join(PicPath, pic_list[i])
                X1.append(read_image(fd))
    pic = np.array(X1)
    seq_list = os.listdir(SeqPath)
    seq_list.sort(key=lambda x: int(x[:-4]))
    print(seq_list)
    seq_length = len(seq_list)
    Z1 = []
    for i in range(0, seq_length):
        if seq_list[i].endswith('.txt'):
            fd = os.path.join(SeqPath, seq_list[i])
            Z1.append(read_seq(fd))
    seq = np.array(Z1)
    pic = pic.astype('float32') / 255
    seq = seq.astype('float32') / 10
    return pic, seq, list

def picture(input_pic):
    input1 = input_pic[:, :, :100, :]
    input2 = input_pic[:, :, 100:200, :]
    A = Conv2D(8, 3, padding='same')(input1)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(8, 3, activation='sigmoid', padding='same')(input2)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv2D(8, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(8, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(16, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(8, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(8, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(16, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv2D(16, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(16, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(32, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(16, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(16, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(32, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv2D(32, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(32, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(64, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(32, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(32, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(64, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv2D(64, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(64, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(128, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(64, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(64, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    pic = MaxPooling2D(pool_size=(2, 2))(A)
    print(pic.shape)
    return pic

def sequence(input_seq):
    print(input_seq.shape)
    A = Conv1D(8, 3, padding='same')(input_seq)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv1D(8, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(8, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(16, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(8, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(16, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv1D(16, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(16, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(32, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(16, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(32, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv1D(32, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(32, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(64, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(32, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(64, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv1D(64, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(64, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(128, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(64, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    seq = MaxPooling1D(pool_size=2)(A)
    print(seq.shape)
    return seq
#mix
def classification(input_shape1,input_shape2,num_classes,droput):
    input_pic = Input(shape=input_shape1)
    input_seq = Input(shape=input_shape2)
    pic = picture(input_pic=input_pic)
    seq = sequence(input_seq=input_seq)

    #edition_1
    hs = GlobalAveragePooling1D()(seq)
    hs = Reshape((1, 1, hs.shape[1]))(hs)
    hs = Conv2D(64 // 4, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True,
                activation="relu")(hs)
    hs = Conv2D(64, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True)(hs)
    hb = GlobalMaxPooling1D()(seq)
    hb = Reshape((1, 1, hb.shape[1]))(hb)
    hb = Conv2D(64 // 4, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True,
                activation="relu")(hb)
    hb = Conv2D(64, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True)(hb)
    out = hs + hb
    out = tf.nn.sigmoid(out)
    input4 = out * pic
    input4 = Flatten()(input4)

    # # edition_2
    # pic = Flatten()(pic)
    # print(pic.shape)
    # seq = Flatten()(seq)
    # input4 = concatenate([pic,seq])


    print(input4.shape)
    y = Dropout(droput)(input4)
    print(y.shape)
    y = Dense(128, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(y)
    y1 = Dense(64, kernel_regularizer=l2(1e-4), use_bias=True)(y)
    y1 = Dense(128, kernel_regularizer=l2(1e-4), use_bias=True)(y1)
    y1 = tf.nn.sigmoid(y1)
    y = y * y1
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(64, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(y)
    y1 = Dense(32, kernel_regularizer=l2(1e-4), use_bias=True)(y)
    y1 = Dense(64, kernel_regularizer=l2(1e-4), use_bias=True)(y1)
    y1 = tf.nn.sigmoid(y1)
    y = y * y1
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(32, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(y)
    y1 = Dense(16, kernel_regularizer=l2(1e-4), use_bias=True)(y)
    y1 = Dense(32, kernel_regularizer=l2(1e-4), use_bias=True)(y1)
    y1 = tf.nn.sigmoid(y1)
    y = y * y1
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(16, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(y)
    y1 = Dense(8, kernel_regularizer=l2(1e-4), use_bias=True)(y)
    y1 = Dense(16, kernel_regularizer=l2(1e-4), use_bias=True)(y1)
    y1 = tf.nn.sigmoid(y1)
    y = y * y1
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(10, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(y)
    y = BatchNormalization(momentum=0.8)(y)
    y = LeakyReLU(alpha=0.2)(y)
    outputs = Dense(num_classes, activation='softmax')(y)
    model = Model([input_pic,input_seq], outputs)
    return model

def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision
def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (pp + K.epsilon())
    return recall
def F1(y_true, y_pred):
    Precision = precision(y_true, y_pred)
    Recall = recall(y_true, y_pred)
    f1 = 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))
    return f1
    
def merge(pred):
    for k in range(0, len(pred)-4):
        if((pred[k]==1)and(pred[k+1]==0)and(pred[k+2]==1)):
            pred[k+1]=1
        elif ((pred[k]==1)and(pred[k+1]==0)and(pred[k+2]==0)and(pred[k+3]==1)):
            pred[k+1]=1
            pred[k+2]=1
        elif ((pred[k]==1)and(pred[k+1]==0)and(pred[k+2]==0)and(pred[k+3]==0)and(pred[k+4]==1)):
            pred[k + 1] = 1
            pred[k + 2] = 1
            pred[k + 3] = 1
    return pred

if __name__ == '__main__':
    Picpath = r'./assemblies/test/picture-attention'
    Seqpath = r'./assemblies/test/information'
    txtpath = r'./assemblies/test/ending/point'
    weight_path = r'./model/0815-model.h5'
    chid ='chr4'
    mymodel = classification(input_shape1=(100, 200, 3),input_shape2=(100,100),num_classes=2,droput=0.4)
    
    from tensorflow.keras.utils import plot_model
    plot_model(mymodel, to_file='mymodel1019-noshape.png', show_shapes=False, show_layer_names=False)
    mymodel.load_weights(weight_path)
    picture, sequence, list = load(Picpath, Seqpath)
    y_pred = mymodel.predict([picture,sequence])
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = merge(y_pred)
    change = []
    text_src = txtpath + '/' + str(chid) +'.txt'
    f = open(text_src, 'w+')
    for i in range(0, len(y_pred)):
        change.append(list[i] +' ' + str(y_pred[i]))
        print(list[i] +'\t' + str(y_pred[i]),file=f)
        
    print(change)
    f.close()    
