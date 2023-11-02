import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.layers import Conv2D,Dropout,LeakyReLU,MaxPooling2D,Reshape,Conv1D,MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add, Flatten
import tensorflow.keras.backend as K

def read_image(img_name):
    im=Image.open(img_name).convert('RGB')
    data=np.array(im)
    return data
def read_seq(filename):
    tracks = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
    tracks = tracks.T
    return tracks

def load_data():
    rate1=0.6
    rate2=0.8
    pic_train0=r'./hg002-pic/zero-zero'
    path_list0 = os.listdir(pic_train0)
    path_list0.sort(key=lambda x: int(x[3:-4]))
    length0 = len(path_list0)
    X1 = []
    Y1 = []
    for i in range(0, length0):
        if path_list0[i].endswith('.png'):
            fd = os.path.join(pic_train0, path_list0[i])
            X1.append(read_image(fd))
            Y1.append(0)
    pic1 = np.array(X1)
    pic1_index = len(pic1)
    label1 = np.array(Y1)
    pic1_train = pic1[0:int(pic1_index*rate1), :, :, :]
    label1_train = label1[0:int(pic1_index*rate1)]
    pic1_test = pic1[int(pic1_index*rate1):int(pic1_index*rate2), :, :, :]
    label1_test = label1[int(pic1_index*rate1):int(pic1_index*rate2)]
    pic1_validation = pic1[int(pic1_index*rate2):, :, :, :]
    label1_validation = label1[int(pic1_index*rate2):]

    seq_train0 = r'./information/zero'
    seq_list0 = os.listdir(seq_train0)
    seq_list0.sort(key=lambda x: int(x[3:-4]))
    seq_length0 = len(seq_list0)
    Z1 = []
    for i in range(0, seq_length0):
        if seq_list0[i].endswith('.txt'):
            fd = os.path.join(seq_train0, seq_list0[i])
            Z1.append(read_seq(fd))
    seq1 = np.array(Z1)
    seq1_index = len(Z1)
    seq1_train = seq1[0:int(seq1_index * rate1), :, :, ]
    seq1_test = seq1[int(seq1_index * rate1):int(seq1_index * rate2), :, :, ]
    seq1_validation = seq1[int(seq1_index * rate2):, :, :, ]

    pic_train1 =r'./hg002-pic/one-one'
    path_list1 = os.listdir(pic_train1)
    path_list1.sort(key=lambda x: int(x[3:-4]))
    length1 = len(path_list1)
    X2 = []
    Y2 = []
    for i in range(0, length1):
        if path_list1[i].endswith('.png'):
            fd = os.path.join(pic_train1, path_list1[i])
            X2.append(read_image(fd))
            Y2.append(1)
    pic2 = np.array(X2)
    pic2_index = len(pic2)
    label2 = np.array(Y2)
    pic2_train = pic2[0:int(pic2_index * rate1), :, :, :]
    label2_train = label2[0:int(pic2_index * rate1)]
    pic2_test = pic2[int(pic2_index * rate1):int(pic2_index * rate2), :, :, :]
    label2_test = label2[int(pic2_index * rate1):int(pic2_index * rate2)]
    pic2_validation = pic2[int(pic2_index * rate2):, :, :, :]
    label2_validation = label2[int(pic2_index * rate2):]

    seq_train1 = r'./information/one'
    seq_list1 = os.listdir(seq_train1)
    seq_list1.sort(key=lambda x: int(x[3:-4]))
    seq_length1 = len(seq_list1)
    Z2 = []
    for i in range(0, seq_length1):
        if seq_list1[i].endswith('.txt'):
            fd = os.path.join(seq_train1, seq_list1[i])
            Z2.append(read_seq(fd))
    seq2 = np.array(Z2)
    seq2_index = len(Z2)
    seq2_train = seq2[0:int(seq2_index * rate1), :, :, ]
    seq2_test = seq2[int(seq2_index * rate1):int(seq2_index * rate2), :, :, ]
    seq2_validation = seq2[int(seq2_index * rate2):, :, :, ]

    X_pic_train = np.concatenate((pic1_train,pic2_train),axis=0)
    X_seq_train = np.concatenate((seq1_train, seq2_train), axis=0)
    Y_pic_train = np.concatenate((label1_train, label2_train), axis=0)
    X_pic_test = np.concatenate((pic1_test, pic2_test), axis=0)
    X_seq_test = np.concatenate((seq1_test, seq2_test), axis=0)
    Y_pic_test = np.concatenate((label1_test, label2_test), axis=0)
    X_pic_validation = np.concatenate((pic1_validation, pic2_validation), axis=0)
    X_seq_validation = np.concatenate((seq1_validation, seq2_validation), axis=0)
    Y_pic_validation = np.concatenate((label1_validation, label2_validation), axis=0)

    x3_index = len(X_pic_train)
    index3 = np.arange(x3_index)
    np.random.seed(10)
    np.random.shuffle(index3)
    X_pic_train = X_pic_train[index3, :, :, :]
    X_seq_train = X_seq_train[index3, :, :]
    Y_pic_train = Y_pic_train[index3]

    x4_index = len(X_pic_test)
    index4 = np.arange(x4_index)
    np.random.seed(10)
    np.random.shuffle(index4)
    X_pic_test = X_pic_test[index4, :, :, :]
    X_seq_test = X_seq_test[index4, :, :]
    Y_pic_test = Y_pic_test[index4]

    x5_index = len(X_pic_validation)
    index5 = np.arange(x5_index)
    np.random.seed(10)
    np.random.shuffle(index5)
    X_pic_validation = X_pic_validation[index5, :, :, :]
    X_seq_validation = X_seq_validation[index5, :, :]
    Y_pic_validation = Y_pic_validation[index5]

    input_shape1 = X_pic_train[0].shape
    input_shape2 = X_seq_train[0].shape
    X_pic_train = X_pic_train.astype('float32') / 255
    X_pic_test = X_pic_test.astype('float32') / 255
    X_pic_validation = X_pic_validation.astype('float32') / 255
    X_seq_train = X_seq_train.astype('float32') /10
    X_seq_test = X_seq_test.astype('float32') / 10
    X_seq_validation = X_seq_validation.astype('float32') / 10
    Y_pic_train = to_categorical(Y_pic_train)
    Y_pic_test = to_categorical(Y_pic_test)
    Y_pic_validation = to_categorical(Y_pic_validation)
    num_classes = 2
    return X_pic_train,X_seq_train,X_pic_test,X_seq_test,Y_pic_train,Y_pic_test,\
           X_pic_validation,X_seq_validation,Y_pic_validation,input_shape1,input_shape2,num_classes

def picture(input_pic):
    input1 = input_pic[:, :, :100, :]
    input2 = input_pic[:, :, 100:200, :]
    A = Conv2D(10, 3, padding='same')(input1)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(10, 3, activation='sigmoid', padding='same')(input2)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv2D(10, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(10, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(20, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(10, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(10, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(30, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv2D(30, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(30, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(60, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(30, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(30, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(60, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv2D(60, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(60, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(120, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(60, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(60, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    A = MaxPooling2D(pool_size=(2, 2))(A)
    B = MaxPooling2D(pool_size=(2, 2))(B)
    A = Conv2D(100, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv2D(100, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(100, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(200, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv2D(100, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    B = Conv2D(100, 3, activation='sigmoid', padding='same')(B)
    A1 = tf.multiply(A, B)
    A = Add()([A1, A])
    print(A.shape)
    pic = MaxPooling2D(pool_size=(2, 2))(A)
    print(pic.shape)
    return pic

def sequence(input_seq):
    print(input_seq.shape)
    A = Conv1D(10, 3, padding='same')(input_seq)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv1D(10, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(10, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(20, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(10, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(30, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv1D(30, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(30, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(60, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(30, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(60, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 3
    for i1 in range(time):
        A1 = Conv1D(60, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(60, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(120, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(60, kernel_size=2, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A = Add()([A1, A])
        A = BatchNormalization(momentum=0.8)(A)
        A = LeakyReLU(alpha=0.2)(A)
    print(A.shape)
    A = MaxPooling1D(pool_size=2)(A)
    A = Conv1D(100, 3, padding='same')(A)
    A = BatchNormalization(momentum=0.8)(A)
    A = LeakyReLU(alpha=0.2)(A)
    time = 2
    for i1 in range(time):
        A1 = Conv1D(100, kernel_size=2, padding='same')(A)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(100, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(200, kernel_size=3, padding='same')(A1)
        A1 = BatchNormalization(momentum=0.8)(A1)
        A1 = LeakyReLU(alpha=0.2)(A1)
        A1 = Conv1D(100, kernel_size=2, padding='same')(A1)
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
    hs = Conv2D(100 // 4, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True,
                activation="relu")(hs)
    hs = Conv2D(100, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True)(hs)
    hb = GlobalMaxPooling1D()(seq)
    hb = Reshape((1, 1, hb.shape[1]))(hb)
    hb = Conv2D(100 // 4, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True,
                activation="relu")(hb)
    hb = Conv2D(100, kernel_size=1, strides=1, padding="same", kernel_regularizer=l2(1e-4), use_bias=True)(hb)
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

def build_and_train_models():
    batch_size = 2
    droput = 0.4
    X_pic_train, X_seq_train, X_pic_test, X_seq_test, Y_pic_train, Y_pic_test, \
    X_pic_validation, X_seq_validation, Y_pic_validation, input_shape1, input_shape2, num_classes=load_data()
    model = classification(input_shape1=input_shape1,input_shape2=input_shape2, num_classes=num_classes, droput=droput)
    optimizer = RMSprop(lr=0.000001, decay=1e-8, momentum=0.9)
    model.compile(loss='categorical_crossentropy', metrics=['acc',F1],optimizer=optimizer)
    model.summary()
    # callbacks_list = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath='E:/PyCharmDocument/yxy/mini/model/mymodel.h5' ,
    #         monitor='val_loss',
    #         save_best_only=True,
    #     )
    # ]
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/mynewmodel.h5',
            monitor='val_F1',
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
        )
    ]
    history = model.fit([X_pic_train,X_seq_train], Y_pic_train, batch_size=batch_size, epochs=25, shuffle=True,callbacks=callbacks_list,validation_data=([X_pic_test, X_seq_test], Y_pic_test))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig('./model/acc.jpg')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./model/loss.jpg')
    loss, acc, f1= model.evaluate([X_pic_validation, X_seq_validation],Y_pic_validation, batch_size=batch_size)
    print('Test accuracy: %.1f%%' % (100.0 * acc))
if __name__ == '__main__':
    build_and_train_models()