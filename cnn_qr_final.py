from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import os
import qrcode
import glob
from PIL import Image
import random

train_size = 360  # 20000
test_size = 40  # 100
list_size = 30
max_num = 122 + 1  # 正解に使われる最大が122だから0～122で123分類？
size = 73

model_weight = 'cnn_qr.hdf5'  # モデルを保存する名前

np.set_printoptions(precision=2, suppress=True)  # 指数表示禁止、少数表示
np.set_printoptions(threshold=300000)  # 要素の省略禁止

# Trainデータ作成
max_url_length = 30
min_url_length = 25
folder = "C:/Users/****/QRcode"
list = "abcdefghijklmnopqrstuvwxyz0123456789:/.-http.www.//.co.jp//"  # 使える文字のリスト


def np_chr2dec(text):
    y = np.array([])
    for j in text:
        i = ord(j)
        y = np.append(y, i)
    return y


def dec2chr(num):
    text = ""
    for i in num:
        i = int(i)
        chr(i)
        text += chr(i)
    return text


def makeQR(url, name):
    qr = qrcode.QRCode(version=12, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=2, border=4)
    print(url)
    qr.add_data(url)
    qr.make()
    img = qr.make_image(fill_color="black", back_color="white")
    os.chdir(folder)
    img.save(name + '.jpg')


def make_dataset(count):
    for cnt in range(count):
        y_bin = np.zeros((max_url_length))
        url_length = random.randint(min_url_length, max_url_length)
        try:
            for i in range(max_url_length):
                if i < url_length:
                    rand = random.randint(0, len(list) - 1)
                    ii = ord(list[rand])  # chr2dec
                    y_bin[i] = ii  # 10進数バイナリ
                elif i == url_length:
                    url = dec2chr(y_bin)
                    y_bin[i] = 32  # スペースで埋める
                else:
                    y_bin[i] = 32  # スペースで埋める
            cnt = str(cnt).zfill(5)
            makeQR(url, cnt)
            file = open('list.txt', 'a', encoding='utf')  # 書き込みモードでオープン
            file.write(url + "\n")
        except:
            print("Error= ", url)
    return y_bin

def model_build():
    x_train = []  # 普通のリスト
    x_test = []
    y_train = []
    y_test = []

    file = open('list.txt', 'r', encoding='utf')  # ファイルオープン
    url_list = []
    while (True):
        Line = file.readline()
        if Line == "":
            break
        print(Line)
        Line = Line.replace("\n", "")
        url_list.append(np_chr2dec(Line))  # 10進数に変換して追加

    # Trainデータ作成
    image_count = 0
    os.chdir(folder + "/data")
    filelist = glob.glob("./*")
    print(type(url_list))
    for picture in filelist:
        if ".jpg" in picture:
            print("picture = " + picture)
            image = np.array(Image.open(folder + "/data/" + picture).convert('L'))  # モノクロ　'L'
            # reshapeを使って開かれた配列を1次元配列に変換する
            image_risize = image.reshape(image.size)
            image_risize = image_risize.astype('float32') / 255
            if len(url_list[image_count]) == 0:
                break
            for i in range(max_url_length):
                image_risize_plus = np.append(image_risize, i)  # 画素 + 答えのインデックス
                if image_count <= train_size:
                    # バッチリストに追加していく
                    x_train.append(image_risize_plus)
                    y_train.append(url_list[image_count][i])
                elif image_count <= train_size + test_size:
                    x_test.append(image_risize_plus)
                    y_test.append(url_list[image_count][i])
                elif image_count > train_size + test_size:
                    break
        image_count += 1

    # arrayに変換
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 正解ラベルをone-hot-encoding
    y_train = to_categorical(y_train, max_num)
    y_test = to_categorical(y_test, max_num)

    model = model_read()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1460, epochs=60, verbose=1)  # データを学習

    model.save_weights(model_weight)  # 学習結果を保存
    model.evaluate(x_test, y_test)


def model_read():
    os.chdir(folder)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=size * size + 1))
    model.add(Dense(64, activation='relu', input_dim=64))
    model.add(Dense(64, activation='relu', input_dim=64))
    model.add(Dense(64, activation='relu', input_dim=64))
    model.add(Dense(max_num, activation='softmax'))  # softmax
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("") #改行が足りない
    # 学習済みモデルの重みのデータを読み込み
    if os.path.exists(model_weight):
        model.load_weights(model_weight)  # 最初は重みデータファイルがないとErrorになる
    else:
        print("Fail to read model weigth!")
    return model


def check():
    check_folder = "C:/Users/****/QRcode/check"
    model = model_read()
    os.chdir(check_folder)
    filelist = glob.glob("./*")

    for picture in filelist:
        if ".jpg" in picture:
            image = np.array(Image.open(check_folder + "/" + picture).convert('L'))  # モノクロ　'L'
            # reshapeを使って開かれた配列を1次元配列に変換する
            image_risize = image.reshape(image.size)
            image_risize = image_risize.astype('float32') / 255
            answer = ""  # 復元されたURL
            bin = ""
            for i in range(max_url_length):
                image_risize_plus = np.append(image_risize, i)  # 画素 + 答えのインデックス
                image_risize_plus = image_risize_plus.reshape(1, size * size + 1)  # 1次元化
                # 判定
                res = model.predict([image_risize_plus])
                y = res.argmax()  # 値の中で最も値が大きいものが答え
                bin += str(y)
                answer += chr(y)
            print(picture)
            print(bin)
            print(answer)


if __name__ == "__main__":
    make_dataset(20)
    model_build()
    check()
