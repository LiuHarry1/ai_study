import os as os
import numpy as np
from sklearn.model_selection import train_test_split

datapath = r'./aclImdb'
save_dir = r'./data'

def get_data(datapath):
    pos_files = os.listdir(datapath + '/pos')
    neg_files = os.listdir(datapath + '/neg')
    print(len(pos_files))
    print(len(neg_files))

    pos_all = []
    neg_all = []
    for pf, nf in zip(pos_files, neg_files):
        with open(datapath + '/pos' + '/' + pf, encoding='utf-8') as f:
            s = f.read()
            pos_all.append(s)
        with open(datapath + '/neg' + '/' + nf, encoding='utf-8') as f:
            s = f.read()
            neg_all.append(s)

    X_orig= np.array(pos_all + neg_all)
    Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])
    print("X_orig:", X_orig.shape)
    print("Y_orig:", Y_orig.shape)

    return X_orig, Y_orig

def generate_train_data():
    X_orig, Y_orig = get_data(datapath+r'/train')
    X_test, Y__test = get_data(datapath+r'/test')
    X = np.concatenate([X_orig, X_test])
    Y = np.concatenate([Y_orig, Y__test])
    np.random.seed = 1
    random_indexs = np.random.permutation(len(X))
    X = X[random_indexs]
    Y = Y[random_indexs]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("x_val:", X_val.shape)
    print("y_val:", y_val.shape)
    np.savez(save_dir + '/imdb_train', x=X_train, y=y_train)
    np.savez(save_dir + '/imdb_test', x=X_test, y=y_test)
    np.savez(save_dir + '/imdb_val', x=X_val, y=y_val)

if __name__ == '__main__':
    generate_train_data()
