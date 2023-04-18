from option import Option
from load import Loader
from preprocess import PreProcessor
from preprocess_3d import PreProcessor_3d
from RBM import RBM
from DBN import DBN
from DBN import DBN_last_layer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tsai.all import *

def my_main():

    option=Option()

    loader = Loader(option)
    x = loader.load_data()

    preProccessor = PreProcessor(x, option)

    x=preProccessor.getx()
    y=preProccessor.gety()
    origin_x=x

    if option.preprecess_opt.is_ssa:
        ssa_result=preProccessor.ssa()
        x = preProccessor.dataAdd(x,ssa_result)
    if option.preprecess_opt.is_fft:
        fft_result=preProccessor.fft(origin_x)
        x=preProccessor.dataAdd(x,fft_result)
    if option.preprecess_opt.is_concatenate:
        x = preProccessor.normalize(x)
    if option.preprecess_opt.is_concatenate:
        x,y=preProccessor.dataConcatenate(x,y)

    train_x, train_y, test_x, test_y = preProccessor.generateTrainTest(x,y)

    dbn = DBN(train_x.shape[1], option)
    dbn.train_DBN(train_x)


    model = torch.nn.Sequential(dbn.initialize_model(), torch.nn.Softmax(dim=1))
    torch.save(model, 'mnist_trained_dbn_classifier.pt')


    y_new, x_new = dbn.reconstructor(test_x)

    # dbn_last_layer = DBN_last_layer(model, train_x, train_y, test_x, test_y, 100, 32, 0.0001)
    # dbn_last_layer.train(model,train_x,train_y,train_x,train_y,test_x,test_y,100)
    print(model)



def fft_dbn_train():
    computer_setup()
    option=Option()
    loader = Loader(option)
    x_3d,y_3d=loader.load_3d()

    preprocessor=PreProcessor_3d(x_3d,y_3d,option)
    fft_x_3d,_=preprocessor.fft_3d(x_3d)
    fft_x_2d=preprocessor.x_3d_to_2d(fft_x_3d)

    fft_x_2d_tensor=torch.from_numpy(fft_x_2d).float()
    dbn = DBN(fft_x_2d_tensor.shape[1], option)
    dbn.train_DBN(fft_x_2d_tensor)


    _, fft_x_2d_features = dbn.reconstructor(fft_x_2d_tensor)
    samples=preprocessor.num_person*preprocessor.num_gesture
    fft_x_3d_features=preprocessor.x_2d_to_3d(fft_x_2d_features,samples=samples)
    return fft_x_3d_features


def model_compare():
    computer_setup()
    option=Option()
    loader = Loader(option)
    x_3d,y_3d=loader.load_3d()

    preprocessor=PreProcessor_3d(x_3d,y_3d,option)


    x,y=preprocessor.window_3d(x_3d)

    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, [Categorize()]]
    batch_tfms=[TSStandardize(),TSNormalize()]
    dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=True)
    dsets
    bs=64
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs*2])

    archs = [(FCN, {}), (ResNet, {}), (xresnet1d34, {}), (ResCNN, {}), 
            (LSTM, {'n_layers':1, 'bidirectional': False}), (LSTM, {'n_layers':2, 'bidirectional': False}), (LSTM, {'n_layers':3, 'bidirectional': False}), 
            (LSTM, {'n_layers':1, 'bidirectional': True}), (LSTM, {'n_layers':2, 'bidirectional': True}), (LSTM, {'n_layers':3, 'bidirectional': True}),
            (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}), (OmniScaleCNN, {}), (mWDN, {'levels': 4})]

    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls, **k)
        print(model.__class__.__name__)
        learn = Learner(dls, model,  metrics=accuracy)
        start = time.time()
        learn.fit_one_cycle(100, 1e-3)
        elapsed = time.time() - start
        vals = learn.recorder.values[-1]
        results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
        results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
        os.system('cls' if os.name == 'nt' else 'clear')
        display(results)

def two_model_train():

    computer_setup()
    option=Option()
    loader = Loader(option)
    x_3d,y_3d=loader.load_3d()

    preprocessor=PreProcessor_3d(x_3d,y_3d,option)
    fft_x_3d_temp,_=preprocessor.fft_3d(x_3d)
    fft_x_3d,y=preprocessor.window_3d(fft_x_3d_temp)

    x_3d,y=preprocessor.window_3d(x_3d)
    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, [Categorize()]]
    batch_tfms=[TSStandardize(),TSNormalize()]
    x_dsets = TSDatasets(x_3d, y, tfms=tfms, splits=splits, inplace=True)
    bs=64
    x_dls   = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[bs, bs*2],batch_tfms=batch_tfms)
    x_model = build_ts_model(XceptionTime, dls=x_dls)

    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, [Categorize()]]
    fft_dsets = TSDatasets(fft_x_3d, y, tfms=tfms, splits=splits, inplace=True)
    bs=64
    fft_dls   = TSDataLoaders.from_dsets(fft_dsets.train, fft_dsets.valid, bs=[bs, bs*2])
    fft_model = create_model(OmniScaleCNN, dls=fft_dls)

    learn = Learner(fft_dls, fft_model, metrics=[accuracy, RocAuc()])
    learn.fit_one_cycle(100, 1e-3)
    learn.save_all(path='models', dls_fname='x_dls', model_fname='x_model', learner_fname='x_learner')
    learn = Learner(x_dls, x_model, metrics=[accuracy, RocAuc()])
    learn.fit_one_cycle(100, 1e-3)
    learn.save_all(path='models', dls_fname='fft_dls', model_fname='fft_model', learner_fname='fft_learner')


two_model_train()




# archs = [(FCN, {}), (ResNet, {}), (xresnet1d34, {}), (ResCNN, {}), 
#         (LSTM, {'n_layers':1, 'bidirectional': False}), (LSTM, {'n_layers':2, 'bidirectional': False}), (LSTM, {'n_layers':3, 'bidirectional': False}), 
#         (LSTM, {'n_layers':1, 'bidirectional': True}), (LSTM, {'n_layers':2, 'bidirectional': True}), (LSTM, {'n_layers':3, 'bidirectional': True}),
#         (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}), (OmniScaleCNN, {}), (mWDN, {'levels': 4})]

# results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
# for i, (arch, k) in enumerate(archs):
#     model = create_model(arch, dls=dls, **k)
#     print(model.__class__.__name__)
#     learn = Learner(dls, model,  metrics=accuracy)
#     start = time.time()
#     learn.fit_one_cycle(100, 1e-3)
#     elapsed = time.time() - start
#     vals = learn.recorder.values[-1]
#     results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
#     results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
#     os.system('cls' if os.name == 'nt' else 'clear')
#     display(results)



# fft_x_2d=preprocessor.x_3d_to_2d(fft_x_3d)

# dbn = DBN(fft_x_2d_tensor.shape[1], option)
# dbn.train_DBN(fft_x_2d_tensor)

# model = dbn.initialize_model()
# _, fft_x_2d_features = dbn.reconstructor(fft_x_2d_tensor)

# samples=preprocessor.num_person*preprocessor.num_gesture
# steps=option.preprecess_opt.window_length
# fft_x_3d_features=preprocessor.deconcatenate(fft_x_2d_features,steps=steps)