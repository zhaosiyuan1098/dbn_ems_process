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
from tsai.inference import get_X_preds

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

def preload():

    computer_setup()
    option=Option()
    loader = Loader(option)
    x_3d,y_3d=loader.load_3d()

    preprocessor=PreProcessor_3d(x_3d,y_3d,option)
    fft_x_3d_temp,_=preprocessor.fft_3d(x_3d)
    fft_x_3d,y=preprocessor.window_3d(fft_x_3d_temp)

    x_3d,y=preprocessor.window_3d(x_3d)
    return x_3d,fft_x_3d,y
    
    
    
def train_two_model(x_3d,fft_x_3d,y):
    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, [Categorize()]]
    x_dsets = TSDatasets(x_3d, y, tfms=tfms, splits=splits, inplace=True)
    batch_tfms=[TSStandardize(),TSNormalize()]
    bs=64
    x_dls   = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[bs, bs*2],batch_tfms=batch_tfms)
    x_model = build_ts_model(XceptionTime, dls=x_dls)

    valid_dls_before_learn=x_dls.valid
    valid_dls_before_learn_ds=x_dls.valid_ds

    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, [Categorize()]]
    fft_dsets = TSDatasets(fft_x_3d, y, tfms=tfms, splits=splits, inplace=True)
    bs=64
    fft_dls   = TSDataLoaders.from_dsets(fft_dsets.train, fft_dsets.valid, bs=[bs, bs*2])
    fft_model = create_model(OmniScaleCNN, dls=fft_dls)

    learn = Learner(fft_dls, fft_model, metrics=[accuracy, RocAuc()])
    learn.fit_one_cycle(100, 1e-3)
    learn.save_all(path='models', dls_fname='fft_dls', model_fname='fft_model', learner_fname='fft_learner')

    learn = Learner(x_dls, x_model, metrics=[accuracy, RocAuc()])
    learn.fit_one_cycle(100, 1e-3)
    learn.save_all(path='models', dls_fname='x_dls', model_fname='x_model', learner_fname='x_learner')
    
    dls_after_learn=learn.dls
    valid_dls_after_learn=dls_after_learn.valid
    valid_dls_after_learn_ds=dls_after_learn.valid_ds
    return x_model,fft_model

# two_model_train()
def load_two_model():
    x_learn = load_learner_all(path='models', dls_fname='x_dls', model_fname='x_model', learner_fname='x_learner')
    dls = x_learn.dls
    valid_dl = dls.valid
    valid_probas, valid_targets, valid_preds = x_learn.get_preds(dl=valid_dl, with_decoded=True)
    print("x model accuracy=    "+str((valid_targets == valid_preds).float().mean()))
    print(valid_probas)
    print(valid_targets)
    print(valid_preds)
    
    
    fft_learn=load_learner_all(path='models', dls_fname='fft_dls', model_fname='fft_model', learner_fname='fft_learner')
    dls = fft_learn.dls
    valid_dl = dls.valid
    valid_probas, valid_targets, valid_preds = fft_learn.get_preds(dl=valid_dl, with_decoded=True)
    print("fft model accuracy=    "+str((valid_targets == valid_preds).float().mean()))
    return x_learn,fft_learn
  
    
def dbn_train(x_3d,fft_x_3d,y,):
    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=7, shuffle=True)
    y_test = y[splits[0]]
    
    x_learn = load_learner("./models/x_learner.pkl")
    x_test = x_3d[splits[0]]
    x_probas, x_targets, x_preds,x_loss  = x_learn.get_X_preds(x_test,y_test,bs=64,with_loss=True, with_decoded=True,)
    
    fft_learn = load_learner("./models/fft_learner.pkl")
    fft_test = fft_x_3d[splits[0]]
    fft_probas, fft_targets, fft_preds,fft_loss  = fft_learn.get_X_preds(fft_test,y_test,bs=64,with_loss=True, with_decoded=True,)
    
    print(x_targets)
    print(x_preds)
    print(fft_targets)
    print(fft_preds)


x_3d,fft_x_3d,y=preload()
# train_two_model(x_3d,fft_x_3d,y)
x_learn,fft_learn=load_two_model()  
dbn_train(x_3d,fft_x_3d,y)


