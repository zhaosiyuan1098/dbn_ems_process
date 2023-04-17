from option import Option
from load import Loader
from preprocess import PreProcessor
from RBM import RBM
from DBN import DBN
from DBN import DBN_last_layer
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

def load():
    option=Option()
    num_samples=option.load_opt.num_person*option.load_opt.num_gesture
    num_features=option.load_opt.num_channel
    num_steps=option.load_opt.num_row_perpage
    x=np.zeros((num_samples,num_features,num_steps))
    y=np.zeros((num_samples))
    for i in range(1,option.load_opt.num_person+1):
        for j in range(1,option.load_opt.num_gesture+1): 
            dftemp=pd.read_excel(option.load_opt.folder_path+'/{}{}.xls'.format(i,j))
            print(dftemp.shape)
            x_index=(i-1)*option.load_opt.num_gesture+j-1
            x[x_index,:,:]=dftemp.T
            y[x_index]=int(x_index+1)
    return x,y

def window(x):
    option=Option()
    window_length = 30
    stride = 30
    x_length=int((option.load_opt.num_row_perpage+stride-window_length)/stride)
    x_new_samples=x_length*option.load_opt.num_person*option.load_opt.num_gesture
    x_new_features=option.load_opt.num_channel
    x_new_steps=window_length
    x_new=np.zeros((x_new_samples,x_new_features,x_new_steps))
    y_new=np.zeros((x_new_samples,))
    for i in range(int(x.shape[0])):
        x_temp=x[i,:,:]
        x_slide,y_slide = SlidingWindow(window_len=window_length, stride=stride,seq_first=False)(x_temp)
        x_new[i*x_length:(i+1)*x_length,:,:]=x_slide
        y_new[i*x_length:(i+1)*x_length,]=int(i%option.load_opt.num_gesture)
    print("windowed x shape:"+str(x_new.shape))
    print("windowed y shape:"+str(y_new.shape))    
    print("window finish")
    return x_new,y_new
    # window_length = 30
    # stride = 5
    # horizon=0
    # columns=[f'var_{i}' for i in range(option.load_opt.num_channel)]+['target']
    # x_pd = pd.DataFrame(x, columns=columns).T
    # x, y = SlidingWindow(window_len=window_length,stride=stride, horizon=horizon, get_x=columns[:-1], get_y='target', seq_first=False)(x_pd)
    return x

def fft_3d(x):
    fft_data_3d = np.zeros(x.shape)
    fft_data_2d=np.zeros((x.shape[0]*x.shape[2],x.shape[1]))
    for i in range(x.shape[0]):
        fft_temp=np.fft.fft(x[i:i+1,:,:],axis=1)
        fft_data_3d[i:i+1,:,:]=fft_temp
        fft_data_2d[i*x.shape[2]:(i+1)*x.shape[2],:]=np.squeeze(fft_temp).T
    return fft_data_3d,fft_data_2d

def x_3d_to_2d(x):
    #x shape=(samples,features,steps)
    #new x shape=(samples*steps,features)
    new_x=np.zeros((x.shape[0]*x.shape[2],x.shape[1]))
    for i in range(x.shape[0]):
        new_x[i*x.shape[2]:(i+1)*x.shape[2],:]=np.squeeze(x[i:(i+1),:,:].T)
    return new_x

def x_2d_to_3d(x,samples):
    x_2d_samples=int(samples)
    x_2d_features=x.shape[1]
    x_2d_steps=int(x.shape[0]/x_2d_samples)
    new_x=np.zeros((x_2d_samples,x_2d_features,x_2d_steps))
    for i in range(x_2d_samples):
        new_x[i:(i+1),:,:]=x[i*x_2d_steps:(i+1)*x_2d_steps,:].T
    return new_x




def tsai_main():
    computer_setup()
    option=Option()
    loader = Loader(option)
    origin_x,y=load()

    # x = loader.load_data()
    x,y=window(origin_x)

    fft_x_3d,fft_x_2d=fft_3d(origin_x)
    fft_x_2d = torch.from_numpy(fft_x_2d).float()
    dbn = DBN(fft_x_2d.shape[1], option)
    dbn.train_DBN(fft_x_2d)

    y_new, x_new = dbn.reconstructor(fft_x_2d)

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
        # clear_output()
        display(results)







    # dls = get_ts_dls(x, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms, bs=128)
    # model = InceptionTime(dls.vars, dls.c)
    # learn = Learner(dls, model, metrics=accuracy)

    # learn.lr_find()
    # learn.fit_one_cycle(100, lr_max=1e-3)
    # learn.save('stage1')
    # learn.recorder.plot_metrics()
    # learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    
    print(111)

def tsai_valid():
    learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    dls = learn.dls
    valid_dl = dls.valid
    valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
    valid_probas, valid_targets, valid_preds
    b=(valid_targets == valid_preds).float().mean()
    print(b)
    learn.show_probas()


tsai_main()
# tsai_valid()
# load()


