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
from torch.utils.data import TensorDataset, DataLoader
from tsai.all import *
from tsai.inference import get_X_preds

def my_main():

    option=Option()

    loader = Loader(option)
    x = loader.load_2d()

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
    bs=64
    x_dls   = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[bs, bs*2])
    x_model = build_ts_model(XceptionTime, dls=x_dls)

    valid_dls_before_learn=x_dls.valid
    valid_dls_before_learn_ds=x_dls.valid_ds

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
 
    fft_learn=load_learner_all(path='models', dls_fname='fft_dls', model_fname='fft_model', learner_fname='fft_learner')
    dls = fft_learn.dls
    valid_dl = dls.valid
    valid_probas, valid_targets, valid_preds = fft_learn.get_preds(dl=valid_dl, with_decoded=True)
    print("fft model accuracy=    "+str((valid_targets == valid_preds).float().mean()))
    return x_learn,fft_learn
  
    
def dbn_pre(x_3d,fft_x_3d,y,):
    splits = get_splits(y, valid_size=.2,test_size=0.1, stratify=True, random_state=55, shuffle=True)
    tfms  = [None, [Categorize()]]
    bs=64
    
    x_dsets = TSDatasets(x_3d, y, tfms=tfms, splits=splits, inplace=True)
    x_dls   = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[bs, bs*2])

    fft_dsets = TSDatasets(fft_x_3d, y, tfms=tfms, splits=splits, inplace=True)
    fft_dls   = TSDataLoaders.from_dsets(fft_dsets.train, fft_dsets.valid, bs=[bs, bs*2])
    
    
    x_learn = load_learner("./models/x_learner.pkl")
    x_probas, x_targets, x_preds = x_learn.get_preds(dl=x_dls.train, with_decoded=True)
    
    fft_learn = load_learner("./models/fft_learner.pkl")
    fft_probas, fft_targets, fft_preds = fft_learn.get_preds(dl=fft_dls.train, with_decoded=True)
    
    print("x model accuracy=    "+str((x_targets == x_preds).float().mean()))
    print("fft model accuracy=    "+str((fft_targets == fft_preds).float().mean()))
    
    
    print(x_targets)
    print(x_preds)
    print(x_probas) 
    
    x_probas_array=toarray(x_probas)
    fft_probas_array=toarray(fft_probas)
    dbn_x=np.zeros((x_probas_array.shape[0],x_probas_array.shape[1]+fft_probas_array.shape[1]))
    dbn_x[:, 0:x_probas_array.shape[1]] = x_probas_array
    dbn_x[:, x_probas_array.shape[1]:x_probas_array.shape[1]+fft_probas_array.shape[1]] = fft_probas_array
    dbn_x=torch.from_numpy(dbn_x).float()
      
    return dbn_x,x_targets
    
    
def dbn_model_gen(dbn_x):
    
    option=Option()
    dbn=DBN(dbn_x.shape[1],option)
    dbn.train_DBN(dbn_x)
    model = dbn.initialize_model()
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    torch.save(model, './models/dbn_pretrained_model.pt')
    
    return model
    
def dbn_train(dbn_x,y):
    # 创建 PyTorch 数据集
    model = torch.load('./models/dbn_pretrained_model.pt')
    dataset = TensorDataset(dbn_x, y)

    # 创建 DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 将梯度归零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            running_loss += loss.item()

        # 打印每个 epoch 的平均损失
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

# my_main()
x_3d,fft_x_3d,y=preload()
# train_two_model(x_3d,fft_x_3d,y)
# x_learn,fft_learn=load_two_model()

dbn_x,y=dbn_pre(x_3d,fft_x_3d,y)
#  model=dbn_model_gen(dbn_x=dbn_x)
dbn_train(dbn_x,y)





