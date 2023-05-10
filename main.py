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
from tqdm import trange
import matplotlib.pyplot as plt

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


def plot_losses(learn):
    losses = learn.recorder.values
    train_losses = [x[0] for x in losses]
    valid_losses = [x[1] for x in losses]
    
    plt.plot(train_losses, label='Train loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
    
    
    x_learn.show_probas()
    x_learn.plot_confusion_matrix()
    x_learn.feature_importance()
    x_learn.step_importance()
 
    fft_learn=load_learner_all(path='models', dls_fname='fft_dls', model_fname='fft_model', learner_fname='fft_learner')
    dls = fft_learn.dls
    valid_dl = dls.valid
    valid_probas, valid_targets, valid_preds = fft_learn.get_preds(dl=valid_dl, with_decoded=True)

    fft_learn.show_probas()
    fft_learn.plot_confusion_matrix()
    fft_learn.feature_importance()
    fft_learn.step_importance()
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
    x_probas_train, y_train, _ = x_learn.get_preds(dl=x_dls.train, with_decoded=True)
    x_probas_valid, y_valid, _ = x_learn.get_preds(dl=x_dls.valid, with_decoded=True)
    
    fft_learn = load_learner("./models/fft_learner.pkl")
    fft_probas_train, _, _ = fft_learn.get_preds(dl=fft_dls.train, with_decoded=True)
    fft_probas_valid, _, _ = fft_learn.get_preds(dl=fft_dls.valid, with_decoded=True)

    
    print(x_probas_train) 
    
    x_probas_train_array=toarray(x_probas_train)
    fft_probas_train_array=toarray(fft_probas_train)
    dbn_x_train=np.zeros((x_probas_train_array.shape[0],x_probas_train_array.shape[1]+fft_probas_train_array.shape[1]))
    dbn_x_train[:, 0:x_probas_train_array.shape[1]] = x_probas_train_array
    dbn_x_train[:, x_probas_train_array.shape[1]:x_probas_train_array.shape[1]+fft_probas_train_array.shape[1]] = fft_probas_train_array
    dbn_x_train=torch.from_numpy(dbn_x_train).float()
    
    x_probas_valid_array=toarray(x_probas_valid)
    fft_probas_valid_array=toarray(fft_probas_valid)
    dbn_x_valid=np.zeros((x_probas_valid_array.shape[0],x_probas_valid_array.shape[1]+fft_probas_valid_array.shape[1]))
    dbn_x_valid[:, 0:x_probas_valid_array.shape[1]] = x_probas_valid_array
    dbn_x_valid[:, x_probas_valid_array.shape[1]:x_probas_valid_array.shape[1]+fft_probas_valid_array.shape[1]] = fft_probas_valid_array
    dbn_x_valid=torch.from_numpy(dbn_x_valid).float()
      
    return dbn_x_train,y_train,dbn_x_valid,y_valid
    
    
def dbn_model_gen(dbn_x_train):
    
    option=Option()
    dbn=DBN(dbn_x_train.shape[1],option)
    dbn.train_DBN(dbn_x_train)
    model = dbn.initialize_model()
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    torch.save(model, './models/dbn_pretrained_model.pt')
    
    return model
    
def dbn_train(dbn_x_train,y_train,dbn_x_valid,y_valid):
    # 创建 PyTorch 数据集
    model = torch.load('./models/dbn_pretrained_model.pt')
    dataset = TensorDataset(dbn_x_train, y)

    # 创建 DataLoader
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 500

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
    torch.save(model, './models/dbn_trained_model.pt')
    
    
def dbn_train2(dbn_x_train,y_train,dbn_x_valid,y_valid):
    model = torch.load('./models/dbn_pretrained_model.pt')
    model, progress = train(model, dbn_x_train,y_train, dbn_x_train,y_train,dbn_x_valid,y_valid)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('DBN_with_pretraining_classifier.csv', index=False)
    torch.save(model, './models/dbn_trained_model.pt')



def generate_batches(x, y, batch_size=64):
	x = x[:int(x.shape[0] - x.shape[0]%batch_size)]
	x = torch.reshape(x, (x.shape[0]//batch_size, batch_size, x.shape[1]))
	y = y[:int(y.shape[0] - y.shape[0]%batch_size)]
	y = torch.reshape(y, (y.shape[0]//batch_size, batch_size))
	return {'x':x, 'y':y}

def test(model, train_x, train_y, test_x, test_y, epoch):
	criterion = torch.nn.CrossEntropyLoss()

	output_test = model(test_x)
	loss_test = criterion(output_test, test_y).item()
	output_test = torch.argmax(output_test, axis=1)
	acc_test = torch.sum(output_test == test_y).item()/test_y.shape[0]

	output_train = model(train_x)
	loss_train = criterion(output_train, train_y).item()
	output_train = torch.argmax(output_train, axis=1)
	acc_train = torch.sum(output_train == train_y).item()/train_y.shape[0]

	return epoch, loss_test, loss_train, acc_test, acc_train


def train(model, x, y, train_x, train_y, test_x, test_y, epochs=500):
	dataset = generate_batches(x, y)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	training = trange(epochs)
	progress = []
	for epoch in training:
		running_loss = 0
		acc = 0
		for batch_x, target in zip(dataset['x'], dataset['y']):
			output = model(batch_x)
			loss = criterion(output, target)
			output = torch.argmax(output, dim=1)
			acc += torch.sum(output == target).item()/target.shape[0]
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		running_loss /= len(dataset['y'])
		acc /= len(dataset['y'])
		progress.append(test(model, train_x, train_y, test_x, test_y, epoch+1))
		training.set_description(str({'epoch': epoch+1, 'loss': round(running_loss, 4), 'acc': round(acc, 4)}))

	return model, progress


# my_main()
x_3d,fft_x_3d,y=preload()
# train_two_model(x_3d,fft_x_3d,y)
x_learn,fft_learn=load_two_model()

dbn_x_train,y_train,dbn_x_valid,y_valid=dbn_pre(x_3d,fft_x_3d,y)
# model=dbn_model_gen(dbn_x_train=dbn_x_train)
dbn_train2(dbn_x_train,y_train,dbn_x_valid,y_valid)




