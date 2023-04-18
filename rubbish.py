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

def train_fft_dbn(origin_x,option):
    fft_x_3d,fft_x_2d=fft_3d(origin_x)
    fft_x_2d = torch.from_numpy(fft_x_2d).float()

    dbn = DBN(fft_x_2d.shape[1], option)
    dbn.train_DBN(fft_x_2d)

    y_new, x_new = dbn.reconstructor(fft_x_2d)
    print(y_new.shape)
    print(x_new.shape)
    return y_new,x_new




    # dls = get_ts_dls(x, y, tfms=tfms, splits=splits, batch_tfms=batch_tfms, bs=128)
    # model = InceptionTime(dls.vars, dls.c)
    # learn = Learner(dls, model, metrics=accuracy)

    # learn.lr_find()
    # learn.fit_one_cycle(100, lr_max=1e-3)
    # learn.save('stage1')
    # learn.recorder.plot_metrics()
    # learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    

