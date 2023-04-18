from option import Option
from sklearn.model_selection import StratifiedShuffleSplit
from tsai.all import *
import torch
import numpy as np
option = Option()

class PreProcessor_3d:
    def __init__(self, x,y, Option: option):
        self.origin_x = x
        self.origin_y = y
        self.num_person = option.preprecess_opt.num_person
        self.num_gesture = option.preprecess_opt.num_gesture
        self.num_channel = option.preprecess_opt.num_channel
        self.num_row_perpage = option.preprecess_opt.num_row_perpage
        self.window_length=option.preprecess_opt.window_length
        self.window_stride=option.preprecess_opt.window_stride

    def fft_2d(self, data):
        fft_data = np.zeros(data.shape)
        length = int(data.shape[0]/self.num_person)
        for i in range(0, self.num_person):
            fft_data[i*length:(i+1)*length, :] = np.fft.fft(data[i *
                                                                length:(i+1)*length, :], axis=0)
        return fft_data
    
    def fft_3d(self,x):
        fft_data_3d = np.zeros(x.shape)
        fft_data_2d=np.zeros((x.shape[0]*x.shape[2],x.shape[1]))
        for i in range(x.shape[0]):
            fft_temp=np.fft.fft(x[i:i+1,:,:],axis=1)
            fft_data_3d[i:i+1,:,:]=fft_temp
            fft_data_2d[i*x.shape[2]:(i+1)*x.shape[2],:]=np.squeeze(fft_temp).T
        return fft_data_3d,fft_data_2d
    
    def x_3d_to_2d(self,x):
        #x shape=(samples,features,steps)
        #new x shape=(samples*steps,features)
        new_x=np.zeros((x.shape[0]*x.shape[2],x.shape[1]))
        for i in range(x.shape[0]):
            new_x[i*x.shape[2]:(i+1)*x.shape[2],:]=np.squeeze(x[i:(i+1),:,:].T)
        return new_x

    def x_2d_to_3d(self,x,samples):
        x_3d_samples=int(samples)
        x_3d_features=x.shape[1]
        x_3d_steps=int(x.shape[0]/x_3d_samples)
        new_x=np.zeros((x_3d_samples,x_3d_features,x_3d_steps))
        for i in range(x_3d_samples):
            new_x[i:(i+1),:,:]=x[i*x_3d_steps:(i+1)*x_3d_steps,:].T
        return new_x

    def concatenate(self,x):
        new_x=np.zeros((x.shape[0],x.shape[1]*x.shape[2]))
        for i in range(x.shape[2]):
            new_x[:,i*x.shape[1]:(i+1)*x.shape[1]]=x[:,:,i]
        return new_x
    
    def deconcatenate(self,x,steps):
        # x shape =(samples,features*steps)
        #new x shape =(samples,features,steps)
        x_3d_samples=x.shape[0]
        x_3d_features=int(x.shape[1]/steps)
        x_3d_steps=int(steps)
        new_x=np.zeros((x_3d_samples,x_3d_features,x_3d_steps))
        for i in range(x_3d_steps):
            new_x[:,:,i]=x[:,i*x_3d_features:(i+1)*x_3d_features]
        return new_x

    def window_3d(self,x):
        window_length = self.window_length
        stride = self.window_stride
        x_length=int((self.num_row_perpage+stride-window_length)/stride)
        x_new_samples=x_length*self.num_person*self.num_gesture
        x_new_features=x.shape[1]
        x_new_steps=window_length
        x_new=np.zeros((x_new_samples,x_new_features,x_new_steps))
        y_new=np.zeros((x_new_samples,))
        for i in range(int(x.shape[0])):
            x_temp=x[i,:,:]
            x_slide,y_slide = SlidingWindow(window_len=window_length, stride=stride,seq_first=False)(x_temp)
            x_new[i*x_length:(i+1)*x_length,:,:]=x_slide
            y_new[i*x_length:(i+1)*x_length,]=int(i%self.num_gesture)
        print("windowed x shape:"+str(x_new.shape))
        print("windowed y shape:"+str(y_new.shape))    
        print("window finish")
        return x_new,y_new
        
    def generateTensor(self,x,y):
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).long()


    def getx(self):
        return self.originX

    def gety(self):
        return self.origin_y 