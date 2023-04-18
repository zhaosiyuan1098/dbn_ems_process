from option import Option
import numpy as np
import pandas as pd
option=Option()
class Loader :
    def __init__(self,Option:option):
        self.num_person = option.load_opt.num_person
        self.num_gesture = option.load_opt.num_gesture
        self.num_channel= option.load_opt.num_channel
        self.num_row_perpage = option.load_opt.num_row_perpage
        self.folder_path=option.load_opt.folder_path

    def load_2d(self):
        df=np.zeros((1,self.num_channel+1))
        for i in range(1,self.num_person+1):
            for j in range(1,self.num_gesture+1): 
                dftemp=pd.read_excel(self.folder_path+'/{}{}.xls'.format(i,j))
                dftemp=np.concatenate((dftemp,np.full((self.num_row_perpage,1),j-1)),axis=1)
                df=np.concatenate((df,dftemp),axis=0)
                # if __debug__ :
                #     print("current loading excel:    "+self.folder_path+'/{}{}.xls'.format(i,j))
                #     print("current excel shape:    "+str(dftemp.shape))
        df=np.delete(df, 0, axis=0)
        # if __debug__ :
        #     print("load complete,all excel shape:   "+str(df.shape))
        if df.shape[0]*(df.shape[1]-1)==self.num_person*self.num_gesture*self.num_channel*self.num_row_perpage:
            if __debug__ :
                print("load success")
                print(df)
            return df
        else: 
            if __debug__ :
                print("load error")
            return False
    
    def load_3d(self):
        num_samples=self.num_person*self.num_gesture
        num_features=self.num_channel
        num_steps=self.num_row_perpage
        x=np.zeros((num_samples,num_features,num_steps))
        y=np.zeros((num_samples))
        for i in range(1,self.num_person+1):
            for j in range(1,self.num_gesture+1): 
                dftemp=pd.read_excel(self.folder_path+'/{}{}.xls'.format(i,j))
                print(dftemp.shape)
                x_index=(i-1)*self.num_gesture+j-1
                x[x_index,:,:]=dftemp.T
                y[x_index]=int(x_index+1)
        return x,y
