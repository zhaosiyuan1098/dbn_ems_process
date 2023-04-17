from option import Option
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import numpy as np
option = Option()


class PreProcessor:
    def __init__(self, data, Option: option):
        self.originData = data
        self.originX = np.delete(data, -1, axis=1)
        self.y = self.originData[:, [-1]]
        self.num_person = option.preprecess_opt.num_person
        self.num_gesture = option.preprecess_opt.num_gesture
        self.num_channel = option.preprecess_opt.num_channel
        self.num_row_perpage = option.preprecess_opt.num_row_perpage
        self.test_rate = option.preprecess_opt.test_rate
        self.concatenate_length=option.preprecess_opt.concatenate_length
        self.max_concatenate_length=option.preprecess_opt.max_concatenate_length
        self.ssa_save_length=option.preprecess_opt.ssa_save_length
        self.ssa_window_length=option.preprecess_opt.ssa_window_length


        if __debug__:
            print(self.originX.shape)
            print(self.y.shape)
            print(len(self.y))

    def generateTrainTest(self, x, y):
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_rate, random_state=42)
        for train_index, test_index in split.split(x, y):
            x_train = x[train_index, :]
            x_test = x[test_index, :]
            y_train = y[train_index, :]
            y_test = y[test_index, :]
            # 将 numpy 数组转换为 PyTorch 的 Tensor 类型
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).long()
            x_test = torch.from_numpy(x_test).float()
            y_test = torch.from_numpy(y_test).long()

        return x_train, y_train, x_test, y_test

    def ssa(self):
        """对时间序列x进行SSA分解"""
        K = int(self.num_row_perpage - self.ssa_window_length + 1)
        X = np.zeros((self.ssa_window_length, K))
        saved_ssa = np.zeros((int(len(self.y)), int(
            self.num_channel*self.ssa_save_length)))
        for w in range(0, self.num_person*self.num_gesture):
            saved_rec = np.zeros(
                (int(self.num_row_perpage), int(self.num_channel*self.ssa_save_length)))
            # saved_rec=saved_rec.reshape(self.num_row_perpage,1)
            for q in range(0, self.num_channel):
                series = self.originX[w *
                                      self.num_row_perpage:(w+1)*self.num_row_perpage, q]
                for i in range(K):
                    X[:, i] = series[i:i + self.ssa_window_length]
                U, sigma, VT = np.linalg.svd(X, full_matrices=False)

                for i in range(VT.shape[0]):
                    VT[i, :] *= sigma[i]
                A = VT

                rec = np.zeros((self.ssa_window_length, self.num_row_perpage))

                for i in range(self.ssa_window_length):
                    for j in range(self.ssa_window_length-1):
                        for m in range(j+1):
                            rec[i, j] += A[i, j-m] * U[m, i]
                        rec[i, j] /= (j+1)
                    for j in range(self.ssa_window_length-1, self.num_row_perpage - self.ssa_window_length + 1):
                        for m in range(self.ssa_window_length):
                            rec[i, j] += A[i, j-m] * U[m, i]
                        rec[i, j] /= self.ssa_window_length
                    for j in range(self.num_row_perpage - self.ssa_window_length + 1, self.num_row_perpage):
                        for m in range(j-self.num_row_perpage+self.ssa_window_length, self.ssa_window_length):
                            rec[i, j] += A[i, j - m] * U[m, i]
                        rec[i, j] /= (self.num_row_perpage - j)

                saved_rec_temp = rec[:self.ssa_save_length, :]
                denoised_series_temp = np.sum(saved_rec_temp, axis=0)
                saved_rec_temp = np.transpose(saved_rec_temp)
                if __debug__:
                    print("gesture of  "+str(w+1))
                    print("channel of    "+str(q+1))
                    # plt.figure()
                    # for i in range(int(self.ssa_window_length/2)):
                    #     ax = plt.subplot(int(self.ssa_window_length/4)+1,2,i+1)
                    #     ax.plot(rec[i, 0:600],color='black')

                    # plt.figure()
                    # plt.plot(series[0:600])
                    # plt.plot(denoised_series_temp[0:600])
                    # plt.show()
                    # print(111)
                saved_rec[:, q*self.ssa_save_length:(q+1)
                          * self.ssa_save_length] = saved_rec_temp

            saved_ssa[w*self.num_row_perpage:(w+1)
                      * self.num_row_perpage, :] = saved_rec
            self.ssaRusult = saved_ssa
        return saved_ssa

    def fft(self, data):
        fft_data = np.zeros(data.shape)
        length = int(data.shape[0]/self.num_person)
        for i in range(0, self.num_person):
            fft_data[i*length:(i+1)*length, :] = np.fft.fft(data[i *
                                                                length:(i+1)*length, :], axis=0)
        return fft_data

    def dataAdd(self, dataToAdd, dataExisted):
        if(dataToAdd.shape[0] == dataExisted.shape[0]):
            a = int(dataToAdd.shape[0])
            b = int(dataToAdd.shape[1])
            c = int(dataExisted.shape[1])
            result = np.zeros((a, b+c))
            result[:, 0:b] = dataToAdd
            result[:, b:b+c] = dataExisted

            return result
        else:
            print("dataToAdd cant connect with dataExisted")
            return -1

    def normalize(self, data):
        if (data != 0).any():
            min_value = data.min()
            max_value = data.max()
            normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data
    
    def dataConcatenate(self,x,y):
        
        if self.concatenate_length>self.max_concatenate_length:
            return False
            
        if __debug__:
            print("the length for concatenate is:   "+str(self.concatenate_length))
        x=x.reshape(int(x.shape[0]/self.concatenate_length),int(x.shape[1]*self.concatenate_length))
        y_new=np.zeros((int(y.shape[0]/self.concatenate_length),1))
        class_length=self.num_row_perpage/self.concatenate_length
        for i in range(self.num_person):
            for j in range(self.num_gesture):
                y_new[int((i*self.num_gesture+j)*class_length):int((i*self.num_gesture+j+1)*class_length)] = j
        return x,y_new


    def getx(self):
        return self.originX

    def gety(self):
        return self.y