class Option:
    class Load_opt:
        def __init__(self):

            # load
            self.folder_path = "./data"
            self.num_person = 1
            self.num_gesture = 12
            self.num_channel = 6
            self.num_row_perpage = 4165

    class Preprocess_opt:
        def __init__(self,load_opt):


            self.num_person = load_opt.num_person
            self.num_gesture = load_opt.num_gesture
            self.num_channel = load_opt.num_channel
            self.num_row_perpage = load_opt.num_row_perpage
            # preprocess
            self.is_ssa = True
            self.ssa_window_length=12
            self.ssa_save_length=2
            self.is_fft = True
            self.is_normalize = True
            self.is_concatenate = True    
            self.min_concatenate_length = 10
            self.max_concatenate_length = 20

            self.window_length=30
            self.window_stride=30
            self.test_rate=0.1

            if self.is_concatenate:
                self.concatenate_length = self.min_concatenate_length
                while self.num_row_perpage % self.concatenate_length != 0:
                    self.concatenate_length += 1
            else:
                self.concatenate_length = 1


    class Rbm_opt:
        def __init__(self,preprocess_opt) -> None:

            # rbm
            self.batch_size = 64
            self.learning_rate = 0.0001
            self.epochs = 3000
            self.mode='bernoulli' 
            self.k=3
            self.optimizer='adam'
            self.gpu=True
            self.savefile=None
            self.early_stopping_patience=50
            self.is_shuffle = True
            self.visible_units = (1+preprocess_opt.ssa_save_length*int(preprocess_opt.is_ssa) +
                                    int(preprocess_opt.is_fft))*preprocess_opt.num_channel*preprocess_opt.concatenate_length



    class Dbn_opt:
        def __init__(self,rbm_opt,load_opt) -> None:
            # dbn
            self.layers = [20,16]
            self.mode='bernoulli'
            self.gpu=True
            self.k=3
            self.savefile='./models/dbn_feature_extractor.pt'

    def __init__(self):
        self.load_opt = Option.Load_opt()
        self.preprecess_opt=Option.Preprocess_opt(self.load_opt)
        self.rbm_opt=Option.Rbm_opt(self.preprecess_opt)
        self.dbn_opt=Option.Dbn_opt(self.rbm_opt,self.load_opt)


