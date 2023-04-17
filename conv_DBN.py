from conv_RBM import conv_RBM
from option import Option
option=Option()
class conv_DBN():
    def __init__(self, Option:option) -> None:
        self.rbm1 = conv_RBM(option)
        self.rbm2 = conv_RBM(option)
        self.rbm3 = conv_RBM(option)

    def forward(self, input_data):
        hidden1 = self.rbm1.sample_hidden(input_data)
        hidden2 = self.rbm2.sample_hidden(hidden1)
        hidden3 = self.rbm3.sample_hidden(hidden2)
        return hidden3

    def pretrain(self, input_data, epochs):
        for epoch in range(epochs):
            error1 = self.rbm1.contrastive_divergence(input_data)
            hidden1 = self.rbm1.sample_hidden(input_data)
            error2 = self.rbm2.contrastive_divergence(hidden1)
            hidden2 = self.rbm2.sample_hidden(hidden1)
            error3 = self.rbm3.contrastive_divergence(hidden2)
            print("Epoch %d: error1 = %f, error2 = %f, error3 = %f" % (epoch+1, error1, error2, error3))