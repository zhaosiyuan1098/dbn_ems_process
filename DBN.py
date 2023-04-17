import numpy as np
import torch
import random
from tqdm import trange
from RBM import RBM
from option import Option
option=Option()


class DBN:
	def __init__(self,input_size, Option:option):
		self.layers = option.dbn_opt.layers
		self.input_size = input_size
		self.layer_parameters = [{'W': None, 'hb': None, 'vb': None}
		    for _ in range(len(self.layers))]
		self.k = option.dbn_opt.k
		self.mode = option.dbn_opt.mode
		self.savefile = option.dbn_opt.savefile

	def sample_v(self, y, W, vb):
		wy = torch.mm(y, W)
		activation = wy + vb
		p_v_given_h = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_v_given_h, torch.bernoulli(p_v_given_h)
		else:
			return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))

	def sample_h(self, x, W, hb):
		wx = torch.mm(x, W.t())
		activation = wx + hb
		p_h_given_v = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_h_given_v, torch.bernoulli(p_h_given_v)
		else:
			return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

	def generate_input_for_layer(self, index, x):
		if index > 0:
			x_gen = []
			for _ in range(self.k):
				x_dash = x.clone()
				for i in range(index):
					_, x_dash = self.sample_h(
					    x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
				x_gen.append(x_dash)

			x_dash = torch.stack(x_gen)
			x_dash = torch.mean(x_dash, dim=0)
		else:
			x_dash = x.clone()
		return x_dash

	def train_DBN(self, x):
		for index, layer in enumerate(self.layers):
			if index == 0:
				vn = self.input_size
			else:
				vn = self.layers[index-1]
			hn = self.layers[index]

			rbm = RBM(vn, hn, option)
			x_dash = self.generate_input_for_layer(index, x)
			rbm.train(x_dash)
			self.layer_parameters[index]['W'] = rbm.W.cpu()
			self.layer_parameters[index]['hb'] = rbm.hb.cpu()
			self.layer_parameters[index]['vb'] = rbm.vb.cpu()
			print("Finished Training Layer:", index, "to", index+1)
		if self.savefile is not None:
			torch.save(self.layer_parameters, self.savefile)

	def reconstructor(self, x):
		x_gen = []
		for _ in range(self.k):
			x_dash = x.clone()
			for i in range(len(self.layer_parameters)):
				_, x_dash = self.sample_h(
				    x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
			x_gen.append(x_dash)
		x_dash = torch.stack(x_gen)
		x_dash = torch.mean(x_dash, dim=0)

		y = x_dash

		y_gen = []
		for _ in range(self.k):
			y_dash = y.clone()
			for i in range(len(self.layer_parameters)):
				i = len(self.layer_parameters)-1-i
				_, y_dash = self.sample_v(
				    y_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['vb'])
			y_gen.append(y_dash)
		y_dash = torch.stack(y_gen)
		y_dash = torch.mean(y_dash, dim=0)

		return y_dash, x_dash

	def initialize_model(self):
		print("The Last layer will not be activated. The rest are activated using the Sigoid Function")
		modules = []
		for index, layer in enumerate(self.layer_parameters):
			modules.append(torch.nn.Linear(layer['W'].shape[1], layer['W'].shape[0]))
			if index < len(self.layer_parameters) - 1:
				modules.append(torch.nn.Sigmoid())
		model = torch.nn.Sequential(*modules)

		for layer_no, layer in enumerate(model):
			if layer_no//2 == len(self.layer_parameters)-1:
				break
			if layer_no % 2 == 0:
				model[layer_no].weight = torch.nn.Parameter(
				    self.layer_parameters[layer_no//2]['W'])
				model[layer_no].bias = torch.nn.Parameter(
				    self.layer_parameters[layer_no//2]['hb'])

		return model


def trial_dataset():
	dataset = []
	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random() > 0.75:
				t.append(0)
			else:
				t.append(1)
		dataset.append(t)

	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random() > 0.75:
				t.append(1)
			else:
				t.append(0)
		dataset.append(t)

	dataset = np.array(dataset, dtype=np.float32)
	np.random.shuffle(dataset)
	dataset = torch.from_numpy(dataset)
	return dataset


class DBN_last_layer():
	def __init__(self, model, train_x, train_y, test_x, test_y, epoch,batch_size,learning_rate):
		test_y = torch.flatten(test_y)
		train_y = torch.flatten(train_y)
		train_x = train_x > torch.mean(train_x)
		test_x = test_x > torch.mean(test_x)
		train_x, test_x = train_x.int().float(), test_x.int().float()

		self.model=model
		self.train_x=train_x
		self.train_y=train_y
		self.test_x=test_x
		self.test_y=test_y
		self.epoch=epoch
		self.batch_size=batch_size
		self.learning_rate=learning_rate

	def generate_batches(self,x, y,batch_size):
		x = x[:int(x.shape[0] - x.shape[0] % batch_size)]
		x = torch.reshape(x, (x.shape[0]//batch_size, batch_size, x.shape[1]))
		y = y[:int(y.shape[0] - y.shape[0] % batch_size)]
		y = torch.reshape(y, (y.shape[0]//batch_size, batch_size))
		return {'x': x, 'y': y}


	def test(self,model, train_x, train_y, test_x, test_y, epoch):
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

	def train(self,model, x, y, train_x, train_y, test_x, test_y, epochs):
		dataset = self.generate_batches(x, y,self.batch_size)

		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
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
			progress.append(self.test(model, train_x, train_y, test_x, test_y, epoch+1))
			training.set_description(
				str({'epoch': epoch+1, 'loss': round(running_loss, 4), 'acc': round(acc, 4)}))

		return model, progress 
		

if __name__ == '__main__':
	dataset = trial_dataset()

	layers = [7, 5, 2]

	dbn = DBN(10, layers)
	dbn.train_DBN(dataset)

	model = dbn.initialize_model()

	y, x = dbn.reconstructor(dataset)

	print(y.shape, x.shape, dataset.shape)