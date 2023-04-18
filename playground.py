from tsai.basics import *
from tsai.data.all import *
from tsai.models.utils import *
from tsai.models.InceptionTimePlus import *
from tsai.models.TabModel import *
from tsai.all import *

dsid = 'NATOPS'
X, y, splits = get_UCR_data(dsid, split_data=False)
ts_features_df = get_ts_features(X, y)

# raw ts
tfms  = [None, [TSCategorize()]]
batch_tfms = TSStandardize()
ts_dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
ts_model = build_ts_model(InceptionTimePlus, dls=ts_dls)

# ts features
cat_names = None
cont_names = ts_features_df.columns[:-2]
y_names = 'target'
tab_dls = get_tabular_dls(ts_features_df, cat_names=cat_names, cont_names=cont_names, y_names=y_names, splits=splits)
tab_model = build_tabular_model(TabModel, dls=tab_dls)

# mixed
mixed_dls = get_mixed_dls(ts_dls, tab_dls)
MultiModalNet = MultiInputNet(ts_model, tab_model)
learn = Learner(mixed_dls, MultiModalNet, metrics=[accuracy, RocAuc()])
learn.fit_one_cycle(1, 1e-3)

(ts, (cat, cont)),yb = mixed_dls.one_batch()
learn.model((ts, (cat, cont))).shape







import torch
import torch.nn as nn
import torch.optim as optim

# 假设 ModelA 和 ModelB 是两个预训练模型的类
class ModelA(nn.Module):
    # ...

class ModelB(nn.Module):
    # ...

# 加载预训练模型和权重
modelA = ModelA()
modelB = ModelB()
modelA.load_state_dict(torch.load('modelA_weights.pth'))
modelB.load_state_dict(torch.load('modelB_weights.pth'))

# 自定义一个新的模型，结合两个预训练模型的输出
class ParallelModel(nn.Module):
    def __init__(self, modelA, modelB, num_classes):
        super(ParallelModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(modelA.output_size + modelB.output_size, num_classes)
        
    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        out = torch.cat((outA, outB), dim=1)
        out = self.classifier(out)
        return out

# 创建并行模型实例
num_classes = 10  # 假设有10个分类
parallel_model = ParallelModel(modelA, modelB, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(parallel_model.parameters(), lr=0.001)
