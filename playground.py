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