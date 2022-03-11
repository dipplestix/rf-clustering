import math
import h2o
import pandas as pd
from h2o.estimators.aggregator import H2OAggregatorEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from random import choices


def find_dissim(l1, l2, measure='exp'):
    # return 1 - sum([a == b for a, b in zip(l1, l2)]) / len(l1)
    # return math.sqrt(1 - sum([a == b for a, b in zip(l1, l2)]) / len(l1))
    if measure == 'exp':
        return math.exp(1 - sum([a == b for a, b in zip(l1, l2)]) / len(l1))


def addcl1(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    new_cols = {}
    nrows = len(df)
    df['label'] = [1] * nrows
    df = df[[col for col in df.columns if col not in drop_cols]]
    for col in df.columns:
        if col not in drop_cols:
            col_list = df[col].to_list()
            samples = choices(col_list, k=nrows)
            new_cols[col] = samples
    new_df = pd.DataFrame(new_cols)
    new_df['label'] = [2] * nrows
    return pd.concat([df, new_df])


class RFClusterer:
    def __init__(self, df, dist_measure, depth=10, ntrees=1000, drop_cols=None, agg=False, **kwargs):
        h2o.init()
        self.agg = agg
        self.depth = depth
        self.df = h2o.H2OFrame(pd.DataFrame(df))
        self.drop_cols = drop_cols
        self.matrix = None
        self.model = None
        self.ntrees = ntrees
        self.synth_df = None
        self.dist_measure = dist_measure
        self.kwargs = kwargs
        self.clusters = None
        self.best3 = None
        self.best2 = None
        self.dfp = pd.DataFrame(df)
        if agg is True:
            ag = H2OAggregatorEstimator(save_mapping_frame=True)
            ag.train(training_frame=self.df)
            self.agg_df = ag.aggregated_frame
            # noinspection PyProtectedMember
            self.mapping_frame = h2o.get_frame(ag._model_json["output"]["mapping_frame"]["name"])
            self.dfa = self.agg_df
        else:
            self.dfa = self.df

    def synth_data(self, method=addcl1):
        synth_df = h2o.H2OFrame(method(self.dfa.as_data_frame(), self.drop_cols))
        synth_df['label'] = synth_df['label'].asfactor()
        self.synth_df = synth_df

    def make_model(self, y=None):
        if y is None:
            hdf = self.synth_df
            y = 'label'
        else:
            hdf = self.dfa
        x = hdf.columns
        x.remove(y)
        model = H2ORandomForestEstimator(ntrees=self.ntrees, max_depth=self.depth,
                                         nbins=1024,
                                         nbins_top_level=1024,
                                         nbins_cats=1024,
                                         stopping_tolerance=0.00000001
                                         )
        model.train(x=x, y=y, training_frame=hdf)
        self.model = model
        self.best3 = [x[0] for x in model.varimp()[:3]]
        self.best2 = [x[0] for x in model.varimp()[:2]]

    def make_matrix(self):
        df = self.dfa
        tree_nodes = self.model.predict_leaf_node_assignment(df)
        tree_nodes = tree_nodes.as_data_frame().values.tolist()
        matrix = [[0 for _ in range(len(df))] for _ in range(len(df))]
        for i in range(len(df)):
            for j in range(i, len(df)):
                if i == j:
                    res = 0
                else:
                    res = find_dissim(tree_nodes[i], tree_nodes[j])
                matrix[i][j] = res
                matrix[j][i] = res
        self.matrix = matrix

    def make_clusters(self):
        if 'min_k' in self.kwargs:
            clusters = {}
            for k in range(self.kwargs['min_k'], self.kwargs['max_k']+1):
                temp_dist = self.dist_measure(**self.kwargs, a=self.matrix, k=k)
                temp_dist.run()
                clusters[k] = temp_dist.clusters
            self.clusters = clusters
        else:
            temp_dist = self.dist_measure(a=self.matrix, **self.kwargs)
            temp_dist.run()
            self.clusters = temp_dist.clusters

    def run(self):
        self.synth_data()
        self.make_model()
        self.make_matrix()
        self.make_clusters()
