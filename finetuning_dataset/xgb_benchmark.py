from hpobench.benchmarks.ml import XGBoostBenchmark
import pandas as pd
from typing import Dict, Union
import numpy as np
import time

class CustomXGBoostBenchmark(XGBoostBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.valid_y.dtype == 'object' or self.valid_y.dtype.name == 'category':
            self.valid_y = self.valid_y.astype('category').cat.codes
        if self.test_y.dtype == 'object' or self.test_y.dtype.name == 'category':
            self.test_y = self.test_y.astype('category').cat.codes
        if self.train_y.dtype == 'object' or self.train_y.dtype.name == 'category':
            self.train_y = self.train_y.astype('category').cat.codes

    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):
        
        if rng is not None:
            rng = get_rng(rng, self.rng)

        # initializing model
        model = self.init_model(config, fidelity, rng)

        # preparing data
        if evaluation == "valid":
            train_X = self.train_X
            train_y = self.train_y
            train_idx = self.train_idx
        else:
            train_X = np.vstack((self.train_X, self.valid_X))
            train_y = pd.concat((self.train_y, self.valid_y))
            train_idx = np.arange(len(train_X))

        # Convert categorical labels to numeric if necessary
        if train_y.dtype == 'object' or train_y.dtype.name == 'category':
            train_y = train_y.astype('category').cat.codes

        # shuffling data
        if shuffle:
            train_idx = self.shuffle_data_idx(train_idx, rng)
            train_X = train_X.iloc[train_idx]
            train_y = train_y.iloc[train_idx]

        # subsample here:
        if self.lower_bound_train_size is None:
            self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
            self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))
        subsample = np.max((fidelity['subsample'], self.lower_bound_train_size))
        train_idx = self.rng.choice(
            np.arange(len(train_X)), size=int(
                subsample * len(train_X)
            )
        )
        # fitting the model with subsampled data
        start = time.time()
        model.fit(train_X[train_idx], train_y.iloc[train_idx])
        model_fit_time = time.time() - start

        # computing statistics on training data
        scores = dict()
        score_cost = dict()
        for k, v in self.scorers.items():
            scores[k] = 0.0
            score_cost[k] = 0.0
            if evaluation == "test":
                _start = time.time()
                scores[k] = v(model, train_X, train_y)
                score_cost[k] = time.time() - _start
        train_loss = 1 - scores["acc"]
        return model, model_fit_time, train_loss, scores, score_cost
