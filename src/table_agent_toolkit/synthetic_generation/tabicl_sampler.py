import numpy as np
import pandas as pd
from tabicl import TabICLClassifier, TabICLRegressor
from tqdm import tqdm


class TabICLSampler:
    def __init__(
        self, data, discrete_columns=(), order=None, carry_target=False, **kwargs
    ):
        self.data = data
        self.kwargs = kwargs
        self.order = order
        self.carry_target = carry_target
        self.discrete_columns = discrete_columns

    def sample(self, num_rows):
        sampling_order = col_order = list(self.data)
        target_col = col_order[-1]
        if type(self.order) == list:
            sampling_order = self.order
        elif self.order == "full_random":
            sampling_order = np.random.permutation(col_order).tolist()
        elif self.order == "random":
            sampling_order = np.random.permutation(col_order[:-1]).tolist()
            sampling_order.append(col_order[-1])

        X_synth = self.data.iloc[:, -1:].sample(num_rows, replace=True)

        for j, col in tqdm(list(enumerate(sampling_order))):
            y_train = self.data[col]
            X_synth.drop(col, axis=1, inplace=True, errors="ignore")
            X_train = self.data[list(X_synth)]

            if col in self.discrete_columns:
                model = TabICLClassifier()
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_synth).cumsum(axis=1)
                rands = np.random.random(X_synth.shape[0]).reshape(-1, 1)
                X_synth[col] = (probs < rands).sum(axis=1)
            else:
                model = TabICLRegressor()
                model.fit(X_train, y_train)
                quants = model.predict(X_synth, output_type="quantiles")
                rands = np.random.randint(quants.shape[1], size=X_synth.shape[0])
                X_synth[col] = quants[range(len(rands)), rands]

            if j == 0 and not self.carry_target:
                X_synth = X_synth.iloc[:, -1:]

        return X_synth[col_order]
