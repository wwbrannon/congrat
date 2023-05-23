from abc import abstractmethod

import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, LinearRegression


def plot_embeddings(dat, colors=None, title=None, cmap='Set2'):
    z = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(dat.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        fig.suptitle(title)

    ax.scatter(z[:, 0], z[:, 1], c=colors, s=70, cmap=cmap)

    return ax

# Label propagation code from
# https://datascience.stackexchange.com/questions/45459/how-to-use-scikit-learn-label-propagation-on-graph-structured-data
class BaseLabelPropagation:
    def __init__(self, adj_matrix):
        self.norm_adj_matrix = self._normalize(adj_matrix)
        self.n_nodes = adj_matrix.size(0)
        self.one_hot_labels = None
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    @staticmethod
    @abstractmethod
    def _normalize(adj_matrix):
        raise NotImplementedError("_normalize must be implemented")

    @abstractmethod
    def _propagate(self):
        raise NotImplementedError("_propagate must be implemented")

    def _one_hot_encode(self, labels):
        # Get the number of classes
        classes = torch.unique(labels)
        classes = classes[classes != -1]
        self.n_classes = classes.size(0)

        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        labels = labels.clone()  # defensive copying
        labels[unlabeled_mask] = 0
        self.one_hot_labels = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)
        self.one_hot_labels = self.one_hot_labels.scatter(1, labels.unsqueeze(1), 1)
        self.one_hot_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, max_iter, tol):
        """Fits a semi-supervised learning label propagation model.

        labels: torch.LongTensor
            Tensor of size n_nodes indicating the class number of each node.
            Unlabeled nodes are denoted with -1.
        max_iter: int
            Maximum number of iterations allowed.
        tol: float
            Convergence tolerance: threshold to consider the system at steady state.
        """
        self._one_hot_encode(labels)

        self.predictions = self.one_hot_labels.clone()
        prev_predictions = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)

        for i in range(max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = torch.abs(self.predictions - prev_predictions).sum().item()

            if variation < tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_predictions = self.predictions
            self._propagate()

    def predict(self):
        return self.predictions

    def predict_classes(self):
        return self.predictions.max(dim=1).indices


class LabelPropagation(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

    @staticmethod
    def _normalize(adj_matrix):
        """Computes D^-1 * W"""
        degs = adj_matrix.sum(dim=1)
        degs[degs == 0] = 1  # avoid division by 0 error
        return adj_matrix / degs[:, None]

    def _propagate(self):
        self.predictions = torch.matmul(self.norm_adj_matrix, self.predictions)

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.one_hot_labels[self.labeled_mask]

    def fit(self, labels, max_iter=1000, tol=1e-3):
        super().fit(labels, max_iter, tol)


class LabelSpreading(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)
        self.alpha = None

    @staticmethod
    def _normalize(adj_matrix):
        """Computes D^-1/2 * W * D^-1/2"""
        degs = adj_matrix.sum(dim=1)
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 1
        return adj_matrix * norm[:, None] * norm[None, :]

    def _propagate(self):
        self.predictions = (
            self.alpha * torch.matmul(self.norm_adj_matrix, self.predictions)
            + (1 - self.alpha) * self.one_hot_labels
        )

    def fit(self, labels, max_iter=1000, tol=1e-3, alpha=0.5):
        """
        Parameters
        ----------
        alpha: float
            Clamping factor.
        """
        self.alpha = alpha
        super().fit(labels, max_iter, tol)


#
# Table formatting utilities for paper
#


def bold_largest_by_metric(dat):
    dat = dat.copy()
    
    dat.name = 'value'
    dat.index.names = ('metric', 'model')
    dat = pd.DataFrame(dat).reset_index()
    
    for mt in dat['metric'].unique():
        vals = dat.loc[dat['metric'] == mt, 'value']
        ind = vals.argmax()
        
        style = [
            '' if i != ind else 'font-weight: bold'
            for i, s in enumerate(vals)
        ]
        
        style = pd.Series(style, index=vals.index)
        
        dat.loc[dat['metric'] == mt, 'style'] = style
    
    return dat['style'].tolist()

def bold_largest_by_row(row):
    ind = np.nanargmax(row)

    style = [
        '' if i != ind else 'font-weight: bold'
        for i, s in enumerate(row)
    ]

    return pd.Series(style, index=row.index)

def bold_above_thresh(row, thresh):
    style = [
        '' if np.isnan(s) or s <= thresh else 'font-weight: bold'
        for i, s in enumerate(row)
    ]

    return pd.Series(style, index=row.index)