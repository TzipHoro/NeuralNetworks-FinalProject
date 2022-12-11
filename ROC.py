"""
@auther: Tziporah
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, RocCurveDisplay, confusion_matrix, f1_score


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
plt.style.use('seaborn-v0_8')


class ROCMetrics:
    def __init__(self, y_true: pd.Series, p_pred: pd.Series):
        self.y_true = y_true
        self.p_pred = p_pred

    def conf_matrix(self, y_pred: pd.Series) -> np.array:
        return confusion_matrix(self.y_true, y_pred)

    @staticmethod
    def sensitivity(conf_matrix: np.array) -> float:
        tp = conf_matrix[1][1]
        fn = conf_matrix[1][0]

        with np.errstate(divide='ignore', invalid='ignore'):
            sens = np.true_divide(tp, tp + fn)
            if sens == np.inf:
                sens = 0

        return sens

    @staticmethod
    def specificity(conf_matrix: np.array) -> float:
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            spec = np.true_divide(tn, tn + fp)
            if spec == np.inf:
                spec = 0

        return spec

    @staticmethod
    def precision(conf_matrix: np.array) -> float:
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            prec = np.true_divide(tp, tp + fp)
            if prec == np.inf:
                prec = 0

        return prec

    @staticmethod
    def fall_out(conf_matrix: np.array) -> float:
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]

        with np.errstate(divide='ignore', invalid='ignore'):
            fo = np.true_divide(fp, tn + fp)
            if fo == np.inf:
                fo = 0

        return fo

    def accuracy(self, y_pred: pd.Series) -> float:
        return accuracy_score(self.y_true, y_pred)

    def f1_score(self, y_pred: pd.Series) -> float:
        return f1_score(self.y_true, y_pred)

    def threshold_matrix(self, step_size: float) -> pd.DataFrame:
        if not (0 <= step_size <= 1):
            raise ValueError('step_size must be a valid probability')
        cols = np.arange(0, 1, step_size)
        matrix = pd.DataFrame()

        for i in cols:
            y_pred = self.p_pred >= i
            conf = self.conf_matrix(y_pred)
            sens = self.sensitivity(conf)
            spec = self.specificity(conf)
            prec = self.precision(conf)
            fo = self.fall_out(conf)
            accr = self.accuracy(y_pred)
            f1 = self.f1_score(y_pred)

            y_pred = y_pred.append(pd.Series({'sensitivity': sens, 'specificity': spec, 'precision': prec,
                                              'fall_out': fo, 'accuracy': accr, 'f1_score': f1}))
            matrix[i] = y_pred

        return matrix

    def roc_plot(self, threshold_matrix: pd.DataFrame,  path: str = None):
        plt.clf()

        fpr = threshold_matrix.loc['fall_out', :]
        tpr = threshold_matrix.loc['sensitivity', :]
        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.y_true.name)
        display.plot()

        plt.title('ROC Curve')

        if path is not None:
            plt.savefig(path)

    @staticmethod
    def metrics_plot(threshold_matrix: pd.DataFrame, path: str = None, xlim: tuple = None):
        plt.clf()

        threshold_matrix.loc['sensitivity', :].plot(legend=True)
        threshold_matrix.loc['specificity', :].plot(legend=True)
        threshold_matrix.loc['precision', :].plot(legend=True)
        threshold_matrix.loc['accuracy', :].plot(legend=True)
        threshold_matrix.loc['f1_score', :].plot(legend=True)

        plt.title('ROC Metrics')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        if xlim is not None:
            plt.xlim(xlim)
        plt.plot()

        if path is not None:
            plt.savefig(path)


def main(output_data_path: str, plot_path: str = None):
    df = pd.read_csv(r'data/train.csv', index_col='DataItem')
    y_test = df.loc[df.index % 2 == 0, 'TACA'].reset_index(drop=True)

    df_output = pd.read_csv(output_data_path, index_col=0)
    yhat = df_output.loc[:, 'Testing Set 3']

    roc = ROCMetrics(y_test, yhat)
    thresholds = roc.threshold_matrix(step_size=0.0001)
    p_th = thresholds.loc[['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score'], :].sum().idxmax()

    path = plot_path.rsplit('.')
    roc.metrics_plot(thresholds, path=path[0] + '-metrics.' + path[1], xlim=(0.6, 1))

    roc.roc_plot(thresholds, plot_path)

    return thresholds.loc[:, p_th]


if __name__ == '__main__':
    # multi-layer perceptron
    p_th_1 = main(r'data/multilayer_data/testing_activation_1.csv', r'plots/roc-mlp-1.png')
    p_th_2 = main(r'data/multilayer_data/testing_activation_2.csv', r'plots/roc-mlp-2.png')
    p_th_3 = main(r'data/multilayer_data/testing_activation_3.csv', r'plots/roc-mlp-3.png')
    print('threshold 1:', p_th_1, '\nthreshold 2:', p_th_2, '\nthreshold 3:', p_th_3)

    # single-layer perceptron
    p_th_1 = main(r'data/single_data/testing_activation_1.csv', r'plots/roc-slp-1.png')
    p_th_2 = main(r'data/single_data/testing_activation_2.csv', r'plots/roc-slp-2.png')
    print('\nthreshold 1:', p_th_1, '\nthreshold 2:', p_th_2)
