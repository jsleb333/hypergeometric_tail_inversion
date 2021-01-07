import os, sys
sys.path.append(os.getcwd())
import warnings
import csv
from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np

from graal_utils import Timer

from experiments.utils import make_polynomial_features, make_polynomial_dataset
from source.utils import func_to_cmd


def launch_single_run(dataset, classifier_degree, C):
    (Xtr, Ytr), (Xts, Yts) = dataset
    # n_examples = Xtr.shape[0]
    # class_weight = {1:-np.sum(np.sign(Ytr)-1)/2/n_examples, -1: np.sum(np.sign(Ytr)+1)/2/n_examples}
    # sample_weight = np.array([class_weight[y] for y in np.sign(Ytr)])

    classifier = SVC(kernel='poly', degree=1, C=C, max_iter=10e6)
    Xtr_poly = make_polynomial_features(Xtr, classifier_degree)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        classifier.fit(Xtr_poly,
                       np.sign(Ytr),
                    #    sample_weight=sample_weight,
                       )

    Ytr_pred = classifier.predict(Xtr_poly)
    tr_risk = 1 - accuracy_score(np.sign(Ytr), Ytr_pred)
    Yts_pred = classifier.predict(make_polynomial_features(Xts, classifier_degree))
    ts_risk = 1 - accuracy_score(np.sign(Yts), Yts_pred)

    return tr_risk, ts_risk


def launch_single_poly(filename,
                       n_examples,
                       true_degree,
                       seed=11,
                       n_runs=3,
                       classifier_degrees=list(range(1,11)),
                       noise=0.5,
                       C=10e5):
    with open(filename + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        _titles = [[f'{n=}-tr-risk', f'{n=}-ts-risk'] for n in classifier_degrees]
        titles = [t for _t in _titles for t in _t] + ['seed']
        writer.writerow(titles)

        for run in range(n_runs):
            if run+1 < n_runs:
                print(f'Run {run+1}/{n_runs}...', end='\r')
            else:
                print(f'All {n_runs} runs completed.')

            row_data = []
            run_seed = seed + 10*run
            np.random.seed(run_seed)

            dataset = make_polynomial_dataset(n_examples=n_examples,
                                                degree=true_degree,
                                                noise=noise,
                                                root_dist=(.5, 2),
                                                root_margin=2,
                                                poly_scale=1)

            for classifier_degree in classifier_degrees:
                row_data.extend(launch_single_run(dataset=dataset,
                                                  classifier_degree=classifier_degree,
                                                  C=C))
            writer.writerow(row_data + [run_seed])


@func_to_cmd
def launch_experiment(exp_name='',
                      n_examples=250,
                      min_true_degree=2,
                      max_true_degree=7,
                      noise=.5,
                      C=10e5,
                      n_runs=100,
                      ):
    """
    Will launch the experiment with specified parameters. Automatically saves all results in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>.csv". Experiments parameters are saved in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>_params.py". See the README for more usage details.

    Args:
        exp_name (str):
            Name of the experiment. Will be used to save the results on disk. If empty, the date and time of the beginning of the experiment will be used.
    """
    if not exp_name:
        exp_name = exp_name if exp_name else datetime.now().strftime("%Y-%m-%d_%Hh%Mm")

    for true_degree in range(min_true_degree, max_true_degree+1):
        with Timer(f'True degree: {true_degree}'):
            pathname = f'./experiments/results/{exp_name}/'
            os.makedirs(pathname, exist_ok=True)
            filename = f'n={true_degree}-m={n_examples}-noise={noise}-runs={n_runs}-C={C}'
            launch_single_poly(
                filename=pathname+filename,
                n_examples=n_examples,
                true_degree=true_degree,
                n_runs=n_runs,
                C=C,
                noise=noise,
            )


if __name__ == "__main__":
    launch_experiment(exp_name='test',
                      n_examples=100,
                      min_true_degree=2,
                      max_true_degree=4,
                      n_runs=10,
                      noise=2,
                      )