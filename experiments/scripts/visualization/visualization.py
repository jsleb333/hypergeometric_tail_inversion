import os, sys
sys.path.append(os.getcwd())
import warnings
import csv
from datetime import datetime
import numpy as np
from copy import copy
import python2latex as p2l
from graal_utils import Timer

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from experiments.utils import make_polynomial_features, make_polynomial_dataset
from source.utils import func_to_cmd


tr_color_p = p2l.Color(.8, 0, 0, color_name='train-pos-color')
tr_color_n = p2l.Color( 0, 0,.8, color_name='train-neg-color')
ts_color_p = p2l.Color( 1,.5,.5, color_name='test-pos-color')
ts_color_n = p2l.Color(.5,.5, 1, color_name='test-neg-color')
true_poly_color = p2l.Color(.5,.1,.5, color_name='true-poly-color')
decision_color_p = p2l.Color( 1,.8,.8, color_name='decision-pos-color')
decision_color_n = p2l.Color( .8,.8, 1, color_name='decision-neg-color')


n_examples = 100
true_degree = 3
seed = 101
min_degree = 1
max_degree = 10
classifier_degrees = list(range(min_degree, max_degree+1))
noise = 1.5
C = 10e6

path = './experiments/scripts/visualization/'

np.random.seed(seed)

(Xtr, Ytr), (Xts, Yts), polynomial, scaler = make_polynomial_dataset(
    n_examples=n_examples,
    degree=true_degree,
    noise=noise,
    root_dist=(.5, 2),
    root_margin=1,
    poly_scale=1,
    return_poly=True,
)

Xtr_normalized = Xtr.copy()
Xts_normalized = Xts.copy()
Xtr = scaler.inverse_transform(Xtr)
Xts = scaler.inverse_transform(Xts)

for classifier_degree in classifier_degrees:
    doc = p2l.Document(f'visualization_n={classifier_degree}', filepath=path, doc_type='standalone')
    plot = doc.new(p2l.Plot(plot_name=f'n={classifier_degree}', plot_path=path, width='4.5cm', height='4.5cm', marks='1.3pt', grid=False, clip_mode='individual', as_float_env=False))

    classifier = SVC(kernel='poly', degree=1, C=C, max_iter=10e6)
    Xtr_poly = make_polynomial_features(Xtr_normalized, classifier_degree)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        classifier.fit(Xtr_poly, np.sign(Ytr))

    Ytr_pred = classifier.predict(Xtr_poly)
    tr_risk = 1 - accuracy_score(np.sign(Ytr), Ytr_pred)
    print(tr_risk)
    Yts_pred = classifier.predict(make_polynomial_features(Xts_normalized, classifier_degree))
    ts_risk = 1 - accuracy_score(np.sign(Yts), Yts_pred)

    x_min = np.min([np.min(Xtr), np.min(Xts)])
    x_max = np.max([np.max(Xtr), np.max(Xts)])
    y_min = -8
    y_max = 8

    plot.x_min = x_min
    plot.x_max = x_max
    plot.y_min = y_min
    plot.y_max = y_max

    # Decision regions
    decision_regions = {1.:[], -1.:[]}
    step = .01
    current_region = [x_min, x_min+step]
    x = scaler.transform(np.array([[current_region[0]]]))
    region_sign = classifier.predict(make_polynomial_features(x, classifier_degree))[0]
    while current_region[1] < x_max:
        x = scaler.transform(np.array([[current_region[1]]]))
        sign = classifier.predict(make_polynomial_features(x, classifier_degree))[0]
        if region_sign != sign: # We just switched region
            decision_regions[region_sign].append(copy(current_region))
            region_sign = sign
            current_region[0] = current_region[1]

        current_region[1] += step

    decision_regions[region_sign].append(current_region)

    for sign, regions in decision_regions.items():
        if sign == 1.:
            color = decision_color_p
        else:
            color = decision_color_n
        for region in regions:
            plot.axis += f'\\fill[{p2l.build(color, doc)}] (axis cs:{region[0]},{y_min}) -- (axis cs:{region[0]},{y_max}) -- (axis cs:{region[1]},{y_max}) -- (axis cs:{region[1]},{y_min}) -- cycle;'

    # Scatter dataset
    plot.add_plot(Xts[Yts>=0].reshape(-1), Yts[Yts>=0].reshape(-1), 'only marks', color=ts_color_p)
    plot.add_plot(Xts[Yts<0].reshape(-1), Yts[Yts<0].reshape(-1), 'only marks', color=ts_color_n)
    plot.add_plot(Xtr[Ytr>=0].reshape(-1), Ytr[Ytr>=0].reshape(-1), 'only marks', color=tr_color_p)
    plot.add_plot(Xtr[Ytr<0].reshape(-1), Ytr[Ytr<0].reshape(-1), 'only marks', color=tr_color_n)

    # Grid
    for y in np.linspace(-7, 7, 7):
        plot.add_plot([x_min, x_max], [y,y], 'thin', 'dashed', color='gray!70')
    for x in np.linspace(0, 4, 5):
        plot.add_plot([x,x], [y_min, y_max], 'thin', 'dashed', color='gray!70')

    # True polynomial
    X = np.linspace(x_min, x_max, 200)
    Y = polynomial(X)
    plot.add_plot(X, np.clip(Y, y_min-0.5, y_max+0.5), color=true_poly_color)

    # plot.axis += '\\legend{};'

    plot.x_label = '$x$'
    plot.y_label = '$p^*(x)+\eta$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.4cm}'

    plot.title = f"Degree {classifier_degree}"
    plot.axis.kwoptions['title style'] = '{at={(.5,-.35)}, anchor=north}'

    plot.axis += fr"\node[fill=white, draw=black, align=left, anchor=south east] at (axis cs:{x_max},{y_min}) {{\tiny train risk: {tr_risk:.2f}\\[-4pt] \tiny test risk: {ts_risk:.2f}}};"

    doc.build(show_pdf=False, delete_files='all')

    os.remove(path+f'n={classifier_degree}.csv')
