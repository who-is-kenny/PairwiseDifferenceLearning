from Rose import *
from Uncertainty import *
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from pairwise.benchmark_utils import get_ground_truth, NUM_POINTS
# from pairwise.benchmark_utils import generate_x_shape_data as generate_data
# this line is used when uncommented (for generate_data):
# from pairwise.benchmark_utils import generate_x_shape_data_known_ground_truth as generate_data

# from pairwise.padre import PairwiseDifferenceClassifier              (from  pdll) 
from Padre import *
# from pairwise.uncertainty import *
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Cursor
import matplotlib
import seaborn as sns
import warnings

# from s10_pdl_results import c, colors

#from Karim

colors = {'trees': 'C2', 'anchors': 'C1', 'trees-anchors': 'C3', 'trees-anchors-average': 'C4',
          'pdc_trees': 'C2', 'pdc_anchors': 'C1', 'pdc_trees-anchors': 'C3', 'pdc_trees-anchors-average': 'C4',
          'base': 'C0', 'base_trees': 'C0', 'base_':'C0'}


warnings.filterwarnings("ignore", category=UserWarning)  # for seaborn
matplotlib.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

#################### PARAMETERS ####################
# model = DecisionTreeClassifier(min_samples_leaf=4, random_state=1) #doesnt work
# model = RandomForestClassifier(min_samples_leaf= 5, n_estimators=50, random_state=1, n_jobs=-1) # works
# model = RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_leaf=4,max_features='sqrt',bootstrap=True,oob_score=True,random_state=1,n_jobs=-1)
# model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=30, random_state=1)  # doesnt work , works in combination with pdl
# model = BaggingClassifier(estimator=MLPClassifier(random_state=1), n_estimators=20, n_jobs=-1, random_state=1, verbose=2) #works?
# model = PairwiseDifferenceClassifier(model)   # should work with latest code , has default model

# Eyke:
# uncommented this line to test
model = RandomForestClassifier(min_samples_leaf=4, n_estimators=10, random_state=1, n_jobs=-1)  #works
# model = PairwiseDifferenceClassifier(model)


# pdc_perturbation = 'trees'
pdc_perturbation='trees-anchors'
# pdc_perturbation = 'anchors'

#based on shannon

# background = 'total_uncertainty'
# background = 'aleatoric_uncertainty'
# background = 'epistemic_uncertainty'

# based on variance

background = 'var_tu' # 'var_tu' is the same as 'total_uncertainty'
background = 'var_eu'
# background = 'var_au'
####################################################
unc_formula = "entropy" if not 'var' in background else 'variance'

# Generate dataset

# 1. Build the rose KDE once
random_seed = 3
np.random.seed(random_seed)
kde_red, kde_blue, rose_points = build_rose_kde(
    noise_scale=0.01,  # smaller noise => sharper petals
    random_seed=random_seed
)

# X, y = generate_data(num_points=NUM_POINTS, seed=1) (commented out and replaced with own function)

# X, y = create_rose_dataset( n_points=800, random_seed=1)   

# X, y = create_dataset_using_get_ground_truth(n_points=800, noise_scale=0.05, random_seed=42)

# X, y = create_dataset_by_petals(top_n=400, bottom_n=200, bounding_box=1.2, alpha=40.0, random_seed=2)

X, y = create_dataset_weighted_by_sigma(
        n_points=600,
        bounding_box=1.2,
        alpha=40.0,
        random_seed=2
    )

# X_test, y_test = generate_data(num_points=500, seed=2)

# X_test, y_test = create_rose_dataset( n_points=800, random_seed=2)

# X_test, y_test =  create_dataset_using_get_ground_truth(n_points=800, noise_scale=0.05, random_seed=41)

# X_test, y_test  = create_dataset_by_petals(top_n=400, bottom_n=200,bounding_box=1.2, alpha=40.0, random_seed=1)

X_test, y_test = create_dataset_weighted_by_sigma(
        n_points=600,
        bounding_box=1.2,
        alpha=40.0,
        random_seed=1
    )

# given the min and max of the X_test, generate a grid of points
# space = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
space = np.linspace(-1, 1, 100)
X_grid = np.array([[x, y] for x in space for y in space])


def get_id(x1: float, x2: float) -> int:
    """
    Returns the linear index of the point in X_grid
    that is closest to (x1, x2).
    """
    # Find the index of space closest to x1
    i_x = np.argmin(np.abs(space - x1))

    # Find the index of space closest to x2
    i_y = np.argmin(np.abs(space - x2))

    # Compute the linear index in X_grid
    idx = i_x * len(space) + i_y
    return idx


y = y.astype(int)
print('fitting...')
model.fit(X, y)
print('uncertainty estimation...')
# import uncertainty_quantification.Uncertainty as unc (commented this line)
# unc_method='bays'
# total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.model_uncertainty(model, X_grid, X, y, unc_method=unc_method)
# unc_method = "entropy"
# proba = get_tests_classes_perturbators_likelihood_from_model(model, X_grid)
# total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_from_proba(proba, unc_method=unc_method)
# unc_method = "entropy-trees"
# proba = get_tests_classes_perturbators_likelihood_from_model(model, X_grid, pdc_perturbation='trees')
# total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_from_proba(proba, unc_method="entropy")


unc_method = f"{unc_formula}-{pdc_perturbation}"
proba = get_tests_classes_perturbators_likelihood_from_model(model, X_grid, pdc_perturbation=pdc_perturbation)
total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_from_proba(proba, unc_method="entropy")
var_tu, var_eu, var_au = uncertainty_from_proba(proba, unc_method="variance")
tynes_uncertainty, _, _ = uncertainty_from_proba(proba, unc_method="tynes")

inter_corr = pd.DataFrame({'TU': total_uncertainty, 'EU': epistemic_uncertainty, 'AU': aleatoric_uncertainty}
                          ).corr().round(1).loc[['TU', 'EU'], ['EU', 'AU']]
inter_corr.loc['EU', 'EU'] = np.nan
print('inter corr:\n', inter_corr)
if inter_corr.loc['TU', 'EU'] > 0.95:
    print('Warning: high correlation between total and epistemic uncertainty. Meaning Aleatoric uncertainty is absent.')
if inter_corr.loc['TU', 'AU'] > 0.95:
    print('Warning: high correlation between total and aleatoric uncertainty. Meaning Epistemic uncertainty is absent.')

uncertainty = {'total_uncertainty': total_uncertainty, 'epistemic_uncertainty': epistemic_uncertainty,
               'aleatoric_uncertainty': aleatoric_uncertainty, 'tynes_uncertainty': tynes_uncertainty,
               'var_tu': var_tu, 'var_eu': var_eu, 'var_au': var_au,
               }
# selection = 'epistemic_uncertainty'
vmax = np.round(max([uncertainty[key].max() for key in uncertainty.keys()]), 1)
vmax = None  # .4


def get_overall_distribution_distance(proba):
    """ Get an average err on the prediction of the ground truth second order distrib"""
    vals = np.linspace(-1, 1, 50)   # changed this from 0.5 and changed from 10 to 50
    distances = []
    for x1 in vals:
        for x2 in vals:
            ground_truth_mu, ground_truth_sigma = get_ground_truth(
            x1, x2,
            kde_red, kde_blue, rose_points,
            sigma_base=0.01,  # base uncertainty
            k=1             # multiply distance by 3 => bigger sigma far from petals
        )
            id = get_id(x1, x2)
            predicted_distrib = proba[id, 0, :]
            ground_truth_distrib = np.random.normal(ground_truth_mu, ground_truth_sigma, len(predicted_distrib))
            # distance = kl_divergence(ground_truth_distrib, predicted_distrib)  # bad
            # distance = wasserstein_distance(ground_truth_distrib, predicted_distrib)  # bad
            # distance = np.linalg.norm(ground_truth_distrib - predicted_distrib)  # bad
            distances.append(bhattacharyya_distance(ground_truth_distrib, predicted_distrib))
    return np.nanmean(np.array(distances))


# def get_id(x1, x2)-> int:
#     """ might be incorrect """
#     return int((x1 + 0.5) * 100) * 100 + int((x2 + 0.5) * 100)


from scipy.interpolate import griddata


def color_background(x, y, c, ax=None, cmap='PiYG_r', alpha=0.8, vmin=None, vmax=None, grid_res=100):
    """
    Plots a continuous heatmap as a background using pcolormesh instead of scatter.

    Parameters:
    - x, y: Coordinates of the data points.
    - c: Color values (uncertainty or any other metric).
    - ax: Matplotlib axis to plot on (if None, creates a new figure).
    - cmap: Colormap for the heatmap.
    - alpha: Transparency of the background.
    - vmin, vmax: Color scale limits.
    - grid_res: Resolution of the background grid (higher = smoother).
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Define grid limits
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Create mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))

    # Interpolate values over the grid
    zz = griddata((x, y), c, (xx, yy), method='cubic')  # 'linear' or 'cubic'

    # Plot heatmap
    heatmap = ax.pcolormesh(xx, yy, zz, cmap=cmap, shading='gouraud', alpha=alpha, vmin=vmin, vmax=vmax)

    return heatmap  # Allows for colorbar addition outside the function if needed


def plot(selection='total_uncertainty'):
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 10))
    # fig, axs = plt.subplots(2, 1, figsize=(4, 5))  # for a small distrib for the paper
    ax = axs[0]
    plt.sca(ax)

    model_name = model.__class__.__name__
    if model_name == 'PairwiseDifferenceClassifier':
        model_name = f'PDC({model.estimator.__class__.__name__})'
        model_name = "c(model_name)"   #changed to string

    # plot a heatmap of the uncertainty
    # heatmap = ax.scatter(X_grid[:, 0], X_grid[:, 1], c=uncertainty[selection], cmap='PiYG_r', s=100, edgecolors=None, alpha=0.5, vmin=0, vmax=vmax)  # would leave empty space
    if model_name == 'PDC(RandomForest)' and pdc_perturbation == 'trees-anchors' and 'var' in selection:
        vmax = 0.4
    else:
        vmax = None
    heatmap = color_background(X_grid[:, 0], X_grid[:, 1], c=uncertainty[selection], ax=ax, vmax=vmax)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(selection)
    import matplotlib.ticker as ticker  # Import ticker for formatting
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # Format colorbar ticks to 1 decimal place

    # Plot the training points
    point_colors = ["#FF0000", "#0000FF"]
    cm_bright = ListedColormap(point_colors)
    perm = np.random.permutation(len(X))
    ax.scatter(X[perm, 0], X[perm, 1], c=y[perm], cmap=cm_bright, alpha=1, edgecolors="k")  # , label=f'Class {y[perm][0]}'
    for i in [0, -1]:
        ax.scatter(X[[i], 0], X[[i], 1], c=point_colors[i], alpha=1, edgecolors="k", label=f'Class {y[i]}')

    x_min, x_max = - 1, 1   #changed from 0.5 0.5
    y_min, y_max = - 1, 1    #changed from 0.5 0.5  
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_xticks(())
    # ax.set_yticks(())

    u = uncertainty[selection]
    i = pd.Series(u).argmax()
    ax.scatter((i // 100) / 100 + x_min, (i % 100) / 100 + y_min, facecolors='none',
               edgecolors='gold', s=100, marker='D', linewidths=2, label='max uncertainty')
    i = pd.Series(u).argmin()
    ax.scatter((i // 100) / 100 + x_min, (i % 100) / 100 + y_min, facecolors='none',
               edgecolors='#00FFFF', s=200, marker='h', linewidths=2, label='min uncertainty')

    err_predicted_distrib = get_overall_distribution_distance(proba)

    plt.grid()
    if selection in 'epistemic_uncertainty var_eu':
        pass
        # plt.legend(loc='upper right', ncol=2)
        # plt.legend(loc='center', bbox_to_anchor=(0.5, -1.4), ncol=2)

#     plt.title(c(f'{model_name} - {pdc_perturbation}'))
    
    #plt.title("c(f'{model_name} - {pdc_perturbation}')")
    plt.title(f'{model_name} - {pdc_perturbation}')
    
    # Save only the selected subplot for the paper
    extent = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = extent.expanded(1.2, 1.).translated(50, 0)  # Expand slightly for spacing
    extent = extent.transformed(fig.dpi_scale_trans.inverted())
#     name = f'{c(model_name)}_{pdc_perturbation}_{selection}'.replace('.', '').replace(' ', '_')
    name = "name"
    f = f'./z_evaluation/9/img/paper/2d/{name}'
    # fig.savefig(f, dpi=600, bbox_inches=extent)
    # plt.close() # for the paper
    # return # for the paper

    # model_str = str(model).replace("PairwiseDifferenceClassifier", "PDC").replace(
    #     "                                                              ", "")
    # model_str = model_str.replace("1,\n", "1, ").replace("0,\n", "0, ")
    # model_str = model_str.replace('n_jobs=-1, ', '')
    # model_str = model_str.replace('random_state=1', '')
    # plt.title(c(f'{unc_method} {selection}\n{model_str}'))
    cursor = Cursor(ax, horizOn=True, vertOn=True, color='red', linewidth=2)

    ax2 = axs[1]

    live = True

    def describe(event=None):
        nonlocal live
        if not live:
            return
        print(event)
        if event is None:
            x1, x2 = 1,1  #also changed from -.05 -.01
        else:
            x1, x2 = event.xdata, event.ydata
            if x1 is None:
                return
        id = get_id(x1, x2)
        if id < 0 or id > 10000:
            return
        print('id:=', id)
        predicted_distrib = proba[id, 0, :]
        ground_truth_mu, ground_truth_sigma = get_ground_truth(
            x1, x2,
            kde_red, kde_blue, rose_points,
            sigma_base=0.01,  # base uncertainty
            k=1             # multiply distance by 3 => bigger sigma far from petals
        )
        ground_truth_distrib = np.random.normal(ground_truth_mu, ground_truth_sigma, len(predicted_distrib))
        # distance = kl_divergence(ground_truth_distrib, predicted_distrib)  # bad
        # distance = wasserstein_distance(ground_truth_distrib, predicted_distrib)  # bad
        # distance = np.linalg.norm(ground_truth_distrib - predicted_distrib)  # bad
        distance = bhattacharyya_distance(ground_truth_distrib, predicted_distrib)

        proba_red = predicted_distrib.mean()
        plt.sca(ax2)
        ax2.clear()
        x1, x2 = round(X_grid[id][0], 2), round(X_grid[id][1], 2)

        ax2.title.set_text(f'Location {x1} , {x2}\n{unc_method}')  # {model.__class__.__name__}
        sns.distplot(predicted_distrib, kde=True, color=colors[pdc_perturbation], hist=False, rug=True, rug_kws={'alpha': 0.5},
                     label=f'{"c(pdc_perturbation)"}, mean: {round(proba_red, 2)}', axlabel='Probabilities', ax=ax2)
        sns.distplot(ground_truth_distrib, kde=True, color='gray', hist=False, rug=False,
                     rug_kws={'alpha': 0.4}, ax=ax2, label=f'True distrib. distance = {distance:.5f} | avg. distance = {err_predicted_distrib:.5f}')

        # if pdc_perturbation == 'anchors':
        #     sns.distplot(proba[id, 0, :150], kde=True, color='pink', hist=False, rug=True, ax=ax2, label='predictions based on red anchors')
        #     sns.distplot(proba[id, 0, 150:], kde=True, color='C0', hist=False, rug=True, ax=ax2, label='predictions based on blue anchors')
        plt.xlim(0, 1)
        plt.ylim(0, 10)
        ax2.set_yticks([0, 10])
        ax2.set_ylabel("Density", labelpad=-10)
        ax2.set_xticks([0, 0.5, 1])
        # plt.subplots_adjust(bottom=0.21)
        plt.axvline(proba_red, color='red', linestyle='--',)  # label=f'Proba red: {round(proba_red, 2)}'
        # plt.axvline(.5, color='gray', linestyle='-.')
        # plt.axvline(total_uncertainty[id], color='black', label=f'TU: {total_uncertainty[id]:.2f}')
        plt.axvline(aleatoric_uncertainty[id], color='C0', linestyle=':', label=f'Shannon AU: {aleatoric_uncertainty[id]:.2f}')
        plt.axvline(epistemic_uncertainty[id], color='green', linestyle='--', label=f'Shannon EU: {epistemic_uncertainty[id]:.2f}')

        plt.axvline(var_eu[id], color='green', linestyle='-.', label=f'Variance EU: {var_eu[id]:.2f}')
        # plt.axvline(tynes_uncertainty[id], color='C9', linestyle='--', label=f'Tynes: {tynes_uncertainty[id]:.2f}')
        plt.legend(loc='upper left')  # , ncol=2)
        plt.tight_layout()
        plt.show()

    describe()
    f = f'./uncertainty_quantification/{name}_x_dataset.png'
    # f = f'./z_evaluation/9/img/paper/{name}_x_dataset.png'
    print(f'saving 2 subplots at: {f}')
    plt.savefig(f)

    def stop_resume(event):
        if event.key == ' ':
            nonlocal live
            live = not live
            if live:
                describe()
            print(live)

    fig.canvas.mpl_connect('motion_notify_event', describe)
    fig.canvas.mpl_connect('key_press_event', stop_resume)
    plt.show()


plot(selection=background)
# for background in ['total_uncertainty', 'aleatoric_uncertainty', 'epistemic_uncertainty', 'var_tu', 'var_eu', 'var_au', 'tynes_uncertainty']:
#     plot(selection=background)


# acc_rej = accuracy_rejection_from_model(model, X_test, y_test, unc_method="entropy", pdc_perturbation='anchors')
# for k in ('TU', 'AU', 'EU', 'EU_REVERSE', 'TYNES', 'VAR_TU', 'VAR_AU', 'VAR_EU'):
#     plt.plot(acc_rej[k], label=k)
# plt.legend()
# plt.show()


# id_red = 819
# id_empty = 1850
# assert is lower with good verbose on fail
# np.testing.assert_array_less(tynes_variance_uncertainty(proba[[id_red], :, :]), tynes_variance_uncertainty(proba[[id_empty], :, :]))