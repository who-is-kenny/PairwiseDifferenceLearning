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
from matplotlib.colors import LinearSegmentedColormap
from database_handler import initialize_database, save_compute_results, load_compute_results

from typing import Optional, Sequence

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
model = RandomForestClassifier(min_samples_leaf=4, n_estimators=50, random_state=1, n_jobs=-1)  #works
model = PairwiseDifferenceClassifier(model)


# pdc_perturbation = 'trees'
pdc_perturbation='trees-anchors'
# pdc_perturbation = 'anchors'

#based on shannon

# background = 'total_uncertainty'
# background = 'aleatoric_uncertainty'
background = 'epistemic_uncertainty'

# based on variance

# background = 'var_tu' # 'var_tu' is the same as 'total_uncertainty'
#background = 'var_eu'
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
        n_points=400,
        bounding_box=1.2,
        alpha=40.0,
        random_seed=2
    )

# X_test, y_test = generate_data(num_points=500, seed=2)

# X_test, y_test = create_rose_dataset( n_points=800, random_seed=2)

# X_test, y_test =  create_dataset_using_get_ground_truth(n_points=800, noise_scale=0.05, random_seed=41)

# X_test, y_test  = create_dataset_by_petals(top_n=400, bottom_n=200,bounding_box=1.2, alpha=40.0, random_seed=1)

X_test, y_test = create_dataset_weighted_by_sigma(
        n_points=400,
        bounding_box=1.2,
        alpha=40.0,
        random_seed=1
    )

# given the min and max of the X_test, generate a grid of points
# space = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
space = np.linspace(-1, 1, 50)
X_grid = np.array([[x, y] for x in space for y in space])

def compute_kde(
    samples: Sequence[float],
    bandwidth: float = 0.05,
    x_points: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    NumPy-based Gaussian KDE for 1D samples.

    :param samples: 1D sequence of data points.
    :param bandwidth: Gaussian kernel bandwidth (std deviation).
    :param x_points: Points at which to evaluate the KDE.
                     If None, defaults to 100 points spanning the sample range.
    :return: 1D array of KDE values at each x in x_points.
    """
    samples = np.asarray(samples, dtype=float)
    if x_points is None:
        # default grid if not provided
        x_min, x_max = samples.min(), samples.max()
        x_points = np.linspace(x_min, x_max, 100)
    x_points = np.asarray(x_points, dtype=float)

    n = samples.size
    factor = 1.0 / (np.sqrt(2 * np.pi) * bandwidth * n)

    # Compute pairwise differences and apply Gaussian kernel
    # shape of diffs: (len(x_points), n)
    diffs = (x_points[:, None] - samples[None, :]) / bandwidth
    kernel_vals = np.exp(-0.5 * diffs**2)

    # Sum over the sample axis
    kde = factor * kernel_vals.sum(axis=1)
    return kde


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


def compute(model, X, y, X_grid, pdc_perturbation, background):
    """
    Computes all the necessary variables for plotting the graph.

    Parameters:
    - model: The machine learning model to use.
    - X: Training data features.
    - y: Training data labels.
    - X_grid: Grid of points for plotting.
    - pdc_perturbation: Perturbation method for uncertainty computation.
    - background: The type of uncertainty to compute (e.g., 'total_uncertainty').

    Returns:
    - A dictionary containing all the computed variables.
    """
    # Fit the model
    print('Fitting the model...')
    model.fit(X, y)

    # Compute probabilities and uncertainties
    print('Computing uncertainties...')
    unc_method = "entropy" if 'var' not in background else "variance"
    unc_method = f"{unc_formula}-{pdc_perturbation}"
    proba = get_tests_classes_perturbators_likelihood_from_model(model, X_grid, pdc_perturbation=pdc_perturbation)

    total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_from_proba(proba, unc_method="entropy")
    var_tu, var_eu, var_au = uncertainty_from_proba(proba, unc_method="variance")
    tynes_uncertainty, _, _ = uncertainty_from_proba(proba, unc_method="tynes")

    # Compute inter-correlation
    inter_corr = pd.DataFrame({
        'TU': total_uncertainty,
        'EU': epistemic_uncertainty,
        'AU': aleatoric_uncertainty
    }).corr().round(1).loc[['TU', 'EU'], ['EU', 'AU']]
    inter_corr.loc['EU', 'EU'] = np.nan

    # Print warnings for high correlations
    if inter_corr.loc['TU', 'EU'] > 0.95:
        print('Warning: High correlation between total and epistemic uncertainty. Aleatoric uncertainty may be absent.')
    if inter_corr.loc['TU', 'AU'] > 0.95:
        print('Warning: High correlation between total and aleatoric uncertainty. Epistemic uncertainty may be absent.')

    # Prepare uncertainty dictionary
    uncertainty = {
        'total_uncertainty': total_uncertainty,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'tynes_uncertainty': tynes_uncertainty,
        'var_tu': var_tu,
        'var_eu': var_eu,
        'var_au': var_au,
    }

    # Compute vmax for color scaling
    vmax = None  # Set to None or compute dynamically if needed
    if vmax is None:
        vmax = np.round(max([uncertainty[key].max() for key in uncertainty.keys()]), 1)

    # Return all computed variables
    return {
        'proba': proba,
        'uncertainty': uncertainty,
        'vmax': vmax,
        'inter_corr': inter_corr
    }

def compute_v2(model, X, y, X_grid, pdc_perturbation):
    """
    Computes all the necessary variables for plotting the graph.

    Parameters:
    - model: The machine learning model to use.
    - X: Training data features.
    - y: Training data labels.
    - X_grid: Grid of points for plotting.
    - pdc_perturbation: Perturbation method for uncertainty computation.

    Returns:
    - A dictionary containing all the computed variables.
    """
    # Fit the model
    print('Fitting the model...')
    model.fit(X, y)

    # Compute probabilities and uncertainties
    print('Computing uncertainties...')
    # unc_method = "entropy" if 'var' not in background else "variance"
    # unc_method = f"{unc_formula}-{pdc_perturbation}"
    proba = get_tests_classes_perturbators_likelihood_from_model(model, X_grid, pdc_perturbation=pdc_perturbation)

    # Compute the mean of proba
    proba_mean = proba.mean(axis=2)           # shape (n_grid, n_classes)
    mean_class0 = proba_mean[:,0]             # shape (n_grid,)

     # choose the same densityX you’ll use in React:
    density_x = np.linspace(0, 1, 200)

    # build a (n_grid, len(density_x)) array of KDEs of the class-0 samples
    kde_class0 = np.stack([
        compute_kde(proba[i,0,:], bandwidth=0.05, x_points=density_x)
        for i in range(proba.shape[0])
    ], axis=0)


    total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = uncertainty_from_proba(proba, unc_method="entropy")
    var_tu, var_eu, var_au = uncertainty_from_proba(proba, unc_method="variance")

    # Prepare uncertainty dictionary
    uncertainty = {
        'total_uncertainty': total_uncertainty,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'var_tu': var_tu,
        'var_eu': var_eu,
        'var_au': var_au,
    }

    # Compute vmax for each uncertainty type
    vmax = {key: np.round(value.max(), 1) for key, value in uncertainty.items()}

    # Return all computed variables
    return {
        # 'proba': proba,
        'uncertainty': uncertainty,
        'vmax': vmax,
        'kde_class0': kde_class0,
        'mean_class0': mean_class0
    }


def plot(compute_results, selection='total_uncertainty'):
    """
    Plots the graph using the precomputed variables from the compute function.

    Parameters:
    - compute_results: Dictionary containing precomputed variables (output of the compute function).
    - selection: The type of uncertainty to plot (e.g., 'total_uncertainty').
    """

    
    # Extract precomputed variables
    proba = compute_results['proba']
    uncertainty = compute_results['uncertainty']
    vmax = compute_results['vmax']
    vmax = compute_results['vmax'][selection]  # Use vmax for the selected uncertainty type
    vmax = None
    X = compute_results['X']
    y = compute_results['y']
    X_grid = compute_results['X_grid']
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 10))
    ax = axs[0]
    plt.sca(ax)

    # Plot the heatmap of the selected uncertainty
    green_to_orange = LinearSegmentedColormap.from_list("GreenToOrange", ["#006400", "white", "orange"])
    heatmap = color_background(X_grid[:, 0], X_grid[:, 1], c=uncertainty[selection], ax=ax, vmax=vmax)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(selection)
    import matplotlib.ticker as ticker
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # Format colorbar ticks to 1 decimal place

    # Plot the training points
    point_colors = ["#FF0000", "#0000FF"]
    cm_bright = ListedColormap(point_colors)
    perm = np.random.permutation(len(X))
    ax.scatter(X[perm, 0], X[perm, 1], c=y[perm], cmap=cm_bright, alpha=1, edgecolors="k")
    for i in [0, -1]:
        ax.scatter(X[[i], 0], X[[i], 1], c=point_colors[i], alpha=1, edgecolors="k", label=f'Class {y[i]}')

    # Set axis limits
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Highlight max and min uncertainty points
    u = uncertainty[selection]
    i = pd.Series(u).argmax()
    ax.scatter((i // 100) / 100 + x_min, (i % 100) / 100 + y_min, facecolors='none',
               edgecolors='gold', s=100, marker='D', linewidths=2, label='max uncertainty')
    i = pd.Series(u).argmin()
    ax.scatter((i // 100) / 100 + x_min, (i % 100) / 100 + y_min, facecolors='none',
               edgecolors='#00FFFF', s=200, marker='h', linewidths=2, label='min uncertainty')

    # Add grid and title
    plt.grid()
    temp_model_name = model.__class__.__name__
    if temp_model_name == 'PairwiseDifferenceClassifier':
        temp_model_name = f'PDC({model.estimator.__class__.__name__})'

    plt.title(f'{temp_model_name} - {pdc_perturbation}')

    # Add interactive cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, color='red', linewidth=2)

    # Second subplot for detailed distribution
    ax2 = axs[1]
    live = True

    def describe(event=None):
        nonlocal live
        if not live:
            return
        if event is None or event.xdata is None or event.ydata is None:
            return
        x1, x2 = event.xdata, event.ydata
        id = get_id(x1, x2)
        if id < 0 or id >= len(X_grid):
            return
        print(f'Clicked location: ({x1:.2f}, {x2:.2f}), Grid ID: {id}')
        predicted_distrib = proba[id, 0, :]
        ground_truth_mu, ground_truth_sigma = get_ground_truth(
            x1, x2,
            kde_red, kde_blue, rose_points,
            sigma_base=0.01,
            k=1
        )
        ground_truth_distrib = np.random.normal(ground_truth_mu, ground_truth_sigma, len(predicted_distrib))
        distance = bhattacharyya_distance(ground_truth_distrib, predicted_distrib)

        proba_red = predicted_distrib.mean()
        plt.sca(ax2)
        ax2.clear()
        x1, x2 = round(X_grid[id][0], 2), round(X_grid[id][1], 2)

        ax2.title.set_text(f'Location {x1} , {x2}')
        sns.distplot(predicted_distrib, kde=True, color=colors[pdc_perturbation], hist=False, rug=True, rug_kws={'alpha': 0.5},
                     label=f'{pdc_perturbation}, mean: {round(proba_red, 2)}', axlabel='Probabilities', ax=ax2)
        sns.distplot(ground_truth_distrib, kde=True, color='gray', hist=False, rug=False,
                     rug_kws={'alpha': 0.4}, ax=ax2, label=f'True distrib. distance = {distance:.5f}')

        plt.xlim(0, 1)
        plt.ylim(0, 10)
        ax2.set_yticks([0, 10])
        ax2.set_ylabel("Density", labelpad=-10)
        ax2.set_xticks([0, 0.5, 1])
        plt.axvline(proba_red, color='red', linestyle='--')
        plt.axvline(uncertainty['aleatoric_uncertainty'][id], color='C0', linestyle=':', label=f'Shannon AU: {uncertainty["aleatoric_uncertainty"][id]:.2f}')
        plt.axvline(uncertainty['epistemic_uncertainty'][id], color='green', linestyle='--', label=f'Shannon EU: {uncertainty["epistemic_uncertainty"][id]:.2f}')
        plt.axvline(uncertainty['var_eu'][id], color='green', linestyle='-.', label=f'Variance EU: {uncertainty["var_eu"][id]:.2f}')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def stop_resume(event):
        if event.key == ' ':
            nonlocal live
            live = not live
            print(f'Live mode: {live}')

    fig.canvas.mpl_connect('button_press_event', describe)  # Handle mouse clicks
    fig.canvas.mpl_connect('key_press_event', stop_resume)  # Handle spacebar to toggle live mode
    plt.show()

compute_results = compute_v2(
    model=model,
    X=X,
    y=y,
    X_grid=X_grid,
    pdc_perturbation=pdc_perturbation  
)

initialize_database()  # Initialize the database if not already done

temp_model_name = model.__class__.__name__
if temp_model_name == 'PairwiseDifferenceClassifier':
    temp_model_name = f'PDC({model.estimator.__class__.__name__})'
    

# # Save the computed results to the database
save_compute_results(
    model_name=temp_model_name,
    pdc_perturbation=pdc_perturbation,
    compute_results=compute_results,
    X=X,
    y=y,
    X_grid=X_grid
)

# Load the computed results from the database
loaded_results = load_compute_results(
    model_name=temp_model_name,
    pdc_perturbation=pdc_perturbation
)

# Check if loaded results are not None and plot them
if loaded_results:
    plot(loaded_results, selection='var_eu')
else:
    print("No matching results found in the database.")

# plot(compute_results, selection=background)