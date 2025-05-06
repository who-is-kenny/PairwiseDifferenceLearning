import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def build_rose_kde(noise_scale=0.01, random_seed=42):
    """
    Build a single KDE for red and blue segments of the rose curve (r = sin(2θ))
    so that we don't re-randomize on each point.
    
    Returns
    -------
    kde_red : gaussian_kde
        KDE over the red portion of the rose curve (with noise).
    kde_blue : gaussian_kde
        KDE over the blue portion of the rose curve (with noise).
    rose_points : np.ndarray of shape (2, 800)
        The ideal (x, y) points of the rose curve without noise.
    """
    np.random.seed(random_seed)
    
    # Generate the rose curve: r = sin(2θ)
    theta_full = np.linspace(0, 2 * np.pi, 2000)
    r_full = np.sin(2 * theta_full)
    x_full = r_full * np.cos(theta_full)
    y_full = r_full * np.sin(theta_full)
    
    # Define red and blue regions by angle
    #   Red  = θ in [0, π/2] ∪ [π, 3π/2]
    #   Blue = everything else
    red_mask = ((theta_full >= 0) & (theta_full <= np.pi/2)) | \
               ((theta_full >= np.pi) & (theta_full <= 3*np.pi/2))
    blue_mask = ~red_mask
    
    x_red,  y_red  = x_full[red_mask],  y_full[red_mask]
    x_blue, y_blue = x_full[blue_mask], y_full[blue_mask]
    
    # Generate noise to create a "thick" density
    noise_x_red  = np.random.normal(loc=x_red,  scale=noise_scale)
    noise_y_red  = np.random.normal(loc=y_red,  scale=noise_scale)
    noise_x_blue = np.random.normal(loc=x_blue, scale=noise_scale)
    noise_y_blue = np.random.normal(loc=y_blue, scale=noise_scale)
    
    # Combine original points with noise
    x_all_red  = np.concatenate([x_red,  noise_x_red])
    y_all_red  = np.concatenate([y_red,  noise_y_red])
    x_all_blue = np.concatenate([x_blue, noise_x_blue])
    y_all_blue = np.concatenate([y_blue, noise_y_blue])
    
    # Build KDEs for red and blue
    kde_red  = gaussian_kde(np.vstack([x_all_red,  y_all_red]))
    kde_blue = gaussian_kde(np.vstack([x_all_blue, y_all_blue]))
    
    # Ideal (no‐noise) rose points for distance computations
    rose_points = np.vstack([x_full, y_full])
    
    return kde_red, kde_blue, rose_points

#ground truth function 
def get_ground_truth(x, y, kde_red, kde_blue, rose_points,
                     sigma_base=0.01, k=3.0):
    """
    Given a point (x, y) and precomputed KDEs + ideal rose curve points, compute:
      - mu : probability that the point is red
      - sigma : uncertainty, which grows with distance from the rose curve
    """
    # Evaluate densities at (x, y)
    point = np.array([[x], [y]])
    density_red  = kde_red(point)[0]
    density_blue = kde_blue(point)[0]
    total_density = density_red + density_blue
    
    # Probability of being red
#     if total_density < 1e-12:
#         mu = 0.5
#     else:
#         mu = density_red / total_density
        
    mu = density_red / total_density
    
    # Distance-based uncertainty: sigma = sigma_base + k * distance
    distances = np.sqrt((rose_points[0] - x)**2 + (rose_points[1] - y)**2)
    d_min = np.min(distances)
    sigma = sigma_base + k * d_min
    
    return mu, sigma

# creates dataset
def create_dataset_weighted_by_sigma(n_points=800,
                                     bounding_box=1.2,
                                     alpha=15.0,
                                     random_seed=42):
    """
    1. Build the rose KDE exactly once.
    2. Randomly sample points in [-bounding_box, bounding_box]^2.
    3. For each point, compute (mu, sigma). Accept with probability p = exp(-alpha*sigma).
    4. Label accepted points as red if mu>0.5, else blue.
    5. Continue until we have n_points accepted.
    
    Parameters
    ----------
    n_points : int
        Number of points to accept (i.e. final dataset size).
    bounding_box : float
        We'll sample x,y uniformly in [-bounding_box, bounding_box].
    alpha : float
        Factor multiplying sigma inside exp(-alpha*sigma). Larger alpha => steeper dropoff.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    X : np.ndarray, shape (n_points, 2)
        Accepted points' coordinates.
    y : np.ndarray, shape (n_points,)
        Labels (0 for red, 1 for blue).
    """
    # 1. Build the rose KDE once
    np.random.seed(random_seed)
    kde_red, kde_blue, rose_points = build_rose_kde(
        noise_scale=0.01,  # smaller noise => sharper petals
        random_seed=random_seed
    )
    
    accepted_points = []
    accepted_labels = []
    
    # 2. Rejection sampling until n_points are accepted
    while len(accepted_points) < n_points:
        x_candidate = np.random.uniform(-bounding_box, bounding_box)
        y_candidate = np.random.uniform(-bounding_box, bounding_box)
        
        # 3. Evaluate mu and sigma using the PRE-BUILT KDE
        mu, sigma = get_ground_truth(
            x_candidate, y_candidate,
            kde_red, kde_blue, rose_points,
            sigma_base=0.01,  # base uncertainty
            k=1.0             # multiply distance by 3 => bigger sigma far from petals
        )
        
        # Accept with probability p = exp(-alpha * sigma)
        p_accept = np.exp(-alpha * sigma)
        if np.random.rand() < p_accept:
            accepted_points.append([x_candidate, y_candidate])
            label = 0 if mu > 0.5 else 1  # 0 = red, 1 = blue
            accepted_labels.append(label)
    
    X = np.array(accepted_points)
    y = np.array(accepted_labels)
    return X, y

# creates data set where petals differ in points 
def create_dataset_by_petals(top_n=200, bottom_n=100,
                             bounding_box=1.2, alpha=40.0, random_seed=42):
    """
    1. Build the rose KDE exactly once.
    2. Rejection-sample candidate points in [-bounding_box, bounding_box]^2.
    3. For each candidate, compute (mu, sigma) and accept it with probability exp(-alpha*sigma).
    4. If accepted, assign it to:
         - the top half (y >= 0) if we still need top_n points,
         - or the bottom half (y < 0) if we still need bottom_n points.
    5. Continue until we have exactly top_n + bottom_n points.
    
    Returns
    -------
    X : np.ndarray, shape ((top_n+bottom_n), 2)
        Accepted points' coordinates.
    y : np.ndarray, shape ((top_n+bottom_n),)
        Labels (0 for red, 1 for blue).
    """
    np.random.seed(random_seed)
    kde_red, kde_blue, rose_points = build_rose_kde(noise_scale=0.01, random_seed=random_seed)
    
    accepted_points_top = []
    accepted_labels_top = []
    accepted_points_bottom = []
    accepted_labels_bottom = []
    
    # Continue sampling until both regions are filled
    while len(accepted_points_top) < top_n or len(accepted_points_bottom) < bottom_n:
        x_candidate = np.random.uniform(-bounding_box, bounding_box)
        y_candidate = np.random.uniform(-bounding_box, bounding_box)
        
        mu, sigma = get_ground_truth(x_candidate, y_candidate,
                                     kde_red, kde_blue, rose_points,
                                     sigma_base=0.01, k=1.0)
        p_accept = np.exp(-alpha * sigma)
        if np.random.rand() < p_accept:
            label = 0 if mu > 0.5 else 1
            if y_candidate >= 0 and len(accepted_points_top) < top_n:
                accepted_points_top.append([x_candidate, y_candidate])
                accepted_labels_top.append(label)
            elif y_candidate < 0 and len(accepted_points_bottom) < bottom_n:
                accepted_points_bottom.append([x_candidate, y_candidate])
                accepted_labels_bottom.append(label)
    
    X_top = np.array(accepted_points_top)
    X_bottom = np.array(accepted_points_bottom)
    y_top = np.array(accepted_labels_top)
    y_bottom = np.array(accepted_labels_bottom)
    
    X = np.concatenate([X_top, X_bottom], axis=0)
    y = np.concatenate([y_top, y_bottom], axis=0)
    return X, y

# uncomment to visual the plot 

if __name__ == "__main__":
    # Generate and label data with stronger sigma effect
    # - bounding_box=1.2 => narrower region
    # - alpha=15 => points far from petals are almost never accepted
    # - k=3 => sigma grows quickly with distance => strong penalty
    X, y = create_dataset_weighted_by_sigma(
        n_points=600,
        bounding_box=1.2,
        alpha=40.0,
        random_seed=42
    )
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red',  s=15, label='Class 0 (red)')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', s=15, label='Class 1 (blue)')
    plt.title("Rose Curve Dataset Weighted by Stronger Uncertainty (sigma)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()
    
   
