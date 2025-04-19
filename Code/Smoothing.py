REDUCED_IMAGE_FILE = "Input-Images/reduced_lower_lung_area.png"
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import cdist
from scipy.integrate import quad


def load_img(image_path, save_plot = False):
    img = Image.open(image_path)
    
    # Convert to array while handling alpha channels if present
    if img.mode == 'RGBA':
        arr = np.array(img)
        mask_white = (
            (arr[..., :3] == [255, 255, 255]).all(axis=-1) & 
            (arr[..., 3] >= 254)  # Ensure near fully opaque (adjust threshold if needed)
        )
    else:
        arr = np.array(img.convert('RGB'))
        mask_white = (arr == [255, 255, 255]).all(axis=-1)
    
    # Extract coordinates
    rows, cols = np.where(mask_white)
    white_points = list(zip(cols, rows))
    
    print(f"Number of white points: {len(white_points)}")
    print("First few coordinates:", white_points[:5])

    if save_plot == True:
        # Plot the image and overlay white points in red
        plt.figure(figsize=(10, 6))
        plt.imshow(arr)
        plt.scatter(cols, rows, c='red', s=20)  # Adjust size as needed
        plt.title("Cell Locations (Red Marks)")
        plt.axis('off')
        plt.savefig("Images/Identified_Points.png")

    else:
        print("No plot.")

    return np.array(white_points)

def bump_integral_2d():
    #Precompute the 2D integral of the unnormalized bump function
    integrand = lambda r: np.exp(-1 / (1 - r**2)) * r
    return 2 * np.pi * quad(integrand, 0, 1)[0]

C_phi = bump_integral_2d()  # scalar constant

def normalized_bump_sum(x_eval, centers, eps):
    # Sum of normalized bumps, each integrating to 1
    dists = cdist(x_eval, centers) / eps
    bumps = bump_rbf(dists)
    Z = np.sum(bumps, axis=1)
    return Z / (C_phi * eps**2)

def bump_rbf(r):
    # Radial bump function with compact support on [0, 1)
    mask = r < 1.0
    result = np.zeros_like(r)
    result[mask] = np.exp(-1.0 / (1 - r[mask]**2))
    return result

# Normalize bump function over 2D
def normalized_bump(x_eval, centers, eps, weights):
    dists = cdist(x_eval, centers) / eps  # shape: (M_eval, N_cloud)
    bumps = bump_rbf(dists)
    bumps *= weights  # weighted bumps
    return np.sum(bumps, axis=1)  # evaluate at all x_eval points

def bump_sum(x_eval, centers, eps):
    """
    Evaluate unnormalized sum of bump functions at x_eval points.
    
    x_eval: (M, 2) evaluation points
    centers: (N, 2) center locations of bumps (point cloud)
    eps: bump support radius
    """
    dists = cdist(x_eval, centers) / eps
    bumps = bump_rbf(dists)
    return np.sum(bumps, axis=1)

def make_bump_cloud_visualization(points,eps,res,X_range,Y_range,image_file,Z_file=None):
    # Evaluate smoothed field on a grid ---
    eps = eps  # bump support radius
    grid_res = res
    x = np.linspace(X_range[0], X_range[1], grid_res)
    y = np.linspace(Y_range[0], Y_range[1], grid_res)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    weights = np.ones(points.shape[0]) / points.shape[0]

    Z = normalized_bump_sum(grid_points, points, eps)
    Z = Z.reshape((grid_res, grid_res))
    
    if Z_file != None:
        np.savetxt(Z_file,Z)

    dx = (x.max() - x.min()) / (grid_res - 1)
    dy = (y.max() - y.min()) / (grid_res - 1)
    integral = np.sum(Z) * dx * dy
    print(f"Integral of smoothed field: {integral:.4f} (expected: {points.shape[0]})")

    extent = [x.min(), x.max(), y.min(), y.max()]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5),constrained_layout=True)

    # Original point cloud
    ax[0].scatter(points[:, 0], points[:, 1], c='k', s=10)
    ax[0].set_title("Original Point Cloud")
    ax[0].set_xlim(extent[0], extent[1])
    ax[0].set_ylim(extent[2], extent[3])
    ax[0].axis('equal')
    ax[0].axis('off')

    # Smoothed scalar field (heatmap)
    im = ax[1].imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower', cmap='viridis')
    ax[1].set_title("Smoothed Field")
    ax[1].axis('equal')
    ax[1].axis('off')

    # Binary support mask
    binary = (Z == 0).astype(float)
    ax[2].imshow(binary, extent=[x.min(), x.max(), y.min(), y.max()],
                 origin='lower', cmap='gray')
    ax[2].set_title("Support Region (Z > 0)")
    ax[2].axis('equal')
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(image_file)

def main():
    # Load image and handle modes (RGB or RGBA)
    points = load_img(REDUCED_IMAGE_FILE,True)
    print(f"Reduced number of points: {points.shape[0]}")

    make_bump_cloud_visualization(points,7.5,100,[0,175],[0,160],image_file="Images/Bump_Test_7-5.png",Z_file="Intermediate-Objects/T0_map.txt")

if __name__ == "__main__":
    main()

