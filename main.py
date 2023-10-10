import numpy as np
import matplotlib.pyplot as plt


maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
start = (1, 1)
end = (7, 8)


# Solve Laplace's equation
def solve_laplace(maze, start, end, max_iter=5000):
    phi = np.zeros_like(maze, dtype=float)
    phi[start] = -20
    phi[end] = 20
    wall_mask = (maze == 1)
    for _ in range(max_iter):
        old_phi = phi.copy()
        for i in range(1, phi.shape[0]-1):
            for j in range(1, phi.shape[1]-1):
                if not wall_mask[i, j] and (i, j) not in [start, end]:
                    phi[i, j] = 0.25 * \
                        (phi[i-1, j] + phi[i+1, j] + phi[i, j-1] + phi[i, j+1])
        if np.allclose(old_phi, phi, atol=1e-6):
            break
    return phi

# Find the gradient path


def find_gradient_path(phi, start, end):
    path = [start]
    current_pos = start
    iter_count = 0
    while current_pos != end and iter_count < 10000:
        iter_count += 1
        neighbors = [(current_pos[0]+i, current_pos[1]+j)
                     for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neighbors = [pos for pos in neighbors if 0 <= pos[0] <
                     phi.shape[0] and 0 <= pos[1] < phi.shape[1] and phi[pos] != 0]

        print(f"Current position: {current_pos}, Neighbors: {neighbors}")
        print(f"Phi values: {[phi[pos] for pos in neighbors]}")

        next_pos = max(neighbors, key=lambda pos: phi[pos])
        current_pos = next_pos
        path.append(current_pos)
    return path


# Calculate gradient


def calculate_gradient(phi):
    grad_phi = np.zeros((phi.shape[0], phi.shape[1], 2))
    for i in range(1, phi.shape[0]-1):
        for j in range(1, phi.shape[1]-1):
            if phi[i, j] != 0:
                grad_phi[i, j, 0] = (phi[i, j+1] - phi[i, j-1])/2
                grad_phi[i, j, 1] = (phi[i+1, j] - phi[i-1, j])/2
    return grad_phi


phi = solve_laplace(maze, start, end)
print("Done")
path = find_gradient_path(phi, start, end)
print("Done")
path_np = np.array(path)
print("Done")
grad_phi = calculate_gradient(phi)
print("Done")
y, x = np.mgrid[0:phi.shape[0], 0:phi.shape[1]]


def annotate_values(ax, data, fontsize=8):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha='center',
                    va='center', color='black', fontsize=fontsize)


def plot_circles(ax, data, cmap='viridis', s_factor=1000, min_size=10):
    y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    sizes = np.maximum(data.flatten() * s_factor, min_size)
    ax.scatter(x, y, s=sizes, c="white", alpha=0.6)


fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Original Maze
axs[0, 0].imshow(maze, cmap='gray_r')
axs[0, 0].scatter([start[1], end[1]], [start[0], end[0]],
                  color=['green', 'red'], s=100, zorder=2)
axs[0, 0].set_title('Original Maze (Green: Start, Red: End)')
axs[0, 0].axis('off')

# Potential Field Phi
im_phi = axs[0, 1].imshow(phi, cmap='viridis')
axs[0, 1].scatter([start[1], end[1]], [start[0], end[0]],
                  color=['green', 'red'], s=100, zorder=2)
axs[0, 1].set_title(r'Potential Field $\phi$ (Green: Start, Red: End)')
annotate_values(axs[0, 1], phi)
axs[0, 1].axis('off')
fig.colorbar(im_phi, ax=axs[0, 1], label=r'$\phi$', fraction=0.046, pad=0.04)

# Gradient Field Nabla Phi
axs[1, 0].imshow(phi, cmap='viridis')
grad_magnitude_masked = np.linalg.norm(
    grad_phi, axis=2) * (maze == 0)
plot_circles(axs[1, 0], grad_magnitude_masked, s_factor=50, min_size=20)
# axs[1, 0].quiver(x, y, grad_phi[:, :, 0], grad_phi[:, :, 1],
#                   color='white', scale=100, headwidth=3, headlength=2, width=0.005)
axs[1, 0].scatter([start[1], end[1]], [start[0], end[0]],
                  color=['green', 'red'], s=100, zorder=2)
axs[1, 0].set_title(r'Gradient Field $\nabla\phi$ (Green: Start, Red: End)')
# Annotating gradient magnitudes
annotate_values(axs[1, 0], np.linalg.norm(grad_phi, axis=2))
axs[1, 0].axis('off')

# Path Following Nabla Phi
axs[1, 1].imshow(phi, cmap='viridis')
axs[1, 1].plot(path_np[:, 1], path_np[:, 0],
               color='white', linewidth=2, label='Path')
axs[1, 1].scatter([start[1], end[1]], [start[0], end[0]],
                  color=['green', 'red'], s=100, zorder=2)
axs[1, 1].set_title(r'Path Following $\nabla\phi$ (Green: Start, Red: End)')
axs[1, 1].axis('off')

plt.tight_layout()
# plt.show()
plt.savefig("maze_solution.png", dpi=300, bbox_inches='tight')
