# Laplace Maze Solver

This repository contains an innovative approach to solving mazes using Laplace's equation. Instead of traditional pathfinding algorithms, we use physical principles to model the maze as a potential field and then navigate the maze by following the gradient.

## Theory

### Representing the Maze as a System

A maze can be thought of as a 2D domain where certain regions (paths) are accessible and others (walls) are not. From a physics perspective, imagine filling this domain with a conductive material. Walls are perfect insulators, while paths are conductive regions.

Now, if we impose a potential difference across this domain—by setting high potential at the start and low potential at the end—we induce a potential field across the maze. The distribution of this potential field will be governed by Laplace's equation in regions without any sources or sinks.

### Laplace's Equation

Laplace's equation is a second-order partial differential equation defined as:

$$\nabla^2 \phi = 0$$

Where $\nabla^2$ is the Laplacian operator and $\phi$ is the scalar potential. The Laplacian operator, in a Cartesian coordinate system, is given by:

$$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$$

When we solve this equation with appropriate boundary conditions (start and end potentials and insulating walls), we obtain a continuous scalar field $\phi$ that smoothly varies across the conductive regions of the maze.

### Navigating using the Potential Field

Given the potential field, the path from the start to the end can be determined by following the gradient of the potential, which always points in the direction of the steepest increase in potential:

$$\nabla \phi$$

Moving in the direction of $\nabla \phi$ ensures that at every step, we are moving toward regions of higher potential. For our setup, this would mean moving from the start (high potential) to the end (low potential), effectively solving the maze.

## Implementation

### 1. `solve_laplace(maze, start, end, max_iter=5000)`

This function solves the Laplace's equation iteratively for the given maze.

**Math Details**:
We initialize a potential field $\phi$ of the maze's shape. The starting point is given a potential of -20 and the ending point a potential of 20 (Just to get better symmetry). For all non-wall cells, the potential is updated based on the average of its neighboring cells:

$$ \phi_{i,j} = \frac{(\phi_{i-1,j} + \phi_{i+1,j} + \phi_{i,j-1} + \phi_{i,j+1})}{4} $$

This iteration continues until the field converges or the number of iterations reaches `max_iter`.

### 2. `find_gradient_path(phi, start, end)`

This function determines the path from the start to the end of the maze by following the gradient of the potential field.

**Math Details**:
At every point $(i, j)$, we identify the neighboring cell with the highest potential and move to it. Mathematically, the movement is towards the steepest ascent of $\phi$.

### 3. `calculate_gradient(phi)`

Computes the gradient of the potential field.

**Math Details**:
For each cell $(i, j)$ in the maze, the gradient is computed using central differences:

$$ \nabla\phi_{i,j} = \left( \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2}, \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2} \right) $$

## Conclusion

This method provides an interesting approach to maze-solving by leveraging principles from physics. The generated potential field offers insight into how potential flows through the maze, and the gradient-driven path offers a means to navigate from start to end.

---
