# functions used to graph and show NN in all sorts of ways

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Callable, List, Tuple, Dict, Any, Optional

from .nn import NN

def auto_subplot(
    nrows: int = 1,
    ncols: int = 1,
    func_args: List[Tuple[Callable[..., None], Dict[str, Any]]] = [],
    figsize: Optional[tuple] = None,
) -> None:
    """
    Arrange multiple plotting functions into an nrows×ncols grid.

    Args:
        nrows, ncols: subplot grid shape.
        func_args: list of (func, kwargs) pairs. Each func must accept:
            - fig: the matplotlib Figure
            - nrows, ncols: grid dimensions
            - index: 1-based subplot index or (row, col) tuple
            - plus whatever is in kwargs.
        figsize: optional figure size tuple.
    """
    m = len(func_args)
    if nrows * ncols < m:
        raise ValueError(f"Grid {nrows}×{ncols} too small for {m} plots")

    fig = plt.figure(figsize=figsize)

    for idx, (fn, kw) in enumerate(func_args, start=1):
        # inject the subplot parameters
        call_kwargs = dict(fig=fig, nrows=nrows, ncols=ncols, index=idx)
        call_kwargs.update(kw)
        fn(**call_kwargs)

    plt.tight_layout()
    plt.show()

        
def graph_2d(
    model:NN, 
    x_min: float = 0,
    x_max: float = 5, 
    n: int = 20, 
    xlabel : str = "x",
    ylabel : str = "y",
    title : str = "Network plot",
    ax=None, 
    fig=None, 
    nrows: Optional[int] = None, 
    ncols: Optional[int] = None, 
    index: Optional[int | tuple[int, int]] = None
    ) -> None :
    
    X = np.linspace(x_min,x_max,n).reshape(-1,1)  # shape: (n, 1)

    if ax is None:
        if fig is not None:
            ax = fig.add_subplot(nrows,ncols,index)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(nrows,ncols,index)

    ax.plot(X.flatten(),model.use(X).flatten(),) # flatten to (n,)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)



def graph_3d(
    model:NN,
    x_min: float = 0, 
    x_max: float = 5, 
    y_min: float = 0, 
    y_max: float = 5, 
    n: int = 20, 
    xlabel : str = "x",
    ylabel : str = "y",
    zlabel : str = "z",
    title : str = "Network surface",
    ax=None, 
    fig=None, 
    nrows: Optional[int] = None, 
    ncols: Optional[int] = None, 
    index: Optional[int | tuple[int, int]] = None
    ) -> None :

    grid = np.array([
        [x, y] 
        for x in np.linspace(x_min, x_max, n) 
        for y in np.linspace(y_min, y_max, n)
    ]) # shape: (n*n, 2)

    points = model.use(grid).flatten()  # flatten to (n*n,)

    if ax is None:
        if fig is not None:
            ax = fig.add_subplot(nrows,ncols,index,projection="3d")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(nrows,ncols,index,projection="3d")

    ax.scatter(grid[:, 0], grid[:, 1], points, c=points, cmap="winter")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)




def decision_boundary(
    model:NN,
    x_min: float = 0, 
    x_max: float = 5, 
    y_min: float = 0, 
    y_max: float = 5, 
    n: int = 100, 
    boundary: float = 2.5, 
    data_0: Optional[np.ndarray] = None, 
    data_1: Optional[np.ndarray] = None, 
    xlabel : str = "x",
    ylabel : str = "y",
    title : str = "Decision Boundary",
    class0name : str = "Class 0",
    class1name : str = "Class 1",
    ax=None, 
    fig=None, 
    nrows: Optional[int] = None, 
    ncols: Optional[int] = None, 
    index: Optional[int | tuple[int, int]] = None
    ) -> None :

    # Step 1: Create a meshgrid
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y)
    
    # Step 2: Flatten meshgrid for batch model input
    grid = np.c_[xx.ravel(), yy.ravel()]  # shape: (n*n, 2)
    
    # Step 3: Predict and reshape
    z = model.use(grid).flatten()  # shape: (n*n,)
    z = (z > boundary).astype(int)  # Binary classification
    z = z.reshape(xx.shape)  # shape: (n, n)

    # Step 4: Plot
    if ax is None:
        if fig is not None:
            ax = fig.add_subplot(nrows,ncols,index)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(nrows,ncols,index)

    ax.contourf(xx, yy, z, alpha=0.6, cmap='coolwarm')  # Decision boundary

    if data_0 is not None:
        data_0 = np.array(data_0)
        ax.scatter(data_0[:, 0], data_0[:, 1], c='b', label=class0name)
    if data_1 is not None:
        data_1 = np.array(data_1)
        ax.scatter(data_1[:, 0], data_1[:, 1], c='r', label=class1name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

