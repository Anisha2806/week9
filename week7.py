import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import make_regression

def locally_weighted_regression(x0, X, Y, tau):
    """
    Locally Weighted Linear Regression
    Args:
    x0 : array-like, shape (m,)
        The input point where the prediction is to be made.
    X : array-like, shape (n, m)
        The input features.
    Y : array-like, shape (n,)
        The output values.
    tau : float
        The bandwidth parameter.
        
    Returns:
    y0 : float
        The predicted value at x0.
    """
    m = X.shape[0]
    x0 = np.r_[1, x0]  # Add intercept term
    X = np.c_[np.ones(m), X]  # Add intercept term
    
    # Calculate weights
    W = np.exp(-np.sum((X - x0)*2, axis=1) / (2 * tau*2))
    
    # Compute the theta values using normal equation
    theta = np.linalg.inv(X.T @ (W[:, None] * X)) @ (X.T @ (W * Y))
    
    # Predict the value at x0
    y0 = x0 @ theta
    return y0

def main():
    st.write("22AIA-MACHINE MASTERS")
    st.title("Locally Weighted Regression")
    
    # Generate synthetic dataset
    np.random.seed(0)
    X, Y = make_regression(n_samples=100, n_features=1, noise=10.0)
    X = X.flatten()  # Ensure X is 1D
    
    st.subheader("Generated Dataset")
    data = np.vstack((X, Y)).T
    st.write(pd.DataFrame(data, columns=["Feature", "Target"]))
    
    # Define the range of x values for prediction
    x_pred = np.linspace(X.min(), X.max(), 300)
    
    # User input for bandwidth parameter tau
    tau = st.slider("Select Bandwidth (tau)", 0.01, 1.0, 0.1)
    
    # Predict y values using locally weighted regression
    y_pred = np.array([locally_weighted_regression(x, X, Y, tau) for x in x_pred])
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, Y, color='blue', label='Data Points')
    ax.plot(x_pred, y_pred, color='red', label='LWR Curve')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    ax.set_title('Locally Weighted Regression')
    
    # Display plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()

   
