# README.md

## Symmetric Kullback-Leibler Divergence Loss

This directory provides an implementation of a **Symmetric Kullback-Leibler (KL) Divergence Loss** tailored for tensors
representing ellipses in matrix form. The loss function is designed to measure the difference between two elliptical
shapes and is particularly useful in optimization and generative modeling tasks.

## Loss Calculation

### **Kullback-Leibler Divergence**

For two ellipses represented by their matrix forms ( $A_1$ ) and ( $A_2$ ), the KL divergence is calculated as:
$$ D_{KL}(A_1 \parallel A_2) = \frac{1}{2} \left( \text{Tr}(C_2^{-1}C_1) + (\mu_1 - \mu_2)^T C_2^{-1} (\mu_1 - \mu_2) - 2 + \log\left(\frac{\det(C_2)}{\det(C_1)}\right) \right) $$
Where:

- ( $C_1$, $C_2$ ): Covariance matrices extracted from ( $A_1$, $A_2$ ).
- ( $\mu_1$, $\mu_2$ ): Centers (means) of the ellipses, computed from the conic representation.
- ( $\text{Tr}$ ): Trace operator.
- ( $C_2^{-1}$ ): Inverse of the covariance matrix of ( $A_2$ ).
- ( $\det(C_1)$, $\det(C_2)$ ): Determinants of covariance matrices.

A regularization term ( $\epsilon$ ) is added to ensure numerical stability when computing inverses and determinants.

### **Symmetric KL Divergence**

The symmetric version of the KL divergence combines the calculations in both directions:
$$ D_{KL}^{\text{sym}}(A_1, A_2) = \frac{1}{2} \left( D_{KL}(A_1 \parallel A_2) + D_{KL}(A_2 \parallel A_1) \right) $$
This ensures a bidirectional comparison, making the function suitable as a loss metric in optimization tasks.

## Features of the Loss

- **Shape-Only Comparison**: Option to ignore translation and compute divergence based purely on the shapes (covariance
  matrices).
- **NaN Handling**: Replaces NaN values with a specified constant, ensuring robust loss evaluation.
- **Normalization**: An optional normalization step that rescales the divergence for certain applications.

### Usage

The loss is encapsulated in the `SymmetricKLDLoss` class, which integrates seamlessly into PyTorch-based workflows.
