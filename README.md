# Antiviral Potency Prediction

[![ASAP Discovery Challenge](https://img.shields.io/badge/Challenge-ASAP%20Discovery%202025-blue)](https://www.asapdiscovery.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.6%2B-red)](https://gpytorch.ai/)

This repository contains advanced machine learning approaches for predicting antiviral potency (pIC50 values) of molecules against SARS-CoV-2 and MERS-CoV main proteases (Mpro). Developed for the ASAP Discovery x OpenADMET Competition (March 2025).

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [About the dataset](#-about-the=dataset)
- [Approaches](#-approaches)
  - [Approach 1: Base GP Model](#approach-1-base-gp-model)
  - [Approach 2: Transfer Learning](#approach-2-transfer-learning)
  - [Approach 3: Active Learning](#approach-3-active-learning)
- [Technical Details](#-technical-details)
  - [Model Architecture](#model-architecture)
  - [Molecular Representation](#molecular-representation)
  - [Training Process](#training-process)
- [Key Features](#-key-features)
- [Results](#-results)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

## 🔍 Project Overview

This project tackles the challenge of predicting molecule potency against coronavirus targets using Gaussian Process Regression with several innovative approaches:

- **Baseline GP Models**: Direct prediction of potency from molecular fingerprints
- **Transfer Learning**: Leveraging larger datasets to improve predictions on smaller target-specific datasets
- **Active Learning**: Intelligent molecule selection strategies to optimize data efficiency

All approaches predict potency for two coronavirus targets:
- SARS-CoV-2 main protease (Mpro)
- MERS-CoV main protease (Mpro)

## 💾 About the dataset
![image](https://github.com/user-attachments/assets/8f1da34c-9840-496a-96d6-b239f2d7b283)

The graph shows a bimodal (two-peaked) distribution, which means compounds fall into roughly two similarity groups:

- The larger peak around 0.2-0.3 indicates that most compound pairs share limited structural features - they're fairly different from each other
- The smaller peak around 0.4-0.5 likely represents compounds within the same chemical series that share common scaffolds
- The long tail extending toward 1.0 represents very similar compound pairs, with the small bar at 1.0 representing identical compounds or extremely close analogs.

![image](https://github.com/user-attachments/assets/eaefaf40-8eb8-45fc-9d29-3f47cbb9d83f)

- The first peak at ~0.2-0.25 represents compound pairs with relatively low structural similarity
- The second peak at ~0.45-0.55 is more pronounced than in the training set and represents compound pairs that share significant structural features

![image](https://github.com/user-attachments/assets/3db9ded8-6484-4991-8292-b98b672e85aa)

- The primary peak at ~0.2-0.25 tells us that most training-test compound pairs share limited structural features - they're generally quite different from each other
- The secondary peak at ~0.45-0.55 indicates a significant subset of test compounds that share common scaffolds or structural elements with the training compounds


![image](https://github.com/user-attachments/assets/1f41e2b9-ea99-4e36-975b-dd1bb0b712d3)
- The main peak around 0.7-0.8 shows that most test compounds have at least one fairly similar analog in the training set. This is promising for prediction accuracy, as your model won't need to extrapolate too far for most compounds.
- The small spike at 1.0 (with ~32 compounds) represents test compounds that are extremely similar or potentially identical to compounds in your training set. These should be predicted very accurately.
- Only a tiny fraction of test compounds have maximum similarities below 0.5, meaning very few test compounds are truly novel compared to your training data.

![image](https://github.com/user-attachments/assets/c918e15d-3b75-4fcf-a2da-1a3378574dae)

- There's a positive correlation between activities against both viral proteases, indicated by the general upward trend. Compounds that work well against SARS-CoV-2 tend to also work against MERS-CoV.
- The red dashed line represents y=x (equal potency against both targets). Many compounds fall close to this line, suggesting similar activity against both proteases.
- Most data points cluster in the 4-6 pIC50 range for both targets, showing moderate activity overall.
- The considerable scatter around the trend indicates that while there's correlation, it's not extremely strong. Many compounds show preferential activity against one target over the other.
- There are some notable outliers, including a few compounds with very low MERS activity (around 1.0) despite moderate SARS activity, and a couple with exceptionally high MERS activity (around 9.0) with moderate SARS activity.

## 🧪 Approaches

### Approach 1: Base GP Model
*Implemented in `GP_only.ipynb`*

A straightforward Gaussian Process Regression model that directly maps molecular fingerprints to potency values:

- Separate models for each viral target
- Multiple kernel options (Linear, Tanimoto, RBF, Matern)
- Standard training procedure with Adam optimizer

**Advantages:**
- Simple implementation
- Built-in uncertainty quantification
- Works well with limited training data

  

### Approach 2: Transfer Learning
*Implemented in `GP_finetuned.ipynb`*

A three-step transfer learning approach:

1. **Base Model Training**: Train on a larger MPRO dataset (2,000 data points)
2. **Model Initialization**: Initialize target-specific models with base model weights
3. **Fine-tuning**: Separately fine-tune for SARS-CoV-2 and MERS-CoV targets

**Advantages:**
- Leverages larger dataset knowledge
- Specializes for each target
- More accurate predictions with uncertainty estimates

### Approach 3: Active Learning
*Combined with transfer learning*

An iterative approach that selects the most informative molecules for training:

- **UCB Selection Strategy**: Balances exploration vs. exploitation
- **Cycle-based Learning**: Iteratively selects batches and retrains models
- **Multiple Selection Protocols**: Options like `ucb-exploit-heavy`, `random`, etc.

**Advantages:**
- Optimizes data efficiency
- Focuses on most informative molecules
- Adaptable selection strategies

## 🔬 Technical Details

### Model Architecture

- **Core Algorithm**: Gaussian Process Regression (GPyTorch)
- **Custom Kernels**:
  - `TanimotoKernel`: Optimized for molecular fingerprint similarity
  - `MaternKernel`: Flexible for structure-activity relationships
  - Also available: RBF, Linear, and RQ kernels
- **Uncertainty Quantification**: Variance estimates for all predictions

### Molecular Representation

Multiple representation options via the `FingerprintFactory`:

- **ECFP**: Extended-Connectivity Fingerprints (Morgan fingerprints)
- **MACCS**: MACCS keys for structural features 
- **ChemBERTa**: Pre-trained molecular embeddings

### Training Process

- **Base Model**: 1,500 epochs (MPRO dataset)
- **Fine-tuning**: 700 epochs per target
- **Active Learning**: Multiple cycles with different selection strategies
- **GPU Acceleration**: Batch processing for memory efficiency
- **Learning Rate Scheduling**: Exponential decay

## 🌟 Key Features

- **Custom Tanimoto Kernel**: Specifically designed for binary fingerprint similarity
- **Uncertainty Analysis**: Confidence estimates for each prediction
- **Memory Management**: Efficient batch prediction for large datasets
- **Experiment Tracking**: Progress monitoring with Weights & Biases
- **Molecular Property Analysis**: Correlating features like molecular weight with potency

## 📊 Results

The models produce:

- Potency predictions for both viral targets
- Uncertainty estimates for each prediction
- Visualizations:
  - Training loss curves
  - Prediction distributions
  - Molecular weight analyses
  - Performance metrics across selection cycles
  



## 📂 Project Structure

- **Data loading & preprocessing**: Molecular representation pipeline
- **Model definitions**: GP model classes and kernel implementations
- **Training functions**: Optimization procedures and hyperparameter settings
- **Active learning**: Selection strategies and cycle-based training
- **Transfer learning**: Knowledge transfer between models
- **Prediction & analysis**: Uncertainty quantification and visualization

## 🔮 Future Improvements

- Finetuning BALM

## 👤 Author

Satya Pratik Srivastava

