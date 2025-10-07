# Social Network Analysis

A collection of network analysis projects implementing various graph algorithms and machine learning techniques for social network analysis.

## Overview

This repository contains implementations of network analysis techniques including centrality measures, connectivity analysis, link prediction, and graph classification. The projects use NetworkX for graph operations and scikit-learn for machine learning tasks.

## Projects

### 1. Network Connectivity Analysis
**File:** `network_connectivity.ipynb`

Analyzes an internal email communication network from a mid-sized manufacturing company.

**Key Features:**
- Directed multigraph analysis of email communications
- Strong and weak connectivity detection
- Connected components identification
- Network diameter and eccentricity calculations
- Node centrality and periphery analysis
- Minimum cut-set computation
- Transitivity and clustering coefficient analysis

**Techniques Used:**
- Strongly/weakly connected components
- Shortest path algorithms
- Graph diameter and radius
- Node connectivity

### 2. Centrality Measures
**File:** `centrality_measures.ipynb`

Explores multiple centrality measures on friendship and political blog networks.

**Key Features:**
- **Part 1 - Friendship Network:**
  - Degree, closeness, and betweenness centrality
  - Optimal node selection for information diffusion
  - Strategic node identification for network disruption

- **Part 2 - Political Blog Network:**
  - PageRank algorithm implementation
  - HITS (Hyperlink-Induced Topic Search) algorithm
  - Hub and authority score computation

**Applications:**
- Voucher distribution optimization
- Information propagation analysis
- Influence maximization

### 3. Graph Classification and Link Prediction
**File:** `graph_classification_link_prediction.ipynb`

Implements machine learning models for graph analysis and prediction tasks.

**Key Features:**
- **Part 1 - Random Graph Identification:**
  - Classification of graph generation algorithms
  - Detection of Preferential Attachment (PA) networks
  - Identification of Small World networks (low/high rewiring probability)
  - Features: degree distribution, clustering coefficient

- **Part 2A - Management Salary Prediction:**
  - Neural network classifier (MLPClassifier)
  - Node feature engineering using centrality measures
  - AUC-based model evaluation

- **Part 2B - Link Prediction:**
  - Gradient Boosting Classifier
  - Multiple link prediction features:
    - Common neighbors
    - Jaccard coefficient
    - Resource allocation index
    - Adamic-Adar index
    - Preferential attachment score

**Machine Learning Models:**
- Multi-layer Perceptron (MLP) for node classification
- Gradient Boosting for link prediction

## Technologies Used

- **NetworkX**: Graph creation, analysis, and algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning models and preprocessing
  - MLPClassifier
  - GradientBoostingClassifier
  - StandardScaler
  - train_test_split

## Key Concepts Covered

### Graph Theory
- Directed and undirected graphs
- Multigraphs
- Graph connectivity
- Shortest paths
- Centrality measures

### Network Analysis
- Degree centrality
- Closeness centrality
- Betweenness centrality
- PageRank
- HITS algorithm
- Clustering coefficient
- Transitivity

### Link Prediction
- Common neighbors
- Jaccard coefficient
- Resource allocation index
- Adamic-Adar index
- Preferential attachment

### Graph Classification
- Preferential attachment networks
- Small world networks
- Degree distribution analysis
- Clustering patterns

## Performance Metrics

- **Classification Tasks**: Area Under ROC Curve (AUC)
- **Target Performance**:
  - Full points: AUC ≥ 0.88
  - Passing: AUC ≥ 0.82

## Course Information

These projects are part of a Python Social Network Analysis course, focusing on practical applications of graph theory and network science using Python.

## License

MIT License - see [LICENSE](LICENSE) file for details.
