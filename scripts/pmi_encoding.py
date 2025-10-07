#!/usr/bin/env python3
"""
Create PMI encoding with proper handling of different feature types
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def create_proper_encoding():
    """Create encoding with appropriate transformations for each feature type"""
    
    # Paths
    csv_path = "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv"
    output_dir = Path("/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading PMI features...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} parts with {len(df.columns)} columns")
    
    # Get numeric features only
    exclude_cols = ['part_name', 'Processes']
    numeric_df = df.drop(columns=exclude_cols)
    print(f"Processing {len(numeric_df.columns)} numeric features")
    
    # Fill missing values with 0
    data = numeric_df.fillna(0).copy()
    
    # Categorize features
    binary_features = []
    count_features = []
    continuous_features = []
    
    for col in data.columns:
        # Check if binary (only 0 and 1 values)
        unique_vals = data[col].unique()
        if set(unique_vals).issubset({0, 1}):
            binary_features.append(col)
        # Features starting with 'has_' are binary
        elif col.startswith('has_'):
            binary_features.append(col)
        # Count features
        elif 'count' in col or col.startswith('n_') or col == 'total_tolerance_count':
            count_features.append(col)
        # Continuous features (stats, min, max, avg)
        elif 'stats_' in col or '_min' in col or '_max' in col or '_avg' in col:
            continuous_features.append(col)
        # Default: treat as count
        else:
            count_features.append(col)
    
    print(f"\nFeature categorization:")
    print(f"  Binary features: {len(binary_features)}")
    print(f"  Count features: {len(count_features)}")
    print(f"  Continuous features: {len(continuous_features)}")
    
    # Initialize encoded array
    encoded = np.zeros_like(data.values, dtype=float)
    
    # 1. BINARY FEATURES - keep as is
    print("\n1. Processing binary features (no transformation)...")
    for col in binary_features:
        if col in data.columns:
            idx = data.columns.get_loc(col)
            encoded[:, idx] = data[col].values
    
    # 2. COUNT FEATURES - log transform + standardize
    print("2. Processing count features (log + standardization)...")
    if count_features:
        count_indices = [data.columns.get_loc(col) for col in count_features]
        
        # Step 1: Log transform
        for idx, col in zip(count_indices, count_features):
            encoded[:, idx] = np.log1p(data[col].values)  # log(x + 1)
        
        # Step 2: Standardize the log-transformed values
        scaler = StandardScaler()
        encoded[:, count_indices] = scaler.fit_transform(encoded[:, count_indices])
    
    # 3. CONTINUOUS FEATURES - standardize only
    print("3. Processing continuous features (standardization only)...")
    if continuous_features:
        continuous_indices = [data.columns.get_loc(col) for col in continuous_features]
        
        # Direct standardization
        scaler = StandardScaler()
        encoded[:, continuous_indices] = scaler.fit_transform(data[continuous_features].values)
    
    # Save encoding
    output_path = output_dir / "standard_encoding.npy"
    np.save(output_path, encoded)
    
    # Print statistics
    print(f"\n✅ SUCCESS!")
    print(f"Saved to: {output_path}")
    print(f"Shape: {encoded.shape}")
    print(f"Range: [{encoded.min():.3f}, {encoded.max():.3f}]")
    print(f"Mean: {encoded.mean():.3f}, Std: {encoded.std():.3f}")
    
    # Statistics per feature type
    print("\nStatistics by feature type:")
    if binary_features:
        binary_indices = [data.columns.get_loc(col) for col in binary_features]
        binary_vals = encoded[:, binary_indices]
        print(f"  Binary: range=[{binary_vals.min():.2f}, {binary_vals.max():.2f}]")
    
    if count_features:
        count_indices = [data.columns.get_loc(col) for col in count_features]
        count_vals = encoded[:, count_indices]
        print(f"  Count: range=[{count_vals.min():.2f}, {count_vals.max():.2f}], mean={count_vals.mean():.3f}, std={count_vals.std():.3f}")
    
    if continuous_features:
        continuous_indices = [data.columns.get_loc(col) for col in continuous_features]
        cont_vals = encoded[:, continuous_indices]
        print(f"  Continuous: range=[{cont_vals.min():.2f}, {cont_vals.max():.2f}], mean={cont_vals.mean():.3f}, std={cont_vals.std():.3f}")
    
    # Important info for models
    print(f"\n📌 IMPORTANT FOR YOUR MODELS:")
    print(f"   pmi_dim={encoded.shape[1]}")
    print(f"   pmi_path='encoding_results/standard_encoding.npy'")
    
    return encoded

if __name__ == "__main__":
    create_proper_encoding()