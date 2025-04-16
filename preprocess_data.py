import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setting the paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "Thesis", "figures")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_all_datasets(verbose=True):
    """Load and combine all datasets"""
    if verbose:
        print("=== Loading Datasets ===")
    
    # Get all dataset files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if verbose:
        print(f"Found {len(csv_files)} dataset files")
    
    # Load and combine all datasets
    combined_df = pd.DataFrame()
    
    for file in csv_files:
        file_path = os.path.join(DATA_DIR, file)
        if verbose:
            print(f"Loading {file}...")
        
        # Load the entire dataset
        temp_df = pd.read_csv(file_path, delimiter='|', low_memory=False)
        
        # Add source file information
        temp_df['source_file'] = file
        
        # Append to combined dataframe
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        
        if verbose:
            print(f"  Added {len(temp_df)} rows from {file}")
    
    if verbose:
        print(f"\nCombined dataset shape: {combined_df.shape}")
    
    return combined_df

def clean_and_transform_data(df, verbose=True):
    """Clean and transform the dataset"""
    if verbose:
        print("\n=== Cleaning and Transforming Data ===")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle missing values
    if verbose:
        print("Handling missing values...")
        missing_before = df.isna().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
    
    # Convert appropriate columns to numeric
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 
                   'orig_ip_bytes', 'resp_ip_bytes']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate missing percentage for each column
    missing_percentages = df.isna().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > 20].index.tolist()
    
    if high_missing_cols and verbose:
        print(f"Dropping columns with >20% missing values: {high_missing_cols}")
    
    # Drop columns with high percentage of missing values
    df = df.drop(columns=high_missing_cols, errors='ignore')
    
    # Identify columns that are mostly zeros (>95% zeros)
    numeric_df = df.select_dtypes(include=['number'])
    zero_percentages = (numeric_df == 0).mean() * 100
    mostly_zero_cols = zero_percentages[zero_percentages > 95].index.tolist()
    
    if mostly_zero_cols and verbose:
        print(f"Dropping columns with >95% zeros: {mostly_zero_cols}")
    
    # Drop columns that are mostly zeros
    df = df.drop(columns=mostly_zero_cols, errors='ignore')
    
    # For remaining missing values in numeric columns, fill with median
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with the most common value
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['label', 'detailed-label', 'source_file'] and df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Special handling for detailed-label if it wasn't dropped
    if 'detailed-label' in df.columns:
        # For malicious traffic, replace NaNs with 'unknown'
        malicious_mask = (df['label'] == 'Malicious') & df['detailed-label'].isna()
        df.loc[malicious_mask, 'detailed-label'] = 'unknown'
        
        # For benign traffic, fill NaNs with empty string
        benign_mask = (df['label'] == 'Benign') & df['detailed-label'].isna()
        df.loc[benign_mask, 'detailed-label'] = ''
    
    if verbose:
        missing_after = df.isna().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        print(f"Removed {missing_before - missing_after} missing values")
    
    # Feature Engineering
    if verbose:
        print("\nPerforming feature engineering...")
    
    # Convert timestamp to datetime and extract features
    if 'ts' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Compute additional traffic features
    if all(col in df.columns for col in ['orig_bytes', 'resp_bytes']):
        df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
        # Log transformation for skewed features
        df['log_total_bytes'] = np.log1p(df['total_bytes'])
    
    if all(col in df.columns for col in ['orig_pkts', 'resp_pkts']):
        df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']
        df['log_total_pkts'] = np.log1p(df['total_pkts'])
    
    # Bytes per packet (traffic efficiency)
    if all(col in df.columns for col in ['orig_bytes', 'orig_pkts']):
        safe_orig_pkts = df['orig_pkts'].replace(0, 1)
        df['orig_bytes_per_pkt'] = df['orig_bytes'] / safe_orig_pkts
    
    if all(col in df.columns for col in ['resp_bytes', 'resp_pkts']):
        safe_resp_pkts = df['resp_pkts'].replace(0, 1)
        df['resp_bytes_per_pkt'] = df['resp_bytes'] / safe_resp_pkts
    
    # Traffic ratio features
    if all(col in df.columns for col in ['orig_bytes', 'resp_bytes']):
        df['bytes_ratio'] = df['orig_bytes'] / (df['resp_bytes'] + 1)
    
    if all(col in df.columns for col in ['orig_pkts', 'resp_pkts']):
        df['pkts_ratio'] = df['orig_pkts'] / (df['resp_pkts'] + 1)
    
    # Clean up the labels
    if 'label' in df.columns:
        # Standardize label formats
        df['label'] = df['label'].str.strip()
        df['label'] = df['label'].apply(lambda x: 'Malicious' if 'Malicious' in str(x) else x)
        
        # Extract detailed labels from main label if needed
        label_parts = df['label'].str.split(r'\s{2,}', expand=True)
        if label_parts.shape[1] > 1:
            df['label'] = label_parts[0]
            if 'detailed-label' not in df.columns:
                df['detailed-label'] = label_parts[1]
            else:
                empty_detailed = df['detailed-label'].isna() | (df['detailed-label'] == '')
                df.loc[empty_detailed, 'detailed-label'] = label_parts.loc[empty_detailed, 1]
    
    # Drop unnecessary columns
    cols_to_drop = ['uid', 'id.orig_h', 'id.resp_h', 'ts', 'tunnel_parents', 'timestamp']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df

def encode_categorical_features(df, verbose=True):
    """Encode categorical features for machine learning"""
    if verbose:
        print("\n=== Encoding Categorical Features ===")
    
    df = df.copy()
    encoders = {}
    
    # Identify categorical columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols = ['source_file', 'label', 'detailed-label']
    categorical_cols = [col for col in object_cols if col not in exclude_cols]
    
    if verbose:
        print(f"Categorical columns to encode: {categorical_cols}")
    
    # Encode each categorical column
    for col in categorical_cols:
        if verbose:
            print(f"Encoding {col}...")
        
        encoder = LabelEncoder()
        df[col] = df[col].fillna('unknown')
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Special handling for the label column
    if 'label' in df.columns:
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
        encoders['label'] = label_encoder
    
    return df, encoders, categorical_cols

def prepare_train_test_data(df, test_size=0.25, stratify=True, balance=True, verbose=True):
    """Prepare data for training and testing"""
    if verbose:
        print("\n=== Preparing Training and Testing Data ===")
    
    # Get feature columns
    exclude_cols = ['label', 'detailed-label', 'source_file', 'label_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Make sure all feature columns are numeric
    for col in feature_cols:
        if df[col].dtype == 'object':
            print(f"Warning: Column {col} is not numeric. Converting to category codes.")
            df[col] = df[col].astype('category').cat.codes
    
    # Prepare features and target
    X = df[feature_cols]
    y = df['label_encoded'] if 'label_encoded' in df.columns else df['label']
    
    # Split the data
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    if verbose:
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Print class distribution
        train_class_dist = pd.Series(y_train).value_counts(normalize=True) * 100
        test_class_dist = pd.Series(y_test).value_counts(normalize=True) * 100
        
        print("\nClass distribution in training set:")
        for cls, pct in train_class_dist.items():
            print(f"  Class {cls}: {pct:.2f}%")
        
        print("\nClass distribution in testing set:")
        for cls, pct in test_class_dist.items():
            print(f"  Class {cls}: {pct:.2f}%")
    
    # Apply SMOTE to balance the training data if requested
    if balance:
        if verbose:
            print("\nBalancing training data using SMOTE...")
        
        try:
            # Try without n_jobs parameter
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            if verbose:
                balanced_class_dist = pd.Series(y_train).value_counts(normalize=True) * 100
                print("\nClass distribution after balancing:")
                for cls, pct in balanced_class_dist.items():
                    print(f"  Class {cls}: {pct:.2f}%")
        
        except Exception as e:
            print(f"Warning: SMOTE balancing encountered an issue: {str(e)}")
            print("Continuing with imbalanced data.")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    
    if verbose:
        print("\nFeature scaling applied and scaler saved to models directory")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def analyze_preprocessing_results(X_train, X_test, y_train, y_test, features, verbose=True):
    """Analyze and visualize preprocessing results"""
    if verbose:
        print("\n=== Preprocessing Results Analysis ===")
    
    # Check for any remaining missing values
    train_missing = np.isnan(X_train).sum().sum()
    test_missing = np.isnan(X_test).sum().sum()
    
    if verbose:
        print(f"Missing values in training data: {train_missing}")
        print(f"Missing values in testing data: {test_missing}")
    
    # Handle any remaining missing values
    if train_missing > 0 or test_missing > 0:
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Create correlation heatmap
    if len(features) > 0:
        plt.figure(figsize=(14, 12))
        X_train_df = pd.DataFrame(X_train, columns=features)
        
        # Take top 15 features if there are too many
        if len(features) > 15:
            # Get most correlated features with target
            if hasattr(y_train, 'shape'):
                X_train_df['target'] = y_train
                correlations = X_train_df.corr()['target'].abs().sort_values(ascending=False)
                top_features = correlations.index[:16].tolist()
                if 'target' in top_features:
                    top_features.remove('target')
                X_train_df.drop(columns=['target'], inplace=True)
                corr_df = X_train_df[top_features].corr()
            else:
                corr_df = X_train_df.iloc[:, :15].corr()
        else:
            corr_df = X_train_df.corr()
        
        # Plot correlation heatmap
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_correlation.png'))
        plt.close()
        
        if verbose:
            print(f"Feature correlation heatmap saved to {os.path.join(FIGURES_DIR, 'feature_correlation.png')}")
    
    # Class distribution visualization
    plt.figure(figsize=(12, 6))
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    bar_width = 0.35
    index = np.arange(len(train_counts.index))
    
    plt.bar(index, train_counts.values, bar_width, label='Training Set', color='#3498db')
    plt.bar(index + bar_width, test_counts.values, bar_width, label='Testing Set', color='#e74c3c')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training and Testing Sets')
    plt.xticks(index + bar_width/2, train_counts.index)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution_train_test.png'))
    plt.close()
    
    if verbose:
        print(f"Class distribution visualization saved to {os.path.join(FIGURES_DIR, 'class_distribution_train_test.png')}")

def save_processed_data(X_train, X_test, y_train, y_test, features, encoders=None, verbose=True):
    """Save the processed data for later use"""
    if verbose:
        print("\n=== Saving Processed Data ===")
    
    # Save the data
    np.savez(os.path.join(PROCESSED_DIR, 'processed_data.npz'),
            X_train=X_train, X_test=X_test, 
            y_train=y_train, y_test=y_test,
            feature_names=features)
    
    # Save encoders if provided
    if encoders:
        joblib.dump(encoders, os.path.join(PROCESSED_DIR, 'encoders.pkl'))
    
    if verbose:
        print(f"Processed data saved to {os.path.join(PROCESSED_DIR, 'processed_data.npz')}")
        if encoders:
            print(f"Encoders saved to {os.path.join(PROCESSED_DIR, 'encoders.pkl')}")

# Execute the main preprocessing pipeline
print("=== Starting Data Preprocessing Pipeline ===")

# 1. Load all datasets (full dataset)
df_combined = load_all_datasets()

# 2. Clean and transform the data
df_cleaned = clean_and_transform_data(df_combined)

# 3. Ensure binary classification
if 'label' in df_cleaned.columns:
    df_cleaned['label'] = df_cleaned['label'].apply(
        lambda x: 'Malicious' if x.startswith('Malicious') else x
    )
    
    # Check labels
    unique_labels = df_cleaned['label'].unique()
    print(f"\nUnique labels after cleaning: {unique_labels}")
    
    # Filter to keep only Benign and Malicious categories if needed
    if len(unique_labels) > 2:
        print(f"Warning: Found {len(unique_labels)} unique labels. Filtering to keep only Benign and Malicious.")
        df_cleaned = df_cleaned[df_cleaned['label'].isin(['Benign', 'Malicious'])]

# Save the processed dataset
processed_path = os.path.join(PROCESSED_DIR, 'processed_dataset.csv')
df_cleaned.to_csv(processed_path, index=False)
print(f"\nSaved clean processed dataset to {processed_path}")
print(f"Dataset shape: {df_cleaned.shape}")

# 4. Encode categorical features
df_encoded, encoders, cat_cols = encode_categorical_features(df_cleaned)

# 5. Prepare train/test data
X_train, X_test, y_train, y_test, features = prepare_train_test_data(df_encoded, balance=True)

# 6. Analyze preprocessing results
analyze_preprocessing_results(X_train, X_test, y_train, y_test, features)

# 7. Save processed data for modeling
save_processed_data(X_train, X_test, y_train, y_test, features, encoders)

print("\n=== Data Preprocessing Complete ===")
print(f"A single cleaned dataset has been created with {df_cleaned.shape[0]} samples and {df_cleaned.shape[1]} features")
print(f"The processed data is ready for modeling with {X_train.shape[1]} features")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")