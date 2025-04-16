import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['savefig.dpi'] = 300  # High resolution for publication-quality figures

# Setting the paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "Thesis", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define color palettes for different plot types
BINARY_PALETTE = ['#3498db', '#e74c3c']
MULTI_PALETTE = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

def load_dataset(file_name):
    """
    Load a specified dataset file from the data directory
    """
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"Loading dataset from: {file_path}")
    
    try:
        # Load the dataset with the pipe delimiter
        df = pd.read_csv(file_path, delimiter='|', low_memory=False)
        
        # Clean up the dataset - convert appropriate columns to numeric
        numeric_cols = ['duration', 'orig_bytes', 'resp_bytes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        if 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
        
        # Print basic information about the dataset
        print(f"\nDataset shape: {df.shape}")
        print("\nColumns in the dataset:")
        print(df.columns.tolist())
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nSummary statistics for numeric columns:")
        print(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
        
        # Additional analysis for detailed labels
        if 'detailed-label' in df.columns:
            print("\nDetailed label distribution:")
            print(df['detailed-label'].value_counts())
        
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def perform_basic_analysis(df):
    """
    Perform basic analysis on the dataset
    """
    print("\n=== Basic Analysis ===")
    
    # Check if 'label' column exists for class distribution analysis
    if 'label' in df.columns:
        print("\nClass distribution:")
        label_counts = df['label'].value_counts()
        print(label_counts)
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y=df['label'], palette=BINARY_PALETTE)
        plt.title('Distribution of Network Traffic Labels', fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Label')
        
        # Add count labels to the bars
        for i, count in enumerate(label_counts.values):
            ax.text(count + 10, i, f"{count:,} ({count/len(df)*100:.1f}%)", va='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'))
        plt.close()
        print(f"Class distribution plot saved to {os.path.join(FIGURES_DIR, 'class_distribution.png')}")
    
    # Check protocol distribution if 'proto' column exists
    if 'proto' in df.columns:
        print("\nProtocol distribution:")
        proto_counts = df['proto'].value_counts()
        print(proto_counts)
        
        # Visualize protocol distribution - use a simple palette based on number of categories
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y=df['proto'])
        plt.title('Distribution of Network Protocols', fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Protocol')
        
        # Add count labels to the bars
        for i, count in enumerate(proto_counts.values):
            ax.text(count + 10, i, f"{count:,} ({count/len(df)*100:.1f}%)", va='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'protocol_distribution.png'))
        plt.close()
        print(f"Protocol distribution plot saved to {os.path.join(FIGURES_DIR, 'protocol_distribution.png')}")
        
        # Create stacked bar chart showing protocol distribution by label
        if 'label' in df.columns:
            plt.figure(figsize=(12, 7))
            protocol_by_label = pd.crosstab(df['proto'], df['label'], normalize='index') * 100
            protocol_by_label.plot(kind='bar', stacked=True, colormap='viridis')
            plt.title('Protocol Distribution by Traffic Type', fontweight='bold')
            plt.xlabel('Protocol')
            plt.ylabel('Percentage')
            plt.legend(title='Traffic Type')
            plt.xticks(rotation=0)
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for n, x in enumerate([*protocol_by_label.index.values]):
                for (proportion, y_loc) in zip(protocol_by_label.loc[x], protocol_by_label.loc[x].cumsum()):
                    if proportion > 5:  # Only show labels for segments > 5%
                        plt.text(n, y_loc - proportion/2, f'{proportion:.1f}%', 
                                ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'protocol_by_label.png'))
            plt.close()
            print(f"Protocol distribution by label plot saved to {os.path.join(FIGURES_DIR, 'protocol_by_label.png')}")
    
    # Analyze connection states if available
    if 'conn_state' in df.columns:
        print("\nConnection state distribution:")
        conn_counts = df['conn_state'].value_counts()
        print(conn_counts)
        
        plt.figure(figsize=(12, 8))
        ax = sns.countplot(y=df['conn_state'], order=conn_counts.index)
        plt.title('Distribution of Connection States', fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Connection State')
        
        # Add count labels to the bars
        for i, count in enumerate(conn_counts.values):
            ax.text(count + 10, i, f"{count:,} ({count/len(df)*100:.1f}%)", va='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'connection_states.png'))
        plt.close()
        print(f"Connection states plot saved to {os.path.join(FIGURES_DIR, 'connection_states.png')}")
        
        # Connection state by label
        if 'label' in df.columns:
            plt.figure(figsize=(14, 10))
            conn_by_label = pd.crosstab(df['conn_state'], df['label'], normalize='index') * 100
            conn_by_label.plot(kind='bar', stacked=True, colormap='viridis')
            plt.title('Connection States by Traffic Type', fontweight='bold')
            plt.xlabel('Connection State')
            plt.ylabel('Percentage')
            plt.legend(title='Traffic Type')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels for significant segments
            for n, x in enumerate([*conn_by_label.index.values]):
                for (proportion, y_loc) in zip(conn_by_label.loc[x], conn_by_label.loc[x].cumsum()):
                    if proportion > 10:  # Only show labels for segments > 10%
                        plt.text(n, y_loc - proportion/2, f'{proportion:.1f}%', 
                                ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'connection_states_by_label.png'))
            plt.close()
            print(f"Connection states by label plot saved to {os.path.join(FIGURES_DIR, 'connection_states_by_label.png')}")

def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in the network data
    """
    print("\n=== Temporal Analysis ===")
    
    if 'timestamp' in df.columns:
        # Add hour of day column
        df['hour'] = df['timestamp'].dt.hour
        
        # Traffic volume by hour
        plt.figure(figsize=(14, 8))
        hourly_traffic = df.groupby(['hour', 'label']).size().unstack()
        hourly_traffic.plot(kind='bar', stacked=True)
        plt.title('Network Traffic Volume by Hour of Day', fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Number of Connections')
        plt.xticks(range(24), range(24))
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Traffic Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'hourly_traffic.png'))
        plt.close()
        print(f"Hourly traffic volume plot saved to {os.path.join(FIGURES_DIR, 'hourly_traffic.png')}")
        
        # Traffic volume time series
        plt.figure(figsize=(16, 8))
        df['time_bin'] = df['timestamp'].dt.floor('10min')
        time_series = df.groupby(['time_bin', 'label']).size().unstack().fillna(0)
        
        time_series.plot()
        plt.title('Network Traffic Volume Over Time', fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Number of Connections')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Traffic Type')
        
        # Format x-axis for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'traffic_timeseries.png'))
        plt.close()
        print(f"Traffic time series plot saved to {os.path.join(FIGURES_DIR, 'traffic_timeseries.png')}")

def analyze_traffic_features(df):
    """
    Analyze network traffic features
    """
    print("\n=== Traffic Feature Analysis ===")
    
    # Distribution of bytes transferred
    if all(col in df.columns for col in ['orig_bytes', 'resp_bytes']):
        # Convert to numeric if needed
        for col in ['orig_bytes', 'resp_bytes']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create total bytes column
        df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
        
        # Plot distribution of bytes (log scale)
        plt.figure(figsize=(12, 8))
        
        # Using log scale due to potential extreme values
        for label, color in zip(['Benign', 'Malicious'], BINARY_PALETTE):
            data = df[df['label'] == label]['total_bytes'].replace(0, 0.1)  # Replace zeros for log scale
            sns.kdeplot(data, shade=True, label=label, color=color, log_scale=True)
        
        plt.title('Distribution of Data Transfer Volume by Traffic Type (Log Scale)', fontweight='bold')
        plt.xlabel('Total Bytes Transferred (log scale)')
        plt.ylabel('Density')
        plt.legend(title='Traffic Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'bytes_distribution.png'))
        plt.close()
        print(f"Bytes distribution plot saved to {os.path.join(FIGURES_DIR, 'bytes_distribution.png')}")
        
    # Packets vs Bytes scatter plot
    if all(col in df.columns for col in ['orig_pkts', 'orig_bytes']):
        plt.figure(figsize=(12, 8))
        
        # Create a sample of the data to avoid overcrowding the plot
        sample_size = min(10000, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        # Create the scatter plot
        sns.scatterplot(data=sample_df, x='orig_pkts', y='orig_bytes', hue='label', 
                        palette=BINARY_PALETTE, alpha=0.6, s=50)
        
        plt.title('Relationship between Packets and Bytes Sent', fontweight='bold')
        plt.xlabel('Number of Packets')
        plt.ylabel('Bytes Sent')
        plt.legend(title='Traffic Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'packets_vs_bytes.png'))
        plt.close()
        print(f"Packets vs bytes plot saved to {os.path.join(FIGURES_DIR, 'packets_vs_bytes.png')}")

def perform_dimensionality_reduction(df):
    """
    Perform PCA on numeric features for visualization
    """
    print("\n=== Dimensionality Reduction Analysis ===")
    
    # Select numeric columns for PCA
    numeric_cols = ['id.orig_p', 'id.resp_p', 'orig_pkts', 'orig_ip_bytes', 
                    'resp_pkts', 'resp_ip_bytes']
    
    # Additional columns if they exist and are numeric
    additional_cols = ['duration', 'orig_bytes', 'resp_bytes']
    for col in additional_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    # Check if we have the necessary columns
    if all(col in df.columns for col in numeric_cols):
        print(f"Performing PCA on columns: {numeric_cols}")
        
        # Sample data to manage memory usage
        sample_size = min(50000, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        # Extract features and standardize
        X = sample_df[numeric_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['label'] = sample_df['label'].reset_index(drop=True)
        
        # Plot the results
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette=BINARY_PALETTE, alpha=0.6, s=50)
        
        # Add some analytics to the plot
        explained_variance = pca.explained_variance_ratio_ * 100
        plt.title('PCA of Network Traffic Features', fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}% variance explained)')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}% variance explained)')
        plt.legend(title='Traffic Type')
        plt.grid(True, alpha=0.3)
        
        # Draw markers to show the distribution of each class
        for label, color in zip(['Benign', 'Malicious'], BINARY_PALETTE):
            class_data = pca_df[pca_df['label'] == label]
            mean_x = class_data['PC1'].mean()
            mean_y = class_data['PC2'].mean()
            plt.plot(mean_x, mean_y, marker='o', markersize=10, color=color)
            plt.annotate(f"Mean {label}", (mean_x, mean_y), xytext=(5, 5), 
                         textcoords='offset points', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'pca_visualization.png'))
        plt.close()
        print(f"PCA visualization saved to {os.path.join(FIGURES_DIR, 'pca_visualization.png')}")
        
        # Feature importance based on PCA
        plt.figure(figsize=(12, 8))
        loadings = pca.components_.T
        df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=numeric_cols)
        
        # Plot heatmap of loadings
        sns.heatmap(df_loadings, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('Feature Contributions to Principal Components', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'pca_loadings.png'))
        plt.close()
        print(f"PCA loadings visualization saved to {os.path.join(FIGURES_DIR, 'pca_loadings.png')}")

def analyze_multiple_files(analyze_all=False):
    """
    Analyze and compare multiple files
    
    Parameters:
    analyze_all (bool): If True, analyze all files; if False, just compare statistics
    """
    print("\n=== Multiple Dataset Comparison ===")
    
    # Get all dataset files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} dataset files")
    
    # Collect basic statistics across files
    stats = []
    detailed_labels_counts = {}
    
    for file in csv_files:
        try:
            print(f"\nProcessing {file}...")
            if analyze_all:
                # Full analysis on each file
                df = load_dataset(file)
                if df is not None:
                    print(f"\n--- Performing analysis on {file} ---")
                    perform_basic_analysis(df)
                    analyze_temporal_patterns(df)
                    analyze_traffic_features(df)
                    perform_dimensionality_reduction(df)
                    analyze_detailed_labels(df)
                    print(f"--- Completed analysis on {file} ---\n")
            else:
                # Just get statistics
                df = pd.read_csv(os.path.join(DATA_DIR, file), delimiter='|', low_memory=False, nrows=10000)
                if 'label' in df.columns:
                    malicious_count = df[df['label'] == 'Malicious'].shape[0]
                    benign_count = df[df['label'] == 'Benign'].shape[0]
                    total_count = df.shape[0]
                    
                    malicious_pct = (malicious_count / total_count) * 100 if total_count > 0 else 0
                    benign_pct = (benign_count / total_count) * 100 if total_count > 0 else 0
                    
                    # Count missing values
                    missing_data_cols = df.columns[df.isna().any()].tolist()
                    missing_data_counts = df.isna().sum().sum()
                    
                    stats.append({
                        'file': file,
                        'total_rows': total_count,
                        'malicious_count': malicious_count,
                        'benign_count': benign_count,
                        'malicious_pct': malicious_pct,
                        'benign_pct': benign_pct,
                        'missing_data_columns': missing_data_cols,
                        'missing_data_count': missing_data_counts
                    })
                    
                    # Collect detailed label statistics
                    if 'detailed-label' in df.columns:
                        label_counts = df['detailed-label'].value_counts().to_dict()
                        for label, count in label_counts.items():
                            if label in detailed_labels_counts:
                                detailed_labels_counts[label] += count
                            else:
                                detailed_labels_counts[label] = count
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Create a comparison bar chart for class distribution
    if stats and not analyze_all:
        stats_df = pd.DataFrame(stats)
        
        print("\n=== Dataset Statistics Summary ===")
        print(f"Total files analyzed: {len(stats_df)}")
        print("\nClass distribution across datasets:")
        print(stats_df[['file', 'malicious_count', 'benign_count', 'malicious_pct', 'benign_pct']])
        
        print("\nMissing data summary:")
        for idx, row in stats_df.iterrows():
            print(f"{row['file']}: {row['missing_data_count']} missing values in {len(row['missing_data_columns'])} columns")
            if row['missing_data_columns']:
                print(f"  Columns with missing data: {', '.join(row['missing_data_columns'])}")
        
        # Class distribution chart
        plt.figure(figsize=(14, 8))
        bar_width = 0.35
        index = np.arange(len(stats_df))
        
        plt.bar(index, stats_df['malicious_pct'], bar_width, label='Malicious', color='#e74c3c')
        plt.bar(index, stats_df['benign_pct'], bar_width, bottom=stats_df['malicious_pct'], label='Benign', color='#3498db')
        
        plt.title('Class Distribution Across Multiple Datasets', fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Percentage')
        plt.xticks(index, stats_df['file'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_DIR, 'dataset_comparison.png'))
        plt.close()
        print(f"Dataset comparison plot saved to {os.path.join(FIGURES_DIR, 'dataset_comparison.png')}")
        
        # Missing data visualization
        plt.figure(figsize=(14, 8))
        sns.barplot(x=stats_df['file'], y=stats_df['missing_data_count'])
        plt.title('Missing Data Count Across Datasets', fontweight='bold')
        plt.xlabel('Dataset')
        plt.ylabel('Count of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'missing_data_comparison.png'))
        plt.close()
        print(f"Missing data comparison plot saved to {os.path.join(FIGURES_DIR, 'missing_data_comparison.png')}")
        
        # Detailed label distribution
        if detailed_labels_counts:
            detailed_labels_df = pd.DataFrame(list(detailed_labels_counts.items()), columns=['Label', 'Count'])
            detailed_labels_df = detailed_labels_df.sort_values('Count', ascending=False)
            
            plt.figure(figsize=(16, 10))
            sns.barplot(x='Count', y='Label', data=detailed_labels_df.head(15))
            plt.title('Top 15 Detailed Labels Across All Datasets', fontweight='bold')
            plt.xlabel('Count')
            plt.ylabel('Detailed Label')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'detailed_labels_all_datasets.png'))
            plt.close()
            print(f"Detailed labels plot saved to {os.path.join(FIGURES_DIR, 'detailed_labels_all_datasets.png')}")

def analyze_detailed_labels(df):
    """
    Analyze the detailed labels in the dataset
    """
    print("\n=== Detailed Label Analysis ===")
    
    if 'detailed-label' in df.columns:
        print("\nDetailed label distribution:")
        detailed_counts = df['detailed-label'].value_counts()
        print(detailed_counts.head(10))  # Show top 10 most common detailed labels
        
        # Create a bar plot of top detailed labels
        plt.figure(figsize=(14, 10))
        top_labels = detailed_counts.head(10)
        ax = sns.barplot(x=top_labels.values, y=top_labels.index)
        plt.title('Top 10 Detailed Attack Types', fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Attack Type')
        
        # Add count labels to the bars
        for i, count in enumerate(top_labels.values):
            ax.text(count + 10, i, f"{count:,} ({count/len(df)*100:.1f}%)", va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'detailed_labels_top10.png'))
        plt.close()
        print(f"Detailed labels plot saved to {os.path.join(FIGURES_DIR, 'detailed_labels_top10.png')}")

def combined_dataset_analysis():
    """
    Analyze a combined dataset from multiple files to get a comprehensive view
    """
    print("\n=== Combined Dataset Analysis ===")
    
    # Get all dataset files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    # Load and combine samples from each dataset
    combined_df = pd.DataFrame()
    sample_size_per_file = 10000  # Adjust as needed based on memory constraints
    
    for file in csv_files:
        try:
            print(f"Loading sample from {file}...")
            temp_df = pd.read_csv(os.path.join(DATA_DIR, file), delimiter='|', low_memory=False, nrows=sample_size_per_file)
            temp_df['source_file'] = file  # Add source file information
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not combined_df.empty:
        print(f"\nCombined dataset shape: {combined_df.shape}")
        
        # Convert numeric columns
        numeric_cols = ['duration', 'orig_bytes', 'resp_bytes']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Convert timestamp
        if 'ts' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['ts'], unit='s')
        
        # Check for missing values
        missing_values = combined_df.isnull().sum()
        print("\nMissing values per column in combined dataset:")
        print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")
        
        # Visualize missing data
        plt.figure(figsize=(14, 8))
        sns.heatmap(combined_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Data in Combined Dataset', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'combined_missing_data.png'))
        plt.close()
        print(f"Missing data visualization saved to {os.path.join(FIGURES_DIR, 'combined_missing_data.png')}")
        
        # Brief analysis of the combined dataset
        if 'label' in combined_df.columns:
            print("\nClass distribution in combined dataset:")
            label_counts = combined_df['label'].value_counts()
            print(label_counts)
        
        if 'detailed-label' in combined_df.columns:
            print("\nDetailed label distribution in combined dataset:")
            detailed_counts = combined_df['detailed-label'].value_counts().head(15)
            print(detailed_counts)
            
            plt.figure(figsize=(14, 10))
            sns.countplot(y=combined_df['detailed-label'], order=combined_df['detailed-label'].value_counts().index[:15])
            plt.title('Top 15 Detailed Labels in Combined Dataset', fontweight='bold')
            plt.xlabel('Count')
            plt.ylabel('Detailed Label')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'combined_detailed_labels.png'))
            plt.close()
            print(f"Combined detailed labels plot saved to {os.path.join(FIGURES_DIR, 'combined_detailed_labels.png')}")
        
        return combined_df
    else:
        print("Failed to create combined dataset.")
        return None

if __name__ == "__main__":
    print("=== Network Malware Analysis ===")
    print("Starting comprehensive analysis of all datasets...")
    
    # Option 1: Analyze a single dataset
    # dataset_file = "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"
    # df = load_dataset(dataset_file)
    # if df is not None:
    #     perform_basic_analysis(df)
    #     analyze_temporal_patterns(df)
    #     analyze_traffic_features(df)
    #     perform_dimensionality_reduction(df)
    #     analyze_detailed_labels(df)
    
    # Option 2: Compare statistics across all datasets
    analyze_multiple_files(analyze_all=False)
    
    # Option 3: Perform analysis on a combined dataset
    combined_df = combined_dataset_analysis()
    
    print("\nAnalysis complete!")
    print(f"Generated visualizations have been saved in the {FIGURES_DIR} directory.")