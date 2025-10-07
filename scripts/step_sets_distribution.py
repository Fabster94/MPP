# standard imports
import json
import logging
from pathlib import Path
from collections import Counter

# third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# custom imports
from mpp.constants import PATHS, TKMS_VOCAB
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def analyze_tkms_pmi_distribution(
    pmi_path="/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy",
    pmi_csv_path="/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv"
):
    """
    Analyze the distribution of manufacturing steps in TKMS_PMI dataset across train/valid/test splits
    
    Parameters
    ----------
    pmi_path : str
        Path to encoded PMI features
    pmi_csv_path : str
        Path to PMI CSV file with part names
    
    Returns
    -------
    tuple
        (step_distributions, sample_counts, raw_data)
    """
    
    print("\n" + "="*80)
    print("TKMS PMI DATASET - MANUFACTURING STEP DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Initialize storage for all splits
    results = {}
    raw_data = {}
    
    # Analyze each split
    for mode in ['train', 'valid', 'test']:
        print(f"\nAnalyzing {mode.upper()} split...")
        
        try:
            # Load dataset
            dataset = TKMS_PMI_Dataset(
                mode=mode,
                target_type="step-set",  # Use step-set to get multi-label targets
                pmi_path=pmi_path,
                pmi_csv_path=pmi_csv_path
            )
            
            # Initialize counters
            step_counter = Counter()
            total_samples = len(dataset)
            samples_with_processes = 0
            samples_without_processes = 0
            
            # Collect all targets for detailed analysis
            all_targets = []
            
            # Process each sample
            for idx in range(len(dataset)):
                (vecset, pmi), target = dataset[idx]
                all_targets.append(target)
                
                # target is a multi-label vector [Bohren, Drehen, Fräsen]
                # Map to process names (excluding START, STOP, PAD)
                process_names = ['Bohren', 'Drehen', 'Fräsen']
                
                has_any_process = False
                for i, process in enumerate(process_names):
                    if target[i] == 1:
                        step_counter[process] += 1
                        has_any_process = True
                
                if has_any_process:
                    samples_with_processes += 1
                else:
                    samples_without_processes += 1
            
            # Store results
            results[mode] = {
                'total_samples': total_samples,
                'samples_with_processes': samples_with_processes,
                'samples_without_processes': samples_without_processes,
                'step_counts': dict(step_counter),
                'all_targets': torch.stack(all_targets)
            }
            
            # Store raw data for further analysis
            raw_data[mode] = all_targets
            
            print(f"  Total samples: {total_samples}")
            print(f"  Samples with processes: {samples_with_processes}")
            print(f"  Samples without processes: {samples_without_processes}")
            
        except Exception as e:
            logger.error(f"Error analyzing {mode} split: {e}")
            results[mode] = None
    
    return results, raw_data


def calculate_step_distributions(results):
    """
    Calculate percentage distributions for each manufacturing step
    
    Parameters
    ----------
    results : dict
        Results from analyze_tkms_pmi_distribution
    
    Returns
    -------
    pd.DataFrame
        DataFrame with distribution statistics
    """
    
    # Process names
    process_names = ['Bohren', 'Drehen', 'Fräsen']
    
    # Create distribution table
    data = []
    
    for process in process_names:
        row = {'Process': process}
        
        for mode in ['train', 'valid', 'test']:
            if results[mode]:
                count = results[mode]['step_counts'].get(process, 0)
                total = results[mode]['samples_with_processes']
                percentage = (count / results[mode]['total_samples']) * 100 if results[mode]['total_samples'] > 0 else 0
                
                row[f'{mode}_count'] = count
                row[f'{mode}_pct'] = percentage
        
        data.append(row)
    
    # Add row for "No Process" samples
    no_process_row = {'Process': 'No Process'}
    for mode in ['train', 'valid', 'test']:
        if results[mode]:
            count = results[mode]['samples_without_processes']
            total = results[mode]['total_samples']
            percentage = (count / total) * 100 if total > 0 else 0
            
            no_process_row[f'{mode}_count'] = count
            no_process_row[f'{mode}_pct'] = percentage
    
    data.append(no_process_row)
    
    df = pd.DataFrame(data)
    
    # Calculate standard deviation across splits for each process
    for _, row in df.iterrows():
        percentages = [row['train_pct'], row['valid_pct'], row['test_pct']]
        row['std_dev'] = np.std(percentages)
    
    return df


def visualize_distributions(results, df):
    """
    Create comprehensive visualizations of the distribution analysis
    
    Parameters
    ----------
    results : dict
        Results from analyze_tkms_pmi_distribution
    df : pd.DataFrame
        Distribution statistics DataFrame
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Grouped bar chart for process distributions
    ax1 = axes[0, 0]
    processes = df[df['Process'] != 'No Process']['Process'].tolist()
    x = np.arange(len(processes))
    width = 0.25
    
    for i, mode in enumerate(['train', 'valid', 'test']):
        values = df[df['Process'] != 'No Process'][f'{mode}_pct'].tolist()
        ax1.bar(x + i * width, values, width, label=mode.capitalize(), color=colors[i])
    
    ax1.set_xlabel('Manufacturing Process')
    ax1.set_ylabel('Percentage of Samples (%)')
    ax1.set_title('Process Distribution Across Splits')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(processes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pie charts for samples with/without processes
    for i, mode in enumerate(['train', 'valid', 'test']):
        ax = axes[0, i+1] if i < 2 else axes[1, 0]
        if results[mode]:
            sizes = [results[mode]['samples_with_processes'], 
                    results[mode]['samples_without_processes']]
            labels = ['With Processes', 'No Processes']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{mode.capitalize()} Split - Process Coverage')
    
    # 3. Heatmap of percentage distribution
    ax3 = axes[1, 1]
    heatmap_data = []
    for mode in ['train', 'valid', 'test']:
        row = []
        for process in ['Bohren', 'Drehen', 'Fräsen', 'No Process']:
            pct = df[df['Process'] == process][f'{mode}_pct'].values[0]
            row.append(pct)
        heatmap_data.append(row)
    
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.1f',
                xticklabels=['Bohren', 'Drehen', 'Fräsen', 'No Process'],
                yticklabels=['Train', 'Valid', 'Test'],
                cmap='YlOrRd',
                ax=ax3,
                cbar_kws={'label': 'Percentage (%)'})
    ax3.set_title('Distribution Heatmap')
    
    # 4. Standard deviation visualization
    ax4 = axes[1, 2]
    processes_all = df['Process'].tolist()
    std_devs = [np.std([df[df['Process'] == p]['train_pct'].values[0],
                        df[df['Process'] == p]['valid_pct'].values[0],
                        df[df['Process'] == p]['test_pct'].values[0]]) 
                for p in processes_all]
    
    bars = ax4.bar(processes_all, std_devs, color=['green' if s < 2 else 'orange' if s < 5 else 'red' for s in std_devs])
    ax4.set_xlabel('Process')
    ax4.set_ylabel('Standard Deviation (%)')
    ax4.set_title('Distribution Uniformity (Lower is Better)')
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Good (<2%)')
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<5%)')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, std_devs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.suptitle('TKMS PMI Dataset - Manufacturing Step Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_detailed_statistics(results, df):
    """
    Print detailed statistics and analysis summary
    
    Parameters
    ----------
    results : dict
        Results from analyze_tkms_pmi_distribution
    df : pd.DataFrame
        Distribution statistics DataFrame
    """
    
    print("\n" + "="*80)
    print("DETAILED DISTRIBUTION STATISTICS")
    print("="*80)
    
    # Sample counts
    print("\n📊 SAMPLE COUNTS:")
    print("-" * 40)
    total_samples = 0
    for mode in ['train', 'valid', 'test']:
        if results[mode]:
            n = results[mode]['total_samples']
            total_samples += n
            print(f"{mode.capitalize():8} {n:5} samples")
    print(f"{'Total':8} {total_samples:5} samples")
    
    # Distribution table
    print("\n📈 PROCESS DISTRIBUTION TABLE:")
    print("-" * 40)
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Uniformity assessment
    print("\n🎯 DISTRIBUTION UNIFORMITY ASSESSMENT:")
    print("-" * 40)
    
    for _, row in df.iterrows():
        process = row['Process']
        percentages = [row['train_pct'], row['valid_pct'], row['test_pct']]
        std_dev = np.std(percentages)
        
        if std_dev < 2:
            assessment = "✅ Excellent"
        elif std_dev < 5:
            assessment = "⚠️  Good"
        else:
            assessment = "❌ Poor"
        
        print(f"{process:12} Std Dev: {std_dev:5.2f}%  {assessment}")
    
    # Overall assessment
    print("\n📋 OVERALL ASSESSMENT:")
    print("-" * 40)
    
    all_std_devs = []
    for _, row in df.iterrows():
        percentages = [row['train_pct'], row['valid_pct'], row['test_pct']]
        all_std_devs.append(np.std(percentages))
    
    mean_std_dev = np.mean(all_std_devs)
    
    if mean_std_dev < 2:
        print(f"Mean Std Dev: {mean_std_dev:.2f}% - Distribution is EXCELLENT ✅")
        print("The manufacturing steps are very evenly distributed across splits.")
    elif mean_std_dev < 5:
        print(f"Mean Std Dev: {mean_std_dev:.2f}% - Distribution is GOOD ⚠️")
        print("The manufacturing steps are reasonably well distributed across splits.")
    else:
        print(f"Mean Std Dev: {mean_std_dev:.2f}% - Distribution is POOR ❌")
        print("Consider re-splitting the dataset for better balance.")


def main():
    """
    Main function to run the complete distribution analysis
    """
    
    # Analyze distributions
    results, raw_data = analyze_tkms_pmi_distribution()
    
    # Calculate statistics
    df = calculate_step_distributions(results)
    
    # Print detailed statistics
    print_detailed_statistics(results, df)
    
    # Create visualizations
    visualize_distributions(results, df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results, df


if __name__ == "__main__":
    results, df = main()