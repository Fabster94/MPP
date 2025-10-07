#!/usr/bin/env python3
"""
Analyze PMI Ablation Study Results

This script analyzes the results from the ablation experiments and generates
visualizations and statistical tests.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon


def load_experiment_results(experiment_dir: Path) -> Dict:
    """Load all results from an experiment directory"""
    results = {}
    
    # Check if we have a nested timestamp directory structure
    timestamp_dirs = [d for d in experiment_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('202')]
    
    if timestamp_dirs:
        # Use the most recent timestamp directory
        timestamp_dir = sorted(timestamp_dirs)[-1]
        print(f"Found timestamp directory: {timestamp_dir.name}")
        search_dir = timestamp_dir
    else:
        # No timestamp directory, search directly
        search_dir = experiment_dir
    
    # Find all variant directories
    for variant_dir in search_dir.iterdir():
        if variant_dir.is_dir():
            final_results_file = variant_dir / 'final_results.json'
            if final_results_file.exists():
                with open(final_results_file, 'r') as f:
                    results[variant_dir.name] = json.load(f)
                print(f"Loaded results for: {variant_dir.name}")
            else:
                print(f"Warning: No final_results.json in {variant_dir}")
    
    # If still no results, try one more level deep
    if not results:
        print("No results found at expected level, checking one level deeper...")
        for parent_dir in search_dir.iterdir():
            if parent_dir.is_dir():
                for variant_dir in parent_dir.iterdir():
                    if variant_dir.is_dir():
                        final_results_file = variant_dir / 'final_results.json'
                        if final_results_file.exists():
                            with open(final_results_file, 'r') as f:
                                results[variant_dir.name] = json.load(f)
                            print(f"Loaded results for: {variant_dir.name}")
                    
    return results


def calculate_deltas(results: Dict) -> pd.DataFrame:
    """Calculate performance deltas relative to FULL baseline"""
    
    # Get baseline results
    if 'FULL_baseline' not in results:
        raise ValueError("FULL_baseline results not found!")
    
    baseline = results['FULL_baseline']
    
    # Debug print
    print(f"DEBUG: Baseline keys: {list(baseline.keys())}")
    print(f"DEBUG: Processing {len(results)-1} ablation experiments")
    
    # Calculate deltas for each ablation
    delta_data = []
    
    for variant_name, variant_results in results.items():
        if variant_name == 'FULL_baseline':
            continue
        
        print(f"DEBUG: Processing {variant_name}")
            
        # Extract group name
        if variant_name.startswith('MINUS_'):
            group_name = variant_name.replace('MINUS_', '')
            ablation_type = 'MINUS'
        elif variant_name.startswith('WITHOUT_'):
            group_name = variant_name.replace('WITHOUT_', '')
            ablation_type = 'MINUS'  # WITHOUT is same as MINUS
        elif variant_name.startswith('ONLY_'):
            group_name = variant_name.replace('ONLY_', '')
            ablation_type = 'ONLY'
        elif variant_name == 'NO_PMI':
            # Special baseline without any PMI
            continue  # Skip NO_PMI in delta calculation
        else:
            print(f"DEBUG: Skipping {variant_name} - unknown prefix")
            continue
        
        # Calculate deltas for each metric
        try:
            base_f1_macro = baseline['aggregated']['f1_macro']['values']
            variant_f1_macro = variant_results['aggregated']['f1_macro']['values']
        except KeyError as e:
            print(f"DEBUG: KeyError for {variant_name}: {e}")
            print(f"DEBUG: variant_results keys: {list(variant_results.keys())}")
            if 'aggregated' in variant_results:
                print(f"DEBUG: aggregated keys: {list(variant_results['aggregated'].keys())}")
            continue
        
        deltas = np.array(variant_f1_macro) - np.array(base_f1_macro)
        
        # Wilcoxon test
        if len(deltas) >= 5:
            stat, p_value = wilcoxon(deltas)
        else:
            p_value = np.nan
        
        # Per-class deltas
        class_deltas = {}
        for class_name in ['bohren', 'drehen', 'fraesen']:
            base_values = baseline['aggregated'][f'f1_{class_name}']['values']
            variant_values = variant_results['aggregated'][f'f1_{class_name}']['values']
            class_deltas[class_name] = np.mean(np.array(variant_values) - np.array(base_values))
        
        delta_data.append({
            'group': group_name,
            'ablation_type': ablation_type,
            'delta_f1_macro_mean': np.mean(deltas),
            'delta_f1_macro_std': np.std(deltas),
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False,
            **{f'delta_{k}': v for k, v in class_deltas.items()}
        })
    
    print(f"DEBUG: Created {len(delta_data)} delta entries")
    if len(delta_data) > 0:
        print(f"DEBUG: First entry: {delta_data[0]}")
    
    df = pd.DataFrame(delta_data)
    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    
    return df


def plot_ablation_results(df: pd.DataFrame, output_path: Path = None):
    """Create bar plot of ablation results"""
    
    if df.empty:
        print("WARNING: DataFrame is empty, cannot create ablation plot")
        return
    
    print(f"DEBUG: plot_ablation_results - df shape: {df.shape}")
    print(f"DEBUG: plot_ablation_results - columns: {list(df.columns)}")
    
    # Filter for MINUS ablations
    minus_df = df[df['ablation_type'] == 'MINUS'].sort_values('delta_f1_macro_mean')
    
    if minus_df.empty:
        print("No MINUS ablation results found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.barh(minus_df['group'], minus_df['delta_f1_macro_mean'])
    
    # Color bars by significance
    colors = ['red' if sig else 'lightcoral' for sig in minus_df['significant']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add error bars
    ax.errorbar(minus_df['delta_f1_macro_mean'], range(len(minus_df)), 
                xerr=minus_df['delta_f1_macro_std'], fmt='none', color='black', alpha=0.5)
    
    # Add significance stars
    for i, (idx, row) in enumerate(minus_df.iterrows()):
        if row['significant']:
            ax.text(row['delta_f1_macro_mean'] - 0.001, i, '*', 
                   fontsize=20, ha='right', va='center')
    
    # Formatting
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('ΔF1-macro (WITHOUT - FULL)', fontsize=12)
    ax.set_ylabel('PMI Feature Group', fontsize=12)
    ax.set_title('PMI Feature Group Importance via Ablation\n'
                 '(Negative values indicate group contributes to performance)', fontsize=14)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'ablation_bars.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_heatmap(df: pd.DataFrame, output_path: Path = None):
    """Create heatmap showing per-class effects"""
    
    # Filter for MINUS ablations
    minus_df = df[df['ablation_type'] == 'MINUS']
    
    if minus_df.empty:
        print("No MINUS ablation results found")
        return
    
    # Create matrix for heatmap
    classes = ['bohren', 'drehen', 'fraesen']
    groups = minus_df['group'].tolist()
    
    heatmap_data = []
    for group in groups:
        row = minus_df[minus_df['group'] == group].iloc[0]
        heatmap_data.append([row[f'delta_{c}'] for c in classes])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(heatmap_data,
                xticklabels=['Bohren', 'Drehen', 'Fräsen'],
                yticklabels=groups,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',  # Red = negative (important), Green = positive
                center=0,
                cbar_kws={'label': 'ΔF1 (WITHOUT - FULL)'})
    
    plt.title('Per-Class Impact of PMI Feature Groups', fontsize=14)
    plt.xlabel('Manufacturing Process', fontsize=12)
    plt.ylabel('PMI Feature Group', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'class_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_only_ablation_results(df: pd.DataFrame, output_path: Path = None):
    """Create bar plot for ONLY ablation results"""
    
    # Filter for ONLY ablations
    only_df = df[df['ablation_type'] == 'ONLY'].sort_values('delta_f1_macro_mean', ascending=False)
    
    if only_df.empty:
        print("No ONLY ablation results found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.barh(only_df['group'], only_df['delta_f1_macro_mean'])
    
    # Color bars - green for positive (good when used alone), red for negative
    colors = ['green' if val > 0 else 'lightcoral' for val in only_df['delta_f1_macro_mean']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add error bars
    ax.errorbar(only_df['delta_f1_macro_mean'], range(len(only_df)), 
                xerr=only_df['delta_f1_macro_std'], fmt='none', color='black', alpha=0.5)
    
    # Formatting
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('ΔF1-macro (ONLY group - FULL baseline)', fontsize=12)
    ax.set_ylabel('PMI Feature Group', fontsize=12)
    ax.set_title('Performance Using Only Single PMI Feature Groups\n'
                 '(Negative values show performance drop vs. using all features)', fontsize=14)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'only_ablation_bars.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of results"""
    
    # Focus on MINUS ablations
    minus_df = df[df['ablation_type'] == 'MINUS'].copy()
    
    if minus_df.empty:
        print("No MINUS ablation results found")
        return pd.DataFrame()
    
    # Create summary
    summary = minus_df[['group', 'delta_f1_macro_mean', 'delta_f1_macro_std', 
                       'p_value', 'significant']].copy()
    
    # Format columns
    summary['Delta F1-macro'] = summary.apply(
        lambda row: f"{row['delta_f1_macro_mean']:.3f} ± {row['delta_f1_macro_std']:.3f}", 
        axis=1
    )
    summary['p-value'] = summary['p_value'].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A")
    summary['Significant'] = summary['significant'].apply(lambda x: "Yes*" if x else "No")
    
    # Select and rename columns
    summary = summary[['group', 'Delta F1-macro', 'p-value', 'Significant']]
    summary.columns = ['PMI Group', 'ΔF1-macro (mean ± std)', 'Wilcoxon p-value', 'Significant']
    
    # Sort by delta
    summary = summary.sort_values('ΔF1-macro (mean ± std)')
    
    return summary


def generate_report(experiment_dir: Path, output_file: Path = None):
    """Generate complete analysis report"""
    
    # Load results
    results = load_experiment_results(experiment_dir)
    print(f"\nLoaded results for {len(results)} experiments")
    
    if len(results) == 0:
        print("ERROR: No experiment results found!")
        print(f"Searched in: {experiment_dir}")
        return None, None
    
    # Check if NO_PMI exists for special comparison
    if 'NO_PMI' in results and 'FULL_baseline' in results:
        no_pmi_metrics = results['NO_PMI']['aggregated']['f1_macro']
        full_metrics = results['FULL_baseline']['aggregated']['f1_macro']
        improvement = full_metrics['mean'] - no_pmi_metrics['mean']
        print("\n" + "="*80)
        print("PMI OVERALL IMPACT:")
        print("="*80)
        print(f"Without PMI (NO_PMI):     F1-macro = {no_pmi_metrics['mean']:.4f} ± {no_pmi_metrics['std']:.4f}")
        print(f"With all PMI (FULL):      F1-macro = {full_metrics['mean']:.4f} ± {full_metrics['std']:.4f}")
        print(f"Improvement from PMI:     ΔF1 = +{improvement:.4f}")
        print("="*80)
    
    # Calculate deltas
    delta_df = calculate_deltas(results)
    
    # Create output directory if needed
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    plot_ablation_results(delta_df, experiment_dir)
    plot_class_heatmap(delta_df, experiment_dir)
    
    # Create ONLY ablation plot if available
    if not delta_df.empty and 'ONLY' in delta_df['ablation_type'].values:
        plot_only_ablation_results(delta_df, experiment_dir)
    
    # Create summary table
    summary_table = create_summary_table(delta_df)
    
    print("\n" + "="*80)
    print("PMI ABLATION STUDY RESULTS")
    print("="*80)
    
    if not summary_table.empty:
        print("\nSummary Table:")
        print(summary_table.to_string(index=False))
        print("\n* Significant at p < 0.05 (Wilcoxon signed-rank test)")
    
    # Save detailed results
    if not delta_df.empty:
        delta_df.to_csv(experiment_dir / 'ablation_analysis.csv', index=False)
    
    # Generate HTML report if requested
    if output_file and output_file.suffix == '.html' and not delta_df.empty:
        generate_html_report(results, delta_df, summary_table, output_file)
    
    return delta_df, summary_table


def generate_html_report(results: Dict, delta_df: pd.DataFrame, 
                        summary_table: pd.DataFrame, output_file: Path):
    """Generate an HTML report with all results"""
    
    html_content = f"""
    <html>
    <head>
        <title>PMI Ablation Study Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .significant {{ background-color: #ffe6e6; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>PMI Ablation Study Results</h1>
        
        <h2>Summary</h2>
        <p>This report summarizes the results of the PMI feature group ablation study.</p>
        
        <h2>Main Results Table</h2>
        {summary_table.to_html(index=False, classes='summary-table') if not summary_table.empty else "<p>No results available</p>"}
        
        <h2>Visualizations</h2>
        <h3>Feature Group Importance</h3>
        <img src="ablation_bars.png" alt="Feature Group Importance">
        
        <h3>Per-Class Impact</h3>
        <img src="class_heatmap.png" alt="Per-Class Impact Heatmap">
        
        <h2>Interpretation</h2>
        <ul>
            <li>Negative ΔF1 values indicate that removing the group hurts performance 
                (i.e., the group contributes positively)</li>
            <li>More negative values = more important feature group</li>
            <li>Statistical significance tested via Wilcoxon signed-rank test across 5 folds</li>
        </ul>
        
        <hr>
        <p><small>Generated: {pd.Timestamp.now()}</small></p>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_file', type=str,
                       help='Output file for report (optional, .html for HTML report)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    output_file = Path(args.output_file) if args.output_file else None
    
    if not experiment_dir.exists():
        print(f"ERROR: Experiment directory does not exist: {experiment_dir}")
        return
    
    # Generate report
    delta_df, summary_table = generate_report(experiment_dir, output_file)
    
    if delta_df is None:
        return
    
    # Print most important findings
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Find most important groups
    minus_df = delta_df[delta_df['ablation_type'] == 'MINUS'].sort_values('delta_f1_macro_mean')
    
    if len(minus_df) > 0:
        most_important = minus_df.iloc[0]
        print(f"\nMost important PMI group: {most_important['group']}")
        print(f"  ΔF1-macro = {most_important['delta_f1_macro_mean']:.3f} "
              f"± {most_important['delta_f1_macro_std']:.3f}")
        print(f"  p-value = {most_important['p_value']:.3f}")
        
        # Per-class analysis
        print(f"\n  Per-class impact:")
        for class_name in ['bohren', 'drehen', 'fraesen']:
            delta = most_important[f'delta_{class_name}']
            print(f"    {class_name.capitalize()}: ΔF1 = {delta:.3f}")


if __name__ == "__main__":
    main()