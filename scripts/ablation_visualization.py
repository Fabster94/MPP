
#!/usr/bin/env python3
"""
Visualization script for ablation study results
Creates separate vector graphics (SVG) for F1 scores and feature importance
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cycler import cycler

# Colorblind-friendly color definitions
COLORS = {
    'primary': '#0173B2',    # Blue
    'secondary': '#DE8F05',  # Orange
    'tertiary': '#029E73',   # Teal/Green
    'quaternary': '#CC78BC', # Purple
    'accent': '#CA9161'      # Brown
}

# Set default color cycle for matplotlib
plt.rc('axes', prop_cycle=cycler('color', [
    COLORS['primary'],
    COLORS['secondary'],
    COLORS['tertiary'],
    COLORS['quaternary'],
    COLORS['accent']
]))

# Set other default styles
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['grid.alpha'] = 0.3

# Configuration
OUTPUT_DIR = Path("/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/ablation_results_cv/20251011_203751")  # Adjust timestamp!
sns.set_style("whitegrid")

def load_results():
    """Load saved ablation results"""
    with open(OUTPUT_DIR / 'ablation_raw_results.json', 'r') as f:
        data = json.load(f)
    
    df = pd.read_csv(OUTPUT_DIR / 'ablation_analysis_paired.csv')
    return data, df

def create_f1_scores_plot(data, df):
    """Create separate F1 scores comparison plot"""
    
    # Sort by performance drop (importance)
    df_sorted = df.sort_values('Performance_Drop', ascending=False)
    
    # Calculate baseline scores from raw data
    geom_scores = [r['f1_macro'] for r in data['results']['GEOMETRY_ONLY']]
    full_scores = [r['f1_macro'] for r in data['results']['FULL']]
    geom_mean = np.mean(geom_scores)
    full_mean = np.mean(full_scores)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    groups = ['Baseline'] + df_sorted['Group'].tolist()
    geometry_scores = [geom_mean] + [geom_mean] * len(df_sorted)
    full_scores_plot = [full_mean] + [full_mean] * len(df_sorted)
    without_scores = [full_mean] + df_sorted['F1_WITHOUT_mean'].tolist()
    
    x = np.arange(len(groups))
    width = 0.25
    
    # Create bars with consistent colors
    bars1 = ax.bar(x - width, geometry_scores[:1], width, label='Geometry Only', color=COLORS['tertiary'])
    bars2 = ax.bar(x, full_scores_plot[:1], width, label='Full PMI', color=COLORS['primary'])
    bars3 = ax.bar(x[1:] + width, without_scores[1:], width, label='WITHOUT_X', color=COLORS['secondary'])
    
    # Add horizontal line for baselines
    ax.axhline(y=geom_mean, color=COLORS['tertiary'], linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=full_mean, color=COLORS['primary'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    ax.text(0 - width, geom_mean + 0.005, f'{geom_mean:.3f}', ha='center', fontsize=9)
    ax.text(0, full_mean + 0.005, f'{full_mean:.3f}', ha='center', fontsize=9)
    for i, score in enumerate(without_scores[1:], 1):
        ax.text(i + width, score + 0.005, f'{score:.3f}', ha='center', fontsize=9)
    
    ax.set_ylabel('F1-Macro Score')
    ax.set_xlabel('Removed PMI Group')
    ax.set_title('F1-Macro Scores: Impact of Removing Each PMI Group', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0.7, 0.9)
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save as vector graphics (SVG)
    output_file = OUTPUT_DIR / 'ablation_f1_scores.svg'
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Saved F1 scores plot to {output_file}")
    
    # Also save as PNG for quick viewing
    output_file_png = OUTPUT_DIR / 'ablation_f1_scores.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_feature_importance_plot(df):
    """Create separate feature importance plot with values on bars"""
    
    # Sort by performance drop (importance)
    df_sorted = df.sort_values('Performance_Drop', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create bar chart
    x_pos = np.arange(len(df_sorted))
    bars = ax.bar(x_pos, df_sorted['Performance_Drop'], alpha=0.7)
    
    # Color bars by significance (using colorblind-friendly colors)
    colors = [COLORS['secondary'] if sig else '#CCCCCC' for sig in df_sorted['Significant']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        value = row['Performance_Drop']
        # Position label above or below bar depending on sign
        if value >= 0:
            va = 'bottom'
            y_offset = 0.001
        else:
            va = 'top'
            y_offset = -0.001
        
        ax.text(i, value + y_offset, f'{value:.4f}', 
                ha='center', va=va, fontsize=10, fontweight='bold')
    
    # Add error bars (confidence intervals)
    ax.errorbar(x_pos, df_sorted['Performance_Drop'], 
                yerr=[df_sorted['Performance_Drop'] - df_sorted['Drop_CI_Low'],
                      df_sorted['Drop_CI_High'] - df_sorted['Performance_Drop']],
                fmt='none', color='black', capsize=4, linewidth=1)
    
    # Add significance stars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['Wilcoxon_p'] < 0.001:
            ax.text(i, row['Drop_CI_High'] + 0.003, '***', ha='center', fontsize=12, fontweight='bold')
        elif row['Wilcoxon_p'] < 0.01:
            ax.text(i, row['Drop_CI_High'] + 0.003, '**', ha='center', fontsize=12)
        elif row['Wilcoxon_p'] < 0.05:
            ax.text(i, row['Drop_CI_High'] + 0.003, '*', ha='center', fontsize=12)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['Group'], rotation=45, ha='right')
    ax.set_ylabel('Performance Drop (Full - WITHOUT_X)')
    ax.set_xlabel('Removed PMI Feature Group')
    ax.set_title('Feature Importance: Drop When Removed\n(Orange = Significant, p<0.05)', 
                  fontweight='bold')
    ax.grid(True, axis='y')
    
    # Add text box with key findings
    textstr = f'Key Finding:\nOnly "dimensions" is significant\n(p < 0.001, drop = {df_sorted.iloc[0]["Performance_Drop"]:.3f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save as vector graphics (SVG)
    output_file = OUTPUT_DIR / 'ablation_feature_importance.svg'
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Saved feature importance plot to {output_file}")
    
    # Also save as PNG for quick viewing
    output_file_png = OUTPUT_DIR / 'ablation_feature_importance.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_summary_table(df):
    """Create and save a summary table"""
    
    df_sorted = df.sort_values('Performance_Drop', ascending=False)
    
    # Create summary DataFrame
    summary = df_sorted[['Group', 'F1_WITHOUT_mean', 'Performance_Drop', 'Drop_CI_Low', 'Drop_CI_High', 'Wilcoxon_p', 'Significant']].copy()
    summary.columns = ['Feature Group', 'F1 Score', 'Drop', 'CI Low', 'CI High', 'p-value', 'Significant']
    
    # Format numbers
    summary['F1 Score'] = summary['F1 Score'].map('{:.4f}'.format)
    summary['Drop'] = summary['Drop'].map('{:.4f}'.format)
    summary['CI Low'] = summary['CI Low'].map('{:.4f}'.format)
    summary['CI High'] = summary['CI High'].map('{:.4f}'.format)
    summary['p-value'] = summary['p-value'].map(lambda x: f'{x:.4f}' if x >= 0.001 else '<0.001')
    summary['Significant'] = summary['Significant'].map(lambda x: '✓' if x else '')
    
    # Print to console
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70)
    
    # Save as CSV
    summary.to_csv(OUTPUT_DIR / 'ablation_summary.csv', index=False)
    print(f"\nSummary saved to {OUTPUT_DIR / 'ablation_summary.csv'}")
    
    # Create LaTeX table for paper
    latex_summary = summary.copy()
    latex_summary['95% CI'] = '[' + summary['CI Low'] + ', ' + summary['CI High'] + ']'
    latex_summary = latex_summary[['Feature Group', 'F1 Score', 'Drop', '95% CI', 'p-value', 'Significant']]
    
    latex_output = latex_summary.to_latex(index=False, 
                                          escape=False,
                                          column_format='lrrrrr',
                                          caption='PMI Feature Ablation Results',
                                          label='tab:ablation_results')
    
    with open(OUTPUT_DIR / 'ablation_summary.tex', 'w') as f:
        f.write(latex_output)
    print(f"LaTeX table saved to {OUTPUT_DIR / 'ablation_summary.tex'}")

def main():
    """Main execution"""
    print("Loading ablation study results...")
    data, df = load_results()
    
    print(f"Found results for {len(df)} feature groups")
    print(f"Baseline Geometry: {np.mean([r['f1_macro'] for r in data['results']['GEOMETRY_ONLY']]):.4f}")
    print(f"Baseline Full PMI: {np.mean([r['f1_macro'] for r in data['results']['FULL']]):.4f}")
    
    # Create separate plots
    print("\nCreating visualizations...")
    create_f1_scores_plot(data, df)
    create_feature_importance_plot(df)
    create_summary_table(df)
    
    print("\nAll visualizations created successfully!")
    print("\nFiles created:")
    print("  - ablation_f1_scores.svg (vector graphics)")
    print("  - ablation_f1_scores.png (raster preview)")
    print("  - ablation_feature_importance.svg (vector graphics)")
    print("  - ablation_feature_importance.png (raster preview)")
    print("  - ablation_summary.csv (data table)")
    print("  - ablation_summary.tex (LaTeX table)")

if __name__ == "__main__":
    main()