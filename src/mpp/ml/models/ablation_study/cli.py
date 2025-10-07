#!/usr/bin/env python3
"""
Leave-One-Out Ablation Study CLI for PMI Feature Groups

This script runs ablation experiments by removing one PMI group at a time
to assess each group's contribution to model performance.
"""

import argparse
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List
import pandas as pd


# PMI Feature Groups
PMI_GROUPS = ['dimensions', 'geometric_tolerances', 'dimensional_tolerances', 
              'surface_finish', 'fits', 'datums']


def run_experiment(tag: str, excluded_group: str = None, output_dir: str = None, 
                   dry_run: bool = False, ablation_mode: str = "none", 
                   mask_fill: str = "mean"):
    """Run a single ablation experiment"""
    
    if dry_run:
        print(f"[DRY RUN] Would execute experiment: '{tag}'")
        if excluded_group:
            print(f"           Excluding group: {excluded_group}")
        else:
            print(f"           Including: All PMI groups (baseline)")
        print(f"           Output directory: {output_dir}")
        return True, 0
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    pmi_ablation_script = script_dir / "pmi_ablation_study.py"
    
    # Check if script exists
    if not pmi_ablation_script.exists():
        print(f"Error: Cannot find {pmi_ablation_script}")
        print(f"Looking in: {script_dir}")
        return False, 0
    
    # Build command
    cmd = [
        sys.executable,
        str(pmi_ablation_script),
        "--tag", tag,
    ]
    
    if output_dir:
        # Pass the full path for this specific experiment
        experiment_output_dir = f"{output_dir}/{tag}"
        cmd.extend(["--output_dir", experiment_output_dir])
    
    if excluded_group:
        # For leave-one-out, we use "minus" ablation mode
        cmd.extend([
            "--ablation", "minus",
            "--group", excluded_group,
            "--mask_fill", "mean"
        ])
    else:
        # Baseline with all groups
        cmd.extend(["--ablation", "none"])
    
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}")
    if excluded_group:
        print(f"Excluding: {excluded_group}")
    else:
        print("Including: All PMI groups (baseline)")
    print(f"{'='*60}")
    
    # Run
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    success = result.returncode == 0
    status = "✓ Success" if success else "✗ Failed"
    print(f"{status} - Completed in {elapsed/60:.1f} minutes")
    
    return success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Run Leave-One-Out Ablation Study for PMI Feature Groups'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: experiments/ablation/TIMESTAMP)'
    )
    
    parser.add_argument(
        '--groups',
        type=str,
        nargs='+',
        choices=PMI_GROUPS,
        default=PMI_GROUPS,
        help='PMI groups to test (default: all groups)'
    )
    
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip the baseline experiment (use if already run)'
    )
    
    parser.add_argument(
        '--ablation-type',
        type=str,
        choices=['minus', 'only', 'both'],
        default='minus',
        help='Type of ablation to perform (default: minus)'
    )
    
    parser.add_argument(
        '--include-no-pmi',
        action='store_true',
        help='Include a NO_PMI baseline experiment (all PMI features zeroed)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without executing'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"experiments/ablation/leave_one_out_{timestamp}"
    
    # Create directory (only if not dry-run)
    if not args.dry_run:
        Path(output_base).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Leave-One-Out PMI Ablation Study")
    print("="*80)
    print(f"Started: {datetime.now()}")
    print(f"Output: {output_base}")
    print(f"Groups to test: {', '.join(args.groups)}")
    print(f"Total experiments: {1 + len(args.groups) if not args.skip_baseline else len(args.groups)}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]")
    
    # Track results
    results = []
    experiment_plan = []
    
    # 0. No PMI baseline (if requested)
    if args.include_no_pmi:
        experiment_plan.append(("NO_PMI", "ALL", "minus", "zero"))
    
    # 1. Baseline experiment (all groups)
    if not args.skip_baseline:
        experiment_plan.append(("FULL_baseline", None, "none", "mean"))
    
    # 2. Leave-one-out experiments (MINUS)
    if args.ablation_type in ['minus', 'both']:
        for group in args.groups:
            tag = f"WITHOUT_{group}"
            experiment_plan.append((tag, group, "minus", "mean"))
    
    # 3. Keep-only-one experiments (ONLY)
    if args.ablation_type in ['only', 'both']:
        for group in args.groups:
            tag = f"ONLY_{group}"
            experiment_plan.append((tag, group, "only", "mean"))
    
    # Show plan
    print(f"\nExperiment Plan ({len(experiment_plan)} experiments):")
    for i, exp in enumerate(experiment_plan, 1):
        if len(exp) == 4:  # New format with ablation mode
            tag, group, mode, fill = exp
            if group is None:
                desc = "all PMI groups"
            elif group == "ALL":
                desc = "NO PMI (all features zeroed)"
            elif mode == "minus":
                desc = f"excluding {group}"
            elif mode == "only":
                desc = f"only {group}"
            else:
                desc = f"{mode} {group}"
            print(f"  {i}. {tag:<30} ({desc}, fill={fill})")
        else:  # Legacy format
            tag, excluded = exp
            exc_str = f"excluding {excluded}" if excluded else "all groups"
            print(f"  {i}. {tag:<30} ({exc_str})")
    
    if args.dry_run:
        # Execute dry-run for each experiment
        print("\n" + "="*80)
        print("DRY RUN EXECUTION")
        print("="*80)
        
        for i, exp in enumerate(experiment_plan, 1):
            print(f"\n[{i}/{len(experiment_plan)}]")
            if len(exp) == 4:
                tag, group, mode, fill = exp
                run_experiment(tag, group, output_base, dry_run=True, 
                             ablation_mode=mode, mask_fill=fill)
            else:
                tag, group = exp
                run_experiment(tag, group, output_base, dry_run=True)
        
        print("\n" + "="*80)
        print("DRY RUN COMPLETED - No actual experiments were executed")
        print("="*80)
        return
    
    # Execute experiments
    print("\n" + "="*80)
    print("EXECUTING EXPERIMENTS")
    print("="*80)
    
    for i, exp in enumerate(experiment_plan, 1):
        if len(exp) == 4:
            tag, group, mode, fill = exp
            print(f"\n[{i}/{len(experiment_plan)}] {tag}")
            
            success, elapsed = run_experiment(tag, group, output_base, 
                                            ablation_mode=mode, mask_fill=fill)
        else:
            tag, excluded_group = exp
            print(f"\n[{i}/{len(experiment_plan)}] {tag}")
            
            success, elapsed = run_experiment(tag, excluded_group, output_base)
        
        results.append({
            'experiment': tag,
            'excluded_group': group if len(exp) == 4 else (excluded_group or 'none'),
            'success': success,
            'duration_minutes': elapsed / 60
        })
    
    # Save experiment summary
    summary_df = pd.DataFrame(results)
    summary_file = Path(output_base) / "experiment_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = summary_df[summary_df['success']]['experiment'].tolist()
    failed = summary_df[~summary_df['success']]['experiment'].tolist()
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for exp in successful:
        print(f"  ✓ {exp}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for exp in failed:
            print(f"  ✗ {exp}")
    
    print(f"\nTotal runtime: {summary_df['duration_minutes'].sum():.1f} minutes")
    print(f"Summary saved to: {summary_file}")
    
    # Generate analysis (only if we have results)
    if len(successful) > 1:
        print("\n" + "="*80)
        print("GENERATING ANALYSIS")
        print("="*80)
        
        # Get analyze script path
        script_dir = Path(__file__).parent
        analysis_script = script_dir / "analyze_ablation_results.py"
        
        if not analysis_script.exists():
            print(f"Warning: Cannot find {analysis_script}")
            print("Skipping analysis generation")
        else:
            analysis_cmd = [
                sys.executable,
                str(analysis_script),
                "--experiment_dir", output_base,
                "--output_file", f"{output_base}/leave_one_out_analysis.html"
            ]
            
            print("Running analysis script...")
            result = subprocess.run(analysis_cmd)
            
            if result.returncode == 0:
                print("✓ Analysis complete")
                
                # Quick preview of results
                print("\nQuick Preview:")
                print("-" * 50)
                
                # Try to load and show key results
                try:
                    # Import locally to avoid issues if script is moved
                    sys.path.insert(0, str(script_dir))
                    from analyze_ablation_results import load_experiment_results, calculate_deltas
                    
                    exp_results = load_experiment_results(Path(output_base))
                    delta_df = calculate_deltas(exp_results)
                    
                    if not delta_df.empty:
                        minus_df = delta_df[delta_df['ablation_type'] == 'MINUS'].sort_values('delta_f1_macro_mean')
                        
                        print("\nImpact of removing each group (ΔF1-macro):")
                        for _, row in minus_df.iterrows():
                            sig = "*" if row.get('significant', False) else " "
                            print(f"  {row['group']:25} {row['delta_f1_macro_mean']:+.4f} "
                                  f"± {row['delta_f1_macro_std']:.4f} {sig}")
                        
                        print("\n* = statistically significant (p < 0.05)")
                        
                        # Find most important group
                        if len(minus_df) > 0:
                            most_important = minus_df.iloc[0]
                            print(f"\nMost important group: {most_important['group']} "
                                  f"(ΔF1 = {most_important['delta_f1_macro_mean']:.4f})")
                            
                except Exception as e:
                    print(f"Could not generate preview: {e}")
            else:
                print("✗ Analysis failed")
    
    print(f"\nCompleted at: {datetime.now()}")
    print(f"All results saved to: {output_base}/")


if __name__ == "__main__":
    main()