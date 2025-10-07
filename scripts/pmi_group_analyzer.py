import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

class PMIGroupAnalyzer:
    """Analyze PMI feature groups impact on OOF predictions (group-wise)."""

    def __init__(self, oof_path: str, pmi_path: str, *,
                 classes: List[str] = None,
                 model_type: str = "pmi",
                 id_col_oof: str = "id",
                 id_col_pmi: str = "id",
                 use_oof_folds: bool = False,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.oof_df = pd.read_csv(oof_path)
        self.pmi_df = pd.read_csv(pmi_path)
        self.model_type = model_type
        self.id_col_oof = id_col_oof
        self.id_col_pmi = id_col_pmi
        self.use_oof_folds = use_oof_folds
        self.n_splits = n_splits
        self.random_state = random_state

        self.classes = classes or ['Bohren', 'Drehen', 'Fräsen']
        self.results: Dict[str, Dict] = {}
        self.thresholds: Dict[str, float] = {}  # Fallback thresholds

        # PMI feature groups
        self.pmi_groups = {
            'dimensions': [
                'total_dimension_count', 'linear_dimension_count', 
                'diameter_dimension_count', 'radius_dimension_count', 
                'angular_dimension_count'
            ],
            'geometric_tolerances': [
                'has_angularity', 'has_circular_runout', 'has_concentricity',
                'has_cylindricity', 'has_flatness', 'has_parallelism',
                'has_perpendicularity', 'has_position', 'has_profile_of_line',
                'has_profile_of_surface', 'has_roundness', 'has_straightness',
                'has_symmetry', 'has_total_runout', 'tightest_geom_tol_ultra_tight',
                'tightest_geom_tol_very_tight', 'tightest_geom_tol_tight',
                'tightest_geom_tol_medium', 'tightest_geom_tol_coarse',
                'stats_geom_tol_min', 'stats_geom_tol_max', 'stats_geom_tol_avg'
            ],
            'dimensional_tolerances': [
                'n_dia_tol', 'n_lin_tol', 'tightest_dim_tol_lin_ultra_tight',
                'tightest_dim_tol_lin_very_tight', 'tightest_dim_tol_lin_tight',
                'tightest_dim_tol_lin_medium', 'tightest_dim_tol_lin_coarse',
                'tightest_dim_tol_dia_ultra_tight', 'tightest_dim_tol_dia_very_tight',
                'tightest_dim_tol_dia_tight', 'tightest_dim_tol_dia_medium',
                'tightest_dim_tol_dia_coarse', 'stats_dim_tol_lin_min',
                'stats_dim_tol_lin_max', 'stats_dim_tol_lin_avg',
                'stats_dim_tol_dia_min', 'stats_dim_tol_dia_max',
                'stats_dim_tol_dia_avg', 'total_tolerance_count'
            ],
            'surface_finish': [
                'surface_spec_count', 'tightest_surface_very_fine',
                'tightest_surface_fine', 'tightest_surface_medium',
                'tightest_surface_rough', 'stats_surface_min',
                'stats_surface_max', 'stats_surface_avg'
            ],
            'fits': ['hole_fit_count', 'shaft_fit_count'],
            'datums': ['gtol_max_datum_refs', 'gtol_share_with_datum']
        }

        # Clean PMI data - convert object columns to numeric if possible
        self.pmi_df = self.pmi_df.apply(
            lambda s: pd.to_numeric(s, errors='ignore') if s.dtype == 'object' else s
        )

    def _fold_threshold(self, data: pd.DataFrame, class_name: str, val_idx: np.ndarray) -> float:
        """Get threshold for specific validation fold to avoid leakage"""
        thr_col = f"thr_{class_name}"
        if thr_col in data.columns:
            # Take mean of threshold values in validation fold (should be constant)
            thr_vals = data.iloc[val_idx][thr_col].dropna().values
            if len(thr_vals) > 0:
                return float(np.mean(thr_vals))
        return self.thresholds.get(class_name, 0.5)

    def get_fallback_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get global thresholds as fallback (for display only)"""
        thresholds = {}
        for class_name in self.classes:
            thr_col = f'thr_{class_name}'
            if thr_col in data.columns:
                thresholds[class_name] = data[thr_col].mean()
            else:
                thresholds[class_name] = 0.5
        return thresholds

    def prepare_data(self) -> pd.DataFrame:
        """Merge OOF predictions with PMI features."""
        df = self.oof_df.copy()

        # Filter by model type if specified
        if "model_type" in df.columns:
            df = df[df["model_type"] == self.model_type]

        assert self.id_col_oof in df.columns, f"'{self.id_col_oof}' missing in OOF"
        assert self.id_col_pmi in self.pmi_df.columns, f"'{self.id_col_pmi}' missing in PMI"

        # Handle potential _PMI suffix in OOF IDs
        df['id_for_merge'] = df[self.id_col_oof].apply(
            lambda x: x[:-4] if x.endswith('_PMI') else x
        )
        
        # Debug: Check overlap
        oof_ids = set(df['id_for_merge'].unique())
        pmi_ids = set(self.pmi_df[self.id_col_pmi].unique())
        overlap = oof_ids & pmi_ids
        
        print(f"OOF IDs (after suffix removal): {len(oof_ids)} unique")
        print(f"PMI part names: {len(pmi_ids)} unique")
        print(f"Overlap: {len(overlap)} parts ({len(overlap)/len(oof_ids)*100:.1f}%)")
        
        if len(overlap) == 0:
            print("WARNING: No overlap found!")
            print(f"Sample OOF IDs: {list(oof_ids)[:5]}")
            print(f"Sample PMI names: {list(pmi_ids)[:5]}")

        merged = pd.merge(
            df, self.pmi_df,
            left_on='id_for_merge',
            right_on=self.id_col_pmi,
            how='inner'
        )
        
        # Drop the temporary merge column
        if 'id_for_merge' in merged.columns:
            merged = merged.drop('id_for_merge', axis=1)
        
        # Keep only numeric PMI columns
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        # Keep metadata
        keep_meta = [self.id_col_oof]
        if "fold" in merged.columns: keep_meta.append("fold")
        if "model_type" in merged.columns: keep_meta.append("model_type")

        # Target/probability/threshold columns
        keep_targets = [f"y_{c}" for c in self.classes if f"y_{c}" in merged.columns]
        keep_probas  = [f"p_{c}" for c in self.classes if f"p_{c}" in merged.columns]
        keep_thresholds = [f"thr_{c}" for c in self.classes if f"thr_{c}" in merged.columns]

        merged = merged[sorted(set(num_cols + keep_meta + keep_targets + keep_probas + keep_thresholds))]
        
        # Get fallback thresholds
        self.thresholds = self.get_fallback_thresholds(merged)
        
        print(f"Merged {len(merged)} samples (OOF ∩ PMI).")
        return merged

    def _cv_iterator(self, y: pd.Series, df_fold: pd.Series = None):
        """Generate CV splits: either fixed OOF folds or StratifiedKFold."""
        if self.use_oof_folds and df_fold is not None and df_fold.notna().all():
            # Use existing folds
            unique_folds = sorted(df_fold.unique())
            for f in unique_folds:
                val_idx = df_fold.index[df_fold == f].to_numpy()
                train_idx = df_fold.index[df_fold != f].to_numpy()
                yield train_idx, val_idx
        else:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=self.random_state)
            for tr, va in skf.split(np.zeros(len(y)), y):
                yield tr, va

    def _build_pipeline(self, all_cols: List[str], p_col: str, pmi_cols: List[str]) -> Tuple[Pipeline, List[str]]:
        """Pipeline: Impute+Scale for PMI only, p_col passthrough, then LR."""
        pmi_cols_in = [c for c in pmi_cols if c in all_cols]
        
        ct = ColumnTransformer(
            transformers=[
                ("pmi", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), pmi_cols_in)
            ],
            remainder='passthrough'  # p_col passes through unchanged
        )
        
        # Use lbfgs for larger feature sets
        solver = "lbfgs" if len(pmi_cols_in) > 100 else "liblinear"
        
        pipe = Pipeline([
            ("prep", ct),
            ("clf", LogisticRegression(
                class_weight="balanced", 
                max_iter=2000, 
                solver=solver,
                random_state=self.random_state
            ))
        ])
        return pipe, pmi_cols_in

    def _cv_f1(self, X: pd.DataFrame, y: pd.Series, class_name: str,
               pipe: Pipeline, folds: List[Tuple[np.ndarray, np.ndarray]], 
               full_data: pd.DataFrame) -> Tuple[float, float, List[float], List[float]]:
        """Cross-validation with per-fold thresholds and additional metrics"""
        f1_scores = []
        auprc_scores = []
        
        for tr, va in folds:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            
            # Get fold-specific threshold
            threshold = self._fold_threshold(full_data, class_name, va)
            
            y_hat = (p >= threshold).astype(int)
            f1 = f1_score(y.iloc[va], y_hat)
            auprc = average_precision_score(y.iloc[va], p)
            
            f1_scores.append(f1)
            auprc_scores.append(auprc)
            
        return float(np.mean(f1_scores)), float(np.std(f1_scores)), f1_scores, auprc_scores

    def train_baseline_model(self, data: pd.DataFrame, class_name: str,
                             folds: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Baseline: only p_{class}."""
        y = data[f'y_{class_name}'].astype(int)
        p_col = f'p_{class_name}'
        X = data[[p_col]].copy()
        pipe, _ = self._build_pipeline(all_cols=X.columns.tolist(), p_col=p_col, pmi_cols=[])
        
        f1_mean, f1_std, f1_folds, auprc_folds = self._cv_f1(X, y, class_name, pipe, folds, data)
        
        return {
            'mean': f1_mean, 
            'std': f1_std,
            'fold_scores': f1_folds,
            'auprc_mean': np.mean(auprc_folds),
            'auprc_std': np.std(auprc_folds)
        }

    def train_group_model(self, data: pd.DataFrame, class_name: str, 
                          group_name: str, feature_list: List[str],
                          folds: List[Tuple[np.ndarray, np.ndarray]],
                          baseline_results: Dict) -> Dict:
        """Group: p_{class} + PMI group."""
        y = data[f'y_{class_name}'].astype(int)
        p_col = f'p_{class_name}'

        avail = [f for f in feature_list if f in data.columns]
        if not avail:
            return {
                'mean': 0.0, 'std': 0.0, 'improvement': 0.0, 
                'n_features': 0, 'fold_scores': [], 
                'auprc_mean': 0.0, 'auprc_std': 0.0
            }

        X = data[[p_col] + avail].copy()
        pipe, used_pmi = self._build_pipeline(all_cols=X.columns.tolist(), p_col=p_col, pmi_cols=avail)
        
        f1_mean, f1_std, f1_folds, auprc_folds = self._cv_f1(X, y, class_name, pipe, folds, data)
        improvement = f1_mean - baseline_results['mean']
        
        return {
            'mean': f1_mean, 
            'std': f1_std, 
            'improvement': improvement, 
            'n_features': len(used_pmi),
            'fold_scores': f1_folds,
            'auprc_mean': np.mean(auprc_folds),
            'auprc_std': np.std(auprc_folds),
            'auprc_improvement': np.mean(auprc_folds) - baseline_results['auprc_mean']
        }

    def analyze_all_groups(self):
        """Run analysis for all classes and PMI groups with statistical testing."""
        data = self.prepare_data()

        # CV splits
        fold_series = data["fold"] if (self.use_oof_folds and "fold" in data.columns) else None

        for class_name in self.classes:
            assert f"y_{class_name}" in data.columns and f"p_{class_name}" in data.columns, \
                f"OOF must contain y_{class_name} & p_{class_name}"

            print(f"\n{'='*60}")
            print(f"Analyzing class: {class_name}")
            print(f"{'='*60}")
            
            y = data[f'y_{class_name}'].astype(int)
            folds = list(self._cv_iterator(y, fold_series))
            print(f"Using {len(folds)} folds ({'OOF' if self.use_oof_folds and fold_series is not None else 'StratifiedKFold'})")

            self.results[class_name] = {}

            # Baseline
            base_results = self.train_baseline_model(data, class_name, folds)
            self.results[class_name]['baseline'] = base_results
            print(f"\nBaseline (p_{class_name} only):")
            print(f"  F1: {base_results['mean']:.4f} ± {base_results['std']:.4f}")
            print(f"  AUPRC: {base_results['auprc_mean']:.4f} ± {base_results['auprc_std']:.4f}")

            # Groups
            print(f"\nPMI Groups:")
            for group_name, features in self.pmi_groups.items():
                group_results = self.train_group_model(
                    data, class_name, group_name, features, folds, base_results
                )
                self.results[class_name][group_name] = group_results
                
                # Statistical test if enough folds
                p_value_str = ""
                if len(base_results['fold_scores']) >= 5 and len(group_results['fold_scores']) >= 5:
                    try:
                        stat, p_val = wilcoxon(
                            np.array(group_results['fold_scores']) - np.array(base_results['fold_scores'])
                        )
                        p_value_str = f" p={p_val:.3f}"
                        group_results['p_value'] = p_val
                    except:
                        pass
                
                print(f"  {group_name:25s}: F1={group_results['mean']:.4f} ± {group_results['std']:.4f} "
                      f"(Δ={group_results['improvement']:+.4f}{p_value_str}, n={group_results['n_features']})")

    def plot_results(self, save_path: str = None):
        """Plot improvements with statistical significance markers."""
        fig, axes = plt.subplots(1, len(self.classes), figsize=(6*len(self.classes), 6))
        if len(self.classes) == 1:
            axes = [axes]
            
        for ax, class_name in zip(axes, self.classes):
            groups = list(self.pmi_groups.keys())
            improvements = []
            significances = []
            n_features = []
            
            for g in groups:
                imp = self.results[class_name].get(g, {}).get('improvement', 0.0)
                p_val = self.results[class_name].get(g, {}).get('p_value', 1.0)
                n_feat = self.results[class_name].get(g, {}).get('n_features', 0)
                
                improvements.append(imp)
                significances.append(p_val < 0.05)
                n_features.append(n_feat)

            bars = ax.bar(range(len(groups)), improvements, alpha=0.8)
            
            for i, (bar, imp, sig) in enumerate(zip(bars, improvements, significances)):
                bar.set_color('lightgreen' if imp > 0 else 'lightcoral')
                # Add significance marker
                sig_marker = "*" if sig else ""
                ax.text(i, imp + (0.001 if imp >= 0 else -0.001), 
                       f'{imp:.3f}{sig_marker}\n(n={n_features[i]})',
                       ha='center', va='bottom' if imp>=0 else 'top', fontsize=9)

            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.set_ylabel('ΔF1 (vs. baseline)')
            ax.set_title(f'{class_name}')
            ax.axhline(0, color='black', lw=0.8, alpha=0.4)
            
            # Dynamic y-limits
            if improvements:
                pad = max(0.002, 0.05*max(1e-6, np.max(np.abs(improvements))))
                ax.set_ylim(np.min(improvements)-pad, np.max(improvements)+pad)

        cv_method = 'OOF Folds' if self.use_oof_folds else f'{self.n_splits}-Fold CV'
        plt.suptitle(f'PMI Feature Group Impact on Meta-Model Performance\n'
                     f'({cv_method}, * = p<0.05 via Wilcoxon test)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_heatmap(self, save_path: str = None):
        """Create heatmap with improvements and AUPRC changes."""
        # F1 improvements
        f1_mat = []
        auprc_mat = []
        
        for class_name in self.classes:
            f1_row = []
            auprc_row = []
            for g in self.pmi_groups.keys():
                f1_row.append(self.results[class_name].get(g, {}).get('improvement', 0.0))
                auprc_row.append(self.results[class_name].get(g, {}).get('auprc_improvement', 0.0))
            f1_mat.append(f1_row)
            auprc_mat.append(auprc_row)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # F1 heatmap
        sns.heatmap(f1_mat,
                    xticklabels=list(self.pmi_groups.keys()),
                    yticklabels=self.classes,
                    annot=True, fmt='.4f',
                    cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'ΔF1 vs. baseline'},
                    ax=ax1)
        ax1.set_title('F1 Score Improvements')
        ax1.set_xlabel('PMI Groups')
        ax1.set_ylabel('Process Class')
        
        # AUPRC heatmap
        sns.heatmap(auprc_mat,
                    xticklabels=list(self.pmi_groups.keys()),
                    yticklabels=self.classes,
                    annot=True, fmt='.4f',
                    cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'ΔAUPRC vs. baseline'},
                    ax=ax2)
        ax2.set_title('AUPRC Improvements')
        ax2.set_xlabel('PMI Groups')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_top_groups(self) -> pd.DataFrame:
        """Get ranking with statistical significance."""
        rows = []
        for group_name in self.pmi_groups.keys():
            # Average improvements
            f1_imps = []
            auprc_imps = []
            sig_count = 0
            
            for cls in self.classes:
                f1_imp = self.results.get(cls, {}).get(group_name, {}).get('improvement', 0.0)
                auprc_imp = self.results.get(cls, {}).get(group_name, {}).get('auprc_improvement', 0.0)
                p_val = self.results.get(cls, {}).get(group_name, {}).get('p_value', 1.0)
                
                f1_imps.append(f1_imp)
                auprc_imps.append(auprc_imp)
                if p_val < 0.05:
                    sig_count += 1
            
            row = {
                'group': group_name, 
                'avg_f1_improvement': np.mean(f1_imps),
                'avg_auprc_improvement': np.mean(auprc_imps),
                'significant_classes': sig_count
            }
            
            # Add per-class improvements
            for cls in self.classes:
                row[f'{cls}_f1'] = self.results.get(cls, {}).get(group_name, {}).get('improvement', 0.0)
                
            rows.append(row)
            
        return pd.DataFrame(rows).sort_values('avg_f1_improvement', ascending=False)

# --- Usage ---
if __name__ == "__main__":
    analyzer = PMIGroupAnalyzer(
        oof_path="/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/cv_results/20250925_102257/oof_predictions.csv",
        pmi_path="/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv",
        model_type="pmi",
        id_col_oof="id",
        id_col_pmi="part_name",
        use_oof_folds=True,  # Use OOF folds for consistent evaluation
        n_splits=5,
        random_state=42
    )
    
    # Main analysis
    analyzer.analyze_all_groups()
    
    # Visualizations
    analyzer.plot_results(save_path='pmi_group_impact_bars.png')
    analyzer.create_summary_heatmap(save_path='pmi_group_impact_heatmap.png')
    
    # Rankings with significance
    ranks = analyzer.get_top_groups()
    print("\n📊 Overall PMI Group Rankings:")
    print(ranks.to_string(index=False))
    
    # Save detailed results
    import json
    with open('pmi_group_analysis_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = {}
        for cls, cls_results in analyzer.results.items():
            results_for_json[cls] = {}
            for group, group_results in cls_results.items():
                results_for_json[cls][group] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in group_results.items()
                    if k != 'fold_scores'  # Don't save individual fold scores
                }
        
        output = {
            'method': 'OOF Folds' if analyzer.use_oof_folds else f'{analyzer.n_splits}-Fold CV',
            'fallback_thresholds': analyzer.thresholds,
            'results': results_for_json
        }
        json.dump(output, f, indent=2)