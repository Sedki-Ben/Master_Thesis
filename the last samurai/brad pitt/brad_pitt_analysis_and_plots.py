#!/usr/bin/env python3
"""
Brad Pitt Analysis and Visualization System
==========================================

Comprehensive analysis and visualization of physics-informed CNN results
and comparison with classical baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BradPittAnalyzer:
    """
    Comprehensive analysis and visualization of Brad Pitt results
    """
    
    def __init__(self, results_dir="brad_pitt_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        print("📊 Brad Pitt Analyzer Initialized")
        print(f"📁 Results directory: {self.results_dir}")
    
    def load_all_results(self):
        """Load all experimental results"""
        results = {}
        
        # Load CNN results
        cnn_path = self.results_dir / "brad_pitt_comprehensive_results.csv"
        if cnn_path.exists():
            results['cnn'] = pd.read_csv(cnn_path)
            print(f"✅ Loaded CNN results: {len(results['cnn'])} models")
        else:
            print(f"⚠️  CNN results not found: {cnn_path}")
        
        # Load classical results
        classical_path = self.results_dir / "physics_informed_classical_results.csv"
        if classical_path.exists():
            results['classical'] = pd.read_csv(classical_path)
            print(f"✅ Loaded classical results: {len(results['classical'])} models")
        else:
            print(f"⚠️  Classical results not found: {classical_path}")
        
        return results
    
    def create_performance_comparison_plot(self, results):
        """Create comprehensive performance comparison plot"""
        print("📊 Creating performance comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Prepare data for plotting
        plot_data = []
        
        # Add CNN results
        if 'cnn' in results:
            for _, row in results['cnn'].iterrows():
                plot_data.append({
                    'method': f"CNN_{row['model_type']}",
                    'dataset_size': row['dataset_size'],
                    'median_error': row['median_error_m'],
                    'accuracy_1m': row['accuracy_1m'],
                    'accuracy_2m': row['accuracy_2m'],
                    'type': 'CNN',
                    'features': 'physics_informed'
                })
        
        # Add classical results
        if 'classical' in results:
            for _, row in results['classical'].iterrows():
                plot_data.append({
                    'method': row['method'],
                    'dataset_size': row['dataset_size'],
                    'median_error': row['median_error_m'],
                    'accuracy_1m': row['accuracy_1m'],
                    'accuracy_2m': row['accuracy_2m'],
                    'type': 'Classical',
                    'features': row['features']
                })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            print("❌ No data available for plotting")
            return
        
        # Plot 1: Median Error by Method Type
        ax = axes[0, 0]
        sns.boxplot(data=df, x='type', y='median_error', ax=ax)
        ax.set_title('Median Localization Error by Method Type', fontweight='bold')
        ax.set_ylabel('Median Error (m)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy <1m by Dataset Size
        ax = axes[0, 1]
        for method_type in df['type'].unique():
            subset = df[df['type'] == method_type]
            if not subset.empty:
                grouped = subset.groupby('dataset_size')['accuracy_1m'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['dataset_size'], grouped['mean'], yerr=grouped['std'], 
                           marker='o', label=method_type, linewidth=2, markersize=8)
        
        ax.set_title('Sub-meter Accuracy vs Dataset Size', fontweight='bold')
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Accuracy <1m (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Feature Type Comparison (Classical only)
        ax = axes[0, 2]
        if 'classical' in results:
            classical_df = df[df['type'] == 'Classical']
            sns.boxplot(data=classical_df, x='features', y='median_error', ax=ax)
            ax.set_title('Classical Methods: Feature Type Impact', fontweight='bold')
            ax.set_ylabel('Median Error (m)')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Best Methods Comparison
        ax = axes[1, 0]
        best_methods = df.groupby(['method', 'dataset_size'])['median_error'].min().reset_index()
        best_overall = best_methods.groupby('method')['median_error'].min().sort_values()[:10]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_overall)))
        bars = ax.barh(range(len(best_overall)), best_overall.values, color=colors)
        ax.set_yticks(range(len(best_overall)))
        ax.set_yticklabels(best_overall.index, fontsize=10)
        ax.set_xlabel('Best Median Error (m)')
        ax.set_title('Top 10 Methods (Best Performance)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, best_overall.values)):
            ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}m', va='center', fontsize=9)
        
        # Plot 5: Accuracy <2m Comparison
        ax = axes[1, 1]
        method_acc = df.groupby('method')['accuracy_2m'].max().sort_values(ascending=False)[:10]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(method_acc)))
        bars = ax.barh(range(len(method_acc)), method_acc.values, color=colors)
        ax.set_yticks(range(len(method_acc)))
        ax.set_yticklabels(method_acc.index, fontsize=10)
        ax.set_xlabel('Best Accuracy <2m (%)')
        ax.set_title('Top 10 Methods (2m Accuracy)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, method_acc.values)):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}%', va='center', fontsize=9)
        
        # Plot 6: CNN vs Best Classical
        ax = axes[1, 2]
        if 'cnn' in results and 'classical' in results:
            # Best CNN per dataset size
            cnn_best = results['cnn'].groupby('dataset_size')['median_error_m'].min()
            
            # Best classical per dataset size
            classical_best = results['classical'].groupby('dataset_size')['median_error_m'].min()
            
            dataset_sizes = sorted(set(cnn_best.index) | set(classical_best.index))
            
            cnn_errors = [cnn_best.get(ds, np.nan) for ds in dataset_sizes]
            classical_errors = [classical_best.get(ds, np.nan) for ds in dataset_sizes]
            
            ax.plot(dataset_sizes, cnn_errors, 'o-', label='Best CNN', linewidth=3, markersize=8)
            ax.plot(dataset_sizes, classical_errors, 's-', label='Best Classical', linewidth=3, markersize=8)
            
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Best Median Error (m)')
            ax.set_title('CNN vs Classical: Best Performance', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.results_dir / "brad_pitt_performance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved performance comparison: {output_path}")
        
        plt.show()
    
    def create_feature_importance_analysis(self, results):
        """Analyze the impact of physics-informed features"""
        print("🔬 Creating feature importance analysis...")
        
        if 'classical' not in results:
            print("⚠️  No classical results for feature analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        classical_df = results['classical']
        
        # Plot 1: Feature Type Performance
        ax = axes[0, 0]
        feature_performance = classical_df.groupby('features')['median_error_m'].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(feature_performance['features'], feature_performance['mean'], 
                     yerr=feature_performance['std'], capsize=5, alpha=0.8,
                     color=['lightcoral', 'lightblue', 'lightgreen'])
        
        ax.set_title('Feature Type Impact on Performance', fontweight='bold')
        ax.set_ylabel('Mean Median Error (m)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, feature_performance['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{mean_val:.3f}m', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Method Performance by Feature Type
        ax = axes[0, 1]
        pivot_data = classical_df.pivot_table(
            values='median_error_m', 
            index='method', 
            columns='features', 
            aggfunc='min'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis_r', ax=ax)
        ax.set_title('Method × Feature Type Performance Matrix', fontweight='bold')
        ax.set_ylabel('Method')
        
        # Plot 3: Improvement from Physics Features
        ax = axes[1, 0]
        improvements = []
        
        for method in classical_df['method'].unique():
            method_data = classical_df[classical_df['method'] == method]
            
            for ds in method_data['dataset_size'].unique():
                ds_data = method_data[method_data['dataset_size'] == ds]
                
                raw_perf = ds_data[ds_data['features'] == 'raw']['median_error_m']
                physics_perf = ds_data[ds_data['features'] == 'physics']['median_error_m']
                combined_perf = ds_data[ds_data['features'] == 'combined']['median_error_m']
                
                if not raw_perf.empty and not physics_perf.empty:
                    improvement_physics = (raw_perf.iloc[0] - physics_perf.iloc[0]) / raw_perf.iloc[0] * 100
                    improvements.append({
                        'method': method,
                        'dataset_size': ds,
                        'improvement_type': 'Physics vs Raw',
                        'improvement_pct': improvement_physics
                    })
                
                if not raw_perf.empty and not combined_perf.empty:
                    improvement_combined = (raw_perf.iloc[0] - combined_perf.iloc[0]) / raw_perf.iloc[0] * 100
                    improvements.append({
                        'method': method,
                        'dataset_size': ds,
                        'improvement_type': 'Combined vs Raw',
                        'improvement_pct': improvement_combined
                    })
        
        if improvements:
            improvement_df = pd.DataFrame(improvements)
            sns.boxplot(data=improvement_df, x='improvement_type', y='improvement_pct', ax=ax)
            ax.set_title('Performance Improvement from Enhanced Features', fontweight='bold')
            ax.set_ylabel('Performance Improvement (%)')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Best Feature Combination by Method
        ax = axes[1, 1]
        best_features = classical_df.groupby('method').apply(
            lambda x: x.loc[x['median_error_m'].idxmin(), 'features']
        ).value_counts()
        
        colors = ['lightcoral', 'lightblue', 'lightgreen'][:len(best_features)]
        wedges, texts, autotexts = ax.pie(best_features.values, labels=best_features.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax.set_title('Best Feature Type Distribution\n(Across All Methods)', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.results_dir / "feature_importance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved feature importance analysis: {output_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive performance report"""
        print("📝 Generating comprehensive performance report...")
        
        report_path = self.results_dir / "brad_pitt_comprehensive_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Brad Pitt: Physics-Informed CNN Performance Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents the comprehensive evaluation of physics-informed CNN ")
            f.write("architectures against enhanced classical baselines for indoor WiFi localization.\n\n")
            
            # CNN Results Analysis
            if 'cnn' in results:
                f.write("## Physics-Informed CNN Results\n\n")
                cnn_df = results['cnn']
                
                f.write("### Architecture Performance\n\n")
                for model_type in cnn_df['model_type'].unique():
                    model_data = cnn_df[cnn_df['model_type'] == model_type]
                    best_result = model_data.loc[model_data['median_error_m'].idxmin()]
                    
                    f.write(f"**{model_type.replace('_', ' ').title()}**:\n")
                    f.write(f"- Best Performance: {best_result['median_error_m']:.3f}m median error ")
                    f.write(f"({best_result['dataset_size']} samples)\n")
                    f.write(f"- Sub-meter Accuracy: {best_result['accuracy_1m']:.1f}%\n")
                    f.write(f"- 2-meter Accuracy: {best_result['accuracy_2m']:.1f}%\n")
                    f.write(f"- Model Parameters: {best_result['n_parameters']:,}\n\n")
                
                # Overall CNN champion
                cnn_champion = cnn_df.loc[cnn_df['median_error_m'].idxmin()]
                f.write(f"### 🏆 CNN Champion\n\n")
                f.write(f"**{cnn_champion['model_type'].replace('_', ' ').title()}** ")
                f.write(f"({cnn_champion['dataset_size']} samples)\n")
                f.write(f"- **Median Error**: {cnn_champion['median_error_m']:.3f}m\n")
                f.write(f"- **Sub-meter Accuracy**: {cnn_champion['accuracy_1m']:.1f}%\n")
                f.write(f"- **2-meter Accuracy**: {cnn_champion['accuracy_2m']:.1f}%\n\n")
            
            # Classical Results Analysis
            if 'classical' in results:
                f.write("## Enhanced Classical Baselines\n\n")
                classical_df = results['classical']
                
                f.write("### Feature Type Impact\n\n")
                feature_impact = classical_df.groupby('features')['median_error_m'].agg(['mean', 'std', 'min'])
                
                for feature_type in ['raw', 'physics', 'combined']:
                    if feature_type in feature_impact.index:
                        stats = feature_impact.loc[feature_type]
                        f.write(f"**{feature_type.title()} Features**:\n")
                        f.write(f"- Mean Performance: {stats['mean']:.3f} ± {stats['std']:.3f}m\n")
                        f.write(f"- Best Performance: {stats['min']:.3f}m\n\n")
                
                # Classical champion
                classical_champion = classical_df.loc[classical_df['median_error_m'].idxmin()]
                f.write(f"### 🏆 Classical Champion\n\n")
                f.write(f"**{classical_champion['method']}** with {classical_champion['features']} features ")
                f.write(f"({classical_champion['dataset_size']} samples)\n")
                f.write(f"- **Median Error**: {classical_champion['median_error_m']:.3f}m\n")
                f.write(f"- **Sub-meter Accuracy**: {classical_champion['accuracy_1m']:.1f}%\n")
                f.write(f"- **2-meter Accuracy**: {classical_champion['accuracy_2m']:.1f}%\n\n")
            
            # Overall Comparison
            if 'cnn' in results and 'classical' in results:
                f.write("## CNN vs Classical Comparison\n\n")
                
                cnn_best = results['cnn']['median_error_m'].min()
                classical_best = results['classical']['median_error_m'].min()
                
                improvement = (classical_best - cnn_best) / classical_best * 100
                
                f.write(f"- **Best CNN**: {cnn_best:.3f}m\n")
                f.write(f"- **Best Classical**: {classical_best:.3f}m\n")
                f.write(f"- **CNN Improvement**: {improvement:.1f}%\n\n")
                
                if improvement > 0:
                    f.write("✅ **Physics-informed CNNs outperform classical methods**\n\n")
                else:
                    f.write("⚠️  **Classical methods competitive with CNNs**\n\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            f.write("### 1. Physics-Informed Features Impact\n\n")
            
            if 'classical' in results:
                raw_mean = results['classical'][results['classical']['features'] == 'raw']['median_error_m'].mean()
                physics_mean = results['classical'][results['classical']['features'] == 'physics']['median_error_m'].mean()
                
                if not np.isnan(raw_mean) and not np.isnan(physics_mean):
                    physics_improvement = (raw_mean - physics_mean) / raw_mean * 100
                    f.write(f"- Physics features improve classical methods by {physics_improvement:.1f}% on average\n")
                
                combined_mean = results['classical'][results['classical']['features'] == 'combined']['median_error_m'].mean()
                if not np.isnan(combined_mean):
                    combined_improvement = (raw_mean - combined_mean) / raw_mean * 100
                    f.write(f"- Combined features improve classical methods by {combined_improvement:.1f}% on average\n\n")
            
            f.write("### 2. Architecture Insights\n\n")
            if 'cnn' in results:
                freq_adaptive_perf = results['cnn'][results['cnn']['model_type'] == 'frequency_adaptive']['median_error_m'].min()
                enhanced_hybrid_perf = results['cnn'][results['cnn']['model_type'] == 'enhanced_hybrid']['median_error_m'].min()
                
                if not np.isnan(freq_adaptive_perf) and not np.isnan(enhanced_hybrid_perf):
                    if freq_adaptive_perf < enhanced_hybrid_perf:
                        f.write("- Frequency-adaptive architecture shows superior performance\n")
                        f.write("- Multi-scale processing and attention mechanisms are effective\n")
                    else:
                        f.write("- Enhanced hybrid architecture with physics features performs best\n")
                        f.write("- Combined CSI, physics, and RSSI features are optimal\n")
            
            f.write("\n### 3. Dataset Size Effects\n\n")
            if 'cnn' in results:
                size_performance = results['cnn'].groupby('dataset_size')['median_error_m'].min()
                if len(size_performance) > 1:
                    if size_performance.iloc[-1] < size_performance.iloc[0]:
                        f.write("- Performance improves with larger datasets\n")
                    else:
                        f.write("- Model may be overfitting with larger datasets\n")
                        f.write("- Optimal performance achieved with smaller datasets\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("### For Production Deployment\n\n")
            
            if 'cnn' in results and 'classical' in results:
                if cnn_best < classical_best:
                    f.write("- **Deploy physics-informed CNN** for best accuracy\n")
                    f.write("- Consider computational requirements vs. accuracy trade-offs\n")
                else:
                    f.write("- **Classical methods with physics features** may be sufficient\n")
                    f.write("- Lower computational requirements with competitive accuracy\n")
            
            f.write("\n### For Future Research\n\n")
            f.write("- Investigate transfer learning from physics-informed features\n")
            f.write("- Explore ensemble methods combining CNN and classical approaches\n")
            f.write("- Study temporal dynamics and environmental variations\n")
            f.write("- Implement real-time coherence bandwidth adaptation\n\n")
            
        print(f"✅ Generated comprehensive report: {report_path}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🎬 Brad Pitt: Complete Analysis Pipeline")
        print("=" * 50)
        
        # Load results
        results = self.load_all_results()
        
        if not results:
            print("❌ No results found for analysis")
            return
        
        # Create visualizations
        self.create_performance_comparison_plot(results)
        self.create_feature_importance_analysis(results)
        
        # Generate report
        self.generate_comprehensive_report(results)
        
        print("\n🎉 Complete analysis finished!")
        print(f"📁 All outputs saved to: {self.results_dir}")
        
        return results

def main():
    """Main execution function"""
    analyzer = BradPittAnalyzer()
    results = analyzer.run_complete_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
