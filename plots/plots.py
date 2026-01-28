"""
Unified plotting script for Q-learning results from the Neoclassical Growth Model
Handles both 'rl' and 'cpu' prefixed output files
Reads CSV output files and creates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re

# Configuration
DATA_DIR = Path("out/data")
OUTPUT_DIR = Path("out/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#C73E1D'
}


def load_final_results(n_k, prefix='rl'):
    """Load the final policy and value function results"""
    filepath = DATA_DIR / f"{prefix}_{n_k}_vfi_final.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Final results file not found: {filepath}")
    return pd.read_csv(filepath)


def load_convergence_diffs(n_k, prefix='rl'):
    """Load the convergence differences across iterations"""
    filepath = DATA_DIR / f"{prefix}_{n_k}_vfi_diffs.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Diffs file not found: {filepath}")
    return pd.read_csv(filepath)


def load_snapshots(n_k, prefix='rl'):
    """Load all value function snapshots during training"""
    pattern = str(DATA_DIR / f"{prefix}_{n_k}_vfi_iter*.csv")
    files = sorted(glob.glob(pattern))
    
    snapshots = []
    for f in files:
        # Extract iteration number from filename (use basename)
        name = Path(f).name
        match = re.search(r'iter(\d+|final)', name)
        if match:
            iter_num = match.group(1)
            df = pd.read_csv(f)
            df['iteration'] = iter_num if iter_num == 'final' else int(iter_num)
            snapshots.append(df)
    
    return snapshots


def plot_value_function_evolution(snapshots, n_k, prefix='rl'):
    """Plot how the value function evolves over iterations"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to show a reasonable number of snapshots
    if len(snapshots) > 20:
        indices = np.linspace(0, len(snapshots)-1, 20, dtype=int)
        snapshots_to_plot = [snapshots[i] for i in indices]
    else:
        snapshots_to_plot = snapshots
    
    # Plot each snapshot
    for snap in snapshots_to_plot: 
        iter_label = snap['iteration'].iloc[0]
        alpha = 0.3 if iter_label != 'final' else 1.0
        linewidth = 1 if iter_label != 'final' else 2.5
        color = COLORS['primary'] if iter_label != 'final' else COLORS['accent']
        label = f"Iter {iter_label}" if iter_label == 'final' or iter_label == 0 else None
        
        ax.plot(snap['K'], snap['V'], alpha=alpha, linewidth=linewidth, 
                color=color, label=label)
    
    ax.set_xlabel('Capital Stock (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value Function V(K)', fontsize=12, fontweight='bold')
    ax.set_title(f'Value Function Evolution [{prefix. upper()}] (n_k={n_k})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{prefix}_value_function_evolution_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved:  {prefix}_value_function_evolution_{n_k}.png")


def plot_convergence(diffs_df, n_k, prefix='rl', epsilon=0.001):
    """Plot convergence of the value function"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale
    ax1.plot(diffs_df['iter'], diffs_df['diff'], color=COLORS['primary'], linewidth=2)
    ax1.axhline(y=epsilon, color=COLORS['danger'], linestyle='--', 
                linewidth=2, label=f'Tolerance (epsilon={epsilon})')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Absolute Difference', fontsize=12, fontweight='bold')
    ax1.set_title(f'Convergence [{prefix.upper()}] (Linear Scale)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(diffs_df['iter'], diffs_df['diff'], color=COLORS['secondary'], linewidth=2)
    ax2.axhline(y=epsilon, color=COLORS['danger'], linestyle='--', 
                linewidth=2, label=f'Tolerance (epsilon={epsilon})')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Absolute Difference (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Convergence [{prefix.upper()}] (Log Scale)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{prefix}_convergence_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_convergence_{n_k}.png")


def plot_policy_function(final_df, n_k, prefix='rl'):
    """Plot the policy function K' vs K"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot policy function
    ax.plot(final_df['K'], final_df['Kp'], color=COLORS['primary'], 
            linewidth=2.5, label="Policy K'(K)")
    
    # Plot 45-degree line (steady state reference)
    K_min, K_max = final_df['K'].min(), final_df['K'].max()
    ax.plot([K_min, K_max], [K_min, K_max], 'k--', 
            linewidth=2, alpha=0.7, label="45 degree line (K'=K)")
    
    # Find and mark steady state (where policy crosses 45-degree line)
    diff = final_df['Kp'] - final_df['K']
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) > 0:
        # Mark the approximate steady state
        ss_idx = sign_changes[0]
        ss_K = final_df.iloc[ss_idx]['K']
        ss_Kp = final_df.iloc[ss_idx]['Kp']
        ax.plot(ss_K, ss_Kp, 'o', color=COLORS['accent'], 
                markersize=12, label=f'Steady State approx {ss_K:.2f}', zorder=5)
    
    ax.set_xlabel('Current Capital K', fontsize=12, fontweight='bold')
    ax.set_ylabel("Next Period Capital K'", fontsize=12, fontweight='bold')
    ax.set_title(f'Policy Function [{prefix.upper()}] (n_k={n_k})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{prefix}_policy_function_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_policy_function_{n_k}. png")


def plot_consumption_and_savings(final_df, n_k, prefix='rl'):
    """Plot consumption and savings rate"""
    # Calculate savings rate
    final_df['savings_rate'] = (final_df['Kp'] - final_df['K']) / final_df['K']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Consumption
    ax1.plot(final_df['K'], final_df['c'], color=COLORS['success'], linewidth=2.5)
    ax1.set_xlabel('Capital Stock K', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Consumption c(K)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Consumption Function [{prefix.upper()}] (n_k={n_k})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Savings rate
    ax2.plot(final_df['K'], final_df['savings_rate'] * 100, 
             color=COLORS['secondary'], linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Capital Stock K', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Savings Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Savings Rate [{prefix.upper()}] (n_k={n_k})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{prefix}_consumption_savings_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {prefix}_consumption_savings_{n_k}.png")


def plot_comprehensive_dashboard(final_df, diffs_df, n_k, prefix='rl', epsilon=0.001):
    """Create a comprehensive 4-panel dashboard"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Value Function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(final_df['K'], final_df['V'], color=COLORS['primary'], linewidth=2.5)
    ax1.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Value Function V(K)', fontsize=11, fontweight='bold')
    ax1.set_title('Value Function', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Policy Function
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(final_df['K'], final_df['Kp'], color=COLORS['secondary'], linewidth=2.5)
    K_min, K_max = final_df['K'].min(), final_df['K'].max()
    ax2.plot([K_min, K_max], [K_min, K_max], 'k--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Next Period Capital K'", fontsize=11, fontweight='bold')
    ax2.set_title('Policy Function', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(diffs_df['iter'], diffs_df['diff'], color=COLORS['accent'], linewidth=2)
    ax3.axhline(y=epsilon, color=COLORS['danger'], linestyle='--', linewidth=2)
    ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Max Abs Difference (log)', fontsize=11, fontweight='bold')
    ax3.set_title('Convergence', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Consumption
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(final_df['K'], final_df['c'], color=COLORS['success'], linewidth=2.5)
    ax4.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Consumption c(K)', fontsize=11, fontweight='bold')
    ax4.set_title('Consumption Function', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Results Dashboard [{prefix.upper()}] (n_k={n_k})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(OUTPUT_DIR / f'{prefix}_dashboard_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved:  {prefix}_dashboard_{n_k}.png")


def generate_summary_stats(final_df, diffs_df, n_k, prefix='rl', epsilon=0.001):
    """Generate and save summary statistics"""
    # Find steady state
    diff = final_df['Kp'] - final_df['K']
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    stats = {
        'Prefix': prefix. upper(),
        'Grid Size (n_k)': n_k,
        'Total Iterations': len(diffs_df),
        'Final Diff': diffs_df['diff'].iloc[-1],
        'Convergence Achieved': diffs_df['diff'].iloc[-1] < epsilon,
        'Min Value':  final_df['V'].min(),
        'Max Value': final_df['V'].max(),
        'Min Capital': final_df['K'].min(),
        'Max Capital': final_df['K'].max(),
    }
    
    if len(sign_changes) > 0:
        ss_idx = sign_changes[0]
        stats['Steady State Capital'] = final_df.iloc[ss_idx]['K']
        stats['Steady State Consumption'] = final_df.iloc[ss_idx]['c']
    
    # Find when convergence was achieved
    converged_iter = diffs_df[diffs_df['diff'] < epsilon]['iter'].min()
    if not np.isnan(converged_iter):
        stats['Converged at Iteration'] = int(converged_iter)
    
    # Save to file
    stats_file = OUTPUT_DIR / f'{prefix}_summary_stats_{n_k}.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Results Summary [{prefix.upper()}] (n_k={n_k})\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved: {prefix}_summary_stats_{n_k}.txt")
    return stats


def plot_comparison(rl_df, cpu_df, n_k, metric='V', ylabel='Value Function V(K)'):
    """Compare RL and CPU results side by side"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # RL results
    ax1.plot(rl_df['K'], rl_df[metric], color=COLORS['primary'], linewidth=2.5)
    ax1.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax1.set_title(f'RL Results (n_k={n_k})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # CPU results
    ax2.plot(cpu_df['K'], cpu_df[metric], color=COLORS['secondary'], linewidth=2.5)
    ax2.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax2.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax2.set_title(f'CPU Results (n_k={n_k})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Overlay comparison
    ax3.plot(rl_df['K'], rl_df[metric], color=COLORS['primary'], 
             linewidth=2.5, label='RL', alpha=0.8)
    ax3.plot(cpu_df['K'], cpu_df[metric], color=COLORS['secondary'], 
             linewidth=2.5, label='CPU', alpha=0.8, linestyle='--')
    ax3.set_xlabel('Capital Stock K', fontsize=11, fontweight='bold')
    ax3.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax3.set_title(f'Comparison (n_k={n_k})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    metric_name = ylabel.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'comparison_{metric_name}_{n_k}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: comparison_{metric_name}_{n_k}.png")


def plot_convergence_comparison(rl_diffs, cpu_diffs, n_k):
    """Compare convergence between RL and CPU"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale
    ax1.plot(rl_diffs['iter'], rl_diffs['diff'], color=COLORS['primary'], 
             linewidth=2, label='RL', alpha=0.8)
    ax1.plot(cpu_diffs['iter'], cpu_diffs['diff'], color=COLORS['secondary'], 
             linewidth=2, label='CPU', alpha=0.8, linestyle='--')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Absolute Difference', fontsize=12, fontweight='bold')
    ax1.set_title(f'Convergence Comparison (Linear) (n_k={n_k})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(rl_diffs['iter'], rl_diffs['diff'], color=COLORS['primary'], 
                 linewidth=2, label='RL', alpha=0.8)
    ax2.semilogy(cpu_diffs['iter'], cpu_diffs['diff'], color=COLORS['secondary'], 
                 linewidth=2, label='CPU', alpha=0.8, linestyle='--')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Absolute Difference (log)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Convergence Comparison (Log) (n_k={n_k})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_convergence_{n_k}. png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: comparison_convergence_{n_k}. png")


def process_results(n_k, prefix='rl'):
    """Process results for a given grid size and prefix"""
    print(f"\n{'='*60}")
    print(f"Processing {prefix.upper()} results for n_k = {n_k}")
    print(f"{'='*60}")
    
    try:
        # Load data
        print("Loading data...")
        final_df = load_final_results(n_k, prefix)
        diffs_df = load_convergence_diffs(n_k, prefix)
        snapshots = load_snapshots(n_k, prefix)
        
        print(f"  - Final results: {len(final_df)} grid points")
        print(f"  - Convergence data: {len(diffs_df)} iterations")
        print(f"  - Value function snapshots: {len(snapshots)}")
        
        # Generate plots
        print("\nGenerating plots...")
        plot_convergence(diffs_df, n_k, prefix)
        plot_policy_function(final_df, n_k, prefix)
        plot_consumption_and_savings(final_df, n_k, prefix)
        
        if snapshots:
            plot_value_function_evolution(snapshots, n_k, prefix)
        
        plot_comprehensive_dashboard(final_df, diffs_df, n_k, prefix)
        
        # Generate statistics
        print("\nGenerating summary statistics...")
        stats = generate_summary_stats(final_df, diffs_df, n_k, prefix)
        
        print(f"\nSuccessfully processed {prefix.upper()} results for n_k = {n_k}")
        
        return final_df, diffs_df
        
    except Exception as e: 
        print(f"\nError processing {prefix.upper()} n_k = {n_k}:  {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to generate all plots"""
    print("=" * 60)
    print("Unified Q-Learning Results Plotting Script")
    print("Handles both 'rl' and 'cpu' prefixed files")
    print("=" * 60)
    
    # Diagnostic: list files found in data directory
    print(f"\nLooking in data directory: {DATA_DIR.resolve()}")
    data_files = sorted(DATA_DIR.glob('*'))
    if not data_files:
        print("  No files found in data directory.")
    else:
        print("  Files present:")
        for f in data_files:
            print("   -", f.name)
    
    # Detect all available result files (with .csv extension)
    all_files = {
        'rl': list(DATA_DIR.glob("rl_*_vfi_final.csv")),
        'cpu': list(DATA_DIR.glob("cpu_*_vfi_final.csv"))
    }
    
    # Extract grid sizes for each prefix
    results = {'rl': [], 'cpu':  []}
    
    for prefix in ['rl', 'cpu']: 
        grid_sizes = []
        for f in all_files[prefix]:
            match = re.search(rf'{prefix}_(\d+)_vfi_final', f.name)
            if match:
                grid_sizes. append(int(match.group(1)))
        results[prefix] = sorted(set(grid_sizes))
    
    if not results['rl'] and not results['cpu']:
        print(f"\nError: No result files found in {DATA_DIR}")
        print("Please ensure the C++ programs have been run and output files exist.")
        return
    
    print(f"\nFound RL results for grid sizes: {results['rl']}")
    print(f"\nFound CPU results for grid sizes:  {results['cpu']}")
    
    # Process each prefix and grid size
    for prefix in ['rl', 'cpu']: 
        for n_k in results[prefix]: 
            process_results(n_k, prefix)
    
    # Generate comparison plots if both RL and CPU exist for the same grid size
    common_sizes = set(results['rl']) & set(results['cpu'])
    
    if common_sizes:
        print(f"\n{'='*60}")
        print("Generating comparison plots")
        print(f"{'='*60}")
        print(f"Common grid sizes: {sorted(common_sizes)}")
        
        for n_k in sorted(common_sizes):
            print(f"\nComparing RL vs CPU for n_k = {n_k}...")
            try:
                rl_final = load_final_results(n_k, 'rl')
                cpu_final = load_final_results(n_k, 'cpu')
                rl_diffs = load_convergence_diffs(n_k, 'rl')
                cpu_diffs = load_convergence_diffs(n_k, 'cpu')
                
                plot_comparison(rl_final, cpu_final, n_k, 'V', 'Value Function V(K)')
                plot_comparison(rl_final, cpu_final, n_k, 'Kp', "Next Period Capital K'")
                plot_comparison(rl_final, cpu_final, n_k, 'c', 'Consumption c(K)')
                plot_convergence_comparison(rl_diffs, cpu_diffs, n_k)
                
                print(f"Comparison plots generated for n_k = {n_k}")
                
            except Exception as e:
                print(f"Error generating comparisons for n_k = {n_k}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"All plots saved to:  {OUTPUT_DIR. absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()