from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set matplotlib style for print-friendly plots (white background)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'

# Load data once at module level
df_repro = pd.read_csv('reproduction/speaker-verification.csv')
df_paper = pd.read_csv('paper/speaker-verification.csv')

# Default configurations
DEFAULT_ARCHS = ['CNN', 'CNN-VTLP', 'RNN', 'ResNet', 'F-ResNet', 'Conformer', 'Conformer-NoPE', 'Conformer-NoPE-K1', 'W2V-Bert']
DEFAULT_COLORMAP = {'OS': '#1f77b4', 'SS': '#ff7f0e', 'SU': '#2ca02c'}
DEFAULT_GREEN = '#2ca02c'

# Style configurations
BAR_STYLE_SOLID = {'alpha': 1.0, 'edgecolor': 'black'}
BAR_STYLE_SHADOW = {'alpha': 0.4, 'edgecolor': 'none'}
GRID_STYLE = {'axis': 'y', 'alpha': 0.3, 'linestyle': '--'}
LEGEND_STYLE = {'loc': 'upper center', 'fontsize': 8}
TITLE_STYLE = {'fontsize': 12, 'pad': 10}
LABEL_STYLE = {'fontsize': 10}
TICK_STYLE = {'rotation': 45, 'ha': 'right', 'fontsize': 9}


def _filter_dataframe(df: pd.DataFrame, 
                      datasets: Optional[List[str]] = None,
                      train_strategies: Optional[List[str]] = None,
                      test_strategies: Optional[List[str]] = None,
                      architectures: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter dataframe by multiple criteria."""
    filtered = df.copy()
    if datasets:
        filtered = filtered[filtered['dataset'].isin(datasets)]
    if train_strategies:
        filtered = filtered[filtered['train_strategy'].isin(train_strategies)]
    if test_strategies:
        filtered = filtered[filtered['test_strategy'].isin(test_strategies)]
    if architectures:
        filtered = filtered[filtered['architecture'].isin(architectures)]
    return filtered


def _get_filtered_architecture_data(subset: pd.DataFrame, 
                                   architectures: List[str],
                                   test_strategy: str) -> Tuple[List[float], List[float], List[float]]:
    """Extract filtered EER means, stds, and x positions (excluding None values)."""
    arch_to_eer = {}
    arch_to_std = {}
    
    for _, row in subset[subset['test_strategy'] == test_strategy].iterrows():
        arch_to_eer[row['architecture']] = row['eer_mean']
        arch_to_std[row['architecture']] = row['eer_std']
    
    x_positions = []
    means_filt = []
    stds_filt = []
    
    for idx, arch in enumerate(architectures):
        mean = arch_to_eer.get(arch, None)
        std = arch_to_std.get(arch, None)
        if mean is not None:
            x_positions.append(idx)
            means_filt.append(mean)
            stds_filt.append(std)
    
    return x_positions, means_filt, stds_filt


def _setup_subplots(n_rows: int, n_cols: int, figsize: Tuple[int, int], sharey: str = None) -> Tuple[plt.Figure, np.ndarray]:
    """Create subplot grid with proper reshaping for single row/column cases."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    
    # Handle single row or column cases
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1 and n_rows > 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    
    return fig, axes


def _get_axis(axes, i: int, j: int, n_rows: int, n_cols: int):
    """Get the appropriate axis from a potentially nested axes array."""
    if n_rows == 1 and n_cols == 1:
        return axes[0, 0]
    elif n_rows == 1:
        return axes[0, j]
    elif n_cols == 1:
        return axes[i, 0]
    else:
        return axes[i, j]


def _style_axis(ax, title: str = None, ylabel: str = None, 
                ylim: Tuple[float, float] = None, xticks: np.ndarray = None,
                xticklabels: List[str] = None, show_grid: bool = True,
                legend_ncol: int = 1):
    """Apply consistent styling to a plot axis."""
    if title:
        ax.set_title(title, **TITLE_STYLE)
    if ylabel:
        ax.set_ylabel(ylabel, **LABEL_STYLE)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels, **TICK_STYLE)
    if show_grid:
        ax.grid(**GRID_STYLE)
    
    # Add legend if there are labeled artists
    if len(ax.get_legend_handles_labels()[0]) > 0:
        legend_style = LEGEND_STYLE.copy()
        if legend_ncol > 1:
            legend_style['ncol'] = legend_ncol
        ax.legend(**legend_style)


def paper_vs_repro(archs: Optional[List[str]] = None,
                   datasets: Optional[List[str]] = None,
                   train_strategies: Optional[List[str]] = None,
                   test_strategies: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (16, 8),
                   show_paper: bool = True,
                   show_repro: bool = True):
    """
    Compare paper vs reproduction results.
    
    Parameters:
    -----------
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet']
    datasets : List[str]
        Datasets to include. Default: all datasets in filtered data
    train_strategies : List[str]
        Training strategies to include. Default: ['OS', 'SS']
    test_strategies : List[str]
        Test strategies to include. Default: all test strategies in filtered data
    figsize : Tuple[int, int]
        Figure size. Default: (16, 8)
    show_paper : bool
        Whether to show paper results. Default: True
    show_repro : bool
        Whether to show reproduction results. Default: True
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if train_strategies is None:
        train_strategies = ['OS', 'SS']
    
    # Filter dataframes
    df_filtered_repro = _filter_dataframe(df_repro, train_strategies=train_strategies)
    df_filtered_paper = _filter_dataframe(df_paper, train_strategies=train_strategies)
    
    # Get unique values if not provided
    if datasets is None:
        datasets = sorted(df_filtered_repro['dataset'].unique())
    if test_strategies is None:
        test_strategies = sorted(df_filtered_repro['test_strategy'].unique())
    
    # Set up subplots
    fig, axes = _setup_subplots(len(datasets), len(train_strategies), figsize, sharey='row')
    
    # Color schemes
    colors = DEFAULT_COLORMAP.copy()
    labels = {'OS': 'OS (repro)', 'SS': 'SS (repro)', 'SU': 'SU (repro)'}
    
    # Calculate y-axis max per dataset
    dataset_max = {}
    for dataset in datasets:
        dataset_data_repro = df_filtered_repro[df_filtered_repro['dataset'] == dataset]
        dataset_data_paper = df_filtered_paper[df_filtered_paper['dataset'] == dataset]
        eer_max_repro = dataset_data_repro['eer_mean'] + dataset_data_repro['eer_std']
        eer_max_paper = dataset_data_paper['eer_mean'] + dataset_data_paper['eer_std']
        all_eer_max = pd.concat([eer_max_repro, eer_max_paper])
        dataset_max[dataset] = max(all_eer_max) * 1.15
    
    # Determine bar width based on number of test strategies
    n_test = len(test_strategies)
    bar_width = 0.8 / (n_test * 2)
    
    # Plot for each dataset and training strategy
    for i, dataset in enumerate(datasets):
        for j, train_strategy in enumerate(train_strategies):
            ax = _get_axis(axes, i, j, len(datasets), len(train_strategies))
            
            # Get data subsets
            subset_repro = df_filtered_repro[(df_filtered_repro['dataset'] == dataset) &
                                           (df_filtered_repro['train_strategy'] == train_strategy)]
            subset_paper = df_filtered_paper[(df_filtered_paper['dataset'] == dataset) &
                                           (df_filtered_paper['train_strategy'] == train_strategy)]
            
            # Get architectures (preserving order)
            subset_architectures = [arch for arch in archs if arch in subset_repro['architecture'].unique()]
            
            if not subset_architectures:
                ax.set_title(f'dataset = {dataset}, train = {train_strategy}', **TITLE_STYLE)
                continue
            
            n_arch = len(subset_architectures)
            x = np.arange(n_arch)
            
            # Plot each test strategy
            for k, test_strategy in enumerate(test_strategies):
                paper_offset = (k * 2 - (n_test * 2 - 1) / 2) * bar_width
                repro_offset = (k * 2 + 1 - (n_test * 2 - 1) / 2) * bar_width
                
                # Paper shadow bars
                if show_paper and len(subset_paper) > 0:
                    paper_x, paper_means_filt, paper_stds_filt = _get_filtered_architecture_data(
                        subset_paper, subset_architectures, test_strategy)
                    
                    if len(paper_x) > 0:
                        ax.bar(np.array(paper_x) + paper_offset, paper_means_filt, bar_width,
                               yerr=paper_stds_filt, capsize=5,
                               color=colors[test_strategy],
                               label=f'{test_strategy} (paper)',
                               **BAR_STYLE_SHADOW)
                
                # Repro solid bars
                if show_repro:
                    test_data_repro = subset_repro[subset_repro['test_strategy'] == test_strategy]
                    arch_to_eer = {}
                    arch_to_std = {}
                    for _, row in test_data_repro.iterrows():
                        arch_to_eer[row['architecture']] = row['eer_mean']
                        arch_to_std[row['architecture']] = row['eer_std']
                    
                    eer_means = [arch_to_eer.get(arch, 0) for arch in subset_architectures]
                    eer_stds = [arch_to_std.get(arch, 0) for arch in subset_architectures]
                    
                    ax.bar(x + repro_offset, eer_means, bar_width,
                           yerr=eer_stds, capsize=5,
                           color=colors[test_strategy],
                           label=labels[test_strategy],
                           **BAR_STYLE_SOLID)
            
            # Style the axis
            _style_axis(ax,
                       title=f'dataset = {dataset}, train = {train_strategy}',
                       ylabel='EER',
                       ylim=(0, dataset_max[dataset]),
                       xticks=x,
                       xticklabels=subset_architectures,
                       legend_ncol=len(test_strategies))
    
    plt.tight_layout()


def abs_eer(comparison_type: str = 'test_strategy',
                  archs: Optional[List[str]] = None,
                  datasets: Optional[List[str]] = None,
                  train_strategies: Optional[List[str]] = None,
                  test_strategies: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (16, 8),
                  ylabel: str = 'EER'):
    """
    Plot absolute EER comparisons.
    
    Parameters:
    -----------
    comparison_type : str
        Type of comparison: 'test_strategy' or 'training_strategy'
        - 'test_strategy': Compare test strategies (OS vs SS) for same training strategy
        - 'training_strategy': Compare training strategies (OS-OS vs SS-SS)
    archs : List[str]
        Architectures to include. Default: Depends on comparison_type
    datasets : List[str]
        Datasets to include. Default: ['voxceleb']
    train_strategies : List[str]
        Training strategies for test_strategy comparison. Default: ['OS']
    test_strategies : List[str]
        Test strategies for test_strategy comparison. Default: ['OS', 'SS']
    figsize : Tuple[int, int]
        Figure size. Default: (16, 8) for test_strategy, (14, 6) for training_strategy
    ylabel : str
        Y-axis label. Default: 'EER'
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if datasets is None:
        datasets = ['voxceleb']
    
    if comparison_type == 'training_strategy':
        # Compare OS-OS vs SS-SS
        if figsize == (16, 8):  # Use smaller default for training strategy
            figsize = (14, 6)
        
        # Filter for OS and SS training strategies
        df_filtered = df_repro[df_repro['train_strategy'].isin(['OS', 'SS'])].copy()
        
        # Define colors for the two strategies
        colors = {'OS-OS': '#1f77b4', 'SS-SS': '#ff7f0e'}
        labels = {'OS-OS': 'EER(train=OS, test=OS)', 'SS-SS': 'EER(train=SS, test=SS)'}
        
        # Calculate global max for y-axis alignment
        all_eers = []
        for dataset in datasets:
            os_os_data = df_filtered[(df_filtered['dataset'] == dataset) &
                                 (df_filtered['train_strategy'] == 'OS') &
                                 (df_filtered['test_strategy'] == 'OS')]
            ss_ss_data = df_filtered[(df_filtered['dataset'] == dataset) &
                                 (df_filtered['train_strategy'] == 'SS') &
                                 (df_filtered['test_strategy'] == 'SS')]
            for arch in archs:
                for data, key in [(os_os_data, 'OS-OS'), (ss_ss_data, 'SS-SS')]:
                    row = data[data['architecture'] == arch]
                    if len(row) > 0:
                        all_eers.append(row['eer_mean'].values[0] + row['eer_std'].values[0])
        
        global_max = max(all_eers) * 1.15 if all_eers else 1.0
        
        # Set up subplots
        fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)
        
        # Handle single dataset case
        if len(datasets) == 1:
            axes = [axes]
        
        bar_width = 0.4
        
        # Create bar charts for each dataset
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # Get data for this dataset
            os_os_data = df_filtered[(df_filtered['dataset'] == dataset) &
                             (df_filtered['train_strategy'] == 'OS') &
                             (df_filtered['test_strategy'] == 'OS')]
            ss_ss_data = df_filtered[(df_filtered['dataset'] == dataset) &
                             (df_filtered['train_strategy'] == 'SS') &
                             (df_filtered['test_strategy'] == 'SS')]
            
            # Get architectures present in this dataset
            dataset_architectures = [arch for arch in archs if 
                                     arch in df_filtered[df_filtered['dataset'] == dataset]['architecture'].unique()]
            
            n_arch = len(dataset_architectures)
            x = np.arange(n_arch)
            
            # Plot OS-OS bars
            os_os_means = []
            os_os_stds = []
            for arch in dataset_architectures:
                row = os_os_data[os_os_data['architecture'] == arch]
                if len(row) > 0:
                    os_os_means.append(row['eer_mean'].values[0])
                    os_os_stds.append(row['eer_std'].values[0])
                else:
                    os_os_means.append(0)
                    os_os_stds.append(0)
            
            ax.bar(x - bar_width / 2, os_os_means, bar_width,
                   yerr=os_os_stds, capsize=5,
                   color=colors['OS-OS'], alpha=0.8, edgecolor='black',
                   label=labels['OS-OS'])
            
            # Plot SS-SS bars
            ss_ss_means = []
            ss_ss_stds = []
            for arch in dataset_architectures:
                row = ss_ss_data[ss_ss_data['architecture'] == arch]
                if len(row) > 0:
                    ss_ss_means.append(row['eer_mean'].values[0])
                    ss_ss_stds.append(row['eer_std'].values[0])
                else:
                    ss_ss_means.append(0)
                    ss_ss_stds.append(0)
            
            ax.bar(x + bar_width / 2, ss_ss_means, bar_width,
                   yerr=ss_ss_stds, capsize=5,
                   color=colors['SS-SS'], **BAR_STYLE_SOLID,
                   label=labels['SS-SS'])
            
            # Style the axis
            _style_axis(ax,
                       ylabel=ylabel,
                       ylim=(0, global_max),
                       xticks=x,
                       xticklabels=dataset_architectures,
                       legend_ncol=1)
        
        plt.tight_layout()
        
    else:  # comparison_type == 'test_strategy'
        # Compare test strategies for same training strategy
        if train_strategies is None:
            train_strategies = ['OS']
        if test_strategies is None:
            test_strategies = ['OS', 'SS']
        
        # Filter dataframe
        df_filtered_repro = _filter_dataframe(df_repro, 
                                              datasets=datasets,
                                              train_strategies=train_strategies)
        
        # Set up subplots
        fig, axes = _setup_subplots(len(datasets), len(train_strategies), figsize, sharey='row')
        
        # Color schemes
        colors = DEFAULT_COLORMAP.copy()
        labels = {'OS': 'OS', 'SS': 'SS', 'SU': 'SU'}
        
        # Calculate y-axis max per dataset
        dataset_max = {}
        for dataset in datasets:
            dataset_data_repro = df_filtered_repro[df_filtered_repro['dataset'] == dataset]
            eer_max_repro = dataset_data_repro['eer_mean'] + dataset_data_repro['eer_std']
            dataset_max[dataset] = max(eer_max_repro) * 1.15
        
        # Determine bar width
        n_test = len(test_strategies)
        bar_width = 0.8 / n_test
        
        # Plot for each dataset and training strategy
        for i, dataset in enumerate(datasets):
            for j, train_strategy in enumerate(train_strategies):
                ax = _get_axis(axes, i, j, len(datasets), len(train_strategies))
                
                # Get data
                subset_repro = df_filtered_repro[(df_filtered_repro['dataset'] == dataset) &
                                               (df_filtered_repro['train_strategy'] == train_strategy)]
                
                # Get architectures (preserving order)
                subset_architectures = [arch for arch in archs if arch in subset_repro['architecture'].unique()]
                
                if not subset_architectures:
                    ax.set_title(f'dataset = {dataset}, train = {train_strategy}', **TITLE_STYLE)
                    continue
                
                n_arch = len(subset_architectures)
                x = np.arange(n_arch)
                
                # Plot each test strategy
                for k, test_strategy in enumerate(test_strategies):
                    offset = (k - (n_test - 1) / 2) * bar_width
                    
                    # Repro solid bars
                    test_data_repro = subset_repro[subset_repro['test_strategy'] == test_strategy]
                    arch_to_eer = {}
                    arch_to_std = {}
                    for _, row in test_data_repro.iterrows():
                        arch_to_eer[row['architecture']] = row['eer_mean']
                        arch_to_std[row['architecture']] = row['eer_std']
                    
                    eer_means = [arch_to_eer.get(arch, 0) for arch in subset_architectures]
                    eer_stds = [arch_to_std.get(arch, 0) for arch in subset_architectures]
                    
                    ax.bar(x + offset, eer_means, bar_width,
                           yerr=eer_stds, capsize=5,
                           color=colors[test_strategy],
                           label=labels[test_strategy],
                           **BAR_STYLE_SOLID)
                
                # Style the axis
                _style_axis(ax,
                           title=f'dataset = {dataset}, train = {train_strategy}',
                           ylabel=ylabel,
                           ylim=(0, dataset_max[dataset]),
                           xticks=x,
                           xticklabels=subset_architectures,
                           legend_ncol=len(test_strategies))
        
        plt.tight_layout()


def relative_eer(comparison_type: str = 'test_strategy',
                      archs: Optional[List[str]] = None,
                      datasets: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (14, 6),
                      ylabel: str = 'Relative EER improvement (%)'):
    """
    Plot relative EER comparisons.
    
    Parameters:
    -----------
    comparison_type : str
        Type of comparison: 'test_strategy' or 'training_strategy'
        - 'test_strategy': Compare test strategies (OS vs SS) for same training strategy
        - 'training_strategy': Compare training strategies (OS-OS vs SS-SS)
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
    datasets : List[str]
        Datasets to include. Default: ['voxceleb']
    figsize : Tuple[int, int]
        Figure size. Default: (14, 6)
    ylabel : str
        Y-axis label. Default: 'Relative EER improvement (%)'
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if datasets is None:
        datasets = ['voxceleb']
    
    if comparison_type == 'training_strategy':
        # Compare OS-OS vs SS-SS
        # Filter for OS and SS training strategies with matching test strategies
        df_os_os = df_repro[(df_repro['train_strategy'] == 'OS') & (df_repro['test_strategy'] == 'OS')].copy()
        df_ss_ss = df_repro[(df_repro['train_strategy'] == 'SS') & (df_repro['test_strategy'] == 'SS')].copy()
        
        # Calculate global max for y-axis alignment
        all_rel_diffs = []
        all_rel_errors = []
        
        for dataset in datasets:
            for arch in archs:
                # Get EER for train=OS,test=OS and train=SS,test=SS
                os_os_row = df_os_os[(df_os_os['dataset'] == dataset) & (df_os_os['architecture'] == arch)]
                ss_ss_row = df_ss_ss[(df_ss_ss['dataset'] == dataset) & (df_ss_ss['architecture'] == arch)]
                
                if len(os_os_row) > 0 and len(ss_ss_row) > 0:
                    eer_os_os = os_os_row['eer_mean'].values[0]
                    eer_ss_ss = ss_ss_row['eer_mean'].values[0]
                    std_os_os = os_os_row['eer_std'].values[0]
                    std_ss_ss = ss_ss_row['eer_std'].values[0]
                    
                    # Relative difference
                    rel_diff = ((eer_os_os - eer_ss_ss) / eer_ss_ss) * 100
                    
                    # Error propagation
                    if eer_ss_ss != 0:
                        term1 = (std_os_os / eer_ss_ss) ** 2
                        term2 = ((eer_os_os - eer_ss_ss) / (eer_ss_ss ** 2) * std_ss_ss) ** 2
                        rel_error = (term1 + term2) ** 0.5 * 100
                    else:
                        rel_error = 0
                    
                    all_rel_diffs.append(abs(rel_diff))
                    all_rel_errors.append(rel_error)
        
        # Calculate global max for y-axis alignment
        global_max = (max(all_rel_diffs) + max(all_rel_errors)) * 1.25 if all_rel_diffs and all_rel_errors else 1.0
        
        # Set up subplots
        fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)
        
        # Handle single dataset case
        if len(datasets) == 1:
            axes = [axes]
        
        # Create bar charts for each dataset
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            
            # Get architectures present in this dataset
            dataset_architectures = [arch for arch in archs if 
                                     arch in df_os_os[df_os_os['dataset'] == dataset]['architecture'].unique()]
            
            # Calculate relative differences and errors
            rel_differences = []
            rel_errors = []
            
            for arch in dataset_architectures:
                os_os_row = df_os_os[(df_os_os['dataset'] == dataset) & (df_os_os['architecture'] == arch)]
                ss_ss_row = df_ss_ss[(df_ss_ss['dataset'] == dataset) & (df_ss_ss['architecture'] == arch)]
                
                if len(os_os_row) > 0 and len(ss_ss_row) > 0:
                    eer_os_os = os_os_row['eer_mean'].values[0]
                    eer_ss_ss = ss_ss_row['eer_mean'].values[0]
                    std_os_os = os_os_row['eer_std'].values[0]
                    std_ss_ss = ss_ss_row['eer_std'].values[0]
                    
                    # Relative difference
                    rel_diff = ((eer_os_os - eer_ss_ss) / eer_ss_ss) * 100
                    
                    # Error propagation
                    if eer_ss_ss != 0:
                        term1 = (std_os_os / eer_ss_ss) ** 2
                        term2 = ((eer_os_os - eer_ss_ss) / (eer_ss_ss ** 2) * std_ss_ss) ** 2
                        rel_error = (term1 + term2) ** 0.5 * 100
                    else:
                        rel_error = 0
                    
                    rel_differences.append(rel_diff)
                    rel_errors.append(rel_error)
                else:
                    rel_differences.append(0)
                    rel_errors.append(0)
            
            # Set up x positions
            n_arch = len(dataset_architectures)
            x = np.arange(n_arch)
            
            # Plot bars with error bars
            ax.bar(x, rel_differences, yerr=rel_errors, capsize=5,
                   color=DEFAULT_GREEN, **BAR_STYLE_SOLID)
            
            # Style the axis
            _style_axis(ax,
                       ylabel=ylabel,
                       xticks=x,
                       xticklabels=dataset_architectures)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-global_max, 0)  # Symmetric around zero
        
        plt.tight_layout()
        
    else:  # comparison_type == 'test_strategy'
        # Compare test strategies for same training strategy
        dataset = datasets[0] if datasets else 'voxceleb'
        train_strategy = 'OS'  # Default for test strategy comparison
        test_strategy_1 = 'OS'
        test_strategy_2 = 'SS'
        
        # Filter for specific dataset and training strategy
        df_filtered = df_repro[(df_repro['dataset'] == dataset) & 
                              (df_repro['train_strategy'] == train_strategy)].copy()
        
        # Get architectures present in filtered data
        filtered_archs = [arch for arch in archs if arch in df_filtered['architecture'].unique()]
        
        # Calculate relative differences and errors
        rel_differences = []
        rel_errors = []
        arch_labels = []
        
        for arch in filtered_archs:
            # Get EER for both test strategies
            arch_data = df_filtered[df_filtered['architecture'] == arch]
            eer_row_1 = arch_data[arch_data['test_strategy'] == test_strategy_1]
            eer_row_2 = arch_data[arch_data['test_strategy'] == test_strategy_2]
            
            if len(eer_row_1) > 0 and len(eer_row_2) > 0:
                eer_1 = eer_row_1['eer_mean'].values[0]
                eer_2 = eer_row_2['eer_mean'].values[0]
                std_1 = eer_row_1['eer_std'].values[0]
                std_2 = eer_row_2['eer_std'].values[0]
                
                # Relative difference: (EER_1 - EER_2) / EER_2 * 100%
                rel_diff = ((eer_1 - eer_2) / eer_2) * 100
                
                # Error propagation
                if eer_2 != 0:
                    term1 = (std_1 / eer_2) ** 2
                    term2 = ((eer_1 - eer_2) / (eer_2 ** 2) * std_2) ** 2
                    rel_error = (term1 + term2) ** 0.5 * 100
                else:
                    rel_error = 0
                
                rel_differences.append(rel_diff)
                rel_errors.append(rel_error)
                arch_labels.append(arch)
            else:
                rel_differences.append(0)
                rel_errors.append(0)
                arch_labels.append(arch)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Set up x positions
        n_arch = len(arch_labels)
        x = np.arange(n_arch)
        
        # Plot bars with error bars
        ax.bar(x, rel_differences, yerr=rel_errors, capsize=5,
               color=DEFAULT_GREEN, **BAR_STYLE_SOLID)
        
        # Style the plot
        _style_axis(ax,
                   ylabel=ylabel,
                   xticks=x,
                   xticklabels=arch_labels)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Zero reference line
        
        plt.tight_layout()

def sst_relative_effects(archs: Optional[List[str]] = None,
                         datasets: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (14, 6),
                         ylabel: str = 'Relative EER change (%)'):
    """
    Plot the relative EER effects of SST and distribution shift, based on the model
    EER = SST_L + SST_T + ARCH + SHIFT with
        SST_L = SST learned during training, SST_T = SST leveraged during test,
        ARCH = architecture, SHIFT = distribution shift:

    EER(OS,OS) = SST_L + SST_T + ARCH
    EER(OS,SS) = SST_L + ARCH + SHIFT
    EER(SS,SS) = ARCH
    EER(SS,OS) = ARCH + SHIFT

    Each term is made relative to the EER it is applied to (negative = lower EER):

    SST cost      = SST_L / ARCH
    SST leveraged = SST_T / (SST_L + ARCH)
    Net SST       = EER(OS,OS) / EER(SS,SS) - 1
    Shift         = SHIFT / ARCH

    Parameters:
    -----------
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
    datasets : List[str]
        Datasets to include, one subplot each. Default: all datasets
    figsize : Tuple[int, int]
        Figure size. Default: (14, 6)
    ylabel : str
        Y-axis label. Default: 'Relative EER change (%)'
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if datasets is None:
        datasets = sorted(df_repro['dataset'].unique())

    indexed = df_repro.set_index(['dataset', 'architecture', 'train_strategy', 'test_strategy'])
    mean = indexed['eer_mean']
    std = indexed['eer_std']

    # Model terms solved from (EER(OS,OS), EER(OS,SS), EER(SS,SS), EER(SS,OS))
    ARCH = lambda oo, os, ss, so: ss
    SHIFT = lambda oo, os, ss, so: so - ss
    SST_L = lambda oo, os, ss, so: os - so
    SST_T = lambda oo, os, ss, so: oo - os + so - ss

    effects = [
        ('SST cost: SST_L / ARCH', '#ff7f0e',
         lambda *eer: SST_L(*eer) / ARCH(*eer)),
        ('SST leveraged: SST_T / (SST_L + ARCH)', '#1f77b4',
         lambda *eer: SST_T(*eer) / (SST_L(*eer) + ARCH(*eer))),
        ('Net SST: EER(OS,OS) / EER(SS,SS) - 1', DEFAULT_GREEN,
         lambda oo, os_, ss, so: oo / ss - 1),
        ('Shift: SHIFT / ARCH', 'lightgray',
         lambda *eer: SHIFT(*eer) / ARCH(*eer)),
    ]
    conditions = [('OS', 'OS'), ('OS', 'SS'), ('SS', 'SS'), ('SS', 'OS')]

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)

    # Handle single dataset case
    if len(datasets) == 1:
        axes = [axes]

    bar_width = 0.8 / len(effects)

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        dataset_architectures = [arch for arch in archs if
                                 arch in df_repro[df_repro['dataset'] == dataset]['architecture'].unique()]
        x = np.arange(len(dataset_architectures))

        for k, (label, color, f) in enumerate(effects):
            values = []
            errors = []
            for arch in dataset_architectures:
                m = np.array([mean[(dataset, arch, *c)] for c in conditions])
                sd = np.array([std[(dataset, arch, *c)] for c in conditions])
                value = f(*m)

                # Error propagation assuming independent EER estimates,
                # with partial derivatives from forward differences
                eps = 1e-6
                gradient = np.array([(f(*(m + eps * np.eye(4)[j])) - value) / eps for j in range(4)])

                values.append(value * 100)
                errors.append(np.sqrt(np.sum((gradient * sd) ** 2)) * 100)

            offset = (k - (len(effects) - 1) / 2) * bar_width
            ax.bar(x + offset, values, bar_width,
                   yerr=errors, capsize=3,
                   color=color, label=label, **BAR_STYLE_SOLID)

        _style_axis(ax,
                   title=f'dataset = {dataset}',
                   ylabel=ylabel if i == 0 else None,
                   xticks=x,
                   xticklabels=dataset_architectures)
        ax.legend(**{**LEGEND_STYLE, 'loc': 'upper left'})
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()


def cross_eer(archs: Optional[List[str]] = None,
              dataset: str = 'voxceleb',
              figsize: Tuple[int, int] = (14, 6),
              ylabel: str = 'EER'):
    """
    Plot the out-of-distribution EERs EER(train=OS, test=SS) and EER(train=SS, test=OS).

    Parameters:
    -----------
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
    dataset : str
        Dataset to plot. Default: 'voxceleb'
    figsize : Tuple[int, int]
        Figure size. Default: (14, 6)
    ylabel : str
        Y-axis label. Default: 'EER'
    """
    if archs is None:
        archs = DEFAULT_ARCHS

    indexed = df_repro.set_index(['dataset', 'architecture', 'train_strategy', 'test_strategy'])
    dataset_architectures = [arch for arch in archs if
                             arch in df_repro[df_repro['dataset'] == dataset]['architecture'].unique()]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(dataset_architectures))
    bar_width = 0.4

    for offset, (train, test), color in [
        (-bar_width / 2, ('OS', 'SS'), DEFAULT_COLORMAP['OS']),
        (bar_width / 2, ('SS', 'OS'), DEFAULT_COLORMAP['SS']),
    ]:
        rows = [indexed.loc[(dataset, arch, train, test)] for arch in dataset_architectures]
        ax.bar(x + offset, [row['eer_mean'] for row in rows], bar_width,
               yerr=[row['eer_std'] for row in rows], capsize=5,
               color=color, label=f'EER(train={train}, test={test})',
               **BAR_STYLE_SOLID)

    _style_axis(ax,
               title=f'dataset = {dataset}',
               ylabel=ylabel,
               xticks=x,
               xticklabels=dataset_architectures)

    plt.tight_layout()


def eer_os_os_decomposition(archs: Optional[List[str]] = None,
                            datasets: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (14, 6),
                            ylabel: str = 'Success Rate \n (1 - Error Rate)'):
    """
    Plot 100 - EER(OS,OS) (higher is better) split into the parts responsible for
    the architecture and SST, based on the model of sst_relative_effects():

    EER(OS,OS) = ARCH + (SST_L + SST_T),  ARCH = EER(SS,SS)

    ARCH part = 100 - ARCH
    SST part  = -(SST_L + SST_T) = EER(SS,SS) - EER(OS,OS)

    The label above each bar gives the SST part relative to the ARCH part.

    Parameters:
    -----------
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
    datasets : List[str]
        Datasets to include, one subplot each. Default: all datasets
    figsize : Tuple[int, int]
        Figure size. Default: (14, 6)
    ylabel : str
        Y-axis label. Default: '100 - EER (%)'
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if datasets is None:
        datasets = sorted(df_repro['dataset'].unique())

    indexed = df_repro.set_index(['dataset', 'architecture', 'train_strategy', 'test_strategy'])
    mean = indexed['eer_mean']
    std = indexed['eer_std']

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)

    # Handle single dataset case
    if len(datasets) == 1:
        axes = [axes]

    bar_width = 0.6

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        dataset_architectures = [arch for arch in archs if
                                 arch in df_repro[df_repro['dataset'] == dataset]['architecture'].unique()]
        x = np.arange(len(dataset_architectures))

        eer_os_os = np.array([mean[(dataset, arch, 'OS', 'OS')] for arch in dataset_architectures])
        arch_part = np.array([mean[(dataset, arch, 'SS', 'SS')] for arch in dataset_architectures])
        arch_improvement = 100 - arch_part
        sst_improvement = arch_part - eer_os_os

        ax.bar(x, arch_improvement, bar_width,
               # yerr=[std[(dataset, arch, 'OS', 'OS')] for arch in dataset_architectures],
               capsize=5,
               color=DEFAULT_COLORMAP['OS'], label='Model',
               **BAR_STYLE_SOLID)
        ax.bar(x, sst_improvement, bar_width, bottom=arch_improvement,
               color=DEFAULT_GREEN, label='SST',
               **BAR_STYLE_SOLID)
        # Offset so it doesn't overlap the EER(OS,OS) error bar
        #ax.errorbar(x, arch_part,
        #            yerr=[std[(dataset, arch, 'SS', 'SS')] for arch in dataset_architectures],
        #            fmt='none', capsize=5, ecolor='black')

        for xi, top, sst, arch in zip(x, np.maximum(arch_improvement, 100 - eer_os_os),
                                      sst_improvement, arch_improvement):
            ax.text(xi, top * 1.0, f'{sst / arch:+.1%}',
                    ha='center', va='bottom', fontsize=9)

        _style_axis(ax,
                   title=f'dataset = {dataset}',
                   ylabel=ylabel if i == 0 else None,
                   xticks=x,
                   xticklabels=dataset_architectures)
        ax.legend(**{**LEGEND_STYLE, 'loc': 'upper right'})
        ax.set_ylim(0, 115)  # Headroom so the legend doesn't cover the labels

    plt.tight_layout()


def _sst_reductions(dataset: str, architectures: List[str]):
    """
    Relative EER reduction achieved by leveraging SST (in %) with its run-to-run
    standard deviation (error propagation assuming independent EER estimates):

    Lower bound = 1 - EER(OS,OS) / EER(SS,SS)  (net SST benefit)
    Upper bound = 1 - EER(OS,OS) / EER(OS,SS)  (includes the distribution shift)

    Returns (lower, lower_std, upper, upper_std) as arrays over architectures.
    """
    indexed = df_repro.set_index(['dataset', 'architecture', 'train_strategy', 'test_strategy'])
    m = lambda train, test: np.array([indexed['eer_mean'][(dataset, arch, train, test)] for arch in architectures])
    s = lambda train, test: np.array([indexed['eer_std'][(dataset, arch, train, test)] for arch in architectures])

    def reduction(a, b):
        ratio = m(*a) / m(*b)
        return (1 - ratio) * 100, ratio * np.sqrt((s(*a) / m(*a)) ** 2 + (s(*b) / m(*b)) ** 2) * 100

    lower, lower_std = reduction(('OS', 'OS'), ('SS', 'SS'))
    upper, upper_std = reduction(('OS', 'OS'), ('OS', 'SS'))
    return lower, lower_std, upper, upper_std


def sst_range(archs: Optional[List[str]] = None,
              datasets: Optional[List[str]] = None,
              figsize: Optional[Tuple[int, int]] = None,
              ylabel: Optional[str] = None,
              normalize: bool = False):
    """
    Plot one bar of 100% per architecture, one panel per dataset, stacking:

    Accuracy without SST     0 to 1 - EER(OS,SS)                  (light green)
    SST part (upper bound)   1 - EER(OS,SS) to 1 - EER(SS,SS)     (lighter orange)
    SST part (lower bound)   1 - EER(SS,SS) to 1 - EER(OS,OS)     (light orange)
    Remaining error          1 - EER(OS,OS) to 100, i.e. EER(OS,OS) (red)

    The orange and red parts together are the EER without SST. The label above each bar gives
    the accuracy 1 - EER(OS,OS).

    With normalize=True, each bar is 100% of the EER without SST and shows the two
    estimates as stacked segments instead.

    Parameters:
    -----------
    archs : List[str]
        Architectures to include. Default: ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
    datasets : List[str]
        Datasets to include. Default: all datasets
    figsize : Tuple[int, int]
        Figure size. Default: (16, 6), or (14, 6) if normalize
    ylabel : str
        Y-axis label. Default: 'Accuracy / EER (%)', or 'SST attribution (%)' if normalize
    normalize : bool
        Whether to show the relative EER reductions as 100% bars instead. Default: False
    """
    if archs is None:
        archs = DEFAULT_ARCHS
    if datasets is None:
        datasets = sorted(df_repro['dataset'].unique())
    if ylabel is None:
        ylabel = 'SST attribution (%)' if normalize else 'Success Rate \n (1 - Error Rate)'

    if normalize:
        fig, axes = plt.subplots(1, len(datasets), figsize=figsize or (14, 6), sharey=True)

        # Handle single dataset case
        if len(datasets) == 1:
            axes = [axes]

        bar_width = 0.6

        for i, dataset in enumerate(datasets):
            ax = axes[i]

            dataset_architectures = [arch for arch in archs if
                                     arch in df_repro[df_repro['dataset'] == dataset]['architecture'].unique()]
            x = np.arange(len(dataset_architectures))
            lower, _, upper, _ = _sst_reductions(dataset, dataset_architectures)

            # Remaining EER, the range in which the SST part may start (light), and
            # the minimal SST part on top
            ax.bar(x, 100 - upper, bar_width,
                   color='whitesmoke', edgecolor='none')
            ax.bar(x, upper - lower, bar_width, bottom=100 - upper,
                   color=DEFAULT_COLORMAP['OS'], **BAR_STYLE_SHADOW)
            ax.bar(x, lower, bar_width, bottom=100 - lower,
                   color=DEFAULT_COLORMAP['OS'], edgecolor='none', alpha=0.8)

            for xi, lo, hi in zip(x, lower, upper):
                ax.text(xi, 101, f'{lo:.1f}%–{hi:.1f}%',
                        ha='center', va='bottom', fontsize=9)

            _style_axis(ax,
                       title=f'dataset = {dataset}',
                       ylabel=ylabel if i == 0 else None,
                       xticks=x,
                       xticklabels=dataset_architectures)
            ax.set_ylim(0, 110)  # Headroom for the labels
            ax.set_yticks(range(0, 101, 20))

        plt.tight_layout()
        return

    mean = df_repro.set_index(['dataset', 'architecture', 'train_strategy', 'test_strategy'])['eer_mean']

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize or (16, 6), sharey=True)

    # Handle single dataset case
    if len(datasets) == 1:
        axes = [axes]

    bar_width = 0.7
    arrow_offset = -0.15  # Arrows (and their labels) left of the bar center

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        dataset_architectures = [arch for arch in archs if
                                 arch in df_repro[df_repro['dataset'] == dataset]['architecture'].unique()]
        x = np.arange(len(dataset_architectures))

        eer = lambda train, test: np.array([mean[(dataset, arch, train, test)] for arch in dataset_architectures])
        accuracy_without_sst = 100 - eer('OS', 'SS')
        accuracy = 100 - eer('OS', 'OS')

        # SST part in accuracy percentage points, from the net benefit to the upper bound
        sst_min = eer('SS', 'SS') - eer('OS', 'OS')
        sst_max = accuracy - accuracy_without_sst

        ax.bar(x, accuracy_without_sst, bar_width,
               color='lightgray', alpha=0.4, edgecolor='none',
               label='Accuracy without SST: 1 - EER(OS,SS)')
        # Light green bar and range from the accuracy without SST up to the accuracy
        ax.bar(x, accuracy - accuracy_without_sst, bar_width, bottom=accuracy_without_sst,
               color=DEFAULT_GREEN, alpha=0.3, edgecolor='none')
        for xi, bottom, top, lo, hi in zip(x, accuracy_without_sst, accuracy, sst_min, sst_max):
            ax.annotate('', xy=(xi + arrow_offset, top), xytext=(xi + arrow_offset, bottom),
                        arrowprops=dict(arrowstyle='->', color=DEFAULT_GREEN, lw=1,
                                        linestyle='--', shrinkA=0, shrinkB=0))
            ax.text(xi + arrow_offset + 0.04, (bottom + top) / 2, f'{hi:.1f}pp',
                    color=DEFAULT_GREEN, ha='left', va='center', fontsize=11)

        # Remaining error EER(OS,OS), filling up to 100%
        ax.bar(x, 100 - accuracy, bar_width, bottom=accuracy,
               color='red', alpha=0.1, edgecolor='none')
        for xi, acc in zip(x, accuracy):
            ax.annotate('', xy=(xi + arrow_offset, 100), xytext=(xi + arrow_offset, acc),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1,
                                        linestyle='--', shrinkA=0, shrinkB=0))
            ax.text(xi + arrow_offset + 0.04, (acc + 100) / 2, f'{100 - round(acc, 1):.1f}pp',
                    color='red', ha='left', va='center', fontsize=11)

        # Boundary between the non-SST features and the SST exploitation
        ax.hlines(accuracy_without_sst, x - bar_width / 2, x + bar_width / 2, color='gray', linewidth=0.5, zorder=3)

        # Model performance: line across the bar and label left of it
        ax.hlines(accuracy, x - bar_width / 2, x + bar_width / 2, color='black', linewidth=1.5, zorder=3)
        for xi, acc in zip(x, accuracy):
            ax.text(xi - bar_width / 2 - 0.03, acc, f'{acc:.1f}%',
                    ha='right', va='center', fontsize=12, rotation=90, fontweight='bold')

        _style_axis(ax,
                   title=f'dataset = {dataset}',
                   ylabel=ylabel if i == 0 else None,
                   xticks=x,
                   xticklabels=dataset_architectures)
        ax.set_ylim(0, 108)  # Headroom for the labels
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.5)  # Room for the label left of the first bar

    # One legend for all panels, below the plot
    for ax in axes:
        ax.get_legend().remove()
    arrow = lambda symbol, color: Line2D([], [], marker=symbol, color=color, linestyle='none', markersize=10)
    fig.legend(handles=[Patch(color=DEFAULT_GREEN, alpha=0.3),
                        Patch(color='red', alpha=0.3),
                        Patch(color='lightgray', alpha=0.4)],
               labels=['SST Feature Range', 'Error Range', 'Exploitation of Non-SST Features'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc='lower center', ncol=3, fontsize=11, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
