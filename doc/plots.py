from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
DEFAULT_ARCHS = ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
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
        archs = ['CNN', 'RNN', 'ResNet', 'F-ResNet']
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
        archs = ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
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
        archs = ['CNN', 'RNN', 'ResNet', 'F-ResNet', 'Conformer']
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