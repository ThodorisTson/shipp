"""
Setup Paths for SHIPP + PyWake Integration
===========================================

This module configures Python paths to enable imports from:
1. SHIPP library (in project root)
2. PyWake experiments (in experiments/PyWake/)

Usage:
------
    from setup_paths import configure_paths
    configure_paths()
    
    # Now imports work:
    from shipp.components import Storage, OpSchedule
    from pywake_windfarm import WindFarmModel

Author: Thodoris
Date: January 2026
"""

import sys
from pathlib import Path


def configure_paths(verbose=True):
    """
    Configure Python paths for SHIPP and PyWake experiments.
    
    This function should be called at the start of any experiment script.
    
    Parameters
    ----------
    verbose : bool
        If True, print path configuration details
        
    Returns
    -------
    paths : dict
        Dictionary with configured paths:
        - 'pywake_dir': Path to PyWake experiments folder
        - 'experiments_dir': Path to experiments folder
        - 'shipp_dir': Path to SHIPP repository
        - 'project_root': Path to project root
    """
    # Get current file location (experiments/PyWake/)
    current_file = Path(__file__).resolve()
    pywake_dir = current_file.parent
    
    # Navigate to parent directories
    experiments_dir = pywake_dir.parent  # experiments/
    project_root = experiments_dir.parent  # SHIPP/ (project root)
    
    # SHIPP is at the project root
    shipp_dir = project_root
    
    # Add paths to sys.path if not already there
    paths_to_add = [
        str(pywake_dir),      # For pywake_windfarm.py, pywake_shipp_integration.py
        str(experiments_dir),  # For other experiments
        str(shipp_dir)        # For SHIPP library
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Store paths
    paths = {
        'pywake_dir': pywake_dir,
        'experiments_dir': experiments_dir,
        'shipp_dir': shipp_dir,
        'project_root': project_root
    }
    
    if verbose:
        print("=" * 70)
        print("Python Paths Configured")
        print("=" * 70)
        print(f"PyWake directory:   {pywake_dir}")
        print(f"Experiments folder: {experiments_dir}")
        print(f"SHIPP directory:    {shipp_dir}")
        print(f"Project root:       {project_root}")
        print("=" * 70)
    
    return paths


def check_shipp_structure():
    """
    Check SHIPP directory structure and report what's available.
    
    Returns
    -------
    info : dict
        Information about SHIPP structure
    """
    paths = configure_paths(verbose=False)
    shipp_dir = paths['shipp_dir']
    
    info = {
        'shipp_exists': False,
        'shipp_folder_exists': False,
        'components_exists': False,
        'examples_exist': False,
        'possible_imports': []
    }
    
    # Check if shipp/ folder exists
    shipp_package = shipp_dir / 'shipp'
    if shipp_package.exists():
        info['shipp_folder_exists'] = True
        info['shipp_exists'] = True
        
        # Check for components.py
        components_file = shipp_package / 'components.py'
        if components_file.exists():
            info['components_exists'] = True
            info['possible_imports'].append('from shipp.components import Storage, OpSchedule')
        
        # Check for __init__.py
        init_file = shipp_package / '__init__.py'
        if init_file.exists():
            info['possible_imports'].append('from shipp import Storage, OpSchedule')
    
    # Check for examples
    examples_dir = shipp_dir / 'examples'
    if examples_dir.exists():
        info['examples_exist'] = True
    
    return info


# Auto-configure when imported (optional - can be disabled)
AUTO_CONFIGURE = True

if AUTO_CONFIGURE:
    PATHS = configure_paths(verbose=False)
    PYWAKE_DIR = PATHS['pywake_dir']
    EXPERIMENTS_DIR = PATHS['experiments_dir']
    SHIPP_DIR = PATHS['shipp_dir']
    PROJECT_ROOT = PATHS['project_root']
