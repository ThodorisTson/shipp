"""
Test Imports for SHIPP + PyWake Integration
============================================

This script tests all required imports to ensure the development
environment is properly configured.

Tests:
------
1. Path configuration
2. SHIPP library imports
3. PyWake library (py_wake) imports
4. Custom PyWake model imports
5. Common dependencies (numpy, pandas, matplotlib)

Run this before starting development work.

Author: Thodoris
Date: January 2026
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_test(name, passed, details=""):
    """Print test result."""
    symbol = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{symbol}{reset} {name}")
    if details:
        print(f"  → {details}")


def test_path_configuration():
    """Test 1: Path configuration."""
    print_header("TEST 1: Path Configuration")
    
    try:
        from setup_paths import configure_paths, PYWAKE_DIR, SHIPP_DIR
        paths = configure_paths(verbose=True)
        
        # Verify directories exist
        pywake_exists = paths['pywake_dir'].exists()
        shipp_exists = paths['shipp_dir'].exists()
        
        print_test("Path configuration", True)
        print_test("PyWake directory exists", pywake_exists, str(paths['pywake_dir']))
        print_test("SHIPP directory exists", shipp_exists, str(paths['shipp_dir']))
        
        return True, paths
        
    except Exception as e:
        print_test("Path configuration", False, str(e))
        return False, None


def test_shipp_imports():
    """Test 2: SHIPP library imports."""
    print_header("TEST 2: SHIPP Library Imports")
    
    success_count = 0
    total_tests = 0
    import_methods = []
    
    # Method 1: Import from shipp.components
    total_tests += 1
    try:
        from shipp.components import Storage, OpSchedule
        print_test("from shipp.components import Storage, OpSchedule", True)
        import_methods.append("Method 1: from shipp.components import Storage, OpSchedule")
        success_count += 1
        storage_class = Storage
        opschedule_class = OpSchedule
    except ImportError as e:
        print_test("from shipp.components import Storage, OpSchedule", False, str(e))
    
    # Method 2: Import from shipp directly
    total_tests += 1
    try:
        from shipp.components import Storage, OpSchedule
        print_test("from shipp import Storage, OpSchedule", True)
        import_methods.append("Method 2: from shipp import Storage, OpSchedule")
        success_count += 1
    except ImportError as e:
        print_test("from shipp import Storage, OpSchedule", False, str(e))
    
    # Method 3: Check what's in shipp module
    total_tests += 1
    try:
        import shipp
        print_test("import shipp", True, f"Location: {shipp.__file__}")
        
        # Check what's available
        available = [item for item in dir(shipp) if not item.startswith('_')]
        if available:
            print(f"  → Available in shipp: {', '.join(available[:10])}")
            if 'Storage' in available:
                import_methods.append("Method 3: Storage available in shipp module")
        success_count += 1
    except ImportError as e:
        print_test("import shipp", False, str(e))
    
    # Try to inspect SHIPP structure
    print("\n  Checking SHIPP structure:")
    try:
        from setup_paths import check_shipp_structure
        info = check_shipp_structure()
        
        if info['shipp_folder_exists']:
            print(f"  → shipp/ folder found")
        if info['components_exists']:
            print(f"  → components.py found")
        if info['examples_exist']:
            print(f"  → examples/ folder found")
        
        if info['possible_imports']:
            print(f"\n  Recommended import methods:")
            for method in info['possible_imports']:
                print(f"    • {method}")
    except Exception as e:
        print(f"  → Could not check structure: {e}")
    
    return success_count > 0, import_methods


def test_pywake_library():
    """Test 3: PyWake library (py_wake) imports."""
    print_header("TEST 3: PyWake Library (py_wake)")
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Basic import
    try:
        import py_wake
        print_test("import py_wake", True, f"Version: {py_wake.__version__}")
        success_count += 1
    except ImportError as e:
        print_test("import py_wake", False, "Not installed - run: pip install py_wake")
        return False
    
    # Test 2: Wind turbines
    try:
        from py_wake.wind_turbines import WindTurbine
        print_test("from py_wake.wind_turbines import WindTurbine", True)
        success_count += 1
    except ImportError as e:
        print_test("from py_wake.wind_turbines import WindTurbine", False, str(e))
    
    # Test 3: Wake models
    try:
        from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
        print_test("Wake model (IEA37SimpleBastankhahGaussian)", True)
        success_count += 1
    except ImportError as e:
        print_test("Wake model", False, str(e))
    
    # Test 4: Site models
    try:
        from py_wake.site import UniformSite
        print_test("from py_wake.site import UniformSite", True)
        success_count += 1
    except ImportError as e:
        print_test("from py_wake.site import UniformSite", False, str(e))
    
    return success_count == total_tests


def test_custom_pywake_models():
    """Test 4: Custom PyWake model imports."""
    print_header("TEST 4: Custom PyWake Models")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: pywake_windfarm.py
    try:
        from pywake_windfarm import WindFarmModel, create_reference_offshore_windfarm
        print_test("from pywake_windfarm import WindFarmModel", True)
        success_count += 1
        
        # Try to instantiate
        try:
            wf = create_reference_offshore_windfarm(n_turbines=5, turbine_mw=5.0)
            print_test("  Create wind farm instance", True, 
                      f"{wf.n_turbines} × {wf.turbine_rating_mw}MW")
        except Exception as e:
            print_test("  Create wind farm instance", False, str(e))
            
    except ImportError as e:
        print_test("from pywake_windfarm import WindFarmModel", False, str(e))
    
    # Test 2: pywake_shipp_integration.py
    try:
        from pywake_shipp_integration import prepare_wind_data_for_shipp
        print_test("from pywake_shipp_integration import prepare_wind_data_for_shipp", True)
        success_count += 1
    except ImportError as e:
        print_test("from pywake_shipp_integration import prepare_wind_data_for_shipp", False, str(e))
    
    # Test 3: Check files exist
    try:
        from setup_paths import PYWAKE_DIR
        
        wf_file = PYWAKE_DIR / 'pywake_windfarm.py'
        integration_file = PYWAKE_DIR / 'pywake_shipp_integration.py'
        
        print_test("pywake_windfarm.py exists", wf_file.exists(), str(wf_file))
        print_test("pywake_shipp_integration.py exists", integration_file.exists(), 
                  str(integration_file))
        
        if wf_file.exists() and integration_file.exists():
            success_count += 1
            
    except Exception as e:
        print_test("Check PyWake files", False, str(e))
    
    return success_count >= 2


def test_dependencies():
    """Test 5: Common dependencies."""
    print_header("TEST 5: Common Dependencies")
    
    success_count = 0
    total_tests = 3
    
    # NumPy
    try:
        import numpy as np
        print_test("numpy", True, f"Version: {np.__version__}")
        success_count += 1
    except ImportError:
        print_test("numpy", False, "Install: pip install numpy")
    
    # Pandas
    try:
        import pandas as pd
        print_test("pandas", True, f"Version: {pd.__version__}")
        success_count += 1
    except ImportError:
        print_test("pandas", False, "Install: pip install pandas")
    
    # Matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        print_test("matplotlib", True, f"Version: {matplotlib.__version__}")
        success_count += 1
    except ImportError:
        print_test("matplotlib", False, "Install: pip install matplotlib")
    
    return success_count == total_tests


def run_all_tests():
    """Run all tests and provide summary."""
    print_header("SHIPP + PyWake Import Tests")
    print("Testing development environment setup...")
    
    results = {}
    
    # Run tests
    results['paths'], paths = test_path_configuration()
    results['shipp'], shipp_methods = test_shipp_imports()
    results['pywake_lib'] = test_pywake_library()
    results['custom_models'] = test_custom_pywake_models()
    results['dependencies'] = test_dependencies()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        symbol = "✓" if passed_test else "✗"
        color = "\033[92m" if passed_test else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {test_name.replace('_', ' ').title()}")
    
    print("\n" + "=" * 70)
    print(f"Result: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready for development.")
        print("\nYou can now run:")
        print("  python pywake_windfarm.py          # Test wind farm model")
        print("  python pywake_shipp_integration.py # Test SHIPP integration")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        # Provide specific guidance
        if not results['pywake_lib']:
            print("\n→ Install PyWake: pip install py_wake")
        
        if not results['shipp']:
            print("\n→ Check SHIPP installation:")
            print("  - Verify SHIPP folder exists at project root")
            print("  - Check shipp/components.py exists")
            if shipp_methods:
                print(f"  - Try this import: {shipp_methods[0]}")
        
        if not results['custom_models']:
            print("\n→ Ensure PyWake model files are in experiments/PyWake/:")
            print("  - pywake_windfarm.py")
            print("  - pywake_shipp_integration.py")
    
    print("=" * 70)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
