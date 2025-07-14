"""
Simple script to run FHI-aims tests one by one.
This avoids any JSON serialization issues.
"""

from ase.build import bulk, molecule, fcc111
from quacc.recipes.aims.core import static_job, relax_job, ase_relax_job

def run_test(test_num, test_name, test_function):
    """Run a single test with error handling."""
    print(f"\n{'='*50}")
    print(f"Test {test_num}: {test_name}")
    print(f"{'='*50}")
    
    try:
        test_function()
        print(f"✓ Test {test_num} completed successfully")
    except Exception as e:
        print(f"✗ Test {test_num} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

# Test 1: Bulk silicon with default parameters
def test1():
    si = bulk("Si", "diamond", a=5.431)
    results = static_job(si, species_defaults="light")
    print("Completed bulk Si calculation")

# Test 2: Bulk silicon with custom kspacing
def test2():
    si = bulk("Si", "diamond", a=5.431)
    results = static_job(
        si, 
        species_defaults="intermediate", 
        kspacing=0.05
    )
    print("Completed bulk Si calculation with custom kspacing")

# Test 3: Magnetic iron with automatic spin detection
def test3():
    fe = bulk("Fe", "bcc", a=2.856)
    fe.set_initial_magnetic_moments([2.0] * len(fe))
    results = static_job(fe, species_defaults="light")
    print("Completed Fe calculation with automatic spin detection")

# Test 4: Water molecule (aperiodic system)
def test4():
    h2o = molecule("H2O")
    h2o.center(vacuum=5.0)
    # Let the code detect this is a molecule (non-periodic)
    results = static_job(h2o, species_defaults="tight")
    print("Completed H2O molecule calculation")

# Test 5: Geometry optimization with custom parameter
def test5():
    au = fcc111("Au", size=(2, 2, 3), vacuum=10.0)
    # Let the code detect this is a slab (2D periodic)
    results = relax_job(
        au, 
        species_defaults="light",
        sc_accuracy_rho=1e-6,  # Custom parameter
        output_level="MD_light"  # Another custom parameter
    )
    print("Completed Au slab relaxation")

# Test 6: ASE relaxation with fmax=0.01
def test6():
    co = molecule("CO")
    co.center(vacuum=5.0)
    results = ase_relax_job(
        co,
        species_defaults="light",
        opt_params={"fmax": 0.01, "max_steps": 100}  # Fixed: use steps instead of maxstep
    )
    print("Completed CO molecule ASE relaxation")

# Test 7: Cell relaxation
def test7():
    si = bulk("Si", "diamond", a=5.45)  # Slightly strained
    results = relax_job(
        si,
        species_defaults="light",
        relax_cell=True
    )
    print("Completed Si bulk cell relaxation")

# Test 8: Cell relaxation ASE
def test8():
    si = bulk("Si", "diamond", a=5.45)  # Slightly strained
    results = ase_relax_job(
        si,
        species_defaults="light",
        relax_cell=True,
        opt_params={"fmax": 0.01, "max_steps": 100}
    )
    print("Completed Si bulk cell relaxation with ASE")

# Run tests one by one
if __name__ == "__main__":
    import sys
    
    # Check if a specific test number was provided
    if len(sys.argv) > 1:
        test_num = int(sys.argv[1])
        tests = {
            1: ("Bulk Si default", test1),
            2: ("Bulk Si custom kspacing", test2),
            3: ("Fe with magnetism", test3),
            4: ("H2O molecule", test4),
            5: ("Au relaxation", test5),
            6: ("CO ASE relaxation with fmax=0.01", test6),
            7: ("Si cell relaxation", test7),
            8: ("Si cell relaxation ASE", test8)
        }
        
        if test_num in tests:
            run_test(test_num, tests[test_num][0], tests[test_num][1])
        else:
            print(f"Invalid test number: {test_num}")
            print(f"Available tests: {list(tests.keys())}")
    else:
        # No args, run all tests
        run_test(1, "Bulk Si default", test1)
        run_test(2, "Bulk Si custom kspacing", test2)
        run_test(3, "Fe with magnetism", test3)
        run_test(4, "H2O molecule", test4)
        run_test(5, "Au relaxation", test5)
        run_test(6, "CO ASE relaxation with fmax=0.01", test6)
        run_test(7, "Si cell relaxation", test7)
        run_test(8, "Si cell relaxation ASE", test8)