"""Find feature modules that exist but are never used in build.py"""
from pathlib import Path
import re

features_dir = Path(str(Path(__file__).parent.parent.parent) + '/features')
build_py = features_dir / 'build.py'

# Get all feature modules
all_modules = [f.stem for f in features_dir.glob('*.py')
               if f.stem not in ['__init__', 'build', 'test_features']]

# Read build.py
build_content = build_py.read_text()

# Find imported modules
import_pattern = r'from features\.(\w+) import'
imported_modules = set(re.findall(import_pattern, build_content))

print("=" * 80)
print("FEATURE MODULE USAGE AUDIT")
print("=" * 80)

print(f"\nTotal feature modules: {len(all_modules)}")
print(f"Imported in build.py: {len(imported_modules)}")

unused = sorted(set(all_modules) - imported_modules)
if unused:
    print(f"\nUNUSED MODULES ({len(unused)}):")
    print("These exist but are NEVER imported in build.py:")
    for module in unused:
        module_path = features_dir / f"{module}.py"
        size_kb = module_path.stat().st_size / 1024
        print(f"  - {module}.py ({size_kb:.1f} KB)")

# Check deprecated modules
deprecated_dir = features_dir / '_deprecated'
if deprecated_dir.exists():
    deprecated_modules = list(deprecated_dir.glob('*.py'))
    if deprecated_modules:
        print(f"\nDEPRECATED MODULES ({len(deprecated_modules)}):")
        for module in deprecated_modules:
            size_kb = module.stat().st_size / 1024
            print(f"  - {module.name} ({size_kb:.1f} KB)")

print("\n" + "=" * 80)
print("ACTIVE MODULES (imported in build.py)")
print("=" * 80)
for module in sorted(imported_modules):
    module_path = features_dir / f"{module}.py"
    if module_path.exists():
        size_kb = module_path.stat().st_size / 1024
        print(f"  {module}.py ({size_kb:.1f} KB)")
