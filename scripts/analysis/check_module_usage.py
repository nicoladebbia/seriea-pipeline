"""Check which feature modules are imported but never called in build.py"""
import re
from pathlib import Path

build_py = Path(str(Path(__file__).parent.parent.parent) + '/features/build.py')
content = build_py.read_text()

# Extract imports
import_pattern = r'from features\.(\w+) import ([\w\s,]+)'
imports = re.findall(import_pattern, content)

print("=" * 80)
print("FEATURE MODULE IMPORT vs USAGE ANALYSIS")
print("=" * 80)

# Track all imported functions
imported_functions = {}
for module, functions in imports:
    for func in functions.split(','):
        func = func.strip()
        if func:
            if module not in imported_functions:
                imported_functions[module] = []
            imported_functions[module].append(func)

print(f"\nTotal imported modules: {len(imported_functions)}")
print(f"Total imported functions: {sum(len(v) for v in imported_functions.values())}")

# Check usage for each function
unused_functions = []
for module, functions in sorted(imported_functions.items()):
    print(f"\n{module}:")
    for func in functions:
        # Count occurrences (excluding the import line)
        usage_count = len([m for m in re.finditer(rf'\b{func}\(', content)])
        if usage_count == 0:
            unused_functions.append((module, func))
            print(f"  [{func}] - UNUSED/DEAD CODE")
        else:
            print(f"  [{func}] - called {usage_count} time(s)")

if unused_functions:
    print("\n" + "=" * 80)
    print(f"DEAD CODE SUMMARY - {len(unused_functions)} unused imports")
    print("=" * 80)
    for module, func in unused_functions:
        print(f"  features.{module}.{func}")
