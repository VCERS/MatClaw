"""
Basic usage examples for MatClaw SDK.
"""

# Example 1: Simple tool call
from matclaw_sdk import matgl_predict_bandgap

# This structure would come from a file or database
structure_cif = """
data_example
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha    90.0
_cell_angle_beta     90.0
_cell_angle_gamma    90.0
"""

print("Example 1: Predict band gap")
try:
    result = matgl_predict_bandgap(structure=structure_cif)
    print(f"Band gap prediction: {result}")
except Exception as e:
    print(f"Error: {e}")


# Example 2: Multiple tools
from matclaw_sdk import matgl_relax_structure, structure_validator

print("\nExample 2: Relax and validate")
try:
    relaxed = matgl_relax_structure(structure=structure_cif)
    print(f"Relaxation complete")
    
    validated = structure_validator(input_structure=relaxed)
    print(f"Validation result: {validated}")
except Exception as e:
    print(f"Error: {e}")


# Example 3: Configure for different server
from matclaw_sdk import set_config

print("\nExample 3: Configure client")

# Use HTTP to remote server
set_config('http', url='http://example.com:5000')
print("Configured for HTTP")

# Or use stdio (local subprocess)
# set_config('stdio', command='python /path/to/server.py')
# print("Configured for stdio")
