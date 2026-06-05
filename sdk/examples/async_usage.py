"""
Async usage examples for MatClaw SDK.

Use when you need to call multiple tools in parallel or handle
long-running operations efficiently.
"""

import asyncio
from matclaw_sdk import async_call_tool


async def predict_multiple_bandgaps(structures):
    """
    Predict band gaps for multiple structures in parallel.
    
    Args:
        structures: List of structure CIF strings
    
    Returns:
        List of prediction results
    """
    tasks = []
    for structure in structures:
        task = async_call_tool(
            "matgl_predict_bandgap",
            structure=structure
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results


async def relax_and_validate(structures):
    """
    Relax and validate structures in a pipeline.
    
    Args:
        structures: List of structure CIF strings
    """
    # First relax all structures
    relax_tasks = [
        async_call_tool("matgl_relax_structure", structure=s)
        for s in structures
    ]
    relaxed = await asyncio.gather(*relax_tasks)
    
    # Then validate all relaxed structures
    validate_tasks = [
        async_call_tool("structure_validator", input_structure=r)
        for r in relaxed
    ]
    validated = await asyncio.gather(*validate_tasks)
    
    return validated


async def main():
    """Run examples."""
    print("Async Examples")
    print("=" * 50)
    
    # Example structures
    structures = [
        "data_example\n_cell_length_a 10.0\n...",  # Placeholder
        "data_example2\n_cell_length_a 11.0\n...",  # Placeholder
    ]
    
    print("\nExample 1: Predict multiple band gaps in parallel")
    try:
        results = await predict_multiple_bandgaps(structures)
        for i, result in enumerate(results):
            print(f"Structure {i}: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nExample 2: Relax and validate in pipeline")
    try:
        results = await relax_and_validate(structures)
        for i, result in enumerate(results):
            print(f"Structure {i} validated: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
