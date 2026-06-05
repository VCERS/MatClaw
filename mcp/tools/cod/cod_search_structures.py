"""
Tool for searching the Crystallography Open Database (COD) for crystal structures.
COD contains ~500k experimentally determined structures from published research,
many of which are not in computational databases like Materials Project.

No API key required — fully open access.

COD REST API reference: https://wiki.crystallography.net/RESTful_API/
"""

from typing import Dict, Any, Optional, List, Annotated
from pydantic import Field
import requests
import logging

logger = logging.getLogger(__name__)


def cod_search_structures(
    formula: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Chemical formula to search for. Accepts any standard formula notation — "
                "with or without spaces. Examples: 'Fe2O3', 'Fe2 O3', 'RbCd4Ga3S9', "
                "'Rb Cd4 Ga3 S9', 'C8H10N4O2'. "
                "Internally converted to COD's Hill format automatically."
            )
        )
    ] = None,

    elements: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "Chemical element symbols that must appear in the formula (el1-el8). "
                "Up to 8 elements. Example: ['Rb', 'Cd', 'Ga', 'S']."
            )
        )
    ] = None,

    exclude_elements: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description=(
                "Chemical element symbols that must NOT appear in the formula (nel1-nel4). "
                "Up to 4 elements. Example: ['Pb', 'Hg']."
            )
        )
    ] = None,

    text: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Keyword search in entry metadata — covers bibliographic info, "
                "compound names, mineral names, etc. "
                "Example: 'chalcogenide', 'quartz', 'perovskite'."
            )
        )
    ] = None,

    cod_ids: Annotated[
        Optional[List[int]],
        Field(
            default=None,
            description=(
                "One or more specific COD IDs to retrieve. "
                "Example: [1000000, 1000001]. "
                "When specified, other search filters are ignored."
            )
        )
    ] = None,

    spacegroup: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Hermann-Mauguin space group symbol or superspace group symbol. "
                "Example: 'P n m a', 'F m -3 m', 'P 1'."
            )
        )
    ] = None,

    space_group_number: Annotated[
        Optional[int],
        Field(
            default=None,
            ge=1,
            le=230,
            description=(
                "Space group number (1-230) as per International Tables Vol. A. "
                "Example: 62 for Pnma, 225 for Fm-3m."
            )
        )
    ] = None,

    year: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Publication year to filter by. Example: 2020."
        )
    ] = None,

    journal: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Journal name filter. Example: 'Journal of Solid State Chemistry', "
                "'Inorganic Chemistry', 'Chemistry of Materials'."
            )
        )
    ] = None,

    doi: Annotated[
        Optional[str],
        Field(
            default=None,
            description="DOI filter. Example: '10.1002/anie.202015857'."
        )
    ] = None,

    volume_min: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Minimum cell volume in Angstrom^3 (vmin)."
        )
    ] = None,

    volume_max: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Maximum cell volume in Angstrom^3 (vmax)."
        )
    ] = None,

    min_z: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Minimum Z value (number of formula units per cell)."
        )
    ] = None,

    max_z: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Maximum Z value (number of formula units per cell)."
        )
    ] = None,

    has_fobs: Annotated[
        Optional[bool],
        Field(
            default=None,
            description=(
                "If True, only return entries that have associated structure factor "
                "data (Fobs/Iobs) in the COD."
            )
        )
    ] = None,

    include_duplicates: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, include entries marked as duplicates of other COD entries. "
                "Default: False (duplicates excluded)."
            )
        )
    ] = False,

    include_errors: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, include entries marked as having errors. "
                "Default: False (entries with errors excluded)."
            )
        )
    ] = False,

    include_theoretical: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, include theoretical (calculated, not experimentally determined) entries. "
                "Default: False (theoretical entries excluded)."
            )
        )
    ] = False,

    include_cifs: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, include the full CIF content in each result (as a string in the 'cif' field). "
                "If False, return only metadata (COD ID, formula, space group) without structures. "
                "Default: False."
            )
        )
    ] = False,

    max_results: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            le=200,
            description="Maximum number of results to return (1-200). Default: 20."
        )
    ] = 20,

    timeout: Annotated[
        int,
        Field(
            default=60,
            ge=5,
            le=300,
            description="Request timeout in seconds. Default: 60."
        )
    ] = 60
) -> Dict[str, Any]:
    """
    Search the Crystallography Open Database (COD) for crystal structures.

    COD contains experimentally determined crystal structures from published
    research — the perfect complement to Materials Project (which covers
    computationally validated structures). Many niche compounds, chalcogenides,
    and experimentally synthesized materials are found here but not in MP.

    Search Methods (combine as needed):
        1. By exact formula: formula="Fe2 O3"
        2. By elements present: elements=['Fe', 'O']
        3. By keyword: text="chalcogenide"
        4. By space group: spacegroup="P n m a" or space_group_number=62
        5. By publication: year=2020, journal="Chemistry of Materials"
        6. By COD IDs: cod_ids=[1000000, 2000000]
        7. By cell volume: volume_min=500, volume_max=1000

    Returns:
        Dictionary containing:
            - success (bool): Whether search succeeded
            - query (dict): Search parameters used
            - count (int): Number of results found
            - total_matching (int): Total matching entries in COD (before max_results)
            - structures (list): List of result dicts, each containing:
                - cod_id (int): COD entry ID
                - formula (str): Chemical formula
                - space_group (str): Space group symbol
                - cell_volume (float): Unit cell volume in A^3
                - year (int): Publication year
                - journal (str): Journal name
                - cif (str): CIF content (if include_cifs=True)
                - url (str): URL to the CIF file on COD
            - error (str): Error message if failed
    """
    api_url = "https://www.crystallography.net/cod/result"
    base_url = "https://www.crystallography.net"

    # Build query parameters
    params: Dict[str, Any] = {"format": "json"}

    # Formula (Hill notation with spaces)
    if formula is not None:
        # Convert standard formula notation to COD's Hill format if needed
        # COD expects elements separated by spaces, e.g. "Fe2 O3" not "Fe2O3"
        # pymatgen's Composition.hill_formula already produces this format
        from pymatgen.core import Composition
        try:
            hill = Composition(formula).hill_formula
            # hill_formula already has spaces, e.g. "Fe2 O3"
            params["formula"] = hill
        except Exception:
            # Fall back to raw input
            params["formula"] = formula

    # Element presence (el1-el8)
    if elements is not None:
        for i, el in enumerate(elements[:8]):
            params[f"el{i + 1}"] = el

    # Element exclusion (nel1-nel4)
    if exclude_elements is not None:
        for i, el in enumerate(exclude_elements[:4]):
            params[f"nel{i + 1}"] = el

    # Text search
    if text is not None:
        params["text"] = text

    # Direct COD IDs
    if cod_ids is not None and len(cod_ids) > 0:
        params["id"] = ",".join(str(cid) for cid in cod_ids)

    # Space group
    if spacegroup is not None:
        params["spacegroup"] = spacegroup
    if space_group_number is not None:
        params["space_group_number"] = space_group_number

    # Publication details
    if year is not None:
        params["year"] = str(year)
    if journal is not None:
        params["journal"] = journal
    if doi is not None:
        params["doi"] = doi

    # Cell volume
    if volume_min is not None:
        params["vmin"] = str(volume_min)
    if volume_max is not None:
        params["vmax"] = str(volume_max)

    # Z value
    if min_z is not None:
        params["minZ"] = str(min_z)
    if max_z is not None:
        params["maxZ"] = str(max_z)

    # Structure factor data
    if has_fobs is not None and has_fobs:
        params["has_fobs"] = "true"

    # Include flags
    if include_duplicates:
        params["include_duplicates"] = "true"
    if include_errors:
        params["include_errors"] = "true"
    if include_theoretical:
        params["include_theoretical"] = "true"

    # Validate that at least one search criterion is provided
    search_keys = [
        "formula", "text", "id",
        "el1", "el2", "el3", "el4", "el5", "el6", "el7", "el8",
        "spacegroup", "space_group_number", "year", "journal", "doi",
        "has_fobs"
    ]
    has_search_criterion = any(k in params and k != "format" for k in search_keys)
    if not has_search_criterion:
        return {
            "success": False,
            "error": (
                "At least one search criterion is required. "
                "Provide a formula, elements, text keyword, COD IDs, or other filter."
            )
        }

    try:
        # Step 1: Search for matching COD IDs
        logger.info(f"Searching COD with params: {params}")
        response = requests.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()

        # Parse JSON response
        try:
            entries = response.json()
        except ValueError:
            return {
                "success": False,
                "error": f"Failed to parse COD response as JSON: {response.text[:200]}"
            }

        total_count = len(entries)

        # Apply max_results limit
        entries = entries[:max_results]

        # Step 2: Build result list
        structures = []
        errors = []

        for entry in entries:
            cod_id = entry.get("file")
            if cod_id is None:
                continue

            try:
                cod_id_int = int(cod_id)
            except (ValueError, TypeError):
                errors.append(f"Invalid COD ID: {cod_id}")
                continue

            # Parse formula from response (format: "- Fe2 O3 -")
            raw_formula = entry.get("formula", "")
            # Strip the "- ... -" wrapper if present
            formula_clean = raw_formula.strip("- ").strip() if raw_formula else ""

            result = {
                "cod_id": cod_id_int,
                "formula": formula_clean,
                "space_group": entry.get("sg", ""),
                "cell_volume": entry.get("vol"),
                "year": entry.get("year"),
                "journal": entry.get("journal"),
                "url": f"{base_url}/cod/{cod_id_int}.cif",
            }

            # Optionally download CIF content
            if include_cifs:
                try:
                    cif_response = requests.get(
                        f"{base_url}/cod/{cod_id_int}.cif",
                        timeout=timeout
                    )
                    cif_response.raise_for_status()
                    result["cif"] = cif_response.text
                except requests.RequestException as e:
                    errors.append(f"Failed to download CIF for COD {cod_id_int}: {str(e)}")
                    result["cif"] = None
            else:
                result["cif"] = None

            structures.append(result)

        return {
            "success": True,
            "query": {
                "formula": formula,
                "elements": elements,
                "text": text,
                "cod_ids": cod_ids,
                "spacegroup": spacegroup,
                "space_group_number": space_group_number,
                "year": year,
                "journal": journal,
                "doi": doi,
                "volume_min": volume_min,
                "volume_max": volume_max,
            },
            "count": len(structures),
            "total_matching": total_count,
            "structures": structures,
            "warnings": errors if errors else None,
            "message": f"Found {total_count} matching entries in COD" +
                       (f" (returned {len(structures)})" if total_count > len(structures) else ""),
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": f"COD request timed out after {timeout} seconds. Try increasing timeout or narrowing search."
        }
    except requests.ConnectionError as e:
        return {
            "success": False,
            "error": f"Failed to connect to COD: {str(e)}"
        }
    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"COD request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }