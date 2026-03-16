"""
MatClaw MCP Server
"""

from dotenv import load_dotenv
import logging
from mcp.server.fastmcp import FastMCP
from tools.data_retrieval import (
    pubchem_search_compounds, 
    pubchem_get_compound_properties, 
    mp_search_materials, 
    mp_get_material_properties, 
    mp_get_detailed_property_data,
    mp_search_recipe
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP(name="matclaw-mcp-server")

# Add tools
mcp.tool()(pubchem_search_compounds)
mcp.tool()(pubchem_get_compound_properties)
mcp.tool()(mp_search_materials)
mcp.tool()(mp_get_material_properties)
mcp.tool()(mp_get_detailed_property_data)
mcp.tool()(mp_search_recipe)


if __name__ == "__main__":
    logger.info("Starting MatClaw MCP Server")
    mcp.run()