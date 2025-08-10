"""Main entrypoint for ManyAgents with Hydra configuration."""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Set up logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main ManyAgents entrypoint with Hydra configuration."""
    log.info("Starting ManyAgents...")
    log.info(f"Config: {cfg}")
    
    try:
        # Test ManyLatents import
        log.info("Testing ManyLatents import...")
        import manylatents
        log.info(f"Successfully imported ManyLatents: {manylatents}")
        
        # Test basic ManyLatents functionality if available
        if hasattr(manylatents, '__version__'):
            log.info(f"ManyLatents version: {manylatents.__version__}")
        
        # Simple test run based on config
        if cfg.get("test_run", False):
            log.info("Running test analysis...")
            # This will be expanded to call manylatents.main with overrides
            log.info("Test analysis completed (placeholder)")
        
        log.info("ManyAgents completed successfully")
        
    except ImportError as e:
        log.error(f"Failed to import ManyLatents: {e}")
        log.error("Make sure ManyLatents is installed: pip install -e .")
        raise
    except Exception as e:
        log.error(f"Error in ManyAgents: {e}")
        raise


if __name__ == "__main__":
    main()