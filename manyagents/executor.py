"""Execution skeleton for calling ManyLatents workflows."""
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)


class ManyLatentsExecutor:
    """Handles execution of ManyLatents workflows via subprocess calls."""
    
    def __init__(self, timeout_s: int = 300, dry_run: bool = False):
        self.timeout_s = timeout_s
        self.dry_run = dry_run
    
    def execute_workflow(self, workflow_name: str, overrides: List[str] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute a ManyLatents workflow with given overrides.
        
        Args:
            workflow_name: Name of the workflow to run
            overrides: List of Hydra overrides (e.g., ['data=swissroll', 'seed=42'])
            output_dir: Directory to save outputs
            
        Returns:
            Dict with execution results
        """
        if overrides is None:
            overrides = []
        
        # Build command
        cmd = ["python", "-m", "manylatents.main"]
        
        # Add workflow specification (this will depend on manylatents structure)
        if workflow_name:
            cmd.extend(["workflow=" + workflow_name])
        
        # Add overrides
        cmd.extend(overrides)
        
        # Add output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend([f"hydra.run.dir={output_dir}"])
        
        log.info(f"Executing command: {' '.join(cmd)}")
        
        if self.dry_run:
            log.info("DRY RUN: Command would be executed")
            return {
                "cmd": cmd,
                "returncode": 0,
                "stdout": "DRY RUN - no actual execution",
                "stderr": "",
                "success": True
            }
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=Path.cwd()
            )
            
            success = result.returncode == 0
            
            if success:
                log.info("ManyLatents execution completed successfully")
            else:
                log.error(f"ManyLatents execution failed with code {result.returncode}")
                log.error(f"stderr: {result.stderr}")
            
            return {
                "cmd": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": success
            }
            
        except subprocess.TimeoutExpired:
            log.error(f"ManyLatents execution timed out after {self.timeout_s}s")
            return {
                "cmd": cmd,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Timeout after {self.timeout_s}s",
                "success": False
            }
        except Exception as e:
            log.error(f"Error executing ManyLatents: {e}")
            return {
                "cmd": cmd,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def execute_sequential_workflows(self, workflow_list: List[str], base_overrides: List[str] = None) -> List[Dict[str, Any]]:
        """Execute a sequence of workflows.
        
        Args:
            workflow_list: List of workflow names to execute sequentially
            base_overrides: Base overrides to apply to all workflows
            
        Returns:
            List of execution results
        """
        if base_overrides is None:
            base_overrides = []
        
        results = []
        
        for i, workflow in enumerate(workflow_list):
            log.info(f"Executing workflow {i+1}/{len(workflow_list)}: {workflow}")
            
            result = self.execute_workflow(
                workflow_name=workflow,
                overrides=base_overrides.copy(),
                output_dir=Path("outputs") / f"workflow_{i:02d}_{workflow}"
            )
            
            results.append({
                "workflow": workflow,
                "index": i,
                **result
            })
            
            # Stop on failure if desired
            if not result["success"]:
                log.warning(f"Workflow {workflow} failed, continuing with next...")
        
        return results