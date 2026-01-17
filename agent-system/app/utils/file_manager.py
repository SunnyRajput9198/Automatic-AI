import os
import shutil
import structlog
from typing import List, Optional
from pathlib import Path

logger = structlog.get_logger()

class FileManager:
    """
    Manages persistent file storage across tasks.
    
    FEATURES:
    - Files survive between tasks
    - Workspace isolation per task (optional)
    - Shared workspace for all tasks
    - File cleanup and management
    """
    
    def __init__(self, base_dir: str = "/app/workspace"):
        self.base_dir = Path(base_dir)
        self.shared_workspace = self.base_dir / "shared"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.shared_workspace.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "file_manager_initialized",
            base_dir=str(self.base_dir),
            shared_workspace=str(self.shared_workspace)
        )
    
    def get_task_workspace(self, task_id: str) -> Path:
        """Get or create workspace for specific task"""
        workspace = self.base_dir / task_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace
    
    def get_shared_workspace(self) -> Path:
        """Get shared workspace accessible to all tasks"""
        return self.shared_workspace
    
    def list_files(self, workspace: str = "shared") -> List[str]:
        """List all files in a workspace"""
        if workspace == "shared":
            target = self.shared_workspace
        else:
            target = self.base_dir / workspace
        
        if not target.exists():
            return []
        
        files = []
        for item in target.iterdir():
            if item.is_file():
                files.append(item.name)
        
        logger.info("file_manager_listed", workspace=workspace, count=len(files))
        return files
    
    def read_file(self, filename: str, workspace: str = "shared") -> Optional[str]:
        """Read file content"""
        if workspace == "shared":
            filepath = self.shared_workspace / filename
        else:
            filepath = self.base_dir / workspace / filename
        
        if not filepath.exists():
            logger.warning("file_manager_not_found", filepath=str(filepath))
            return None
        
        try:
            content = filepath.read_text()
            logger.info(
                "file_manager_read",
                filename=filename,
                size=len(content)
            )
            return content
        except Exception as e:
            logger.error("file_manager_read_error", error=str(e))
            return None
    
    def write_file(
        self,
        filename: str,
        content: str,
        workspace: str = "shared"
    ) -> bool:
        """Write content to file"""
        if workspace == "shared":
            filepath = self.shared_workspace / filename
        else:
            workspace_dir = self.base_dir / workspace
            workspace_dir.mkdir(parents=True, exist_ok=True)
            filepath = workspace_dir / filename
        
        try:
            filepath.write_text(content)
            logger.info(
                "file_manager_written",
                filename=filename,
                size=len(content)
            )
            return True
        except Exception as e:
            logger.error("file_manager_write_error", error=str(e))
            return False
    
    def delete_file(self, filename: str, workspace: str = "shared") -> bool:
        """Delete a file"""
        if workspace == "shared":
            filepath = self.shared_workspace / filename
        else:
            filepath = self.base_dir / workspace / filename
        
        if not filepath.exists():
            logger.warning("file_manager_delete_not_found", filepath=str(filepath))
            return False
        
        try:
            filepath.unlink()
            logger.info("file_manager_deleted", filename=filename)
            return True
        except Exception as e:
            logger.error("file_manager_delete_error", error=str(e))
            return False
    
    def cleanup_task_workspace(self, task_id: str) -> bool:
        """Remove all files from task-specific workspace"""
        workspace = self.base_dir / task_id
        
        if not workspace.exists():
            return True
        
        try:
            shutil.rmtree(workspace)
            logger.info("file_manager_cleaned", task_id=task_id)
            return True
        except Exception as e:
            logger.error("file_manager_cleanup_error", error=str(e))
            return False
    
    def get_file_info(self, filename: str, workspace: str = "shared") -> Optional[dict]:
        """Get file metadata"""
        if workspace == "shared":
            filepath = self.shared_workspace / filename
        else:
            filepath = self.base_dir / workspace / filename
        
        if not filepath.exists():
            return None
        
        stat = filepath.stat()
        return {
            "name": filename,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "path": str(filepath)
        }