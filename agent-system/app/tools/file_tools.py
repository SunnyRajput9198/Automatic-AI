import structlog
from typing import Dict, Any

from app.tools.base import Tool, ToolResult
from app.utils.file_manager import FileManager

logger = structlog.get_logger()

class FileReadTool(Tool):
    """Read content from a file in the persistent workspace"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "file_read"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file from the shared workspace. Files persist between tasks."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to read"
                }
            },
            "required": ["filename"]
        }
    
    async def run(self, **kwargs) -> ToolResult:
        filename = kwargs.get("filename", "")
        
        if not filename:
            return ToolResult(
                success=False,
                output="",
                error="Filename is required"
            )
        
        logger.info("file_read_running", filename=filename)
        
        content = self.file_manager.read_file(filename)
        
        if content is None:
            return ToolResult(
                success=False,
                output="",
                error=f"File '{filename}' not found in workspace"
            )
        
        return ToolResult(
            success=True,
            output=content,
            metadata={"filename": filename, "size": len(content)}
        )


class FileWriteTool(Tool):
    """Write content to a file in the persistent workspace"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "file_write"
    
    @property
    def description(self) -> str:
        return "Write content to a file in the shared workspace. File will persist for future tasks."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    
    async def run(self, **kwargs) -> ToolResult:
        filename = kwargs.get("filename", "")
        content = kwargs.get("content", "")
        
        if not filename:
            return ToolResult(
                success=False,
                output="",
                error="Filename is required"
            )
        
        logger.info("file_write_running", filename=filename, size=len(content))
        
        success = self.file_manager.write_file(filename, content)
        
        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write to '{filename}'"
            )
        
        return ToolResult(
            success=True,
            output=f"Successfully wrote {len(content)} characters to {filename}",
            metadata={"filename": filename, "size": len(content)}
        )


class FileListTool(Tool):
    """List all files in the persistent workspace"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "file_list"
    
    @property
    def description(self) -> str:
        return "List all files in the shared workspace that persist between tasks."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def run(self, **kwargs) -> ToolResult:
        logger.info("file_list_running")
        
        files = self.file_manager.list_files()
        
        if not files:
            output = "No files in workspace"
        else:
            output = "\n".join(files)
        
        return ToolResult(
            success=True,
            output=output,
            metadata={"count": len(files), "files": files}
        )


class FileDeleteTool(Tool):
    """Delete a file from the persistent workspace"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "file_delete"
    
    @property
    def description(self) -> str:
        return "Delete a file from the shared workspace."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to delete"
                }
            },
            "required": ["filename"]
        }
    
    async def run(self, **kwargs) -> ToolResult:
        filename = kwargs.get("filename", "")
        
        if not filename:
            return ToolResult(
                success=False,
                output="",
                error="Filename is required"
            )
        
        logger.info("file_delete_running", filename=filename)
        
        success = self.file_manager.delete_file(filename)
        
        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"File '{filename}' not found"
            )
        
        return ToolResult(
            success=True,
            output=f"Successfully deleted {filename}",
            metadata={"filename": filename}
        )