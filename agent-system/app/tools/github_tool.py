"""Read-only GitHub repository metadata and latest commit/release lookup."""
import re
from datetime import datetime, timezone
from typing import Any, Dict

import httpx
import structlog

from app.core.config import settings
from app.tools.base import Tool, ToolResult, classify_tool_failure

logger = structlog.get_logger()
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


class GitHubTool(Tool):
    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return ("Read public GitHub repository metadata: latest commit on the default or named branch, "
                "or latest release. Inputs owner, repo, operation (latest_commit/latest_release), "
                "and optional branch. Use for GitHub repository/release questions instead of web search. "
                "Requires GITHUB_TOKEN only for private repositories; never changes repository state.")

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {
            "owner": {"type": "string", "description": "GitHub account or organization"},
            "repo": {"type": "string", "description": "Repository name"},
            "operation": {"type": "string", "enum": ["latest_commit", "latest_release"]},
            "branch": {"type": "string", "description": "Optional branch for latest_commit"},
        }, "required": ["owner", "repo", "operation"]}

    async def run(self, **kwargs: Any) -> ToolResult:
        owner = str(kwargs.get("owner", "")).strip()
        repo = str(kwargs.get("repo", "")).strip()
        operation = kwargs.get("operation", "latest_commit")
        branch = str(kwargs.get("branch", "")).strip()
        if not _SEGMENT.fullmatch(owner) or owner in {".", ".."} or not _SEGMENT.fullmatch(repo) or repo in {".", ".."}:
            return ToolResult(success=False, output="", error="A valid GitHub owner and repository are required")
        if operation not in {"latest_commit", "latest_release"}:
            return ToolResult(success=False, output="", error="Operation must be latest_commit or latest_release")
        if branch and (len(branch) > 250 or branch.startswith("/") or ".." in branch or "\\" in branch):
            return ToolResult(success=False, output="", error="Invalid branch name")
        path = f"https://api.github.com/repos/{owner}/{repo}/"
        endpoint = path + ("commits" if operation == "latest_commit" else "releases/latest")
        params = {"sha": branch, "per_page": 1} if operation == "latest_commit" else None
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        token = (getattr(settings, "GITHUB_TOKEN", None) or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=False) as client:
                response = await client.get(endpoint, params=params, headers=headers)
                # An expired optional token should not prevent read-only access
                # to a public repository. Retry once anonymously, not endlessly.
                if response.status_code == 401 and token:
                    anonymous_headers = {k: v for k, v in headers.items() if k != "Authorization"}
                    response = await client.get(endpoint, params=params, headers=anonymous_headers)
            if response.status_code == 404:
                return ToolResult(success=False, output="", error="GitHub repository or requested release was not found",
                                  metadata={"tool_name": self.name, "source": "GitHub REST API", "failure_type": "EMPTY_RESULT"})
            if response.status_code in (401, 403) and response.headers.get("X-RateLimit-Remaining") != "0":
                return ToolResult(success=False, output="", error="GitHub authentication failed",
                                  metadata={"tool_name": self.name, "source": "GitHub REST API", "failure_type": "AUTHENTICATION_FAILURE"})
            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                return ToolResult(success=False, output="", error="GitHub API rate limit reached",
                                  metadata={"tool_name": self.name, "source": "GitHub REST API", "failure_type": "RATE_LIMIT"})
            if response.status_code != 200:
                return ToolResult(success=False, output="", error=f"GitHub API returned HTTP {response.status_code}",
                                  metadata={"tool_name": self.name, "source": "GitHub REST API",
                                            "failure_type": classify_tool_failure(None, response.status_code)})
            payload = response.json()
            item = payload[0] if operation == "latest_commit" and payload else payload
            if not item:
                return ToolResult(success=False, output="", error="No commits were returned for the repository",
                                  metadata={"tool_name": self.name, "source": "GitHub REST API", "failure_type": "EMPTY_RESULT"})
            if operation == "latest_commit":
                commit = item.get("commit", {})
                author = commit.get("author", {})
                message = (commit.get("message") or "").splitlines()[0]
                sha = item.get("sha", "")
                url = item.get("html_url", "")
                output = f"Latest commit in {owner}/{repo}: {message}\nSHA: {sha}\nAuthor: {author.get('name', 'Unknown')}\nDate: {author.get('date', 'Unknown')}\nURL: {url}"
            else:
                url = item.get("html_url", "")
                output = f"Latest release in {owner}/{repo}: {item.get('name') or item.get('tag_name', 'Unknown')}\nTag: {item.get('tag_name', 'Unknown')}\nPublished: {item.get('published_at', 'Unknown')}\nURL: {item.get('html_url', '')}\nNotes: {(item.get('body') or '')[:1500]}"
            return ToolResult(success=True, output=output, metadata={
                "tool_name": self.name, "source": "GitHub REST API", "owner": owner,
                "repo": repo, "operation": operation, "query": f"{owner}/{repo} {operation}",
                "source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
        except (httpx.HTTPError, ValueError) as exc:
            failure_type = classify_tool_failure(type(exc).__name__)
            logger.warning("github_tool_failed", failure_type=failure_type)
            return ToolResult(success=False, output="", error=f"GitHub lookup failed ({failure_type})",
                              metadata={"tool_name": self.name, "source": "GitHub REST API", "failure_type": failure_type})
