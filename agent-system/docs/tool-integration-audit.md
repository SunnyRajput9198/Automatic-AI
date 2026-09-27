# Tool and wrapper map

The main task path uses `PlannerAgent.plan_intent` to choose either a direct answer or an explicit list of wrapper calls. Each tool step names one wrapper; `ExecutorAgent` narrows its OpenAI-compatible function schema to that wrapper before invoking its existing `run(**kwargs)` method. Tool results are validated and synthesized from evidence, with failures carried into the final response. The coordinator and research graph remain in the repository for explicit graph/recovery use, but are no longer invoked speculatively before the main planner.

## User-callable wrappers

| Wrapper | Purpose, callable, required inputs, external service | Use / avoid | Invoking agent |
|---|---|---|---|
| `web_search` (`WebSearchTool.run`) | Broad web results from Wikipedia REST plus DuckDuckGo; `query`, optional `max_results`. | Use for fresh or broad source-backed information. Avoid for GitHub-specific data, weather, papers, and requests answerable directly. | Main `ExecutorAgent`; `ResearcherAgent` as a fallback in specialist flows. |
| `web_fetch` (`WebFetchTool.run`) | Fetches up to 50 KB from a known public HTTPS URL; input `url`. | Use when a source URL is already known and page content is needed. Avoid arbitrary/private URLs and use `web_search` when no URL is known. | Main executor and researcher specialist. |
| `github` (`GitHubTool.run`) | Read-only latest commit or release via `api.github.com`; required `owner`, `repo`, `operation`; optional `branch`. Uses optional `GITHUB_TOKEN` header for private repositories. | Use for repository commits/releases, not generic search. No write operations. | Main executor. |
| `get_weather` (`WeatherTool.run`) | Geocodes then returns current conditions through Open-Meteo; on transient/provider failure retries once, then uses wttr.in as a separately attributed fallback; input `city`. | Use for current weather. Do not use for history or forecasts, and never guess weather. | Main executor. |
| `news_search` (`NewsSearchTool.run`) | NewsAPI.org with DuckDuckGo fallback; `query`, optional `num_results`. NewsAPI key is sent in `X-Api-Key`, never in the URL. | Use for news/current headlines, not general web research when other information is requested. | Main executor and researcher specialist. |
| `semantic_scholar_search` (`SemanticScholarTool.run`) | Academic paper search through Semantic Scholar; `query`, optional `limit`. Returns titles, abstracts, authors, citations, URLs. | Use for scholarly papers and citations; avoid for ordinary current-events search. | Main executor and researcher specialist. |
| `wikipedia_search` (`WikipediaTool.run`) | Encyclopedic lookup through Wikipedia REST; input `query`. | Use for stable factual/encyclopedic background, not current weather or latest commits. | Main executor and researcher specialist. |
| `python_executor` (`RestrictedPythonExecutor.run`) | RestrictedPython computation; input executable `code`; stdout captured. | Use for substantive calculation/data processing. Skip trivial arithmetic and ordinary conversation. | Main executor; Engineer specialist. Registered only when `ENABLE_PYTHON_EXECUTOR` is true in the main executor. |
| `file_read` (`FileReadTool.run`) | Reads shared-workspace file; input `filename`. | Use only when the user asks about a workspace file or the task needs requested file input. Avoid filesystem searches for web questions. | Main executor; Engineer specialist. |
| `file_write` (`FileWriteTool.run`) | Creates/overwrites shared-workspace file; inputs `filename`, `content`. | Use only for an explicit save/create request. Avoid unsolicited artifacts. | Main executor; Engineer and Writer specialists. |
| `file_append` (`FileAppendTool.run`) | Appends to shared-workspace file; `filename`, `content`. | Use only when append behavior is requested. | Main executor; Engineer specialist. |
| `file_list` (`FileListTool.run`) | Lists top-level shared-workspace files; no required input. | Use when the user asks to inspect workspace files. | Main executor; Engineer specialist. |
| `file_delete` (`FileDeleteTool.run`) | Deletes one shared-workspace file; `filename`. | Use only on an explicit delete request. Traversal is rejected by `FileManager`. | Main executor; Engineer specialist. |
| `shell_executor` (`ShellExecutor.run`) | Runs a narrow allowlist in the shared workspace; input `command`. Disabled by default (`ENABLE_SHELL=false`). | Use only for explicit shell operations when file tools do not apply. Chaining, traversal, and recursive/executable `find`/copy/move commands are blocked. | Main executor and Engineer specialist, only when enabled. |

## Other external service clients

- `app.utils.llm` provides provider/model abstraction and raw OpenAI-compatible and LangChain `ChatOpenAI` paths. OpenAI-compatible requests, including embeddings in Qdrant memory, use configurable `OPENAI_BASE_URL`, defaulting to `https://aicredits.in/v1`, and default model `gpt-5-mini`. API keys are environment configured and sanitized from errors.
- `QdrantMemory` stores and recalls session context through Qdrant; it is called by orchestration memory flows, not selected as a user-facing tool.
- SQLAlchemy/Postgres persists tasks and execution results; it is application storage, not a planner-selectable tool.
- `ResearcherAgent` currently uses hard-coded query categories internally when explicitly invoked. It is no longer run before the main planner, preventing its automatic extra Wikipedia/search calls from duplicating the selected tool call.

## Fallback relationships

- `get_weather` retries one transient request and can switch from Open-Meteo to wttr.in. The result records both provider attempts and the source that succeeded.
- `news_search` tries NewsAPI first and falls back to DuckDuckGo; the successful result retains the failed NewsAPI attempt when configured.
- `web_search` already combines Wikipedia REST and DuckDuckGo; if both are empty, it reformulates once through DuckDuckGo.
- A failed `github` API request may fall back to web search only for an explicitly public repository; private or visibility-unknown repository data is never inferred from search.
- Planner-declared alternatives are checked against a compatibility map before execution. Weather and Python failures are not substituted with generic web search because it cannot provide equivalent authoritative data.

## Bounded deep-research loop

- Complex plans carry a typed `ResearchPlan`: objective, dynamically derived subquestions, required capabilities, step dependencies/status, completed steps, unresolved questions, conflicts, iteration count, and stopping reason. Planner output is schema-validated and malformed dependencies/capabilities are retried.
- `ResearchSufficiencyReviewer` compares every subquestion with attributed evidence, relevance, failed sources, and possible conflicts. A plan is not sufficient when any subquestion remains unverified or a conflict remains unresolved. If the reviewer cannot respond, the run retains the collected evidence and reports that review limitation.
- The reviewer can propose a targeted follow-up query. It is validated against read-only research wrappers, deduplicated by wrapper and normalized query, appended to the plan, executed, and reviewed again. The loop has configurable iteration, actual tool-attempt, repeated-step, and time limits (`RESEARCH_MAX_ITERATIONS`, `RESEARCH_MAX_TOOL_CALLS`, `RESEARCH_MAX_REPEAT_STEPS`, `RESEARCH_MAX_SECONDS`).
- Evidence attempts retain source, source type, wrapper, query, extracted output, status/failure type, provider attempts, URL, retrieval timestamp, and the subquestion they serve. Final synthesis also receives unresolved gaps, conflicts, and the stopping reason; retrieved-link validation rejects URLs absent from successful evidence.
- Independent research calls currently run sequentially. They share one request-scoped SQLAlchemy session and mutable task/evidence context; serializing them avoids concurrent session writes and makes evidence IDs/status updates deterministic. Per-step critic calls are skipped for this path because the holistic sufficiency review evaluates relevance after collection.
- The cost tracker records planner, wrapper-selection, reviewer, and synthesis LLM calls, tool attempts, elapsed time, retry counts, and estimated cost where token estimates are available.

## Security boundaries

- `web_fetch` permits HTTPS only, rejects credentials, local/private/reserved destinations, resolves hostnames and rejects non-public answers, and checks every redirect target.
- Shell access is disabled by default. The allowlist excludes `find`, `cp`, and `mv`; command chaining, absolute paths, and traversal paths are rejected. Commands run with `shell=False` in the workspace.
- File operations resolve paths under the shared workspace. Task workspace identifiers are validated and symlink escapes are rejected.
- GitHub tokens use an authorization header and are not logged. NewsAPI keys use a header rather than a query string. Wrapper error logs use exception types/status codes instead of request headers or credential-bearing URLs.

## Routing checks

`tests/test_tool_routing.py` covers direct intent, search, GitHub, weather, Python, both wrappers in a multi-tool plan, and a failed wrapper. `tests/test_tool_security.py` covers private URL/redirect rejection, shell traversal/chaining, and task workspace traversal.
