import os
import requests
import json
import time
import pathlib
import argparse
import uvicorn
import threading
import atexit
import langid
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

# 引入 redis
try:
    import redis
except ImportError:
    print("Redis is not installed. Please run: pip install redis")
    exit(1)

from fastapi import FastAPI
from pydantic import BaseModel


# --- Helper: Redis to JSON Persistence Manager ---

class RedisCacheManager:
    """
    Manages the persistence of Redis cache to a JSON file.
    It performs periodic background saves and a final save on exit.
    """

    def __init__(self, redis_client: redis.Redis, cache_file: str, sync_interval_seconds: float = 3600.0):
        if not redis_client:
            raise ValueError("A valid Redis client must be provided.")
        
        self.redis_client = redis_client
        self.cache_file = pathlib.Path(cache_file)
        self.sync_interval = sync_interval_seconds
        self._stop_event = threading.Event()
        
        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Start background sync thread
        self._sync_thread = threading.Thread(target=self._periodic_sync, daemon=True)
        self._sync_thread.start()
        
        # Register a final sync on program exit
        atexit.register(self.stop_and_sync)
        print(f"RedisCacheManager initialized. Will sync to '{self.cache_file}' every {self.sync_interval} seconds.")

    def _periodic_sync(self):
        """Background thread worker for periodic synchronization."""
        while not self._stop_event.wait(self.sync_interval):
            print("Performing periodic Redis to JSON sync...")
            self.sync_to_json()

    def sync_to_json(self):
        """
        Dumps all data from Redis to the JSON file in an atomic and safe manner.
        """
        print("Starting sync from Redis to JSON...")
        start_time = time.time()
        
        try:
            # Use SCAN_ITER to safely iterate over all keys without blocking Redis
            all_keys = [key for key in self.redis_client.scan_iter("*")]
            if not all_keys:
                print("No keys in Redis to sync.")
                return

            # Use MGET for efficient batch retrieval of values
            all_values = self.redis_client.mget(all_keys)
            cache_data = {key: value for key, value in zip(all_keys, all_values) if value is not None}

            # Atomic write: first write to a temporary file, then replace the original
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            temp_file.replace(self.cache_file)

            end_time = time.time()
            print(f"Successfully synced {len(cache_data)} entries to '{self.cache_file}' in {end_time - start_time:.2f} seconds.")

        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis during sync: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during sync: {e}")

    def load_from_json(self):
        """
        Loads data from the JSON file into Redis. Used for "cache warming".
        """
        if not self.cache_file.exists():
            print(f"Cache file '{self.cache_file}' not found. Skipping cache warming.")
            return

        print(f"Warming up cache from '{self.cache_file}'...")
        start_time = time.time()
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Use a pipeline for efficient bulk insertion
            pipeline = self.redis_client.pipeline()
            for key, value in cache_data.items():
                pipeline.set(key, value)
            pipeline.execute()
            
            end_time = time.time()
            print(f"Successfully loaded {len(cache_data)} entries into Redis in {end_time - start_time:.2f} seconds.")

        except json.JSONDecodeError:
            print(f"Cache file '{self.cache_file}' is corrupted. Skipping cache warming.")
        except Exception as e:
            print(f"An error occurred during cache warming: {e}")

    def stop_and_sync(self):
        """Stops the periodic sync and performs one final sync."""
        print("Stopping Redis cache manager and performing final sync...")
        if self._sync_thread.is_alive():
            self._stop_event.set()
            # Perform one last sync immediately
            self.sync_to_json()
            self._sync_thread.join(timeout=10) # Wait for the thread to terminate


# --- Core Search Engine Logic ---

@dataclass
class SearchOutcome:
    """单次查询的结构化结果：状态 + 模型可见文本 + 旁路元数据。

    ``execute`` 仍返回 ``text``（模型可见字符串，兼容现有返回形状）；
    本结构通过 ``get_last_outcome`` 暴露，用于区分失败类型与保留 URL/标题/排名旁路。
    """

    status: str                        # success/not_found/rate_limit/network/server/invalid/auth/empty_query
    text: str                          # 模型可见字符串
    items: List[Dict[str, Any]]        # 旁路：[{"rank","title","url","description"}, ...]
    error_code: Optional[str] = None   # 结构化错误码（如 HTTP 状态码字符串）

    @property
    def ok(self) -> bool:
        return self.status == "success"


class OnlineSearchEngine:
    """
    Bing search tool using Brightdata API, with a high-performance Redis cache
    and JSON file persistence.
    """

    def __init__(
        self,
        api_key: str,
        redis_client: redis.Redis,
        zone: str = "serp_api1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        tool_retry_count: int = 3,
        provider: str = "brightdata",
        cache_version: str = "v1",
    ):
        """
        Initialize the search tool.

        Args:
            api_key: Brightdata API key.
            redis_client: An initialized Redis client instance.
            zone: Brightdata zone name.
            max_results: Maximum number of search results to return.
            result_length: Maximum length of each result snippet.
            location: Country code for search localization.
            tool_retry_count: Number of retries for a failed search.
            provider: Search provider name, used in the cache key namespace.
            cache_version: Cache format version, used in the cache key namespace.
        """
        # API configuration
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._tool_retry_count = tool_retry_count
        self._provider = provider
        self._cache_version = cache_version

        # Redis cache client
        self.redis_client = redis_client

        # 预算口径统计：逻辑搜索、缓存命中、真实 API 请求（含重试）三者分开计数。
        self._logical_queries = 0
        self._cache_hits = 0
        self._api_requests = 0

        # 最近一次查询的结构化结果（旁路元数据 / 失败分类）。
        self._last_outcome: Optional[SearchOutcome] = None

    def _cache_key(self, query: str) -> str:
        """构造带命名空间的缓存键，换 provider/zone/版本/topk 后不命中旧缓存。"""
        return (
            f"{self._provider}:{self._zone}:{self._cache_version}:"
            f"{self._max_results}:{self._result_length}:{query}"
        )

    def get_stats(self) -> Dict[str, int]:
        """返回预算口径统计：逻辑搜索、缓存命中、真实 API 请求（含重试）分开。"""
        return {
            "logical_queries": self._logical_queries,
            "cache_hits": self._cache_hits,
            "api_requests": self._api_requests,
        }

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _make_request(self, query: str, timeout: int) -> requests.Response:
        """Sends a request to the Brightdata API."""
        lang_code, _ = langid.classify(query)
        mkt, setLang = ("zh-CN", "zh") if lang_code == 'zh' else ("en-US", "en")
        
        encoded_query = urlencode({"q": query, "mkt": mkt, "setLang": setLang})
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"zone": self._zone, "url": target_url, "format": "raw"}

        return requests.post("https://api.brightdata.com/request", headers=headers, json=payload, timeout=timeout)

    def execute(self, query: str, timeout: int = 60) -> str:
        """
        Executes a Bing search query, using Redis for caching.

        Args:
            query: The search query string.
            timeout: API request timeout in seconds.

        Returns:
            A formatted string of search results (model-visible).
        """
        outcome = self._execute_structured(query, timeout)
        self._last_outcome = outcome
        return outcome.text

    def get_last_outcome(self) -> Optional[SearchOutcome]:
        """返回最近一次查询的结构化结果（状态 / 文本 / 旁路元数据）。"""
        return self._last_outcome

    @staticmethod
    def _classify_http_error(status_code: Optional[int]) -> str:
        """把 HTTP 状态码归类为结构化 status（限流 / 鉴权 / 服务端 / 其他）。"""
        if status_code in (401, 403):
            return "auth"
        if status_code == 429:
            return "rate_limit"
        if status_code is not None and status_code >= 500:
            return "server"
        return "invalid"

    def _execute_structured(self, query: str, timeout: int = 60) -> SearchOutcome:
        """执行一次查询，返回结构化结果；不改变模型可见返回形状。"""
        query = query.replace('"', '').strip()
        if not query:
            return SearchOutcome(status="empty_query", text="Empty query provided.", items=[])

        # 1. Check Redis cache first
        try:
            cached_result = self.redis_client.get(self._cache_key(query))
            if cached_result:
                self._cache_hits += 1
                print(f"Cache hit for query: '{query}'")
                # 缓存只存了模型可见文本，旁路元数据不可回放，故 items 为空。
                return SearchOutcome(status="success", text=cached_result, items=[])
        except redis.exceptions.ConnectionError as e:
            print(f"Warning: Could not connect to Redis. Proceeding without cache. Error: {e}")
        except Exception as e:
            print(f"Warning: An error occurred while accessing Redis cache. Error: {e}")

        # 2. If not in cache, perform the search
        print(f"Cache miss for query: '{query}'. Performing live search...")
        try:
            # 真实 HTTP 请求（含重试）在此计数，与缓存命中分开统计。
            self._api_requests += 1
            response = self._make_request(query, timeout)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            outcome = self._extract_outcome(data)

            # 3. Store the new result in Redis（仅缓存模型可见文本）。
            try:
                self.redis_client.set(self._cache_key(query), outcome.text, ex=30*24*60*60)
            except redis.exceptions.ConnectionError as e:
                print(f"Warning: Could not save result to Redis. Error: {e}")

            return outcome

        except requests.exceptions.HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            print(f"Bing search failed with HTTP Error: {status_code}")
            return SearchOutcome(
                status=self._classify_http_error(status_code),
                text="",
                items=[],
                error_code=str(status_code) if status_code is not None else None,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Bing search failed with network error: {e}")
            return SearchOutcome(status="network", text="", items=[])
        except Exception as e:
            print(f"Bing search failed with an unexpected error: {e}")
            return SearchOutcome(status="invalid", text="", items=[])

    def _extract_outcome(self, data: Dict) -> SearchOutcome:
        """从 API 响应提取模型可见文本与旁路元数据（rank/title/url/description）。"""
        if 'organic' not in data or not data['organic']:
            return SearchOutcome(status="not_found", text=self._format_results({'chunk_content': []}), items=[])

        chunk_content_list = []
        seen_snippets = set()
        items: List[Dict[str, Any]] = []
        for rank, result in enumerate(data['organic'], 1):
            snippet = result.get('description', '').strip()
            items.append({
                "rank": rank,
                "title": result.get('title', '').strip(),
                "url": result.get('link') or result.get('url') or result.get('displayed_link', ''),
                "description": snippet,
            })
            if snippet and snippet not in seen_snippets:
                chunk_content_list.append(snippet)
                seen_snippets.add(snippet)

        if not chunk_content_list:
            # organic 存在但无可用摘要，视为真实零结果（旁路仍保留 title/url）。
            return SearchOutcome(status="not_found", text=self._format_results({'chunk_content': []}), items=items)

        return SearchOutcome(status="success", text=self._format_results({'chunk_content': chunk_content_list}), items=items)

    def _format_results(self, results: Dict) -> str:
        """Formats search results into a readable text block."""
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        
        return "\n".join(formatted)

    def execute_with_retry(self, query: str) -> str:
        """Executes a search query with a built-in retry mechanism."""
        # 逻辑搜索次数在此计数（每个 query 一次，内部重试不重复计）。
        self._logical_queries += 1
        for i in range(self._tool_retry_count):
            try:
                result_text = self.execute(query)
                if result_text:
                    return result_text
                # 鉴权错误不可重试，立即失败（方案 §4.3）。
                outcome = self._last_outcome
                if outcome is not None and outcome.status == "auth":
                    print(f"Auth error for '{query}', not retrying.")
                    break
                print(f"Attempt {i+1}/{self._tool_retry_count}: Bing Search returned empty output for '{query}'. Retrying...")
            except Exception as e:
                print(f"Attempt {i+1}/{self._tool_retry_count}: Bing Search failed for '{query}'. Error: {e}. Retrying...")
            time.sleep(1) # Wait a bit before retrying

        print(f"All {self._tool_retry_count} retries failed for query: '{query}'")
        return f"Search failed for query: {query}"

    def batch_search(self, queries: List[str]) -> List[str]:
        """Performs a batch of searches concurrently."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.execute_with_retry, queries))
        return results


# --- FastAPI Setup ---
app = FastAPI(title="Online Search Proxy Server")

class SearchRequest(BaseModel):
    queries: List[str]

def connect_to_redis_with_retry(host, port, retries=20, delay=3):
    """
    try to connect Redis
    """
    for i in range(retries):
        try:
            redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            redis_client.ping()
            print(f"Successfully connected to Redis at {host}:{port}")
            return redis_client
        except redis.exceptions.BusyLoadingError as e:
            # catch LOADING
            print(f"Attempt {i+1}/{retries}: Redis is loading data. Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
        except redis.exceptions.ConnectionError as e:
            # else
            print(f"Attempt {i+1}/{retries}: Could not connect to Redis. Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
            
    print(f"FATAL: Could not connect to Redis at {host}:{port} after {retries} attempts.")
    exit(1)

# --- Main Application Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch online search server.")
    # Brightdata API args
    parser.add_argument(
        '--api_key',
        type=str,
        default=os.environ.get("BRIGHTDATA_API_KEY"),
        help="Brightdata API key; defaults to BRIGHTDATA_API_KEY.",
    )
    parser.add_argument('--zone', type=str, default="serp_api1")
    parser.add_argument('--location', type=str, default="cn")
    parser.add_argument('--provider', type=str, default="brightdata", help="Search provider name, used in the cache key namespace.")
    parser.add_argument('--cache_version', type=str, default="v1", help="Cache format version, used in the cache key namespace.")
    # Search behavior args
    parser.add_argument('--max_results', type=int, default=10, help="Number of results to return per query.")
    parser.add_argument('--result_length', type=int, default=1000)
    parser.add_argument('--tool_retry_count', type=int, default=3)
    # Redis and persistence args
    parser.add_argument('--redis_host', type=str, default='localhost', help="Redis server host.")
    parser.add_argument('--redis_port', type=int, default=6397, help="Redis server port.")
    parser.add_argument('--cache_file', type=str, default="search_cache_redis.json", help="Path for JSON persistence.")
    parser.add_argument('--cache_sync_interval', type=float, default=1800.0, help="Interval in seconds to sync Redis to JSON.")
    parser.add_argument('--warm_cache_on_start', action='store_true', help="Load cache from JSON to Redis on startup.")

    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api_key or set BRIGHTDATA_API_KEY")

    # --- Initialization ---
    
    # 1. Connect to Redis
    redis_client = connect_to_redis_with_retry(args.redis_host, args.redis_port)

    # 2. Initialize the Cache Manager for persistence
    cache_file_path = str(pathlib.Path(args.cache_file).expanduser())
    cache_manager = RedisCacheManager(
        redis_client=redis_client,
        cache_file=cache_file_path,
        sync_interval_seconds=args.cache_sync_interval
    )
    
    # 3. (Optional) Warm up the cache
    if args.warm_cache_on_start:
        cache_manager.load_from_json()

    # 4. Instantiate the search engine
    engine = OnlineSearchEngine(
        api_key=args.api_key,
        redis_client=redis_client,
        zone=args.zone,
        location=args.location,
        tool_retry_count=args.tool_retry_count,
        max_results=args.max_results,
        result_length=args.result_length,
        provider=args.provider,
        cache_version=args.cache_version,
    )

    # --- API Routes ---
    @app.post("/retrieve")
    def search_endpoint(request: SearchRequest):
        results = engine.batch_search(request.queries)
        return {"result": results}
    
    # Example route to trigger sync manually
    @app.post("/sync_cache")
    def sync_cache_endpoint():
        cache_manager.sync_to_json()
        return {"message": "Cache sync from Redis to JSON initiated."}

    # 5. Launch the server
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
