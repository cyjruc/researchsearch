#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
: "${BRIGHTDATA_API_KEY:?Set BRIGHTDATA_API_KEY in the retriever environment}"

ZONE="${BRIGHTDATA_ZONE:-serp_api1}"
LOCATION="${BRIGHTDATA_LOCATION:-cn}"
MAX_RESULTS="${WEB_MAX_RESULTS:-3}"
RESULT_LENGTH="${WEB_RESULT_LENGTH:-1000}"
RETRY_COUNT="${WEB_RETRY_COUNT:-2}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6397}"
CACHE_FILE="${WEB_CACHE_FILE:-${REPO_ROOT}/search_cache.json}"
CACHE_SYNC_INTERVAL="${CACHE_SYNC_INTERVAL:-1800}"

mkdir -p "$(dirname -- "${CACHE_FILE}")"
cd "${REPO_ROOT}"
exec python3 search_r1/search/bing_search_server_redis.py \
    --zone "${ZONE}" \
    --location "${LOCATION}" \
    --max_results "${MAX_RESULTS}" \
    --result_length "${RESULT_LENGTH}" \
    --tool_retry_count "${RETRY_COUNT}" \
    --redis_host "${REDIS_HOST}" \
    --redis_port "${REDIS_PORT}" \
    --cache_file "${CACHE_FILE}" \
    --cache_sync_interval "${CACHE_SYNC_INTERVAL}" \
    --warm_cache_on_start
