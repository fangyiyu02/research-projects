#!/usr/bin/env python3
"""Entry point for sequence-chunked FFN and lm_head with no cross-step cache."""

from no_cache_run_common import main


if __name__ == "__main__":
    main("no_cache_chunked")
