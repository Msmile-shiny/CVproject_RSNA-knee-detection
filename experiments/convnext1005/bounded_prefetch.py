"""Prepare at most one next CPU batch while the consumer runs inference.

Use with a deterministic, CPU-only loader. No subprocesses, GPU calls or
exception suppression. Closing waits for the one outstanding read to finish.
"""
from concurrent.futures import ThreadPoolExecutor

def prefetch_one(iterable):
    iterator=iter(iterable)
    end=object()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='next-cpu-batch') as pool:
        pending=pool.submit(next,iterator,end)
        while True:
            item=pending.result()
            if item is end:
                return
            pending=pool.submit(next,iterator,end)
            yield item
