"""
Parallel processing utilities for classla.
Enables multi-core processing for large datasets.
"""

import multiprocessing as mp
from typing import List, Callable, Any, Optional
import os


def _init_worker(lang: str, processors: str, **kwargs):
    """Initialize a classla pipeline in each worker process."""
    global _worker_nlp
    import classla
    _worker_nlp = classla.Pipeline(lang, processors=processors, **kwargs)


def _process_text(text: str) -> dict:
    """Process a single text and return results as dict."""
    global _worker_nlp
    doc = _worker_nlp(text)

    results = []
    for sent in doc.sentences:
        sent_result = []
        for word in sent.words:
            sent_result.append({
                'text': word.text,
                'lemma': word.lemma,
                'upos': word.upos,
                'xpos': word.xpos,
                'feats': word.feats,
            })
        results.append(sent_result)
    return results


def _process_text_lemma_only(text: str) -> List[str]:
    """Process a single text and return only lemmas."""
    global _worker_nlp
    doc = _worker_nlp(text)
    return [word.lemma for sent in doc.sentences for word in sent.words]


class ParallelPipeline:
    """
    Parallel wrapper for classla Pipeline.

    Usage:
        parallel = ParallelPipeline('sl', processors='tokenize,pos,lemma', n_workers=4)
        results = parallel.process_batch(texts)
        parallel.close()

    Or with context manager:
        with ParallelPipeline('sl', n_workers=4) as parallel:
            results = parallel.process_batch(texts)
    """

    def __init__(
        self,
        lang: str = 'sl',
        processors: str = 'tokenize,pos,lemma',
        n_workers: Optional[int] = None,
        **kwargs
    ):
        self.lang = lang
        self.processors = processors
        self.kwargs = kwargs
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self._pool = None

    def _ensure_pool(self):
        """Lazily initialize the worker pool."""
        if self._pool is None:
            self._pool = mp.Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(self.lang, self.processors),
                maxtasksperchild=100  # Restart workers periodically to free memory
            )

    def process_batch(
        self,
        texts: List[str],
        lemma_only: bool = False,
        chunksize: int = 10
    ) -> List[Any]:
        """
        Process multiple texts in parallel.

        Args:
            texts: List of texts to process
            lemma_only: If True, return only lemmas (faster)
            chunksize: Number of texts per worker task

        Returns:
            List of results (dicts or lemma lists)
        """
        self._ensure_pool()

        func = _process_text_lemma_only if lemma_only else _process_text
        results = self._pool.map(func, texts, chunksize=chunksize)
        return results

    def process_batch_async(
        self,
        texts: List[str],
        lemma_only: bool = False,
        chunksize: int = 10
    ):
        """
        Process texts asynchronously (non-blocking).

        Returns:
            AsyncResult object - call .get() to retrieve results
        """
        self._ensure_pool()

        func = _process_text_lemma_only if lemma_only else _process_text
        return self._pool.map_async(func, texts, chunksize=chunksize)

    def close(self):
        """Close the worker pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def process_parallel(
    texts: List[str],
    lang: str = 'sl',
    processors: str = 'tokenize,pos,lemma',
    n_workers: Optional[int] = None,
    lemma_only: bool = False,
    **kwargs
) -> List[Any]:
    """
    Convenience function for one-off parallel processing.

    Usage:
        texts = ["Text 1", "Text 2", "Text 3", ...]
        results = process_parallel(texts, lang='sl', n_workers=4)
    """
    with ParallelPipeline(lang, processors, n_workers, **kwargs) as parallel:
        return parallel.process_batch(texts, lemma_only=lemma_only)
