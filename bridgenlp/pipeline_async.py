import copy
import inspect
import threading
from typing import List

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:  # pragma: no cover - optional dependency
    spacy = None
    Doc = object  # type: ignore

from .pipeline import Pipeline
from .result import BridgeResult


class AsyncPipeline(Pipeline):
    """Asynchronous version of :class:`Pipeline`."""

    async def __aenter__(self):
        for adapter in self.adapters:
            if hasattr(adapter, "__aenter__"):
                await adapter.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for adapter in self.adapters:
            if hasattr(adapter, "__aexit__"):
                await adapter.__aexit__(exc_type, exc_val, exc_tb)
            else:
                adapter.cleanup()
        return False

    async def _call_adapter(self, adapter, method: str, *args):
        func = getattr(adapter, method)
        if inspect.iscoroutinefunction(func):
            return await func(*args)
        return func(*args)

    async def from_text(self, text: str) -> BridgeResult:
        with self._measure_performance():
            if not text or not isinstance(text, str) or not text.strip():
                return BridgeResult(tokens=[])
            with self._pipeline_lock:
                try:
                    cache_key = f"text:{hash(text)}"
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                    local_adapters = list(self.adapters)
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:  # pragma: no cover - defensive
                    import warnings
                    warnings.warn(f"Error initializing pipeline processing: {e}")
                    return BridgeResult(tokens=[])
            try:
                combined_result = await self._call_adapter(local_adapters[0], "from_text", text)
                for i, adapter in enumerate(local_adapters[1:], 1):
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            condition_result = copy.deepcopy(combined_result)
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    if skip_adapter:
                        continue
                    next_result = await self._call_adapter(adapter, "from_text", text)
                    combined_result = self._combine_results(combined_result, next_result)
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                self._update_cache(cache_key, combined_result)
                return copy.deepcopy(combined_result)
            except Exception as e:  # pragma: no cover - defensive
                import warnings
                warnings.warn(f"Error during pipeline text processing: {e}")
                return BridgeResult(tokens=[])

    async def from_tokens(self, tokens: List[str]) -> BridgeResult:
        with self._measure_performance():
            if not tokens or not isinstance(tokens, list):
                return BridgeResult(tokens=[])
            with self._pipeline_lock:
                try:
                    token_tuple = tuple(str(t) for t in tokens)
                    cache_key = f"tokens:{hash(token_tuple)}"
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                    local_adapters = list(self.adapters)
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:  # pragma: no cover - defensive
                    import warnings
                    warnings.warn(f"Error initializing pipeline token processing: {e}")
                    return BridgeResult(tokens=[])
            try:
                safe_tokens = list(tokens)
                combined_result = await self._call_adapter(local_adapters[0], "from_tokens", safe_tokens)
                for i, adapter in enumerate(local_adapters[1:], 1):
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            condition_result = copy.deepcopy(combined_result)
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    if skip_adapter:
                        continue
                    next_result = await self._call_adapter(adapter, "from_tokens", safe_tokens)
                    combined_result = self._combine_results(combined_result, next_result)
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                self._update_cache(cache_key, combined_result)
                return copy.deepcopy(combined_result)
            except Exception as e:  # pragma: no cover - defensive
                import warnings
                warnings.warn(f"Error during pipeline token processing: {e}")
                return BridgeResult(tokens=[])

    async def from_spacy(self, doc: Doc) -> Doc:
        with self._measure_performance():
            if not doc:
                raise ValueError("Cannot process empty Doc")
            with self._pipeline_lock:
                try:
                    try:
                        text_hash = hash(doc.text)
                        tokens_hash = hash(tuple(t.text for t in doc))
                        cache_key = f"spacy:{text_hash}:{tokens_hash}"
                    except Exception as e:
                        import uuid
                        cache_key = f"spacy:uuid:{uuid.uuid4()}"
                        import warnings
                        warnings.warn(f"Error hashing spaCy doc: {e}, using UUID instead")
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        return cached_result.attach_to_spacy(doc)
                    local_adapters = list(self.adapters)
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:  # pragma: no cover - defensive
                    import warnings
                    warnings.warn(f"Error initializing pipeline spaCy processing: {e}")
                    return doc

            with threading.RLock():
                try:
                    for ext_name in [
                        "nlp_bridge_spans",
                        "nlp_bridge_clusters",
                        "nlp_bridge_roles",
                        "nlp_bridge_labels",
                        "nlp_bridge_image_features",
                        "nlp_bridge_audio_features",
                        "nlp_bridge_multimodal_embeddings",
                        "nlp_bridge_detected_objects",
                        "nlp_bridge_captions",
                    ]:
                        if not Doc.has_extension(ext_name):
                            Doc.set_extension(ext_name, default=None)
                        if getattr(doc._, ext_name) is None:
                            default_value = [] if ext_name not in [
                                "nlp_bridge_image_features",
                                "nlp_bridge_audio_features",
                                "nlp_bridge_multimodal_embeddings",
                            ] else None
                            setattr(doc._, ext_name, default_value)
                except Exception as e:  # pragma: no cover - defensive
                    import warnings
                    warnings.warn(f"Error registering spaCy extensions: {e}")

            try:
                combined_result = BridgeResult(tokens=[t.text for t in doc])
                doc = await self._call_adapter(local_adapters[0], "from_spacy", doc)
                with self._result_lock:
                    first_result = BridgeResult(
                        tokens=[t.text for t in doc],
                        spans=copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else [],
                        clusters=copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else [],
                        roles=copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else [],
                        labels=copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else [],
                    )
                combined_result = self._combine_results(combined_result, first_result)
                for i, adapter in enumerate(local_adapters[1:], 1):
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            condition_result = copy.deepcopy(combined_result)
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    if skip_adapter:
                        continue
                    with self._result_lock:
                        current_spans = copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else []
                        current_clusters = copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else []
                        current_roles = copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else []
                        current_labels = copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else []

                    doc = await self._call_adapter(adapter, "from_spacy", doc)
                    with self._result_lock:
                        adapter_result = BridgeResult(
                            tokens=[t.text for t in doc],
                            spans=copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else [],
                            clusters=copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else [],
                            roles=copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else [],
                            labels=copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else [],
                        )
                    combined_result = self._combine_results(combined_result, adapter_result)
                    with self._result_lock:
                        doc._.nlp_bridge_spans = copy.deepcopy(combined_result.spans)
                        doc._.nlp_bridge_clusters = copy.deepcopy(combined_result.clusters)
                        doc._.nlp_bridge_roles = copy.deepcopy(combined_result.roles)
                        doc._.nlp_bridge_labels = copy.copy(combined_result.labels)

                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(doc)
                with self._result_lock:
                    combined_result.image_features = copy.deepcopy(doc._.nlp_bridge_image_features) if doc._.nlp_bridge_image_features else None
                    combined_result.audio_features = copy.deepcopy(doc._.nlp_bridge_audio_features) if doc._.nlp_bridge_audio_features else None
                    combined_result.multimodal_embeddings = copy.copy(doc._.nlp_bridge_multimodal_embeddings) if doc._.nlp_bridge_multimodal_embeddings else None
                    combined_result.detected_objects = copy.deepcopy(doc._.nlp_bridge_detected_objects) if doc._.nlp_bridge_detected_objects else []
                    combined_result.captions = copy.copy(doc._.nlp_bridge_captions) if doc._.nlp_bridge_captions else []
                self._update_cache(cache_key, combined_result)
                return doc
            except Exception as e:  # pragma: no cover - defensive
                import warnings
                warnings.warn(f"Error during pipeline spaCy processing: {e}")
                return doc
