import asyncio

from prompt_cache_kit import CachePolicy, CachedModel, MemoryCacheBackend, cached, wrap_langchain_model


def test_cached_callable_reuses_response():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend, policy=CachePolicy(namespace="test"))
    def model(prompt, temperature=0):
        calls["count"] += 1
        return f"{prompt}:{temperature}:{calls['count']}"

    assert model("hi") == "hi:0:1"
    assert model("hi") == "hi:0:1"
    assert calls["count"] == 1
    assert backend.stats().hits == 1


def test_bypass_forces_fresh_call():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend)
    def model(prompt):
        calls["count"] += 1
        return calls["count"]

    assert model("hi") == 1
    assert model("hi", _cache_bypass=True) == 2
    assert model("hi") == 1


def test_cached_model_wraps_invoke():
    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return {"text": prompt, "calls": self.calls}

    model = Model()
    wrapped = CachedModel(model, backend=MemoryCacheBackend())

    assert wrapped.invoke("x") == {"text": "x", "calls": 1}
    assert wrapped.invoke("x") == {"text": "x", "calls": 1}
    assert model.calls == 1


def test_cached_async_callable():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend)
    async def model(prompt):
        calls["count"] += 1
        await asyncio.sleep(0)
        return calls["count"]

    assert asyncio.run(model("hi")) == 1
    assert asyncio.run(model("hi")) == 1
    assert calls["count"] == 1


def test_cached_callable_can_cache_none():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend)
    def model(prompt):
        calls["count"] += 1
        return None

    assert model("hi") is None
    assert model("hi") is None
    assert calls["count"] == 1


def test_cached_async_callable_can_cache_none():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend)
    async def model(prompt):
        calls["count"] += 1
        await asyncio.sleep(0)
        return None

    assert asyncio.run(model("hi")) is None
    assert asyncio.run(model("hi")) is None
    assert calls["count"] == 1


def test_generator_function_is_not_cached():
    backend = MemoryCacheBackend()
    calls = {"count": 0}

    @cached(backend=backend)
    def stream():
        calls["count"] += 1
        yield calls["count"]

    assert list(stream()) == [1]
    assert list(stream()) == [2]
    assert backend.stats().size == 0


def test_langchain_convenience_wrapper():
    class ChatModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return self.calls

    model = ChatModel()
    wrapped = wrap_langchain_model(model)

    assert wrapped.invoke("same") == 1
    assert wrapped.invoke("same") == 1
    assert model.calls == 1
