"""Use Prompt Cache Kit with a LangGraph-style node.

Prerequisites:
    pip install langgraph langchain-core

This example intentionally keeps the graph minimal. The same pattern works inside
any node that calls a LangChain chat model.
"""

from typing import TypedDict

from prompt_cache_kit import (
    CachePolicy,
    MemoryCacheBackend,
    StablePrefixPromptCachingStrategy,
    apply_cache_points,
    extract_langchain_usage,
    wrap_langchain_model,
)


class State(TypedDict):
    messages: list
    response: object


backend = MemoryCacheBackend()


def build_cached_node(chat_model):
    cached_model = wrap_langchain_model(
        chat_model,
        backend=backend,
        policy=CachePolicy(namespace="langgraph-demo", ttl_seconds=300),
    )
    strategy = StablePrefixPromptCachingStrategy(min_tokens=1024, max_points=4)

    def node(state: State) -> State:
        messages = apply_cache_points(state["messages"], plan=strategy, provider="bedrock")
        response = cached_model.invoke(messages)
        usage = extract_langchain_usage(response)
        print(usage.to_otel_attributes())
        return {"messages": state["messages"], "response": response}

    return node


# In a real LangGraph app:
#
# from langgraph.graph import StateGraph, START, END
# graph = StateGraph(State)
# graph.add_node("llm", build_cached_node(chat_model))
# graph.add_edge(START, "llm")
# graph.add_edge("llm", END)
# app = graph.compile()
