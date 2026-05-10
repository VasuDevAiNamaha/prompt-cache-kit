"""Build LangChain Bedrock-compatible cachePoint message content.

Prerequisites:
    pip install langchain-core langchain-aws
"""

from prompt_cache_kit import apply_cache_points, cache_until


messages = [
    {"role": "system", "content": "Very long stable system prompt...", "stable": True},
    {"role": "human", "content": "Analyze this document.", "stable": False},
]

bedrock_messages = apply_cache_points(messages, plan=cache_until(0, id="system-v1"), provider="bedrock")

print(bedrock_messages)

# With LangChain:
#
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_aws import ChatBedrockConverse
#
# lc_messages = [
#     SystemMessage(content=bedrock_messages[0]["content"]),
#     HumanMessage(content=bedrock_messages[1]["content"]),
# ]
# llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet", region_name="us-east-1")
# response = llm.invoke(lc_messages)
# print(response.usage_metadata)
