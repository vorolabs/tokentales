---
layout: post
title: "Conversation Memory for GenAI Apps: Buffer vs Summary vs Window"
date: 2025-11-03
tags: ["memory","langchain","pydantic-ai","dspy"]
description: "When to use full buffer, rolling summaries, or k-window—and how each framework implements them."
---

## TL;DR
- **Buffer** = best coherence, worst cost over time (grows linearly with conversation length)
- **Summary** = stable cost, risk of information drift in compressed summaries
- **Window** = predictable latency, may forget long-ago facts (fixed-size sliding window)

## The Problem

Without memory, LLMs treat each interaction as isolated. Ask "What is my name?" after introducing yourself, and you'll get a blank stare. Here's how to fix it.

## 1. Conversation Buffer Memory

**What it does:** Stores the entire conversation history verbatim.

**Implementation (LangChain):**

```python
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic import LLMChain

memory = ConversationBufferMemory(memory_key="chat_history")

template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}<|end|>
<|assistant|>"""

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# First interaction
llm_chain.invoke({"input_prompt": "Hi! My name is Abhishek. What is 1 + 1?"})
# Output: "2" (and remembers the name)

# Second interaction - remembers!
llm_chain.invoke({"input_prompt": "What is my name?"})
# Output: "Your name is Abhishek"
```

**Pros:**
- Perfect recall of all context
- No information loss
- Simple to implement

**Cons:**
- Token cost grows linearly with conversation length
- Context window limits (can hit max tokens)
- Slower inference as history grows

**Use when:** Short conversations, critical information accuracy, or when cost isn't a concern.

## 2. Conversation Buffer Window Memory

**What it does:** Keeps only the last `k` conversation turns.

**Implementation:**

```python
from langchain_classic.memory import ConversationBufferWindowMemory

# Retain only the last 2 conversations
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

llm_chain.invoke({"input_prompt": "Hi! My name is Abhishek and I am 30 years old. What is 1+1?"})
llm_chain.invoke({"input_prompt": "What is 3 + 3?"})
llm_chain.invoke({"input_prompt": "What is my name?"})
# ✅ Remembers: "Abhishek"

llm_chain.invoke({"input_prompt": "What is my age?"})
# ❌ Forgot! Only last 2 conversations retained
```

**Pros:**
- Fixed token cost (predictable)
- Fast inference (bounded context)
- Good for recent context

**Cons:**
- Forgets information beyond the window
- May lose important early context
- Requires tuning `k` for your use case

**Use when:** You only need recent context, or want strict token budgets.

## 3. Conversation Summary Memory

**What it does:** Compresses conversation history into summaries, updating them incrementally.

**Implementation:**

```python
from langchain_classic.memory import ConversationSummaryMemory

summary_prompt_template = """<s><|user|>Summarize the conversations and update
with the new lines.
Current summary:
{summary}
new lines of conversation:
{new_lines}
New summary:<|end|>
<|assistant|>"""

summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    output_key="text",
    prompt=summary_prompt
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# Multiple interactions
llm_chain.invoke({"input_prompt": "Hi! My name is Abhishek. What is 1 + 1?"})
llm_chain.invoke({"input_prompt": "What is my name?"})
# ✅ Works: remembers from summary

llm_chain.invoke({"input_prompt": "What was the first question I asked?"})
# ✅ Works: summary contains compressed history
```

**Pros:**
- Stable token cost (doesn't grow with conversation length)
- Retains long-term context
- Good balance between recall and efficiency

**Cons:**
- Requires extra LLM call for summarization (cost + latency)
- Risk of information drift in summaries
- Summarization quality affects memory accuracy

**Use when:** Long conversations where you need both recent and historical context, and you can tolerate summarization overhead.

## Comparison Matrix

| Approach | Token Cost | Recall | Latency | Best For |
|----------|-----------|--------|---------|----------|
| Buffer | Linear growth | Perfect | Fast (initially) | Short conversations |
| Window | Fixed | Recent only | Fast | Recent context only |
| Summary | Stable | Compressed | Slower (2 calls) | Long conversations |

## Takeaways

- **Start simple:** Use buffer memory for short conversations or prototypes
- **Optimize later:** Move to window or summary when you hit token limits or cost concerns
- **Tune the window:** If using window memory, experiment with `k` based on your typical conversation length
- **Monitor summaries:** If using summary memory, check for information loss in compressed summaries

## Next Steps

- Pydantic-AI memory patterns (DIY implementations)
- DSPy `History` abstractions
- Hybrid approaches (buffer + summary)

See the [experiments notebook](https://github.com/yourusername/my-site/tree/main/experiments) for reproducible code.

