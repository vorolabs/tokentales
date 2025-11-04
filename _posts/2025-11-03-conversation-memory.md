---
layout: post
title: "Conversation Memory for GenAI Apps: Buffer vs Summary vs Window"
date: 2025-11-03
tags: ["memory","langchain","pydantic-ai","dspy"]
description: "When to use full buffer, rolling summaries, or k-window—and how each framework implements them."
---

> **Note:** Repo link and reproducible scripts coming next.

## TL;DR
- Buffer = best coherence, worst cost over time.
- Summary = stable cost, risk of drift.
- Window = predictable latency, may forget long-ago facts.

## What we'll compare
LangChain memory classes, DIY memory in Pydantic-AI, and DSPy `History` patterns.

