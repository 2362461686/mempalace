"""LangChain integration for MemPalace.

Provides a retriever and chat memory that use MemPalace as the persistent,
local-first backend. Install with:

    pip install langchain langchain-core

Usage:
    from integrations.langchain import MemPalaceRetriever, MemPalaceChatMemory
"""

from .retriever import MemPalaceRetriever
from .memory import MemPalaceChatMemory

__all__ = ["MemPalaceRetriever", "MemPalaceChatMemory"]
