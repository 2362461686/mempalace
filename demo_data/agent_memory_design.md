# Agent Memory System Design

## The Problem

AI agents face a fundamental challenge: they forget everything between conversations.
Without persistent memory, an agent:
- Cannot build long-term understanding of user preferences
- Loses context about ongoing projects
- Repeats the same questions and mistakes
- Cannot maintain relationships or trust

## Solution Architecture

### Layer 1: Working Memory (Short-term)
- Current conversation context window
- Active tool outputs and intermediate results
- Temporary scratchpad for reasoning

### Layer 2: Episodic Memory (Experience)
- Past conversation sessions stored as episodes
- Semantic indexing for retrieval
- Temporal ordering for chronology

### Layer 3: Semantic Memory (Knowledge)
- Extracted facts, preferences, and knowledge
- Entity-relationship graph
- Continuously updated as new information arrives

### Layer 4: Procedural Memory (Skills)
- Learned workflows and patterns
- Reusable skill templates
- Self-improvement through experience

## Integration with MemPalace

MemPalace implements layers 2 and 3 natively:
- Episodic memory: Rooms per session, drawers for individual messages
- Semantic memory: Knowledge graph with temporal entity relationships
- The palace structure (Wings/Rooms/Drawers) maps naturally to the memory hierarchy

## Performance Requirements

- Search latency: under 100ms for typical queries
- Ingestion throughput: 1000+ messages per second
- Storage efficiency: compact vector representations
- Privacy: zero external API calls, fully local
