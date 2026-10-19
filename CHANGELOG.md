# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-18

### Added
- **Statistical Assertions**: `@pytest.mark.agentcheck` marker supporting trial runs, minimum pass rate thresholds, and Wilson score confidence intervals.
- **Sequential Early Stopping**: Automatically stops tests early when pass rate guarantees or impossibility criteria are reached.
- **Statistical Baselines**: `--agentcheck-baseline=save|compare` supporting two-proportion hypothesis tests ($z$-score and $p$-value) to eliminate false failures on random noise.
- **Flakiness Detection**: Automatic tagging for non-deterministic agents scoring between 5% and 95% pass rate.
- **Fault Injection Engine**: `agentcheck_faults` fixture supporting timeouts, exceptions, empty results, malformed JSON, corrupted values, artificial latency, rate limits, and prompt injection payloads.
- **Resilience Assertions**: Assertions for unhandled agent crashes, maximum retry counts, prompt injection resistance, and false success detection.
- **Record/Replay Cassettes**: Offline testing with secret redaction, request hashing, and diff generation on cassette misses.
- **Deterministic Assertions**: `assert_tool_called`, `assert_tool_not_called`, `assert_tool_order`, `assert_tool_args_match`, `assert_max_steps`, `assert_budget`, `assert_output_matches`, and `assert_no_repeated_loop`.
- **Framework Adapters**:
  - `CallableAdapter`: Default adapter for pure Python functions.
  - `OpenAICompatAdapter`: Client patcher for OpenAI, Groq, Ollama, and vLLM.
  - `AgentCheckCallbackHandler`: Callbacks for LangGraph and LangChain.
  - `ingest_spans` & `AgentCheckSpanProcessor`: OpenTelemetry / OpenInference span ingestion.
- **Reporting**: Aligned terminal tables, JSON, and Markdown summaries written to `.agentcheck/`.
- **CLI**: `agentcheck report` and `agentcheck diff` commands.
