# Goals

Developer AI Lab is a reproducible laboratory for evaluating AI models — local or
cloud — across the **whole software development lifecycle**, not only code generation
and not only tokens per second. The aim is **real engineering productivity** and the
**hardware needed to support a development team** with a self-hosted model.

## Questions we want to answer

- How many concurrent developers can one self-hosted model support on a given hardware config?
- What is the optimal hardware for a model and a team size?
- Which model develops better software and follows a Spec Driven Development flow best?
- Which produces the better architecture?
- When does self-hosting make sense versus commercial APIs?

## What we measure

Two complementary benchmark families, reported separately:

1. **Concurrency & capacity (team sizing)** — drive concurrent load against a model on
   real hardware and find the highest concurrency that holds an SLO (the knee). Comparing
   the knee across hardware configs tells us the optimal hardware. Current focus: **GLM-5.2**.
2. **SDD / productivity** — an agent (Claude Code) builds a real application (first: a
   Todo App) through the full SDD flow, scored by deterministic gates. This answers
   "which model develops better software, and follows the process best".

## Principles

- **Reproducible and deterministic** for the objective measures.
- **Runnable from scratch** on an ephemeral pod.
- **Model-independent** — the same harness works for any model; adding one is a config
  change, never new code.
- **Easy to extend** with new models, hardware, and scenarios.

The goal is not a collection of scripts, but a professional lab for evaluating AI models
applied to software development.
