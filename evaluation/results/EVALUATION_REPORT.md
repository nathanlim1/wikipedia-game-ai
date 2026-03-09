# Wikipedia Game Evaluation Report

## Overview

- **Total runs:** 197
- **Successful:** 178 (90.4%)
- **Excluded:** 22 failed runs (Failed gpt-oss and Llama-70b runs excluded (system issues))
- **Data sources:** failed_configs_retry_20260309_091309.json, eval_results_20260309_012502.json, eval_harness_2runs_20260308_234433.json

**Efficiency (Eff)** = optimal_length / hops (1.0 = perfect path; lower = more detours). **Time** = wall-clock seconds per run.

### Path Optimal Lengths (shortest known path)

| Path | Optimal |
|------|---------|
| Billy Joel -> VANOS | 3 hops |
| David Gilmour -> Michael Phelps | 3 hops |
| John Mulaney -> Apple | 3 hops |
| McDonald's -> Yoga | 2 hops |
| Minecraft -> Chess | 2 hops |
| OpenAI -> Esports | 2 hops |
| Patrick Star -> John Wall | 3 hops |
| Symphony -> United States Navy | 2 hops |

### At a glance

- **Paths where best run achieved optimal:** 4/8
- **Overall % runs reaching optimal:** 33/197 (16.8%)
- **Per-agent avg time (all runs):** default 58.4s, planning 178.0s, tot 44.1s

## Per-Model Per-Agent Per-Path

| Model | Agent | Path | Total | Success | % Optimal | Avg Hops | Avg Time (s) | Eff |
|-------|-------|------|-------|---------|-----------|----------|--------------|-----|
| Qwen3-30B-A3B-Instru | default | Billy Joel -> VANOS | 3 | 3 | 0.0% | 7.0 | 59.9 | 0.43 |
| Qwen3-30B-A3B-Instru | default | David Gilmour -> Michael Phelps | 3 | 3 | 0.0% | 7.0 | 80.5 | 0.43 |
| Qwen3-30B-A3B-Instru | default | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.0 | 39.0 | 0.75 |
| Qwen3-30B-A3B-Instru | default | McDonald's -> Yoga | 3 | 3 | 0.0% | 4.0 | 37.5 | 0.50 |
| Qwen3-30B-A3B-Instru | default | Minecraft -> Chess | 3 | 3 | 0.0% | 5.0 | 25.7 | 0.40 |
| Qwen3-30B-A3B-Instru | default | OpenAI -> Esports | 3 | 3 | 0.0% | 4.0 | 96.5 | 0.50 |
| Qwen3-30B-A3B-Instru | default | Patrick Star -> John Wall | 3 | 3 | 0.0% | 7.33 | 88.8 | 0.44 |
| Qwen3-30B-A3B-Instru | default | Symphony -> United States Navy | 3 | 3 | 100.0% | 2.0 | 23.1 | 1.00 |
| Qwen3-30B-A3B-Instru | planning | Billy Joel -> VANOS | 3 | 2 | 0.0% | 8.0 | 222.3 | 0.38 |
| Qwen3-30B-A3B-Instru | planning | David Gilmour -> Michael Phelps | 3 | 3 | 0.0% | 12.33 | 229.1 | 0.29 |
| Qwen3-30B-A3B-Instru | planning | John Mulaney -> Apple | 3 | 2 | 0.0% | 4.0 | 293.8 | 0.75 |
| Qwen3-30B-A3B-Instru | planning | McDonald's -> Yoga | 3 | 3 | 0.0% | 3.67 | 235.8 | 0.58 |
| Qwen3-30B-A3B-Instru | planning | Minecraft -> Chess | 3 | 3 | 0.0% | 5.33 | 171.3 | 0.40 |
| Qwen3-30B-A3B-Instru | planning | OpenAI -> Esports | 3 | 3 | 0.0% | 3.0 | 53.5 | 0.67 |
| Qwen3-30B-A3B-Instru | planning | Patrick Star -> John Wall | 3 | 0 | 0.0% | - | 433.6 | - |
| Qwen3-30B-A3B-Instru | planning | Symphony -> United States Navy | 3 | 3 | 0.0% | 3.0 | 45.1 | 0.67 |
| Qwen3-30B-A3B-Instru | tot | Billy Joel -> VANOS | 3 | 0 | 0.0% | - | 3.5 | - |
| Qwen3-30B-A3B-Instru | tot | David Gilmour -> Michael Phelps | 3 | 0 | 0.0% | - | 3.3 | - |
| Qwen3-30B-A3B-Instru | tot | John Mulaney -> Apple | 3 | 1 | 0.0% | 5.0 | 107.7 | 0.60 |
| Qwen3-30B-A3B-Instru | tot | McDonald's -> Yoga | 3 | 3 | 0.0% | 4.0 | 87.2 | 0.50 |
| Qwen3-30B-A3B-Instru | tot | Minecraft -> Chess | 3 | 3 | 0.0% | 3.0 | 15.3 | 0.67 |
| Qwen3-30B-A3B-Instru | tot | OpenAI -> Esports | 3 | 3 | 0.0% | 4.0 | 18.2 | 0.50 |
| Qwen3-30B-A3B-Instru | tot | Patrick Star -> John Wall | 3 | 0 | 0.0% | - | 3.5 | - |
| Qwen3-30B-A3B-Instru | tot | Symphony -> United States Navy | 3 | 0 | 0.0% | - | 3.3 | - |
| Llama-3.3-70B-Instru | default | Billy Joel -> VANOS | 3 | 3 | 0.0% | 10.0 | 103.5 | 0.30 |
| Llama-3.3-70B-Instru | default | David Gilmour -> Michael Phelps | 3 | 3 | 0.0% | 11.0 | 114.1 | 0.27 |
| Llama-3.3-70B-Instru | default | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.0 | 70.5 | 0.75 |
| Llama-3.3-70B-Instru | default | McDonald's -> Yoga | 3 | 3 | 0.0% | 4.0 | 41.4 | 0.50 |
| Llama-3.3-70B-Instru | default | Minecraft -> Chess | 3 | 3 | 0.0% | 4.0 | 24.2 | 0.50 |
| Llama-3.3-70B-Instru | default | OpenAI -> Esports | 3 | 3 | 0.0% | 3.0 | 20.1 | 0.67 |
| Llama-3.3-70B-Instru | default | Patrick Star -> John Wall | 3 | 3 | 0.0% | 4.0 | 24.9 | 0.75 |
| Llama-3.3-70B-Instru | default | Symphony -> United States Navy | 3 | 3 | 100.0% | 2.0 | 12.8 | 1.00 |
| Llama-3.3-70B-Instru | planning | Billy Joel -> VANOS | 1 | 1 | 0.0% | 9.0 | 319.4 | 0.33 |
| Llama-3.3-70B-Instru | planning | David Gilmour -> Michael Phelps | 3 | 3 | 0.0% | 7.33 | 220.1 | 0.41 |
| Llama-3.3-70B-Instru | planning | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.67 | 250.4 | 0.65 |
| Llama-3.3-70B-Instru | planning | McDonald's -> Yoga | 3 | 3 | 0.0% | 3.0 | 90.7 | 0.67 |
| Llama-3.3-70B-Instru | planning | Minecraft -> Chess | 3 | 3 | 100.0% | 2.0 | 27.0 | 1.00 |
| Llama-3.3-70B-Instru | planning | OpenAI -> Esports | 3 | 3 | 0.0% | 4.0 | 72.3 | 0.50 |
| Llama-3.3-70B-Instru | planning | Patrick Star -> John Wall | 1 | 1 | 0.0% | 13.0 | 599.0 | 0.23 |
| Llama-3.3-70B-Instru | planning | Symphony -> United States Navy | 3 | 3 | 100.0% | 2.0 | 52.5 | 1.00 |
| Llama-3.3-70B-Instru | tot | Billy Joel -> VANOS | 3 | 3 | 0.0% | 8.33 | 71.1 | 0.37 |
| Llama-3.3-70B-Instru | tot | David Gilmour -> Michael Phelps | 3 | 3 | 0.0% | 4.33 | 24.3 | 0.70 |
| Llama-3.3-70B-Instru | tot | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.67 | 64.1 | 0.67 |
| Llama-3.3-70B-Instru | tot | McDonald's -> Yoga | 3 | 3 | 0.0% | 3.0 | 13.6 | 0.67 |
| Llama-3.3-70B-Instru | tot | Minecraft -> Chess | 3 | 3 | 0.0% | 6.0 | 24.2 | 0.33 |
| Llama-3.3-70B-Instru | tot | OpenAI -> Esports | 3 | 3 | 0.0% | 4.0 | 22.7 | 0.56 |
| Llama-3.3-70B-Instru | tot | Patrick Star -> John Wall | 3 | 3 | 0.0% | 4.0 | 36.5 | 0.75 |
| Llama-3.3-70B-Instru | tot | Symphony -> United States Navy | 3 | 3 | 100.0% | 2.0 | 27.8 | 1.00 |
| gpt-oss-120b | default | Billy Joel -> VANOS | 3 | 3 | 0.0% | 12.33 | 146.4 | 0.28 |
| gpt-oss-120b | default | David Gilmour -> Michael Phelps | 3 | 3 | 100.0% | 3.0 | 29.7 | 1.00 |
| gpt-oss-120b | default | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.0 | 52.5 | 0.75 |
| gpt-oss-120b | default | McDonald's -> Yoga | 3 | 3 | 0.0% | 4.0 | 63.3 | 0.50 |
| gpt-oss-120b | default | Minecraft -> Chess | 3 | 3 | 0.0% | 3.67 | 43.7 | 0.56 |
| gpt-oss-120b | default | OpenAI -> Esports | 3 | 3 | 0.0% | 3.0 | 37.4 | 0.67 |
| gpt-oss-120b | default | Patrick Star -> John Wall | 3 | 3 | 0.0% | 4.33 | 129.4 | 0.70 |
| gpt-oss-120b | default | Symphony -> United States Navy | 3 | 3 | 100.0% | 2.0 | 37.2 | 1.00 |
| gpt-oss-120b | planning | David Gilmour -> Michael Phelps | 2 | 2 | 0.0% | 7.0 | 200.9 | 0.43 |
| gpt-oss-120b | planning | John Mulaney -> Apple | 3 | 3 | 0.0% | 4.33 | 211.1 | 0.70 |
| gpt-oss-120b | planning | McDonald's -> Yoga | 3 | 3 | 0.0% | 3.33 | 91.9 | 0.61 |
| gpt-oss-120b | planning | Minecraft -> Chess | 3 | 3 | 100.0% | 2.0 | 36.0 | 1.00 |
| gpt-oss-120b | planning | OpenAI -> Esports | 3 | 3 | 66.7% | 2.33 | 71.3 | 0.89 |
| gpt-oss-120b | planning | Patrick Star -> John Wall | 1 | 1 | 0.0% | 7.0 | 1124.6 | 0.43 |
| gpt-oss-120b | planning | Symphony -> United States Navy | 3 | 3 | 66.7% | 2.33 | 56.8 | 0.89 |
| gpt-oss-120b | tot | Billy Joel -> VANOS | 1 | 1 | 0.0% | 4.0 | 211.2 | 0.75 |
| gpt-oss-120b | tot | David Gilmour -> Michael Phelps | 3 | 3 | 100.0% | 3.0 | 67.1 | 1.00 |
| gpt-oss-120b | tot | McDonald's -> Yoga | 3 | 3 | 0.0% | 3.0 | 76.4 | 0.67 |
| gpt-oss-120b | tot | Minecraft -> Chess | 3 | 3 | 33.3% | 2.67 | 63.0 | 0.78 |
| gpt-oss-120b | tot | OpenAI -> Esports | 2 | 2 | 50.0% | 2.5 | 36.9 | 0.83 |
| gpt-oss-120b | tot | Patrick Star -> John Wall | 2 | 2 | 0.0% | 4.0 | 119.1 | 0.75 |
| gpt-oss-120b | tot | Symphony -> United States Navy | 1 | 1 | 0.0% | 3.0 | 57.8 | 0.67 |

## Per-Agent Per-Path

| Agent | Path | Total | Success | % Optimal | Avg Hops | Avg Time (s) | Eff |
|-------|------|-------|---------|-----------|----------|--------------|-----|
| default | Billy Joel -> VANOS | 9 | 9 | 0.0% | 9.78 | 103.3 | 0.33 |
| default | David Gilmour -> Michael Phelps | 9 | 9 | 33.3% | 7.0 | 74.8 | 0.57 |
| default | John Mulaney -> Apple | 9 | 9 | 0.0% | 4.0 | 54.0 | 0.75 |
| default | McDonald's -> Yoga | 9 | 9 | 0.0% | 4.0 | 47.4 | 0.50 |
| default | Minecraft -> Chess | 9 | 9 | 0.0% | 4.22 | 31.2 | 0.49 |
| default | OpenAI -> Esports | 9 | 9 | 0.0% | 3.33 | 51.3 | 0.61 |
| default | Patrick Star -> John Wall | 9 | 9 | 0.0% | 5.22 | 81.0 | 0.63 |
| default | Symphony -> United States Navy | 9 | 9 | 100.0% | 2.0 | 24.4 | 1.00 |
| planning | Billy Joel -> VANOS | 4 | 3 | 0.0% | 8.33 | 246.6 | 0.36 |
| planning | David Gilmour -> Michael Phelps | 8 | 8 | 0.0% | 9.12 | 218.7 | 0.37 |
| planning | John Mulaney -> Apple | 9 | 8 | 0.0% | 4.38 | 251.7 | 0.69 |
| planning | McDonald's -> Yoga | 9 | 9 | 0.0% | 3.33 | 139.5 | 0.62 |
| planning | Minecraft -> Chess | 9 | 9 | 66.7% | 3.11 | 78.1 | 0.80 |
| planning | OpenAI -> Esports | 9 | 9 | 22.2% | 3.11 | 65.7 | 0.69 |
| planning | Patrick Star -> John Wall | 5 | 2 | 0.0% | 10.0 | 604.9 | 0.33 |
| planning | Symphony -> United States Navy | 9 | 9 | 55.6% | 2.44 | 51.5 | 0.85 |
| tot | Billy Joel -> VANOS | 7 | 4 | 0.0% | 7.25 | 62.1 | 0.46 |
| tot | David Gilmour -> Michael Phelps | 9 | 6 | 33.3% | 3.67 | 31.5 | 0.85 |
| tot | John Mulaney -> Apple | 6 | 4 | 0.0% | 4.75 | 85.9 | 0.65 |
| tot | McDonald's -> Yoga | 9 | 9 | 0.0% | 3.33 | 59.1 | 0.61 |
| tot | Minecraft -> Chess | 9 | 9 | 11.1% | 3.89 | 34.1 | 0.59 |
| tot | OpenAI -> Esports | 8 | 8 | 12.5% | 3.62 | 24.6 | 0.60 |
| tot | Patrick Star -> John Wall | 8 | 5 | 0.0% | 4.0 | 44.7 | 0.75 |
| tot | Symphony -> United States Navy | 7 | 4 | 42.9% | 2.25 | 21.6 | 0.92 |

## Per-Model Per-Path

| Model | Path | Total | Success | Avg Hops | Avg Time (s) | Eff |
|-------|------|-------|---------|----------|--------------|-----|
| Qwen3-30B-A3B-Instruct | Billy Joel -> VANOS | 9 | 5 | 7.4 | 95.2 | 0.41 |
| Qwen3-30B-A3B-Instruct | David Gilmour -> Michael Phelps | 9 | 6 | 9.67 | 104.3 | 0.36 |
| Qwen3-30B-A3B-Instruct | John Mulaney -> Apple | 9 | 6 | 4.17 | 146.8 | 0.72 |
| Qwen3-30B-A3B-Instruct | McDonald's -> Yoga | 9 | 9 | 3.89 | 120.2 | 0.53 |
| Qwen3-30B-A3B-Instruct | Minecraft -> Chess | 9 | 9 | 4.44 | 70.8 | 0.49 |
| Qwen3-30B-A3B-Instruct | OpenAI -> Esports | 9 | 9 | 3.67 | 56.1 | 0.56 |
| Qwen3-30B-A3B-Instruct | Patrick Star -> John Wall | 9 | 3 | 7.33 | 175.3 | 0.44 |
| Qwen3-30B-A3B-Instruct | Symphony -> United States Navy | 9 | 6 | 2.5 | 23.8 | 0.83 |
| Llama-3.3-70B-Instruct | Billy Joel -> VANOS | 7 | 7 | 9.14 | 120.4 | 0.33 |
| Llama-3.3-70B-Instruct | David Gilmour -> Michael Phelps | 9 | 9 | 7.56 | 119.5 | 0.46 |
| Llama-3.3-70B-Instruct | John Mulaney -> Apple | 9 | 9 | 4.44 | 128.3 | 0.69 |
| Llama-3.3-70B-Instruct | McDonald's -> Yoga | 9 | 9 | 3.33 | 48.6 | 0.61 |
| Llama-3.3-70B-Instruct | Minecraft -> Chess | 9 | 9 | 4.0 | 25.1 | 0.61 |
| Llama-3.3-70B-Instruct | OpenAI -> Esports | 9 | 9 | 3.67 | 38.3 | 0.57 |
| Llama-3.3-70B-Instruct | Patrick Star -> John Wall | 7 | 7 | 5.29 | 111.9 | 0.68 |
| Llama-3.3-70B-Instruct | Symphony -> United States Navy | 9 | 9 | 2.0 | 31.1 | 1.00 |
| gpt-oss-120b | Billy Joel -> VANOS | 4 | 4 | 10.25 | 162.6 | 0.39 |
| gpt-oss-120b | David Gilmour -> Michael Phelps | 8 | 8 | 4.0 | 86.5 | 0.86 |
| gpt-oss-120b | John Mulaney -> Apple | 6 | 6 | 4.17 | 131.8 | 0.72 |
| gpt-oss-120b | McDonald's -> Yoga | 9 | 9 | 3.44 | 77.2 | 0.59 |
| gpt-oss-120b | Minecraft -> Chess | 9 | 9 | 2.78 | 47.5 | 0.78 |
| gpt-oss-120b | OpenAI -> Esports | 8 | 8 | 2.62 | 50.0 | 0.79 |
| gpt-oss-120b | Patrick Star -> John Wall | 6 | 6 | 4.67 | 291.8 | 0.67 |
| gpt-oss-120b | Symphony -> United States Navy | 7 | 7 | 2.29 | 48.6 | 0.90 |

## Per-Agent Results

% Optimal = share of runs reaching optimal path length. Paths Ever Optimal = count of distinct start-target pairs where the agent found an optimal path at least once.

| Agent | Total | Success | Rate | % Optimal | Paths Ever Optimal | Avg Hops | Avg Time (s) | Eff |
|-------|-------|---------|------|-----------|--------------------|----------|--------------|-----|
| default | 72 | 72 | 100.0% | 16.7% | 2 | 4.94 | 58.4 | 0.61 |
| planning | 62 | 57 | 91.9% | 21.0% | 3 | 4.58 | 178.0 | 0.65 |
| tot | 63 | 49 | 77.8% | 12.7% | 4 | 3.94 | 44.1 | 0.67 |

## Per-Model Per-Agent

% Optimal = share of runs (across all paths) that reached the optimal path length.

| Model | Agent | Total | Success | % Optimal | Avg Hops | Avg Time (s) | Eff |
|-------|-------|-------|---------|-----------|----------|--------------|-----|
| Qwen3-30B-A3B-Instruct | default | 24 | 24 | 12.5% | 5.04 | 56.4 | 0.56 |
| Qwen3-30B-A3B-Instruct | planning | 24 | 19 | 0.0% | 5.58 | 210.6 | 0.53 |
| Qwen3-30B-A3B-Instruct | tot | 24 | 10 | 0.0% | 3.8 | 30.3 | 0.56 |
| Llama-3.3-70B-Instruct | default | 24 | 24 | 12.5% | 5.25 | 51.4 | 0.59 |
| Llama-3.3-70B-Instruct | planning | 20 | 20 | 30.0% | 4.55 | 152.9 | 0.66 |
| Llama-3.3-70B-Instruct | tot | 24 | 24 | 12.5% | 4.54 | 35.5 | 0.63 |
| gpt-oss-120b | default | 24 | 24 | 25.0% | 4.54 | 67.4 | 0.68 |
| gpt-oss-120b | planning | 18 | 18 | 38.9% | 3.56 | 162.6 | 0.75 |
| gpt-oss-120b | tot | 15 | 15 | 33.3% | 3.07 | 80.0 | 0.79 |

## Per-Model Per-Agent Per-Candidate-Size

| Model | Agent | Size | Total | Success | Avg Hops | Avg Time (s) | Eff |
|-------|-------|------|-------|---------|----------|--------------|-----|
| Qwen3-30B-A3B-Inst | default | 128 | 24 | 24 | 5.04 | 56.4 | 0.56 |
| Qwen3-30B-A3B-Inst | planning | 12 | 24 | 19 | 5.58 | 210.6 | 0.53 |
| Qwen3-30B-A3B-Inst | tot | 128 | 24 | 10 | 3.8 | 30.3 | 0.56 |
| Llama-3.3-70B-Inst | default | 1024 | 24 | 24 | 5.25 | 51.4 | 0.59 |
| Llama-3.3-70B-Inst | planning | 128 | 20 | 20 | 4.55 | 152.9 | 0.66 |
| Llama-3.3-70B-Inst | tot | 1024 | 24 | 24 | 4.54 | 35.5 | 0.63 |
| gpt-oss-120b | default | 1024 | 24 | 24 | 4.54 | 67.4 | 0.68 |
| gpt-oss-120b | planning | 128 | 18 | 18 | 3.56 | 162.6 | 0.75 |
| gpt-oss-120b | tot | 1024 | 15 | 15 | 3.07 | 80.0 | 0.79 |

## LLM Sweep (Qwen 30B, Default Agent)

Sweep of llm_choices on Qwen/Qwen3-30B-A3B-Instruct-2507 only. Paths: Patrick Star -> John Wall, Billy Joel -> VANOS

| llm_choices | Total | Success | Avg Hops | Avg Time (s) | Avg Efficiency |
|-------------|-------|---------|-----------|--------------|----------------|
| 64 | 4 | 4 | 6.0 | 91.6 | 0.5625 |
| 128 | 4 | 4 | 6.5 | 99.8 | 0.4714 |
| 256 | 4 | 4 | 8.0 | 52.4 | 0.6154 |
| 1024 | 4 | 4 | 8.5 | 93.1 | 0.4513 |

## Failures

| Path | Model | Agent | Reason |
|------|-------|-------|--------|
| John Mulaney -> Apple | Qwen3-30B-A3B-Instru | planning | Stopped by safety timer (420s). |
| John Mulaney -> Apple | Qwen3-30B-A3B-Instru | tot | Agent step failed: |
| John Mulaney -> Apple | Qwen3-30B-A3B-Instru | tot | Agent step failed: |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | planning | Stopped by safety timer (420s). |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | planning | Stopped by safety timer (420s). |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Symphony -> United States Navy | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Symphony -> United States Navy | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Billy Joel -> VANOS | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Billy Joel -> VANOS | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| David Gilmour -> Michael Phelps | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| David Gilmour -> Michael Phelps | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | planning | Stopped by safety timer (420s). |
| Patrick Star -> John Wall | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Symphony -> United States Navy | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| Billy Joel -> VANOS | Qwen3-30B-A3B-Instru | planning | Stopped by safety timer (420s). |
| Billy Joel -> VANOS | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |
| David Gilmour -> Michael Phelps | Qwen3-30B-A3B-Instru | tot | No usable outgoing links. |

## Best Results by Path

All runs tied for fewest hops. **Eff** = optimal/hops. Optimal lengths from Path Optimal Lengths table.

### Billy Joel -> VANOS (optimal 3)
- **Best: 4 hops** (1 run, eff=0.75)
  - gpt-oss-120b | tot | 211.21s, eff=0.75

### David Gilmour -> Michael Phelps (optimal 3)
- **Best: 3 hops** (6 runs, eff=1.0)
  - gpt-oss-120b | default | 19.4s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps
  - gpt-oss-120b | default | 38.89s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps
  - gpt-oss-120b | tot | 77.81s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps
  - gpt-oss-120b | tot | 50.54s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps
  - gpt-oss-120b | default | 30.9s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps
  - gpt-oss-120b | tot | 72.99s, eff=1.0
    Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps

### John Mulaney -> Apple (optimal 3)
- **Best: 4 hops** (16 runs, eff=0.75)
  - Qwen3-30B-A3B-Instruct | default | 27.74s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Qwen3-30B-A3B-Instruct | default | 18.8s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Qwen3-30B-A3B-Instruct | planning | 187.66s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | default | 141.53s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | default | 44.09s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | planning | 270.17s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | tot | 21.37s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | tot | 108.55s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - gpt-oss-120b | default | 56.66s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - gpt-oss-120b | default | 56.34s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - gpt-oss-120b | planning | 110.12s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Qwen3-30B-A3B-Instruct | default | 70.52s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Qwen3-30B-A3B-Instruct | planning | 238.29s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - Llama-3.3-70B-Instruct | default | 25.81s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - gpt-oss-120b | default | 44.39s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple
  - gpt-oss-120b | planning | 300.93s, eff=0.75
    Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple

### McDonald's -> Yoga (optimal 2)
- **Best: 3 hops** (13 runs, eff=0.67)
  - Qwen3-30B-A3B-Instruct | planning | 274.69s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga
  - Qwen3-30B-A3B-Instruct | planning | 208.05s, eff=0.67
    Path: McDonald's → Globalization → Buddhism → Yoga
  - Llama-3.3-70B-Instruct | planning | 81.65s, eff=0.67
    Path: McDonald's → Vegetarianism → Hinduism → Yoga
  - Llama-3.3-70B-Instruct | planning | 94.88s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga
  - Llama-3.3-70B-Instruct | tot | 14.33s, eff=0.67
    Path: McDonald's → Amritsar → Hinduism → Yoga
  - Llama-3.3-70B-Instruct | tot | 13.49s, eff=0.67
    Path: McDonald's → Amritsar → Hinduism → Yoga
  - gpt-oss-120b | planning | 81.32s, eff=0.67
    Path: McDonald's → Vegetarianism → Hinduism → Yoga
  - gpt-oss-120b | tot | 91.51s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga
  - gpt-oss-120b | tot | 86.83s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga
  - Llama-3.3-70B-Instruct | planning | 95.65s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga
  - Llama-3.3-70B-Instruct | tot | 12.95s, eff=0.67
    Path: McDonald's → Amritsar → Hinduism → Yoga
  - gpt-oss-120b | planning | 57.4s, eff=0.67
    Path: McDonald's → Vegetarianism → Hinduism → Yoga
  - gpt-oss-120b | tot | 50.99s, eff=0.67
    Path: McDonald's → India → Hinduism → Yoga

### Minecraft -> Chess (optimal 2)
- **Best: 2 hops** (7 runs, eff=1.0)
  - Llama-3.3-70B-Instruct | planning | 24.93s, eff=1.0
    Path: Minecraft → Game mechanics → Chess
  - Llama-3.3-70B-Instruct | planning | 26.9s, eff=1.0
    Path: Minecraft → Game mechanics → Chess
  - gpt-oss-120b | planning | 27.49s, eff=1.0
    Path: Minecraft → Game mechanics → Chess
  - gpt-oss-120b | planning | 41.07s, eff=1.0
    Path: Minecraft → Artificial intelligence → Chess
  - Llama-3.3-70B-Instruct | planning | 29.07s, eff=1.0
    Path: Minecraft → Game mechanics → Chess
  - gpt-oss-120b | planning | 39.32s, eff=1.0
    Path: Minecraft → Artificial intelligence → Chess
  - gpt-oss-120b | tot | 59.3s, eff=1.0
    Path: Minecraft → Game mechanics → Chess

### OpenAI -> Esports (optimal 2)
- **Best: 2 hops** (3 runs, eff=1.0)
  - gpt-oss-120b | planning | 51.79s, eff=1.0
    Path: OpenAI → IGN → Esports
  - gpt-oss-120b | planning | 55.45s, eff=1.0
    Path: OpenAI → IGN → Esports
  - gpt-oss-120b | tot | 25.94s, eff=1.0
    Path: OpenAI → DeepMind → Esports

### Patrick Star -> John Wall (optimal 3)
- **Best: 4 hops** (10 runs, eff=0.75)
  - Llama-3.3-70B-Instruct | default | 24.18s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall
  - Llama-3.3-70B-Instruct | default | 24.83s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall
  - Llama-3.3-70B-Instruct | tot | 20.65s, eff=0.75
    Path: Patrick Star → Los Angeles Rams → NBA → Washington Wizards → John Wall
  - Llama-3.3-70B-Instruct | tot | 16.02s, eff=0.75
    Path: Patrick Star → Los Angeles Rams → NBA → Washington Wizards → John Wall
  - gpt-oss-120b | default | 36.8s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall
  - gpt-oss-120b | tot | 143.72s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall
  - gpt-oss-120b | tot | 94.4s, eff=0.75
    Path: Patrick Star → Nickelodeon Kids' Choice Awards → LeBron James → List of first overall NBA draft picks → John Wall
  - Llama-3.3-70B-Instruct | default | 25.73s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall
  - Llama-3.3-70B-Instruct | tot | 72.78s, eff=0.75
    Path: Patrick Star → Los Angeles Rams → NBA → Washington Wizards → John Wall
  - gpt-oss-120b | default | 132.24s, eff=0.75
    Path: Patrick Star → The Wall Street Journal → Washington, D.C. → Washington Wizards → John Wall

### Symphony -> United States Navy (optimal 2)
- **Best: 2 hops** (17 runs, eff=1.0)
  - Qwen3-30B-A3B-Instruct | default | 9.81s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Qwen3-30B-A3B-Instruct | default | 9.32s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | default | 15.21s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | default | 11.68s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | planning | 31.95s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | planning | 43.14s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | tot | 8.74s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | tot | 12.47s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - gpt-oss-120b | default | 16.23s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - gpt-oss-120b | default | 57.91s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - gpt-oss-120b | planning | 44.25s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Qwen3-30B-A3B-Instruct | default | 50.02s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | default | 11.64s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | planning | 82.49s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - Llama-3.3-70B-Instruct | tot | 62.3s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - gpt-oss-120b | default | 37.52s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy
  - gpt-oss-120b | planning | 75.29s, eff=1.0
    Path: Symphony → United States Marine Band → United States Navy

## Per-Path Summary

| Path | Optimal | Total | Success | Rate | Avg Hops | Avg Time (s) | Eff |
|------|---------|-------|---------|------|----------|--------------|-----|
| Billy Joel -> VANOS | 3 | 20 | 16 | 80.0% | 8.88 | 117.5 | 0.37 |
| David Gilmour -> Michael Phelps | 3 | 26 | 23 | 88.5% | 6.87 | 104.1 | 0.57 |
| John Mulaney -> Apple | 3 | 24 | 21 | 87.5% | 4.29 | 136.1 | 0.71 |
| McDonald's -> Yoga | 2 | 27 | 27 | 100.0% | 3.56 | 82.0 | 0.58 |
| Minecraft -> Chess | 2 | 27 | 27 | 100.0% | 3.74 | 47.8 | 0.63 |
| OpenAI -> Esports | 2 | 26 | 26 | 100.0% | 3.35 | 48.1 | 0.63 |
| Patrick Star -> John Wall | 3 | 22 | 16 | 72.7% | 5.44 | 186.9 | 0.63 |
| Symphony -> United States Navy | 2 | 25 | 22 | 88.0% | 2.23 | 33.4 | 0.92 |

## Data Files

- `all_runs_compiled.csv` - All runs in flat CSV format
- `aggregated_statistics.json` - Computed statistics
