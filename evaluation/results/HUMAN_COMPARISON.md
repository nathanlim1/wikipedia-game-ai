# Human vs System Evaluation Comparison

Comprehensive comparison of human player results vs AI/LLM system results on the Wikipedia game.

**Efficiency (Eff)** = optimal_length / hops (1.0 = perfect path). **Time** = wall-clock seconds.

## Human Evaluation Report

Full human statistics mirroring EVALUATION_REPORT structure (per person, per path, per person per path).

### Human Overview

- **Total runs:** 40
- **Successful:** 40 (100.0%)
- **Data sources:** Eval Results - Sheet1.csv

**Efficiency (Eff)** = optimal_length / hops (1.0 = perfect path; lower = more detours). **Time** = wall-clock seconds per run.

#### Path Optimal Lengths (shortest known path)

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

#### At a glance

- **Paths where best run achieved optimal:** 2/8
- **Overall % runs reaching optimal:** 3/40 (7.5%)
- **Per-person avg time (all runs):** Caleb 115.5s, Colby 129.6s, James 227.9s, Kyra 219.6s, Nathan 104.6s

### All Trials

| Person | Path | Optimal | Hops | Time (s) | Eff |
|--------|------|---------|------|----------|-----|
| Caleb | Billy Joel -> VANOS | No | 10.00 | 237.0 | 0.30 |
| Caleb | David Gilmour -> Michael Phelps | No | 8.00 | 214.0 | 0.38 |
| Caleb | John Mulaney -> Apple | No | 4.00 | 90.0 | 0.75 |
| Caleb | McDonald's -> Yoga | No | 4.00 | 108.0 | 0.50 |
| Caleb | Minecraft -> Chess | No | 5.00 | 57.0 | 0.40 |
| Caleb | OpenAI -> Esports | No | 3.00 | 37.0 | 0.67 |
| Caleb | Patrick Star -> John Wall | No | 4.00 | 156.0 | 0.75 |
| Caleb | Symphony -> United States Navy | No | 5.00 | 25.0 | 0.40 |
| Colby | Billy Joel -> VANOS | No | 5.00 | 78.0 | 0.60 |
| Colby | David Gilmour -> Michael Phelps | No | 5.00 | 122.0 | 0.60 |
| Colby | John Mulaney -> Apple | No | 4.00 | 209.0 | 0.75 |
| Colby | McDonald's -> Yoga | No | 6.00 | 357.0 | 0.33 |
| Colby | Minecraft -> Chess | No | 4.00 | 85.0 | 0.50 |
| Colby | OpenAI -> Esports | No | 4.00 | 99.0 | 0.50 |
| Colby | Patrick Star -> John Wall | No | 5.00 | 69.0 | 0.60 |
| Colby | Symphony -> United States Navy | Yes | 2.00 | 18.0 | 1.00 |
| James | Billy Joel -> VANOS | No | 4.00 | 64.0 | 0.75 |
| James | David Gilmour -> Michael Phelps | No | 4.00 | 192.0 | 0.75 |
| James | John Mulaney -> Apple | No | 4.00 | 381.0 | 0.75 |
| James | McDonald's -> Yoga | No | 4.00 | 487.0 | 0.50 |
| James | Minecraft -> Chess | No | 3.00 | 238.0 | 0.67 |
| James | OpenAI -> Esports | No | 3.00 | 192.0 | 0.67 |
| James | Patrick Star -> John Wall | No | 5.00 | 124.0 | 0.60 |
| James | Symphony -> United States Navy | No | 3.00 | 145.0 | 0.67 |
| Kyra | Billy Joel -> VANOS | No | 4.00 | 258.0 | 0.75 |
| Kyra | David Gilmour -> Michael Phelps | No | 8.00 | 168.0 | 0.38 |
| Kyra | John Mulaney -> Apple | No | 5.00 | 391.0 | 0.60 |
| Kyra | McDonald's -> Yoga | No | 11.00 | 378.0 | 0.18 |
| Kyra | Minecraft -> Chess | No | 7.00 | 138.0 | 0.29 |
| Kyra | OpenAI -> Esports | No | 5.00 | 152.0 | 0.40 |
| Kyra | Patrick Star -> John Wall | No | 4.00 | 179.0 | 0.75 |
| Kyra | Symphony -> United States Navy | No | 4.00 | 93.0 | 0.50 |
| Nathan | Billy Joel -> VANOS | Yes | 3.00 | 41.0 | 1.00 |
| Nathan | David Gilmour -> Michael Phelps | No | 4.00 | 68.0 | 0.75 |
| Nathan | John Mulaney -> Apple | No | 6.00 | 211.0 | 0.50 |
| Nathan | McDonald's -> Yoga | No | 3.00 | 55.0 | 0.67 |
| Nathan | Minecraft -> Chess | No | 5.00 | 102.0 | 0.40 |
| Nathan | OpenAI -> Esports | No | 7.00 | 127.0 | 0.29 |
| Nathan | Patrick Star -> John Wall | No | 5.00 | 206.0 | 0.60 |
| Nathan | Symphony -> United States Navy | Yes | 2.00 | 27.0 | 1.00 |

### Per-Person Results

% Optimal = share of runs reaching optimal path length. Paths Ever Optimal = count of distinct start-target pairs where the person found an optimal path at least once.

| Person | Total | Success | Rate | % Optimal | Paths Ever Optimal | Avg Hops | Avg Time (s) | Eff |
|--------|-------|---------|------|-----------|--------------------|----------|--------------|-----|
| Caleb | 8 | 8 | 100.0% | 0.0% | 0 | 5.38 | 115.5 | 0.52 |
| Colby | 8 | 8 | 100.0% | 12.5% | 1 | 4.38 | 129.6 | 0.61 |
| James | 8 | 8 | 100.0% | 0.0% | 0 | 3.75 | 227.9 | 0.67 |
| Kyra | 8 | 8 | 100.0% | 0.0% | 0 | 6.0 | 219.6 | 0.48 |
| Nathan | 8 | 8 | 100.0% | 25.0% | 2 | 4.38 | 104.6 | 0.65 |

### Per-Path Summary

| Path | Optimal | Total | Success | Optimal Rate | Avg Hops | Avg Time (s) | Eff |
|------|---------|-------|---------|--------------|----------|--------------|-----|
| Billy Joel -> VANOS | 3 | 5 | 5 | 20.0% | 5.20 | 135.6 | 0.68 |
| David Gilmour -> Michael Phelps | 3 | 5 | 5 | 0.0% | 5.80 | 152.8 | 0.57 |
| John Mulaney -> Apple | 3 | 5 | 5 | 0.0% | 4.60 | 256.4 | 0.67 |
| McDonald's -> Yoga | 2 | 5 | 5 | 0.0% | 5.60 | 277.0 | 0.44 |
| Minecraft -> Chess | 2 | 5 | 5 | 0.0% | 4.80 | 124.0 | 0.45 |
| OpenAI -> Esports | 2 | 5 | 5 | 0.0% | 4.40 | 121.4 | 0.50 |
| Patrick Star -> John Wall | 3 | 5 | 5 | 0.0% | 4.60 | 146.8 | 0.66 |
| Symphony -> United States Navy | 2 | 5 | 5 | 40.0% | 3.20 | 61.6 | 0.71 |

### Best Results by Path

All runs tied for fewest hops. **Eff** = optimal/hops.

#### Billy Joel -> VANOS (optimal 3)
- **Best: 3 hops** (1 run, eff=1.0)
  - Nathan | 41.0s, eff=1.00

#### David Gilmour -> Michael Phelps (optimal 3)
- **Best: 4 hops** (2 runs, eff=0.75)
  - James | 192.0s, eff=0.75
  - Nathan | 68.0s, eff=0.75

#### John Mulaney -> Apple (optimal 3)
- **Best: 4 hops** (3 runs, eff=0.75)
  - Caleb | 90.0s, eff=0.75
  - Colby | 209.0s, eff=0.75
  - James | 381.0s, eff=0.75

#### McDonald's -> Yoga (optimal 2)
- **Best: 3 hops** (1 run, eff=0.67)
  - Nathan | 55.0s, eff=0.67

#### Minecraft -> Chess (optimal 2)
- **Best: 3 hops** (1 run, eff=0.67)
  - James | 238.0s, eff=0.67

#### OpenAI -> Esports (optimal 2)
- **Best: 3 hops** (2 runs, eff=0.67)
  - Caleb | 37.0s, eff=0.67
  - James | 192.0s, eff=0.67

#### Patrick Star -> John Wall (optimal 3)
- **Best: 4 hops** (2 runs, eff=0.75)
  - Caleb | 156.0s, eff=0.75
  - Kyra | 179.0s, eff=0.75

#### Symphony -> United States Navy (optimal 2)
- **Best: 2 hops** (2 runs, eff=1.0)
  - Colby | 18.0s, eff=1.00
  - Nathan | 27.0s, eff=1.00

---

## Human vs System Comparison

### Overview

| Metric | Human | System | Δ |
|--------|-------|--------|---|
| Total runs | 40 | 197 | +157 |
| Successful | 40 | 178 | +138 |
| Success rate | 100.0% | 90.4% | -9.6pp |
| Runs reaching optimal | 3 | 33* | - |
| % runs reaching optimal | 7.5% | ~16.8%* | - |

*System optimal % from EVALUATION_REPORT.md (33/197 runs)

### Per-Path Best Comparison

Best run (fewest hops) per path for humans vs system.

| Path | Optimal | Best Human | Hops | Time (s) | Best System | Hops | Time (s) | Winner |
|------|---------|------------|------|----------|-------------|------|----------|--------|
| Billy Joel -> VANOS | 3 | Nathan | 3 | 41.0 | gpt-oss-120b (tot) | 4 | 211.2 | Human |
| David Gilmour -> Michael Phelps | 3 | James | 4 | 192.0 | gpt-oss-120b (default) | 3 | 19.4 | System |
| John Mulaney -> Apple | 3 | Caleb | 4 | 90.0 | Qwen3-30B-A3B-Inst (default) | 4 | 18.8 | Tie |
| McDonald's -> Yoga | 2 | Nathan | 3 | 55.0 | Llama-3.3-70B-Inst (tot) | 3 | 12.9 | Tie |
| Minecraft -> Chess | 2 | James | 3 | 238.0 | Llama-3.3-70B-Inst (planning) | 2 | 24.9 | System |
| OpenAI -> Esports | 2 | Caleb | 3 | 37.0 | gpt-oss-120b (tot) | 2 | 25.9 | System |
| Patrick Star -> John Wall | 3 | Caleb | 4 | 156.0 | Llama-3.3-70B-Inst (tot) | 4 | 16.0 | Tie |
| Symphony -> United States Navy | 2 | Colby | 2 | 18.0 | Llama-3.3-70B-Inst (tot) | 2 | 8.7 | Tie |

#### Best Paths Detail

**Billy Joel -> VANOS** (optimal 3)
- **Best human:** Nathan — 3 hops, 41.0s — Path: Billy Joel, Rolls-Royce Limited, BMW, VANOS
- **Best system:** gpt-oss-120b (tot) — 4 hops, 211.2s

**David Gilmour -> Michael Phelps** (optimal 3)
- **Best human:** James — 4 hops, 192.0s — Path: david gilmour, university of cambridge, olympic medal, 2004 summer olympics, michael phelps
- **Best system:** gpt-oss-120b (default) — 3 hops, 19.4s — Path: David Gilmour → BBC → 2012 Summer Olympics → Michael Phelps

**John Mulaney -> Apple** (optimal 3)
- **Best human:** Caleb — 4 hops, 90.0s — Path: John Mulaney, Apple TV (streaming service), Apple Inc., Apple (disambiguation), Apple
- **Best system:** Qwen3-30B-A3B-Inst (default) — 4 hops, 18.8s — Path: John Mulaney → Apple TV+ → Apple Inc. → Apple (disambiguation) → Apple

**McDonald's -> Yoga** (optimal 2)
- **Best human:** Nathan — 3 hops, 55.0s — Path: McDonald's, Obesity, Exercise, Yoga
- **Best system:** Llama-3.3-70B-Inst (tot) — 3 hops, 12.9s — Path: McDonald's → Amritsar → Hinduism → Yoga

**Minecraft -> Chess** (optimal 2)
- **Best human:** James — 3 hops, 238.0s — Path: minecraft, video game design, game design, chess
- **Best system:** Llama-3.3-70B-Inst (planning) — 2 hops, 24.9s — Path: Minecraft → Game mechanics → Chess

**OpenAI -> Esports** (optimal 2)
- **Best human:** Caleb — 3 hops, 37.0s — Path: OpenAI, General game playing, Video game, Esports
- **Best system:** gpt-oss-120b (tot) — 2 hops, 25.9s — Path: OpenAI → DeepMind → Esports

**Patrick Star -> John Wall** (optimal 3)
- **Best human:** Caleb — 4 hops, 156.0s — Path: Patric Star, National Football League, National Basketball Association, Washington Wizards, John Wall
- **Best system:** Llama-3.3-70B-Inst (tot) — 4 hops, 16.0s — Path: Patrick Star → Los Angeles Rams → NBA → Washington Wizards → John Wall

**Symphony -> United States Navy** (optimal 2)
- **Best human:** Colby — 2 hops, 18.0s — Path: Symphony, United States Marine Band, United States Navy
- **Best system:** Llama-3.3-70B-Inst (tot) — 2 hops, 8.7s — Path: Symphony → United States Marine Band → United States Navy

### Per-Path Comparison (Combined)

| Path | Optimal | Human Total | Human Hops | Human Time | Human Eff | System Total | System Hops | System Time | System Eff |
|------|---------|-------------|------------|------------|-----------|--------------|-------------|-------------|------------|
| Billy Joel -> VANOS | 3 | 5 | 5.20 | 135.6 | 0.68 | 20 | 8.88 | 117.5 | 0.37 |
| David Gilmour -> Michael Phelps | 3 | 5 | 5.80 | 152.8 | 0.57 | 26 | 6.87 | 104.1 | 0.57 |
| John Mulaney -> Apple | 3 | 5 | 4.60 | 256.4 | 0.67 | 24 | 4.29 | 136.1 | 0.71 |
| McDonald's -> Yoga | 2 | 5 | 5.60 | 277.0 | 0.44 | 27 | 3.56 | 82.0 | 0.58 |
| Minecraft -> Chess | 2 | 5 | 4.80 | 124.0 | 0.45 | 27 | 3.74 | 47.8 | 0.63 |
| OpenAI -> Esports | 2 | 5 | 4.40 | 121.4 | 0.50 | 26 | 3.35 | 48.1 | 0.63 |
| Patrick Star -> John Wall | 3 | 5 | 4.60 | 146.8 | 0.66 | 22 | 5.44 | 186.9 | 0.63 |
| Symphony -> United States Navy | 2 | 5 | 3.20 | 61.6 | 0.71 | 25 | 2.23 | 33.4 | 0.92 |

### Per-Path Summary (Side by Side)

### Human

| Path | Total | Success | Avg Hops | Avg Time (s) | Eff | % Optimal |
|------|-------|---------|----------|--------------|-----|-----------|
| Billy Joel -> VANOS | 5 | 5 | 5.20 | 135.6 | 0.68 | 20.0% |
| David Gilmour -> Michael Phelps | 5 | 5 | 5.80 | 152.8 | 0.57 | 0.0% |
| John Mulaney -> Apple | 5 | 5 | 4.60 | 256.4 | 0.67 | 0.0% |
| McDonald's -> Yoga | 5 | 5 | 5.60 | 277.0 | 0.44 | 0.0% |
| Minecraft -> Chess | 5 | 5 | 4.80 | 124.0 | 0.45 | 0.0% |
| OpenAI -> Esports | 5 | 5 | 4.40 | 121.4 | 0.50 | 0.0% |
| Patrick Star -> John Wall | 5 | 5 | 4.60 | 146.8 | 0.66 | 0.0% |
| Symphony -> United States Navy | 5 | 5 | 3.20 | 61.6 | 0.71 | 40.0% |

### System (all models except Qwen3-30B)

| Path | Total | Success | Avg Hops | Avg Time (s) | Eff | % Optimal |
|------|-------|---------|----------|--------------|-----|-----------|
| Billy Joel -> VANOS | 11 | 11 | 9.54 | 135.8 | 0.36 | 0.0% |
| David Gilmour -> Michael Phelps | 17 | 17 | 5.88 | 104.0 | 0.65 | 35.3% |
| John Mulaney -> Apple | 15 | 15 | 4.33 | 129.7 | 0.70 | 0.0% |
| McDonald's -> Yoga | 18 | 18 | 3.39 | 62.9 | 0.60 | 0.0% |
| Minecraft -> Chess | 18 | 18 | 3.39 | 36.3 | 0.69 | 38.9% |
| OpenAI -> Esports | 17 | 17 | 3.18 | 43.8 | 0.68 | 17.7% |
| Patrick Star -> John Wall | 13 | 13 | 5.00 | 194.9 | 0.67 | 0.0% |
| Symphony -> United States Navy | 16 | 16 | 2.12 | 38.7 | 0.96 | 87.5% |

### Per-Path Deltas (System − Human)

System = all models except Qwen3-30B. Positive Δ = system higher. Negative Δ = system lower (better).

| Path | Δ Avg Hops | Δ Avg Time (s) | Δ Efficiency |
|------|------------|----------------|--------------|
| Billy Joel -> VANOS | +4.34 | +0.2 | -0.32 |
| David Gilmour -> Michael Phelps | +0.08 | -48.8 | +0.08 |
| John Mulaney -> Apple | -0.27 | -126.7 | +0.03 |
| McDonald's -> Yoga | -2.21 | -214.1 | +0.17 |
| Minecraft -> Chess | -1.41 | -87.7 | +0.24 |
| OpenAI -> Esports | -1.22 | -77.6 | +0.17 |
| Patrick Star -> John Wall | +0.40 | +48.1 | +0.01 |
| Symphony -> United States Navy | -1.08 | -22.9 | +0.25 |

### Per-Agent Results

| Agent | Total | Success | Rate | Avg Hops | Avg Time (s) | Eff | % Optimal | Paths Ever Optimal |
|-------|-------|---------|------|----------|--------------|-----|-----------|--------------------|
| Human | 40 | 40 | 100.0% | 4.78 | 159.4 | 0.59 | 7.5% | 2 |
| default | 72 | 72 | 100.0% | 4.94 | 58.4 | 0.61 | 16.7% | 2 |
| planning | 62 | 57 | 91.9% | 4.58 | 178.0 | 0.65 | 21.0% | 3 |
| tot | 63 | 49 | 77.8% | 3.94 | 44.1 | 0.67 | 12.7% | 4 |

### Key Insights

- **Success rate:** Humans 100% (all completed); System 90.4%
- **Avg hops (successful runs):** Human 4.78, System 4.55 (Δ = -0.22)
- **Avg time per run:** Human 159.4s, System 91.5s (Δ = -68.0s)
- **Paths where humans achieved optimal:** 2/8 (Symphony→Navy, Billy Joel→VANOS)
- **Paths where system achieved optimal:** 4/8 (per EVALUATION_REPORT)

### Path-by-Path

- **Fewer hops (human better):** Patrick Star -> John Wall, Billy Joel -> VANOS, David Gilmour -> Michael Phelps
- **Fewer hops (system better):** Minecraft -> Chess, McDonald's -> Yoga, John Mulaney -> Apple, OpenAI -> Esports, Symphony -> United States Navy
- **Faster (human better):** Patrick Star -> John Wall
- **Faster (system better):** Minecraft -> Chess, McDonald's -> Yoga, John Mulaney -> Apple, OpenAI -> Esports, Symphony -> United States Navy, Billy Joel -> VANOS, David Gilmour -> Michael Phelps

## Data Files

- `human_statistics.json` - Human results (from human_results/*.csv)
- `aggregated_statistics.json` - System results (from eval harness)
