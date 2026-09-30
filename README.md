# Automated Customer Support Ticket Routing & Response

A ticket-triage system that receives support tickets (subject, description, priority), categorizes them, analyses sentiment and complexity, routes them to the best-suited agent, predicts resolution time and escalation risk, and drafts an initial reply. It can be used through a **REST API** or a **customer chat in the terminal**.

- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Terminal chat](#terminal-chat)
- [REST API](#rest-api)
- [How each component works](#how-each-component-works)
- [Data and training](#data-and-training)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations and next steps](#limitations-and-next-steps)

## Features

| Requirement | Implementation |
|---|---|
| Receive tickets with subject, description, priority | `POST /tickets` (validated with Pydantic) and the terminal chat |
| Basic keyword categorization | Weighted keyword rules; subject words count double |
| Assign tickets to support queues | Billing, Technical Support, Account Management, Shipping & Delivery, General Support |
| Intelligent categorization | TF-IDF + logistic regression trained on historical tickets, with keyword fallback when not confident |
| Sentiment analysis | Support-tuned lexicon: negation, intensifiers, CAPS, exclamation marks |
| AI agent for pre-written / drafted replies | Template ranking by similarity; Claude drafts a personalised reply grounded in the top templates, with template fallback |
| Routing by complexity and agent expertise | Complexity score sets required seniority; agents are scored on expertise, level fit and current load |
| Predict resolution time and escalations | Gradient-boosting regressor (hours) and classifier (escalation probability) with human-readable reasons |

## Architecture

```
  Customer                               Support staff / integrations
  ┌──────────────────────┐               ┌────────────────────────────┐
  │ Terminal chat        │               │ REST API (FastAPI)         │
  │ scripts/chat.py      │               │ ticket_router/api.py       │
  └──────────┬───────────┘               └──────────────┬─────────────┘
             │        TicketCreate(subject, description, priority)
             └──────────────────────┬───────────────────┘
                                    ▼
              ┌─────────────────────────────────────────────┐
              │ TicketProcessor  (pipeline.py)              │
              │                                             │
              │ 1. Keyword categorizer   basic rules        │
              │ 2. ML categorizer        TF-IDF + LogReg    │
              │ 3. Queue assigner        category → queue   │
              │ 4. Sentiment analyzer    -1 … +1            │
              │ 5. Complexity analyzer   low / medium / high│
              │ 6. Predictors            resolution hours,  │
              │                          escalation risk    │
              │ 7. Agent router          best agent         │
              │ 8. Response agent        top-3 templates +  │
              │                          draft reply        │
              └──────────────────────┬──────────────────────┘
                                     ▼
                     Ticket + full analysis → TicketStore
```

Design principles:

- **Explainable.** Every step returns structured output with its reasoning (keyword scores, escalation reasons, routing rationale), so a routing decision can be audited.
- **Works offline.** Claude improves draft replies but is never required; without an API key the system fills in the best-matching template.
- **Layered intelligence.** The simple keyword categorizer is kept as the baseline and the fallback for the ML model.
- **UI-free core.** The chat logic (`chat.py`) has no terminal code, so it is tested directly and could be reused behind a web UI.

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py   # create synthetic historical tickets (data/historical_tickets.csv)
python scripts/train.py           # train models into artifacts/ (takes a few seconds)
```

Models are also trained automatically on first start if `artifacts/` is empty.

```bash
python scripts/chat.py            # chat as a customer in the terminal
uvicorn ticket_router.api:app     # run the REST API on http://localhost:8000 (docs at /docs)
python scripts/demo.py            # run 5 sample tickets and print the full analysis
```

To have Claude write the draft replies, set an API key first: `export ANTHROPIC_API_KEY=...`.

## Terminal chat

```bash
python scripts/chat.py [--agent-view] [--no-llm]
```

Ava, the support assistant, collects the issue, asks for more detail if the message is too short, asks for a priority (1-4), then creates the ticket and shows a ticket card and the drafted reply.

| Command | Action |
|---|---|
| `/details` | Toggle the internal agent view (category, sentiment, complexity, escalation risk, routing, suggested templates) |
| `/tickets` | List tickets opened in this chat |
| `/new` | Start a new request |
| `/export` | Save the whole chat as a Markdown transcript in `exports/` |
| `/help` | Show commands |
| `/quit` | Leave the chat |

`--agent-view` starts with the internal analysis on; `--no-llm` always drafts from templates.

## REST API

Start with `uvicorn ticket_router.api:app --reload`; interactive docs are at `http://localhost:8000/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/tickets` | Submit a ticket and get the full analysis |
| `GET` | `/tickets` | List tickets; filter with `?queue=Billing&status=open` |
| `GET` | `/tickets/{id}` | Get one ticket |
| `POST` | `/tickets/{id}/resolve` | Mark resolved and free the assigned agent |
| `GET` | `/queues` | Open-ticket count and IDs per queue |
| `GET` | `/agents` | Agents with current workload |
| `GET` | `/health` | Status and whether LLM drafting is active |

Example:

```bash
curl -X POST localhost:8000/tickets -H "Content-Type: application/json" -d '{
  "subject": "Charged twice",
  "description": "I was billed twice this month and nobody answered my last email. This is unacceptable!",
  "priority": "high"
}'
```

`priority` is one of `low`, `medium`, `high`, `urgent`. The response contains the ticket plus an `analysis` object with `categorization`, `sentiment`, `complexity`, `predicted_resolution_hours`, `escalation`, `routing`, `suggested_responses` and `draft_reply`.

## How each component works

**Keyword categorizer** (`keyword_categorizer.py`) - each category has weighted keywords and phrases (e.g. "charged twice" outweighs "plan"). Subject matches count double. No matches means `General`.

**Intelligent categorizer** (`intelligent_categorizer.py`) - word and character n-gram TF-IDF into logistic regression. If the top probability is below `ML_CONFIDENCE_THRESHOLD` (0.55) and the keyword rules found something, the keyword answer wins. The result records which method decided and both answers.

**Sentiment** (`sentiment.py`) - lexicon scoring with negation (flips and dampens), intensifiers, ALL-CAPS and exclamation boosts, normalised to [-1, 1] and labelled positive / neutral / negative / very_negative.

**Complexity** (`complexity.py`) - weighted blend of length, technical terms, multiple issues, prior attempts or repeat contact, and whether strong signals span several categories. Output is a score plus low / medium / high.

**Predictors** (`predictors.py`) - features are category, priority, sentiment, complexity, length and escalation-language count. Resolution time is regressed on log-hours (times are right-skewed). Escalation is a probability with a 0.5 "likely" threshold; reasons are derived from the same signals.

**Agent router** (`routing.py`) - required seniority starts from complexity (low → junior, medium → intermediate, high → senior) and goes up one level for urgent tickets or likely escalations. Candidates must have spare capacity and at least 0.5 proficiency in the category. Among agents at or above the required level the router scores `0.5 × expertise + 0.25 × level fit + 0.25 × availability`; level fit favours the least-senior qualified agent so seniors stay free for hard tickets. If no one at the required level is free, the best available agent is chosen and the reason says so. If nobody qualifies, the ticket waits unassigned in its queue. Workloads are updated on assignment and released on resolve.

**Response agent** (`responses.py`) - ranks the 18 templates by TF-IDF cosine similarity with a boost for the ticket's category and returns the top three. The drafter then either asks Claude for a personalised reply (structured JSON output, customer text treated as data, not instructions, server-side model fallback enabled) or fills in the best template with an opener matched to the customer's sentiment. Any API error, refusal or malformed output falls back to the template draft.

## Data and training

`data/historical_tickets.csv` holds ticket text, priority, true category, resolution hours and whether the ticket escalated. **The bundled file is synthetic**, generated by `scripts/generate_data.py`, so the models run out of the box. To use real history, replace the CSV with the same columns (`subject, description, priority, category, resolution_hours, escalated`) and run `python scripts/train.py`.

Latest held-out results on the synthetic data (`artifacts/metrics.json`):

| Model | Metric | Value |
|---|---|---|
| Categorizer | Accuracy | 1.00 (synthetic text is templated; hand-written unseen tickets in the tests are the more realistic check) |
| Resolution time | Mean absolute error | about 4.7 hours |
| Escalation | ROC-AUC | about 0.81 (base rate 20%) |

Because the data is synthetic, these numbers show the pipeline works, not production accuracy.

`data/agents.json` defines the support team (team, level 1-3, per-category skill, max load). `data/response_templates.json` holds the pre-written replies.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude-drafted replies |
| `TICKET_LLM_MODE` | `auto` | `auto` uses Claude only if credentials exist; `true` / `false` force it |
| `TICKET_LLM_MODEL` | `claude-opus-5` | Model used for drafting |
| `TICKET_LLM_TIMEOUT` | `60` | Seconds before an LLM call gives up (template fallback) |
| `TICKET_ARTIFACTS_DIR` | `artifacts/` | Where trained models are stored |

Thresholds (ML confidence, escalation, complexity cut-offs, suggestions per ticket) are in `ticket_router/config.py`.

## Project structure

```
ticket_router/
  api.py                   FastAPI app
  pipeline.py              TicketProcessor - runs every step
  keyword_categorizer.py   basic keyword rules
  intelligent_categorizer.py  ML categorizer with keyword fallback
  queues.py                category → queue
  sentiment.py             sentiment analysis
  complexity.py            complexity scoring
  predictors.py            resolution-time and escalation models
  routing.py               agent routing
  responses.py             template ranking and draft generation
  training.py              training and model persistence
  chat.py                  chat conversation logic
  chat_cli.py              terminal UI (rich)
  chat_export.py           Markdown transcript export
  store.py                 in-memory ticket store
  schemas.py, config.py, text_utils.py
data/                      agents, templates, historical tickets
scripts/                   generate_data, train, demo, chat
tests/                     pytest suite
```

## Testing

```bash
pytest
```

The 34 tests cover keyword and ML categorization (including unseen wording), sentiment, complexity, routing rules (seniority, expertise, load, release, fallback), the full pipeline, the Claude path with a fake client (success, refusal, malformed output), the API, and the chat flow and export.

## Limitations and next steps

- **In-memory storage.** Tickets and agent workloads reset on restart. Next: a database behind `TicketStore`.
- **No authentication or personal-data masking.** Add API auth, and mask card numbers, emails and phone numbers before storing tickets or sending them to Claude.
- **Synthetic training data.** Retrain on real ticket history before relying on the predictions.
- **Fixed queues.** The five queues and categories follow the spec; a different business (for example food delivery) needs its own categories, templates and training examples, since unknown topics are forced into the closest existing queue.
- **Single process.** Agent load is tracked in memory, so running several API workers would need shared state.
