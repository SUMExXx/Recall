PY := .venv/Scripts/python
BACKEND ?= ollama          # laptop default; event PC uses npu (RECALL_BACKEND)
FULL ?=

.PHONY: setup test demo ask consolidate stats hub clean

setup:
	py -3.12 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

demo:
	$(PY) -m recall_memory --db demo.db --backend $(BACKEND) demo

# usage: make ask Q="Who decided to use JWT?"
ask:
	$(PY) -m recall_memory --db demo.db --backend $(BACKEND) ask "$(Q)"

# usage: make consolidate            (MVP jobs)
#        make consolidate FULL=1     (full Dream tier)
consolidate:
	$(PY) -m recall_memory --db demo.db --backend $(BACKEND) consolidate $(if $(FULL),--full,)

stats:
	$(PY) -m recall_memory --db demo.db --backend $(BACKEND) stats

hub:
	$(PY) scripts/run_hub.py --backend $(BACKEND)

clean:
	-rm -f demo.db recall.db demo_hub.db*
