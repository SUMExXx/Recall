PY := .venv/Scripts/python

.PHONY: setup test demo ask consolidate stats clean

setup:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

demo:
	$(PY) -m recall_memory --db demo.db demo

# usage: make ask Q="Who decided to use JWT?"
ask:
	$(PY) -m recall_memory --db demo.db ask "$(Q)"

consolidate:
	$(PY) -m recall_memory --db demo.db consolidate

stats:
	$(PY) -m recall_memory --db demo.db stats

clean:
	-rm -f demo.db recall.db
