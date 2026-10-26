.PHONY: install demo api test
install: ; pip install -r requirements.txt
demo:    ; python scripts/run_pipeline.py
api:     ; uvicorn climaverse.api.app:app --reload
test:    ; pytest -q
