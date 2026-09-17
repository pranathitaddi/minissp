# Running minissp locally

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Verify

```bash
pytest -q
```

You should see all tests pass.

## Fetch data

```bash
python scripts/get_data.py
```
