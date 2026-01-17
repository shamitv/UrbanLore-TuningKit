# Initial setup script

Use the init script to create required data folders and initialize .env from .env.example (only if .env does not already exist).

## Run

```bash
python init_setup.py
```

## Options

- `--skip-env`: Skip .env creation entirely.

## What it creates

- `corpus/`
- `dataset/`
- `finetune/models/`
- `eval/results/`

## Notes

- The script never overwrites an existing .env.
- Update .env with your API key and preferred settings after initialization.
