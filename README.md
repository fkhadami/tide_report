# Tide Agent

Streamlit app for tide-data analysis and report generation.

It supports:
- CSV upload with configurable time and water-level columns
- summary statistics
- Hampel-based spike cleaning
- FFT spectrum analysis
- UTide harmonic analysis
- station map rendering with high-resolution coastline support
- optional LLM-generated interpretation
- PDF report export

## Repository Notes

This repository is prepared for GitHub without local runtime artifacts.

Not committed by default:
- `tide_env/`
- `outputs/`
- `data/cartopy/` coastline cache
- local secrets

If you need offline coastline rendering, generate the Cartopy cache locally after install.

## Requirements

- Python 3.9+ recommended
- macOS, Linux, or Windows
- internet access on first coastline download unless you pre-populate `data/cartopy`

Install Python packages:

```bash
python3 -m venv tide_env
./tide_env/bin/pip install -r requirements.txt
```

On Windows:

```bat
python -m venv tide_env
tide_env\Scripts\pip install -r requirements.txt
```

## Run The App

macOS/Linux:

```bash
./tide_env/bin/streamlit run app.py
```

Windows:

```bat
tide_env\Scripts\streamlit.exe run app.py
```

The project also includes launcher scripts:
- [`Launch Tide Agent.command`](/Users/fkhadami/Documents/tide_agent/Launch%20Tide%20Agent.command)
- [`Launch Tide Agent.bat`](/Users/fkhadami/Documents/tide_agent/Launch%20Tide%20Agent.bat)

## Deploy Online

The simplest way to publish this app is with Streamlit Community Cloud.

Repository settings:
- Repository: `fkhadami/tide_report`
- Branch: `main`
- Main file path: `app.py`

Steps:
1. Sign in at https://share.streamlit.io/ with your GitHub account.
2. Click `Create app`.
3. Select repository `fkhadami/tide_report`.
4. Set branch to `main`.
5. Set main file path to `app.py`.
6. Deploy.

Optional secret for LLM narrative:

```toml
OPENAI_API_KEY="your_api_key_here"
```

Add that value in the app settings under `Secrets` after deployment. If you skip it, the rest of the app still works and only the LLM-generated narrative is unavailable.

## Input CSV Format

The app expects a CSV containing:
- a timestamp column, default name: `time`
- a water-level column, default name: `wl`

Example:

```csv
time,wl
2024-01-01 00:00:00,1.12
2024-01-01 01:00:00,1.34
2024-01-01 02:00:00,1.08
```

## Coastline Data

The map renderer prefers Natural Earth `10m` land/coastline data and can fall back to other local layers when needed.

To prepare offline coastline data locally:

```bash
./tide_env/bin/python scripts/download_gshhs_offline.py
```

This populates `data/cartopy/` with:
- GSHHS coastline data
- Natural Earth `10m` land
- Natural Earth `10m` coastline

If you want to use a custom Cartopy cache directory, set:

```bash
export TIDE_AGENT_CARTOPY_DATA_DIR=/path/to/cartopy-data
```

## Analysis Constraints

FFT and UTide currently require:
- complete water-level data
- one sample per timestamp
- regular sampling interval

If the uploaded data contains missing values, duplicate timestamps, or irregular gaps, the app will reject FFT/UTide instead of producing misleading results.

## Project Structure

- [`app.py`](/Users/fkhadami/Documents/tide_agent/app.py): Streamlit UI
- [`analyze.py`](/Users/fkhadami/Documents/tide_agent/analyze.py): analysis and plotting logic
- [`report_generator.py`](/Users/fkhadami/Documents/tide_agent/report_generator.py): PDF generation
- [`llm_writer.py`](/Users/fkhadami/Documents/tide_agent/llm_writer.py): OpenAI-based narrative generation
- [`scripts/download_gshhs_offline.py`](/Users/fkhadami/Documents/tide_agent/scripts/download_gshhs_offline.py): offline coastline cache setup
- [`tests/test_analyze.py`](/Users/fkhadami/Documents/tide_agent/tests/test_analyze.py): regression tests for analysis validation

## Testing

Run tests with:

```bash
PYTHONPYCACHEPREFIX=.pycache ./tide_env/bin/python -m unittest discover -s tests -v
```

## Suggested GitHub Workflow

Initialize and push:

```bash
git add .
git commit -m "Prepare Tide Agent for GitHub"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Before Publishing

Recommended follow-ups:
- choose and add a license
- pin dependency versions more strictly if you want reproducible installs
- add screenshots to this README
- add a small sample dataset for demos
