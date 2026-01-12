# Canada Macro–Financial Stress Index

This project constructs a monthly macro–financial stress index for Canada using public data from Statistics Canada, the Bank of Canada, OECD, and FRED.

## Features
- Automated data retrieval via APIs and bulk downloads
- Macro and financial stress sub-indices
- Lead–lag validation against unemployment
- Recession-shaded visualizations
- Power BI–ready outputs

## Data Sources
- Statistics Canada
- Bank of Canada (Valet API)
- OECD
- Federal Reserve Economic Data (FRED)

## Project Structure
- data/processed/    # Final CSV outputs
- src/               # Data ingestion and index construction code
- figures/           # Charts

## How to Run
1. Install requirements
2. Run `Early-Warning Stress Code.py`
3. Outputs are saved to `data/processed/`

## Disclaimer
This index is for educational and analytical purposes only and is not a forecasting model.
