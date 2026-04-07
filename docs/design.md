# Data Pipeline Redesign: NielsenIQ Homescan to K-Aggregator Estimation

## Context

The current pipeline is split across two repos (`Nevo_Replication` and
`K-Aggregators`) with five stages, each maintained by different people
at different times.  Three factual claims in the paper were wrong
(dates, frequency, product count) because the documentation drifted
from the code.  Extending to the full 2004-2024 NielsenIQ homescan
data (available via Globus) requires a ground-up rebuild rather than
patching.

The new pipeline should be a single self-contained project that goes
from raw NielsenIQ extracts to estimation-ready objects for both
market-level and household-level K-aggregator (and BLP) estimation,
with every filtering decision documented in the code itself.

## Design Principles

1. **Single repo, single script** — no DVC, no cross-repo imports, no
   intermediate parquet state that can drift from the code that
   produced it.
2. **Config-driven** — time period, aggregation frequency (quarterly /
   monthly), product-selection threshold, market definition, and
   demographic variables all set in one YAML config, not scattered
   across env vars and hardcoded constants.
3. **Validation at every stage** — row counts, value ranges, rank
   checks, and hash digests printed and logged so silent data loss is
   impossible.
4. **Two output tiers** — market-level objects (for the current
   estimation) and household-level objects (for the microdata
   extension), both from the same pipeline with a shared product/market
   definition.
5. **Instrument construction is estimation-side** — the data pipeline
   produces (S, p, zeta, demog); instrument construction (LF,
   Bartik, GH) lives in the estimation code where it belongs, because
   instrument design is a modeling decision, not a data decision.

## Pipeline Architecture

### Stage 1: Extract (NielsenIQ raw -> working parquets)

**Input**: NielsenIQ Consumer Panel directory on cluster (transferred
via Globus from Kilts Center).

**Tool**: `kiltsreader` (pin version in requirements.txt).

**Config controls**:
- `years: [2004, 2024]` — full available range
- `modules: [1344]` — RTE cereal (extensible to other categories)

**Output**: Four parquets in `data/raw/`:
- `purchases.parquet` — (upc, household, trip, quantity, price, coupon)
- `trips.parquet` — (trip, household, date, store, retailer, dma)
- `panelists.parquet` — (household, year, income, age, children, dma, ...)
- `products.parquet` — (upc, brand_code, description, size, module)

**Validation**: Print year range, row counts, unique household/product
counts.  Hash each output file.

### Stage 2: Product Attributes (nutrition data)

**Current problem**: Hand-maintained `cereal_fdcid.csv` mapping 50
brands to FDC IDs.  No audit trail, silently drops new brands.

**New design**:
- Programmatic lookup against USDA FoodData Central API, keyed by
  brand description.
- Manual overrides in `config/nutrition_overrides.yaml` (versioned).
- The override file documents *why* each override exists (ambiguous
  name, FDC entry missing, etc.).
- Output: `products_with_nutrition.parquet` — all products in the
  module, with nutrition columns (sugar, fiber, protein, fat, sodium)
  and a `has_nutrition` flag.

**Product selection**: Top N brands by total expenditure over the full
sample period.  N is a config parameter (currently 45).  The selection
is logged with expenditure shares so the cutoff is auditable.

### Stage 3: Market-Level Aggregation

**Config controls**:
- `frequency: quarterly` (or `monthly`)
- `market_def: dma` (or `dma_x_retailer`, `store`)
- `n_markets: 65` (top DMAs by household count)
- `market_size_oz_per_day: 4.5` (outside good calibration)
- `price_deflator: CPI` (with base period in config)
- `base_period: Q1Y2014` (for Bartik predetermined shares)

**Process**:
1. Merge purchases x trips x panelists x products
2. Filter to selected products and markets
3. Aggregate to (market, time, product) cells: sum servings, mean price
4. Compute shares: servings / potential market size
5. Deflate prices
6. Compute market-time aggregates: Z_lt = sum_j S_jt * zeta_jl

**Output**: `market_panel.parquet`
- Index: (t, m, j) — same triple index as current
- Columns: S (shares), p (prices), zeta (5 nutrition cols),
  Z_agg (5 market-agg cols), plus identifiers

**Validation**: Check shares sum < 1 per market-time cell.  Flag any
share > 0.3 (implausible concentration).  Print summary stats table
matching the paper's Table 1.

### Stage 4: Household-Level Panel

Same merge as Stage 3 but aggregated to (household, time, product)
cells instead of (market, time, product).

**Output**: `household_panel.parquet`
- Index: (i, t, j) — household x time x product
- Columns: S_ij (household share), p (prices, same within market),
  zeta (nutrition), Z_i (household-level aggregates), dma, demographics

**Additional output**: `household_base_shares.parquet`
- Household-level predetermined shares from base period (for
  household Bartik instruments)
- Index: (i, j)
- Columns: S_ij_0, dma

**Validation**: Print positive-purchase cell count, median products
per household-quarter, fraction of household-quarters with rank >= 2
characteristic span.

### Stage 5: Demographics

**Output**: `demographics.parquet`
- Market-level means (for Pi regressors): (t, m) x demographics
- Household-level (for microdata estimation): (i, t) x demographics

No Bartik instrument construction here — that's estimation-side.

### What Moves to Estimation Code

The current pipeline generates 13 instrument sets in
`generate_instruments.py`.  In the new design, **all instrument
construction lives in the estimation repo**, because:

- The LF vs FL distinction is a modeling insight, not a data property
- The Bartik construction depends on the base period choice
- The FWL transformation is estimation-specific
- Different estimation approaches (market-level, household-level, BLP)
  need different instruments from the same data

The estimation code receives (S, p, zeta, Z_agg, demographics,
household_base_shares) and constructs whatever instruments it needs.

## Key Differences from Current Pipeline

| Aspect | Current | Redesign |
|--------|---------|----------|
| Repos | 2 (Nevo_Replication + K-Aggregators) | 1 self-contained |
| Years | 2014-2018 (frozen) | 2004-2024 (config-driven) |
| Frequency | Quarterly only | Config: quarterly or monthly |
| Products | 45 (implicit in FDC mapping) | Top-N by expenditure (config) |
| Instruments | 13 sets generated in data pipeline | Estimation-side only |
| Demographics | Scattered across agent_data + bartik_demographics | Single demographics.parquet |
| Household data | Not produced | Full (i,j,t) panel |
| Validation | None | Every stage prints counts + hashes |
| Config | Env vars + hardcoded constants | Single YAML |
| Nutrition | Hand-maintained CSV | FDC API + versioned overrides |

## Verification Plan

1. Run the new pipeline on the 2014-2018 window with current config
   and verify it reproduces the existing `market_panel` exactly (same
   N=58,037, same summary statistics, same product list).
2. Run the current estimation code (`cereal_gmm_setup.py` +
   `cereal_gmm_estimate.py`) on the new pipeline's output and verify
   identical results (theta=1.36, lambda_1=2.18, lambda_2=0.48).
3. Extend to 2004-2024 and document: how many products enter/exit,
   how the summary statistics change, whether the 45-product selection
   is stable.
4. Estimate on the extended sample and compare.

## Open Questions

- **Product entry/exit over 2004-2024**: Some of the current 45 brands
  may not exist in early years.  Options: (a) restrict to products
  available throughout, (b) allow unbalanced panel with product FE, (c)
  define cohorts.  This is a modeling decision that should be deferred
  to after seeing the data.
- **Store-level vs DMA-level markets**: The raw data has store
  identifiers.  Finer market definition gives more price variation but
  smaller cells.  Worth exploring but not for the first pass.
- **Monthly aggregation**: More time periods but sparser
  household-quarter cells.  The quarterly choice was never justified;
  the new pipeline should make it easy to compare.
- **Globus transfer automation**: Can the extraction stage pull
  directly from Kilts via Globus, or does it require a manual transfer
  step?  If the latter, document the exact Globus endpoint and
  collection IDs.
