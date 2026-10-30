"""Compare the accepted, xw1, and combined xw2 panel vintages.

The xw1 -> xw2 transition changes only the household panel's deterministic
panel-year tie-break and adds household_size to demographics_household.
The accepted -> xw2 comparison is descriptive because it also includes the
crosswalk route and pinned-brand changes.
"""
import argparse
from pathlib import Path

import pandas as pd


HOUSEHOLD_COLS = ["panel_year", "m", "S", "servings"]


def _changed(a, b, cols, index):
    left, right = a.loc[index], b.loc[index]
    return {c: int((~(left[c].eq(right[c]) |
                       (left[c].isna() & right[c].isna()))).sum())
            for c in cols}


def compare(accepted, xw1, xw2, rms_xw1, rms_xw2):
    accepted, xw1, xw2 = map(Path, (accepted, xw1, xw2))
    rms_xw1, rms_xw2 = map(Path, (rms_xw1, rms_xw2))
    a = pd.read_parquet(accepted / "household_panel.parquet",
                        columns=HOUSEHOLD_COLS)
    one = pd.read_parquet(xw1 / "household_panel.parquet")
    two = pd.read_parquet(xw2 / "household_panel.parquet")
    assert one.index.equals(two.index), "xw1/xw2 household cell keys or order differ"
    assert list(one.columns) == list(two.columns), "xw1/xw2 household columns differ"
    assert one.dtypes.equals(two.dtypes), "xw1/xw2 household dtypes differ"
    for c in one.columns.difference(["panel_year", "m", "S"]):
        assert one[c].equals(two[c]), f"xw1/xw2 household {c} differs"
    changes = _changed(one, two, HOUSEHOLD_COLS, one.index)
    assert changes["servings"] == 0, "xw1/xw2 household servings changed"
    if changes["panel_year"]:
        moved = ~(one["panel_year"].eq(two["panel_year"]))
        calendar_year = two.index.get_level_values("t").str[-4:].astype(int)
        assert (two["panel_year"].to_numpy()[moved.to_numpy()]
                == calendar_year[moved.to_numpy()]).all(), \
            "new dominant panel year did not choose the calendar year"
    assert changes["m"] <= changes["panel_year"]
    assert changes["S"] <= changes["panel_year"]

    base_one = pd.read_parquet(xw1 / "household_base_shares.parquet")
    base_two = pd.read_parquet(xw2 / "household_base_shares.parquet")
    assert base_one.equals(base_two), "base shares differ xw1/xw2"
    for filename in ("market_panel.parquet", "demographics_market.parquet"):
        assert pd.read_parquet(xw1 / filename).equals(pd.read_parquet(xw2 / filename)), \
            f"{filename} differs xw1/xw2"
    assert pd.read_parquet(rms_xw1 / "market_panel_rms.parquet").equals(
        pd.read_parquet(rms_xw2 / "market_panel_rms.parquet")), \
        "RMS panel differs xw1/xw2"

    dh_one = pd.read_parquet(xw1 / "demographics_household.parquet")
    dh_two = pd.read_parquet(xw2 / "demographics_household.parquet")
    assert "household_size" not in dh_one and "household_size" in dh_two
    size = dh_two["household_size"]
    assert size.notna().all() and pd.api.types.is_integer_dtype(size)
    assert size.between(1, 9).all()
    assert dh_two.drop(columns="household_size").equals(dh_one), \
        "other household demographics differ xw1/xw2"

    common = a.index.intersection(two.index)
    accepted_changes = _changed(a, two, HOUSEHOLD_COLS, common)
    def js(frame):
        return set(frame.index.get_level_values("j").astype("int64"))
    L = ["* Panel comparison: accepted, xw1, and combined xw2", "",
         "The UPC-dollar route tables above compare the accepted route with the",
         "crosswalk route.  This section compares built panel files.  The",
         "xw1 -> xw2 assertions isolate the household fixes: market and RMS",
         "panels, market demographics and base shares are exactly equal; only",
         "household panel_year, m and S may change, and household_size is added.",
         "", "| measure | accepted | xw1 | xw2 |",
         "|-|-:|-:|-:|",
         f"| household rows | {len(a):,} | {len(one):,} | {len(two):,} |",
         f"| household brands | {len(js(a))} | {len(js(one))} | {len(js(two))} |",
         f"| household_size non-null rows | 0 | 0 | {int(size.notna().sum()):,} |",
         "", f"Accepted/xw2 shared (i,t,j) cells: {len(common):,}.",
         f"Accepted-only brands: {sorted(js(a) - js(two))}.",
         f"xw2-only brands: {sorted(js(two) - js(a))}.",
         "", "| column | accepted -> xw2, shared cells changed | xw1 -> xw2, all cells changed |",
         "|-|-:|-:|"]
    for c in HOUSEHOLD_COLS:
        L.append(f"| ={c}= | {accepted_changes[c]:,} | {changes[c]:,} |")
    L += ["", "Gates: xw1/xw2 household keys, order and dtypes are identical;",
          "servings and base shares are identical; every changed panel_year",
          "chooses year(t); household_size is integer 1..9 with full coverage;",
          "all other demographics, the market panel, and RMS panel are identical.",
          ""]
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accepted", "xw1", "xw2", "rms-xw1", "rms-xw2", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = compare(args.accepted, args.xw1, args.xw2,
                     args.rms_xw1, args.rms_xw2)
    Path(args.out).write_text(result)
    print(result)


if __name__ == "__main__":
    main()
