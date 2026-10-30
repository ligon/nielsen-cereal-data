"""Identified inside transfers preserve the module and reject unresolved cells."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'src'))
from rms_outside_audit import panel_cells, residual_table
from rms_outside_routing import routed_inputs, KNOWN, UNKNOWN, UNRESOLVED


def world():
    inside = pd.DataFrame(dict(t=['Q1Y2022']*2,m=[501,502],j=[10,10],
        revenue=[100.,100.],servings=[10.,10.])).set_index(['t','m','j'])
    rows = [(501,'modeled_45',100.,100.,10.,1,0),
            (501,'selected_brand_excluded',20.,0.,0.,1,1),
            (501,'other_named',30.,20.,2.,2,1),
            (502,'modeled_45',100.,100.,10.,1,0),
            (502,'selected_brand_excluded',5.,0.,0.,1,1),
            (502,'unmapped',15.,0.,0.,1,1)]
    cats = pd.DataFrame(rows,columns=['m','category','revenue','covered_revenue','covered_ounces','rows','uncovered_rows'])
    cats['tcode'] = 20221
    cats = cats.set_index(['tcode','m','category']).sort_index()
    baseline = residual_table(cats,panel_cells(inside))
    details = pd.DataFrame(dict(file_year=[2022]*4,upc=[1,2,3,4],tcode=[20221]*4,
        m=[501,501,502,502],category=['selected_brand_excluded','other_named','selected_brand_excluded','unmapped'],
        revenue=[20.,10.,5.,15.],units=[2.,1.,1.,5.],rows=[1]*4,uncovered_rows=[1]*4))
    candidates = pd.DataFrame(dict(year=[2022]*4,upc=[1,2,3,4],mass=[4.,5.,np.nan,2.],
        brand_code_uc=[10.,99.,10.,np.nan],catalog_versions=[np.nan,np.nan,np.nan,2.],
        catalog_mass_min_oz=[np.nan]*4,catalog_mass_max_oz=[np.nan]*4,source_abs_gap_oz=[np.nan]*4))
    decisions = pd.DataFrame(dict(year=[2022]*4,upc=[1,2,3,4],
        category=details.category,action=[KNOWN,'retain_existing_outside',UNKNOWN,UNRESOLVED],
        proposed_j=[10.,np.nan,10.,np.nan],revenue=[9999.]*4,units=[9999.]*4))
    return inside,baseline,cats,details,candidates,decisions


def test_transfer_revenue_and_ounces_once_and_preserve_originals():
    args = world()
    originals = [x.copy(deep=True) for x in args]
    audit,cats,inc,summary = routed_inputs(*args,'mass')
    assert inc.loc[(20221,501,10)].tolist() == [20.,8.]
    a = audit.loc[(20221,501)]
    assert a.module_revenue == 150 and a.modeled_revenue == 120
    assert a.residual_revenue == 30 and a.residual_covered_ounces == 7
    assert a.complete_residual_dollars_per_ounce == pytest.approx(30/7)
    assert a.selected_brand_excluded_revenue == 0
    b = audit.loc[(20221,502)]
    assert b.selected_brand_excluded_revenue == 5
    assert b.unresolved_partition_revenue == 15
    assert np.isnan(b.complete_residual_dollars_per_ounce)
    assert summary['transferred_revenue'] == 20
    assert summary['routing_and_mass_eligible_markets'] == 1
    assert cats.loc[(20221,501,'modeled_45'),'covered_ounces'] == 18
    for got,old in zip(args,originals):
        pd.testing.assert_frame_equal(got,old,check_exact=True)


@pytest.mark.parametrize('change,match',[
    ('brand','existing adopted j'),('missing_mass','supported mass'),
    ('decision','explicit routing decision'),('excluded','exhaust selected-brand'),
    ('catalog','explicit unresolved/assortment')])
def test_unsupported_reassignment_fails_closed(change,match):
    args = list(world())
    if change == 'brand': args[5].loc[0,'proposed_j'] = 99
    if change == 'missing_mass': args[4].loc[0,'mass'] = np.nan
    if change == 'decision': args[5] = args[5].iloc[1:]
    if change == 'excluded': args[1].loc[(20221,501),'selected_brand_excluded_revenue'] += 1
    if change == 'catalog': args[5].loc[3,'action'] = 'retain_existing_outside'
    with pytest.raises(ValueError,match=match): routed_inputs(*args,'mass')


def test_mixed_catalog_with_known_mass_still_excludes_market():
    args = list(world())
    args[4].loc[2,'mass'] = 1.
    args[5].loc[2,'action'] = KNOWN
    audit,_,_,summary = routed_inputs(*args,'mass')
    assert audit.complete_residual_dollars_per_ounce.notna().all()
    assert audit.loc[(20221,502),'unresolved_partition_revenue'] == 15
    assert summary['routing_and_mass_eligible_markets'] == 1
