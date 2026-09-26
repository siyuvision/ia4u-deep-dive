"""Check the released evidence hashes and recompute paired metrics; no data/GPU needed."""
import hashlib
import json
from math import isclose
from statistics import mean
from project_env import PROJECT

def main():
    evidence=PROJECT/'evidence'
    load=lambda name:json.loads((evidence/name).read_text(encoding='utf-8'))
    for record in load('provenance.json'):
        actual=hashlib.sha256((PROJECT/record['published_file']).read_bytes()).hexdigest()
        assert actual==record['published_sha256'],record['published_file']
    first=load('two-shot-metrics.json')
    old={r['case']:r['references']['STAPLE'] for r in first['cases']}
    for key in ['dice','iou']:
        assert isclose(mean(r[key] for r in old.values()),first['macro_by_reference']['STAPLE'][key]['mean'],abs_tol=1e-12)
    follow=load('four-shot-summary.json')
    assert set(follow['training_cases']).isdisjoint(follow['evaluation_cases'])
    assert set(follow['evaluation_cases'])<=set(old)
    for stage in ['two_case_raw','four_case_raw','four_case_guarded']:
        rows=[old[r['case']] if stage=='two_case_raw' else r['four_case']['raw' if stage=='four_case_raw' else 'guarded'] for r in follow['cases']]
        for key in ['dice','iou','precision','recall']:
            assert isclose(mean(r[key] for r in rows),follow['macro'][stage][key],abs_tol=1e-12)
    for row in follow['cases']:
        assert row['old_two_case_raw']==old[row['case']]
        assert row['four_case']['raw']['tp']==row['four_case']['guarded']['tp']
    assert sum(r['four_case']['raw']['fp_beyond_10mm'] for r in follow['cases'])==18642
    assert sum(r['four_case']['guarded']['fp_beyond_10mm'] for r in follow['cases'])==0
    print('Published evidence verified: 9 hashes, first five-case mean, three paired stages, no TP removed.')

if __name__=='__main__':main()
