"""
parse_md.py — MD 결과 파일 파서
generate_ppt.py 에서 import 하여 사용
"""
import re


def _strip(s):
    return re.sub(r'\*\*(.+?)\*\*', r'\1', s).strip()


def _f(s):
    """float 파싱: +/−/e-notation/<0.001 모두 처리."""
    s = _strip(s).replace('−', '-').replace('–', '-').lstrip('+')
    if s.startswith('<'):
        try:
            return float(s[1:]) * 0.5
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _table_rows(text, start_marker):
    """start_marker 이후 첫 번째 markdown 테이블의 데이터 행 리스트 반환."""
    idx = text.find(start_marker)
    if idx < 0:
        return []
    chunk = text[idx + len(start_marker):]
    rows = []
    in_table = False
    header_done = False
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            if in_table and rows:
                break
            continue
        in_table = True
        cells = [_strip(c) for c in line.split('|')[1:-1]]
        if not cells:
            continue
        if all(re.match(r'^[-:= ]+$', c) for c in cells):
            header_done = True
            continue
        if not header_done:
            continue
        rows.append(cells)
    return rows


# ─────────────────────────────────────────────────────────────
def load_scaling(md_path):
    """
    scaling_comparison.md 파싱.

    반환:
      R['perf'][model][variant][metric]  = float
      R['ci'][model][variant][metric]    = (lo, hi)
      R['delong'][comp][variant]         = {auc_a, auc_b, delta, pval, sig}

    model  : 'M1' | 'M2' | 'M2_2' | 'M3'
    variant: 'scale_all'(M1) | 'norm' | 'crop80' | 'crop60' | 'len128' | 'excl_extreme'
    metric : 'AUC-ROC' | 'AUPRC' | 'Brier' | 'Accuracy' | 'F1'
    comp   : 'M1vM2' | 'M1vM3' | 'M2vM22' | 'M2vM3'
    """
    with open(md_path, encoding='utf-8') as fh:
        text = fh.read()

    R = {'perf': {}, 'ci': {}, 'delong': {}}

    # Bootstrap CI 테이블
    # | Model | Sub | Case | Metric | Estimate | CI Lower | CI Upper |
    for row in _table_rows(text, "# Test Set — Bootstrap 95% CI"):
        if len(row) < 7:
            continue
        model, _sub, case, metric, est, lo, hi = row[:7]
        R['perf'].setdefault(model, {}).setdefault(case, {})[metric] = _f(est)
        R['ci'].setdefault(model, {}).setdefault(case, {})[metric] = (_f(lo), _f(hi))

    # DeLong 테이블
    # | Comparison | AUC A | AUC B | Δ AUC | z-stat | p-val | sig |
    VARS = ['raw', 'norm', 'global_zscore']
    for marker, comp_key in [
        ('## M1 LR vs M2 CrossAttn',         'M1vM2'),
        ('## M1 LR vs M3 CrossAttn3',        'M1vM3'),
        ('## M1 LR vs M4 AECOnly',           'M1vM4'),
        ('## M2 Matched vs M2_2 Unmatched',  'M2vM22'),
        ('## M2 CrossAttn vs M3 CrossAttn3', 'M2vM3'),
        ('## M4 AECOnly vs M2 CrossAttn',    'M4vM2'),
        ('## M1 LR vs M5 CrossAttn-Feat',    'M1vM5'),
    ]:
        for row in _table_rows(text, marker):
            if len(row) < 7:
                continue
            cmp_str, auc_a, auc_b, delta, _z, pval, sig = row[:7]
            var = next((v for v in VARS if v in cmp_str), None)
            if var is None:
                continue
            R['delong'].setdefault(comp_key, {})[var] = {
                'auc_a': _f(auc_a), 'auc_b': _f(auc_b),
                'delta': _f(delta),  'pval':  _f(pval),
                'sig':   sig.strip(),
            }

    return R


# ─────────────────────────────────────────────────────────────
def load_model_results(md_path):
    """
    개별 model results.md 파싱.

    반환:
      R['overall'] = {metric: float}
      R['sex']['M'|'F'] = {'n': int, metric: float, ...}
      R['ci'][metric]   = (estimate, lo, hi)   # Bootstrap 95% CI
    """
    with open(md_path, encoding='utf-8') as fh:
        text = fh.read()

    R = {'overall': {}, 'sex': {}, 'ci': {}}

    for row in _table_rows(text, "### Overall"):
        if len(row) >= 6 and _f(row[1]) is not None:
            _, auc, auprc, brier, acc, f1 = row[:6]
            R['overall'] = {
                'AUC-ROC': _f(auc), 'AUPRC': _f(auprc),
                'Brier':   _f(brier), 'Accuracy': _f(acc), 'F1': _f(f1),
            }
            break

    for row in _table_rows(text, "### By Sex"):
        if len(row) >= 7 and row[0] in ('M', 'F'):
            sex, n, auc, auprc, brier, acc, f1 = row[:7]
            R['sex'][sex] = {
                'n':       int(_f(n) or 0),
                'AUC-ROC': _f(auc), 'AUPRC': _f(auprc),
                'Brier':   _f(brier), 'Accuracy': _f(acc), 'F1': _f(f1),
            }

    for row in _table_rows(text, "## 4. Bootstrap 95% CI"):
        if len(row) >= 4 and row[0] in ('AUC-ROC', 'AUPRC', 'Brier', 'Accuracy', 'F1'):
            metric, est, lo, hi = row[:4]
            v = (_f(est), _f(lo), _f(hi))
            if all(x is not None for x in v):
                R['ci'][metric] = v

    return R
