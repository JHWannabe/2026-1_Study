# -*- coding: utf-8 -*-
import sys, io, re, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import config

with open('generate_ppt.py', encoding='utf-8') as f:
    src = f.read()

func_src = src.split('prs = Presentation')[0]
exec(func_src)

sites = ['강남', '신초']
for site in sites:
    D = parse_report(site)
    print(f'=== {site} ===')
    print('n_total:', D['n_total'])
    print('ct_dummy_count:', D['ct_dummy_count'])
    print('lin_r2:', D['lin_r2'])
    print('log_auc:', D['log_auc'])
    print('lin_summary:', D['lin_summary'])
    print('resid_sw_stat:', D['resid_sw_stat'])
    print('log_hl_stat/p:', D.get('log_hl_stat'), '/', D.get('log_hl_p'))
    print()

print('PASS')
