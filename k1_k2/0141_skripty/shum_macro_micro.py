# к1 0141: свой свод по noise_summary.json к2 — СКО затравок: macro по исполнителям (тест фолда) против micro (валидация, last)
import json, numpy as np
d = json.load(open('k2_k1/0158_golos_k2/risk_e2/noise_summary.json', encoding='utf-8'))
for ds in ('slovo', 'bukva'):
    C = [c for c in d['conditions'] if c['ds'] == ds and 'sd_test_macro' in c and 'sd_val_last' in c]
    mac = np.array([c['sd_test_macro'] for c in C]); mic = np.array([c['sd_val_last'] for c in C])
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    print(f"{ds}: условий {len(C)}; СКО (RMS по условиям и фолдам): macro теста {rms(mac):.2f}, micro валидации {rms(mic):.2f}, "
          f"отношение {rms(mac) / rms(mic):.2f}; по фолдам: " + ", ".join(f"{rms(mac[:, f]) / rms(mic[:, f]):.2f}" for f in range(5)))
# нижняя граница отношения по одному самому малому исполнителю (биномиально, равные p): √(N/n_min)/K
for ds, v in d['val_composition'].items():
    print(ds, "√(N/n_min)/K по фолдам:", [round(np.sqrt(f['n_records'] / f['min_per_signer']) / f['n_signers'], 2) for f in v.values()])
