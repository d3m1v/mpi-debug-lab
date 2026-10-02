from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ПАРСИНГ TXT-ФАЙЛОВ В ОДИН DataFrame
# ============================================================

INPUT_DIR = Path('.')

PATTERNS = {
    'r':        re.compile(r'\br\s*=\s*(\d+)'),
    'x':        re.compile(r'\bx\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'),
    'N':        re.compile(r'процессов\s*=\s*(\d+)'),
    'q_ref':    re.compile(r'Эталон\s*\(последовательный\)\s*:\s*(\S+)'),
    'q_par':    re.compile(r'Параллельный результат\s*:\s*(\S+)'),
    'abs_diff': re.compile(r'Абсолютная разница\s*:\s*(\S+)'),
    'rel_diff': re.compile(r'Относительная разница\s*:\s*(\S+)'),
    'time_us':  re.compile(r'Время\s*:\s*([\d.]+)'),
}


def to_float(v):
    if v is None:
        return np.nan
    v = v.strip().rstrip(',').rstrip(';')
    if v == '' or v.lower() == 'nan':
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def parse_txt(path: Path):
    """Читает один txt-файл, возвращает dict или None, если не наш."""
    name = path.name
    if name.startswith('deb'):
        os_type = 'deb'
    elif name.startswith('git'):
        os_type = 'git'
    else:
        return None

    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError as e:
        print(f"[!] Не удалось прочитать {path}: {e}")
        return None

    row = {'os': os_type, 'file': name}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        row[key] = m.group(1) if m else None

    row['r'] = int(row['r']) if row['r'] else np.nan
    row['N'] = int(row['N']) if row['N'] else np.nan
    for k in ('x', 'q_ref', 'q_par', 'abs_diff', 'rel_diff', 'time_us'):
        row[k] = to_float(row[k])
    return row


files = sorted(INPUT_DIR.glob('*.txt'))
print(f"Найдено txt-файлов: {len(files)}")
print()

rows = [r for f in files if (r := parse_txt(f)) is not None]
df = pd.DataFrame(rows)

df = df.sort_values(['os', 'N', 'r', 'x'], na_position='last').reset_index(drop=True)

print(f"Обработано: {len(df)} файлов")
print()
print(df.head())
print()

df.drop(columns=['file']).to_csv(
    'polynomial_results_clean.csv',
    index=False, encoding='utf-8-sig',
)

print("Файл polynomial_results_clean.csv сохранен")
print()
print("Процесс построения графиков...")
print()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Масштабирование по N (r=5, x=1.1))
# ============================================================

sub = (df[(df['r'] == 5) & (np.isclose(df['x'], 1.1))]
       .groupby(['os', 'N'], as_index=False)['time_us'].mean()
       .sort_values(['os', 'N']))

fig, ax = plt.subplots(figsize=(8, 5))
for os_name in ['deb', 'git']:
    d = sub[sub['os'] == os_name].sort_values('N')
    if d.empty:
        continue
    ax.plot(d['N'], d['time_us'], marker='o', label=os_name,
            linewidth=2, markersize=8)

ax.set_xlabel('Число процессов N', fontsize=12)
ax.set_ylabel('Время, мкс', fontsize=12)
ax.set_title('Масштабирование по числу процессов (r=5, x=1.1)', fontsize=13)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph1_scaling_N.png', dpi=150)
plt.show()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Ускорение S(N))
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
for os_name in ['deb', 'git']:
    d = sub[sub['os'] == os_name].sort_values('N')
    if d.empty or d['N'].iloc[0] != 1:
        continue
    t1 = d[d['N'] == 1]['time_us'].values[0]
    speedup = t1 / d['time_us'].values
    ax.plot(d['N'], speedup, marker='s', label=f'{os_name} (S(N))',
            linewidth=2, markersize=8)

Ns = np.array(sorted(sub['N'].unique()))
ax.plot(Ns, Ns, 'k--', alpha=0.4, label='Идеальное S(N) = N')

ax.set_xlabel('Число процессов N', fontsize=12)
ax.set_ylabel('Ускорение S(N) = T(1)/T(N)', fontsize=12)
ax.set_title('Ускорение параллельной версии', fontsize=13)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph2_speedup.png', dpi=150)
plt.show()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Сравнение сред (deb vs git))
# ============================================================

common_N = [1, 2, 4, 8, 16]

d_deb = (sub[(sub['os'] == 'deb') & (sub['N'].isin(common_N))]
         .groupby('N', as_index=False)['time_us'].mean()
         .sort_values('N'))
d_git = (sub[(sub['os'] == 'git') & (sub['N'].isin(common_N))]
         .groupby('N', as_index=False)['time_us'].mean()
         .sort_values('N'))

x_pos = np.arange(len(common_N))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_pos - width/2, d_deb['time_us'].values, width,
       label='Debian', color='#3b7dd8')
ax.bar(x_pos + width/2, d_git['time_us'].values, width,
       label='Codespace', color='#d8753b')

ax.set_xlabel('Число процессов N', fontsize=12)
ax.set_ylabel('Время, мкс', fontsize=12)
ax.set_title('Сравнение времени выполнения: Debian vs Codespace', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(common_N)
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph3_comparison.png', dpi=150)
plt.show()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Зависимость времени от размера блока r)
# ============================================================

sub_r = (df[(df['N'] == 4) & (np.isclose(df['x'], 1.1))]
         .groupby(['os', 'r'], as_index=False)['time_us'].mean()
         .sort_values(['os', 'r']))

fig, ax = plt.subplots(figsize=(8, 5))
for os_name in ['deb', 'git']:
    d = sub_r[sub_r['os'] == os_name]
    if d.empty:
        continue
    ax.plot(d['r'], d['time_us'], marker='o', label=os_name,
            linewidth=2, markersize=8)

ax.set_xlabel('Размер блока r', fontsize=12)
ax.set_ylabel('Время, мкс', fontsize=12)
ax.set_title('Зависимость времени от размера блока (N=4, x=1.1)', fontsize=13)
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph4_vs_r.png', dpi=150)
plt.show()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Численная точность vs x)
# ============================================================

sub_x = (df[(df['N'] == 4) & (df['r'] == 6) & df['rel_diff'].notna()]
         .groupby(['os', 'x'], as_index=False)['rel_diff'].mean()
         .sort_values(['os', 'x']))

fig, ax = plt.subplots(figsize=(10, 5))
for os_name in ['deb', 'git']:
    d = sub_x[sub_x['os'] == os_name]
    if d.empty:
        continue
    ax.plot(d['x'], d['rel_diff'], marker='o', linestyle='',
            label=os_name, markersize=6)

eps = np.finfo(float).eps
ax.axhline(y=eps, color='red', linestyle='--', alpha=0.6,
           label=f'Машинный эпсилон ({eps:.1e})')

ax.set_xlabel('Точка x', fontsize=12)
ax.set_ylabel('Относительная разница', fontsize=12)
ax.set_title('Численная точность: относительная разница vs x (N=4, r=6)',
             fontsize=13)
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph5_precision.png', dpi=150)
plt.show()


# ============================================================
# ПОСТРОЕНИЕ ГРАФИКА (Эффективность E(N))
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
for os_name in ['deb', 'git']:
    d = sub[sub['os'] == os_name].sort_values('N')
    if d.empty or d['N'].iloc[0] != 1:
        continue
    t1 = d[d['N'] == 1]['time_us'].values[0]
    N  = d['N'].values
    efficiency = t1 / (d['time_us'].values * N)
    ax.plot(N, efficiency, marker='^', label=f'{os_name} (E(N))',
            linewidth=2, markersize=8)

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5,
           label='Идеальная E(N) = 1')

ax.set_xlabel('Число процессов N', fontsize=12)
ax.set_ylabel('Эффективность E(N) = S(N)/N', fontsize=12)
ax.set_title('Эффективность параллельной версии', fontsize=13)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('graph2b_efficiency.png', dpi=150)
plt.show()

print("Выполнение завершено")
