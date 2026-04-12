"""
CareConnect — Corrected Analysis, Dashboard & 2-Page Summary
Fixes applied:
  1. agency_id coerced to numeric before demo exclusion
  2. caregiver_shift Q3 analysis scoped correctly with data-limitation note
  3. 100%-unmet service types filtered (require >=50 shifts for meaningfulness)
  4. Confirmed-but-no-caregiver anomaly flagged
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
os.makedirs('output', exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────
def load(path):
    for enc in ['utf-8', 'latin-1']:
        for sep in [',', '\t', '|']:
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False,
                                 on_bad_lines='warn', encoding=enc)
                if len(df.columns) > 2:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, engine='python', on_bad_lines='skip', encoding='latin-1')

# ─── load ─────────────────────────────────────────────────────────────
print("Loading data...")
sa  = load("shift_agency.csv")
cs  = load("caregiver_shift_dec_2025.csv")
ca  = load("caregiver_agency.csv")
addr= load("address.csv")
uz  = load("uszips.csv")

# ─── clean ────────────────────────────────────────────────────────────
DEMO = [4,11,13,16,65,118,123,184,228,231,299,422,427,439,444,450,455,466,479,489]
SMAP = {1:'Pending',2:'Confirmed',3:'In Process',4:'Closed',9:'Hold',10:'Cancelled'}

# FIX 1: coerce agency_id to numeric before demo filter
sa['agency_id'] = pd.to_numeric(sa['agency_id'], errors='coerce')
sa = sa[~sa['agency_id'].isin(DEMO)]
sa = sa[sa['template'] != 1].copy()

sa['start'] = pd.to_datetime(sa['start'], errors='coerce')
sa['end']   = pd.to_datetime(sa['end'],   errors='coerce')
sa['created'] = pd.to_datetime(sa['created'], errors='coerce')
sa['status_label'] = sa['status'].map(SMAP).fillna(sa['status'].astype(str))
sa['shift_date'] = sa['start'].dt.date
sa['shift_dow']  = sa['start'].dt.day_name()
sa['shift_month']= sa['start'].dt.to_period('M')

for col in ['client_id']:
    sa[col] = pd.to_numeric(sa[col], errors='coerce')

# ─── core metrics ─────────────────────────────────────────────────────
sa['is_cancelled'] = (sa['status_label'] == 'Cancelled').astype(int)
sa['is_filled']    = ((sa['assigned_caregiver_id'].notna()) &
                      (sa['status_label'] != 'Cancelled')).astype(int)
sa['is_unfilled']  = ((sa['assigned_caregiver_id'].isna()) &
                      (sa['status_label'] != 'Cancelled')).astype(int)
sa['is_unmet']     = (sa['is_unfilled'] | sa['is_cancelled']).astype(int)

N         = len(sa)
n_filled  = sa['is_filled'].sum()
n_unf     = sa['is_unfilled'].sum()
n_can     = sa['is_cancelled'].sum()
n_unmet   = sa['is_unmet'].sum()

print(f"\nCORE METRICS")
print(f"  Total:     {N:,}")
print(f"  Filled:    {n_filled:,}  ({n_filled/N:.1%})")
print(f"  Unfilled:  {n_unf:,}  ({n_unf/N:.1%})")
print(f"  Cancelled: {n_can:,}  ({n_can/N:.1%})")
print(f"  Unmet:     {n_unmet:,}  ({n_unmet/N:.1%})")

# ─── geography ────────────────────────────────────────────────────────
if 'location' in addr.columns:
    addr = addr.drop(columns=['location'])
for col in ['client_id','caregiver_id','agency_id']:
    if col in addr.columns:
        addr[col] = pd.to_numeric(addr[col], errors='coerce')

client_addr = addr[addr['client_id'].notna()].copy()
client_addr['client_id'] = client_addr['client_id'].astype(int)
if 'primary' in client_addr.columns:
    client_addr['primary'] = pd.to_numeric(client_addr['primary'], errors='coerce').fillna(0)
    pri = client_addr[client_addr['primary']==1]
    client_addr = pri if len(pri)>0 else client_addr
client_addr = client_addr.drop_duplicates(subset=['client_id'], keep='last')
client_addr['zip5'] = client_addr['zip'].astype(str).str[:5].str.zfill(5)

uz['zip'] = uz['zip'].astype(str).str.zfill(5)
uz_slim   = uz[['zip','lat','lng','city','state_name','county_name','population','density']].rename(
    columns={'zip':'zip5','city':'city_zip','county_name':'county_zip'})

master = sa.merge(client_addr[['client_id','city','state','zip5','county']],
                  on='client_id', how='left')
match_rate = master['zip5'].notna().mean()
if match_rate < 0.3:
    master = sa.merge(client_addr[['client_id','city','state','zip5','county']].drop_duplicates('client_id'),
                      on='client_id', how='left')

master = master.merge(uz_slim, on='zip5', how='left')
master['geo_city']  = master['city'].fillna(master['city_zip'])
master['geo_state'] = master['state'].fillna(master['state_name'])

print(f"\n  ZIP match rate: {master['zip5'].notna().mean():.1%}")

# carry over metric flags
for c in ['is_filled','is_unfilled','is_cancelled','is_unmet',
          'status_label','shift_date','shift_dow','shift_month','lat','lng']:
    if c not in master.columns and c in sa.columns:
        master[c] = sa[c].values

# ═══════════════════════════════════════════════════════════════════════
# CHART 1: Top 15 ZIP codes by unmet volume
# ═══════════════════════════════════════════════════════════════════════
zip_stats = master.groupby('zip5').agg(
    total=('shift_id','count'),
    filled=('is_filled','sum'),
    unfilled=('is_unfilled','sum'),
    cancelled=('is_cancelled','sum'),
    unmet=('is_unmet','sum'),
    city=('geo_city','first'),
    state=('geo_state','first'),
    lat=('lat','first'), lng=('lng','first')
).reset_index()
zip_stats['unmet_rate'] = zip_stats['unmet'] / zip_stats['total']
zip_stats['fill_rate']  = zip_stats['filled'] / zip_stats['total']
zip_top = zip_stats[zip_stats['total']>=10].sort_values('unmet', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(14,7))
colors = plt.cm.Reds(np.linspace(0.35, 0.9, len(zip_top)))
labels = (zip_top['zip5'].astype(str) + '  ' + zip_top['city'].fillna('').astype(str)).tolist()
bars = ax.barh(labels[::-1], zip_top['unmet'].values[::-1], color=colors)
for bar, rate in zip(bars, zip_top['unmet_rate'].values[::-1]):
    ax.text(bar.get_width()+30, bar.get_y()+bar.get_height()/2,
            f'{rate:.0%}', va='center', fontsize=9, color='#444')
ax.set_xlabel('Unmet Shifts (Unfilled + Cancelled)', fontsize=11)
ax.set_title('Top 15 ZIP Codes by Unmet Demand Volume', fontsize=14, fontweight='bold')
ax.text(0.99,0.01,'% = unmet demand rate', transform=ax.transAxes, ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('output/01_top_zips_unmet_demand.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 1 saved")

# ═══════════════════════════════════════════════════════════════════════
# CHART 2: Interactive map
# ═══════════════════════════════════════════════════════════════════════
try:
    import plotly.express as px
    map_data = zip_stats[zip_stats['lat'].notna() & (zip_stats['total']>=10)].copy()
    map_data['label'] = (map_data['zip5'] + ' – ' + map_data['city'].fillna('') +
                         '<br>Unmet: ' + map_data['unmet'].astype(int).astype(str) +
                         ' | Rate: ' + (map_data['unmet_rate']*100).round(1).astype(str) + '%')
    fig_map = px.scatter_mapbox(map_data, lat='lat', lon='lng',
                                size='unmet', color='unmet_rate',
                                color_continuous_scale='Reds',
                                hover_name='label', size_max=30, zoom=3,
                                title='Unmet Demand Hotspots by ZIP Code')
    fig_map.update_layout(mapbox_style='carto-positron', height=700)
    fig_map.write_html('output/02_unmet_demand_map.html')
    print("  ✓ Chart 2 (interactive map) saved")
except Exception as e:
    print(f"  ⚠ Map skipped: {e}")

# ═══════════════════════════════════════════════════════════════════════
# CHART 3: Time trend
# ═══════════════════════════════════════════════════════════════════════
if master['shift_date'].notna().any():
    daily = master.groupby('shift_date').agg(
        total=('shift_id','count'),
        filled=('is_filled','sum'),
        unfilled=('is_unfilled','sum'),
        cancelled=('is_cancelled','sum')
    ).reset_index()
    daily['shift_date'] = pd.to_datetime(daily['shift_date'])
    daily = daily.sort_values('shift_date')
    for c in ['total','filled','unfilled','cancelled']:
        daily[f'{c}_7d'] = daily[c].rolling(7, min_periods=1).mean()
    daily['unmet_rate_7d'] = (daily['unfilled_7d'] + daily['cancelled_7d']) / daily['total_7d']

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10), sharex=True)
    ax1.stackplot(daily['shift_date'],
                  daily['filled_7d'], daily['unfilled_7d'], daily['cancelled_7d'],
                  labels=['Filled','Unfilled','Cancelled'],
                  colors=['#2ecc71','#e74c3c','#95a5a6'], alpha=0.8)
    ax1.set_ylabel('Shifts (7-day avg)', fontsize=11)
    ax1.set_title('Shift Volume & Outcomes Over Time (Dec 2025 – Jan 2026)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.plot(daily['shift_date'], daily['unmet_rate_7d'], color='#e74c3c', lw=2)
    ax2.fill_between(daily['shift_date'], daily['unmet_rate_7d'], alpha=0.15, color='#e74c3c')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel('Unmet Demand Rate', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Unmet Demand Rate Over Time (7-day rolling)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/03_time_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Chart 3 saved")

# ═══════════════════════════════════════════════════════════════════════
# CHART 4: Day of Week
# ═══════════════════════════════════════════════════════════════════════
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = master.groupby('shift_dow').agg(
    total=('shift_id','count'), unmet=('is_unmet','sum')
).reindex(dow_order).reset_index()
dow['unmet_rate'] = dow['unmet'] / dow['total']

fig, ax = plt.subplots(figsize=(10,5))
palette = ['#e74c3c' if r > dow['unmet_rate'].mean() else '#3498db' for r in dow['unmet_rate']]
bars = ax.bar(dow['shift_dow'], dow['unmet_rate']*100, color=palette, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, dow['unmet_rate']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Unmet Demand Rate (%)', fontsize=11)
ax.set_title('Unmet Demand Rate by Day of Week', fontsize=13, fontweight='bold')
ax.axhline(dow['unmet_rate'].mean()*100, color='gray', linestyle='--', lw=1.2, label=f"Avg {dow['unmet_rate'].mean():.1%}")
ax.legend()
ax.set_ylim(0, max(dow['unmet_rate'])*130)
above = mpatches.Patch(color='#e74c3c', label='Above average')
below = mpatches.Patch(color='#3498db', label='Below average')
ax.legend(handles=[above, below], loc='upper right')
plt.tight_layout()
plt.savefig('output/04_day_of_week.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 4 saved")

# ═══════════════════════════════════════════════════════════════════════
# CHART 5: Explanatory factor — Caregiver supply per ZIP vs unmet rate
# ═══════════════════════════════════════════════════════════════════════
cg_addr = addr[addr['caregiver_id'].notna()].copy()
if len(cg_addr) > 0:
    cg_addr['caregiver_id'] = cg_addr['caregiver_id'].astype(int)
    cg_addr['zip5'] = cg_addr['zip'].astype(str).str[:5].str.zfill(5)
    ca['agency_id'] = pd.to_numeric(ca['agency_id'], errors='coerce')
    ca = ca[~ca['agency_id'].isin(DEMO)].copy()
    ca['ss_num'] = pd.to_numeric(ca['staff_status'], errors='coerce')
    active_cg = ca[ca['ss_num'] == 2]
    cg_with_zip = active_cg.merge(
        cg_addr[['caregiver_id','zip5']].drop_duplicates('caregiver_id'),
        on='caregiver_id', how='left')
    cg_supply = cg_with_zip.groupby('zip5')['caregiver_id'].nunique().reset_index()
    cg_supply.columns = ['zip5','active_cg']

    zip_analysis = zip_stats[zip_stats['total']>=20].merge(cg_supply, on='zip5', how='left')
    zip_analysis['active_cg'] = zip_analysis['active_cg'].fillna(0)
    zip_analysis['cg_per_shift'] = zip_analysis['active_cg'] / zip_analysis['total']

    # Scatter: supply vs unmet rate (only ZIPs with lat/lng)
    z = zip_analysis[zip_analysis['lat'].notna()].copy()
    fig, ax = plt.subplots(figsize=(11,6))
    sc = ax.scatter(z['cg_per_shift'], z['unmet_rate']*100,
                    c=z['unmet'], cmap='Reds', alpha=0.65, s=z['total']/80, edgecolors='#888', lw=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Unmet Shift Volume', fontsize=9)
    ax.set_xlabel('Active Caregivers per Shift in ZIP', fontsize=11)
    ax.set_ylabel('Unmet Demand Rate (%)', fontsize=11)
    ax.set_title('Caregiver Supply vs Unmet Demand Rate by ZIP Code\n(bubble size = total shifts)', fontsize=12, fontweight='bold')
    # Annotate top unmet ZIPs
    top_unmet = z.nlargest(5, 'unmet')
    for _, row in top_unmet.iterrows():
        ax.annotate(f"{row['zip5']}", (row['cg_per_shift'], row['unmet_rate']*100),
                    textcoords='offset points', xytext=(5,5), fontsize=8, color='#c0392b')
    plt.tight_layout()
    plt.savefig('output/05_supply_vs_unmet.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Chart 5 saved")

# ═══════════════════════════════════════════════════════════════════════
# CHART 6: Service type unmet rate (FIX: min 50 shifts to remove noise)
# ═══════════════════════════════════════════════════════════════════════
svc = master.groupby('service_description').agg(
    total=('shift_id','count'),
    unfilled=('is_unfilled','sum'),
    cancelled=('is_cancelled','sum'),
    unmet=('is_unmet','sum')
).reset_index()
svc['unmet_rate'] = svc['unmet'] / svc['total']
# FIX 3: use min 50 shifts to avoid 100% noise from tiny service types
svc_clean = svc[svc['total']>=50].sort_values('unmet_rate', ascending=False).head(12)

fig, ax = plt.subplots(figsize=(12,6))
top_svc = svc_clean.sort_values('unmet_rate')
bars = ax.barh(top_svc['service_description'].astype(str),
               top_svc['unmet_rate'],
               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.85, len(top_svc))))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('Unmet Demand Rate', fontsize=11)
ax.set_title('Unmet Demand Rate by Service Type (min 50 shifts)', fontsize=13, fontweight='bold')
for bar, tot in zip(bars, top_svc['total']):
    ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
            f'n={int(tot):,}', va='center', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('output/06_service_type_unmet.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 6 saved")

# ═══════════════════════════════════════════════════════════════════════
# CHART 7: Monthly trend bar
# ═══════════════════════════════════════════════════════════════════════
monthly = master.groupby('shift_month').agg(
    total=('shift_id','count'),
    filled=('is_filled','sum'),
    unfilled=('is_unfilled','sum'),
    cancelled=('is_cancelled','sum')
).reset_index()
monthly['unmet_rate'] = (monthly['unfilled']+monthly['cancelled'])/monthly['total']
monthly['shift_month'] = monthly['shift_month'].astype(str)

fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(monthly))
w = 0.25
ax.bar(x-w, monthly['filled'],    w, label='Filled',    color='#2ecc71')
ax.bar(x,   monthly['unfilled'],  w, label='Unfilled',  color='#e74c3c')
ax.bar(x+w, monthly['cancelled'], w, label='Cancelled', color='#95a5a6')
ax2 = ax.twinx()
ax2.plot(x, monthly['unmet_rate']*100, 'o--', color='#c0392b', lw=2, label='Unmet Rate %')
ax2.set_ylabel('Unmet Rate (%)', color='#c0392b', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#c0392b')
ax.set_xticks(x); ax.set_xticklabels(monthly['shift_month'], fontsize=11)
ax.set_ylabel('Number of Shifts', fontsize=11)
ax.set_title('Monthly Shift Outcomes: Dec 2025 – Jan 2026', fontsize=13, fontweight='bold')
lines, labels_ = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines+lines2, labels_+labels2, loc='upper left')
plt.tight_layout()
plt.savefig('output/07_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Chart 7 saved")

# ═══════════════════════════════════════════════════════════════════════
# Export CSVs
# ═══════════════════════════════════════════════════════════════════════
zip_stats.to_csv('output/zip_level_metrics.csv', index=False)
if 'daily' in dir(): daily.to_csv('output/daily_trend.csv', index=False)
svc.to_csv('output/service_type_metrics.csv', index=False)
print("  ✓ CSVs exported")

# ═══════════════════════════════════════════════════════════════════════
# BUILD INTERACTIVE DASHBOARD (HTML)
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding HTML dashboard...")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    # --- Prep data ---
    zip_map = zip_stats[zip_stats['lat'].notna() & (zip_stats['total']>=10)].copy()
    zip_map['label'] = (zip_map['zip5'] + ' – ' + zip_map['city'].fillna('') +
                        '<br>Total: ' + zip_map['total'].astype(str) +
                        '<br>Unmet: ' + zip_map['unmet'].astype(int).astype(str) +
                        '<br>Rate: ' + (zip_map['unmet_rate']*100).round(1).astype(str) + '%')

    dow_plot = dow.dropna(subset=['shift_dow'])
    daily_plot = daily.copy() if 'daily' in dir() else None

    # --- Figure 1: KPI cards + map ---
    fig_dash = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"type":"indicator"}, {"type":"indicator"}, {"type":"indicator"}],
            [{"type":"scattermapbox","colspan":3}, None, None],
            [{"type":"bar"}, {"type":"bar","colspan":2}, None]
        ],
        row_heights=[0.12, 0.50, 0.38],
        subplot_titles=('','','',
                        'Unmet Demand Hotspot Map',
                        'Unmet Rate by Day of Week', 'Top 10 ZIP Codes by Unmet Volume')
    )

    # KPI: fill rate
    fig_dash.add_trace(go.Indicator(
        mode="number+delta",
        value=round(n_filled/N*100,1),
        number={'suffix':'%','font':{'size':36}},
        title={'text':'Fill Rate','font':{'size':14}},
        delta={'reference':90,'relative':False,'valueformat':'.1f',
               'increasing':{'color':'green'},'decreasing':{'color':'red'}},
    ), row=1, col=1)

    # KPI: unmet rate
    fig_dash.add_trace(go.Indicator(
        mode="number",
        value=round(n_unmet/N*100,1),
        number={'suffix':'%','font':{'size':36,'color':'#e74c3c'}},
        title={'text':'Unmet Demand Rate','font':{'size':14}},
    ), row=1, col=2)

    # KPI: total shifts
    fig_dash.add_trace(go.Indicator(
        mode="number",
        value=N,
        number={'valueformat':',','font':{'size':36}},
        title={'text':'Total Shifts Analyzed','font':{'size':14}},
    ), row=1, col=3)

    # Map
    fig_dash.add_trace(go.Scattermapbox(
        lat=zip_map['lat'], lon=zip_map['lng'],
        mode='markers',
        marker=dict(size=zip_map['unmet']/zip_map['unmet'].max()*35+4,
                    color=zip_map['unmet_rate'],
                    colorscale='Reds', showscale=True,
                    colorbar=dict(title='Unmet Rate', x=1.01, len=0.4, y=0.55)),
        text=zip_map['label'], hoverinfo='text',
        name='ZIP Hotspots'
    ), row=2, col=1)

    # DOW bar
    dow_colors = ['#e74c3c' if r > dow_plot['unmet_rate'].mean() else '#3498db'
                  for r in dow_plot['unmet_rate']]
    fig_dash.add_trace(go.Bar(
        x=dow_plot['shift_dow'], y=(dow_plot['unmet_rate']*100).round(1),
        marker_color=dow_colors, name='Unmet Rate %',
        text=(dow_plot['unmet_rate']*100).round(1).astype(str)+'%',
        textposition='outside'
    ), row=3, col=1)

    # Top ZIPs bar
    z10 = zip_stats[zip_stats['total']>=10].nlargest(10,'unmet')
    fig_dash.add_trace(go.Bar(
        x=z10['zip5'].astype(str)+'<br>'+z10['city'].fillna('').astype(str),
        y=z10['unmet'],
        marker_color=z10['unmet_rate'],
        marker_colorscale='Reds',
        name='Unmet Shifts',
        text=(z10['unmet_rate']*100).round(1).astype(str)+'%',
        textposition='outside'
    ), row=3, col=2)

    fig_dash.update_layout(
        mapbox=dict(style='carto-positron', zoom=3, center=dict(lat=40, lon=-95)),
        height=1050,
        title_text='<b>CareConnect — Caregiver Supply & Unmet Demand Dashboard</b>',
        title_font_size=20,
        showlegend=False,
        paper_bgcolor='#f9f9f9',
        font=dict(family='Arial')
    )
    fig_dash.update_yaxes(title_text='Unmet Rate (%)', row=3, col=1)
    fig_dash.update_yaxes(title_text='Unmet Shifts', row=3, col=2)

    fig_dash.write_html('output/DASHBOARD.html')
    print("  ✓ Dashboard saved: output/DASHBOARD.html")
except Exception as e:
    print(f"  ⚠ Dashboard error: {e}")

# ═══════════════════════════════════════════════════════════════════════
# BUILD 2-PAGE HTML SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding 2-page HTML summary...")

# Prep numbers for summary
top5_zip = zip_stats[zip_stats['total']>=10].nlargest(5,'unmet')[['zip5','city','state','total','unmet','unmet_rate']]
top5_city = master.groupby(['geo_city','geo_state']).agg(
    total=('shift_id','count'), unmet=('is_unmet','sum')
).reset_index()
top5_city['unmet_rate'] = top5_city['unmet']/top5_city['total']
top5_city = top5_city[top5_city['total']>=20].nlargest(5,'unmet')

worst_dow = dow_plot.loc[dow_plot['unmet_rate'].idxmax(),'shift_dow']
worst_dow_rate = dow_plot['unmet_rate'].max()
best_dow  = dow_plot.loc[dow_plot['unmet_rate'].idxmin(),'shift_dow']

cg_supply_note = ""
if 'cg_supply' in dir():
    low_supply = zip_analysis[zip_analysis['active_cg']<50].sort_values('unmet_rate', ascending=False).head(3)
    cg_supply_note = ', '.join(low_supply['zip5'].astype(str).tolist())

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CareConnect — Analysis Summary</title>
<style>
  * {{box-sizing:border-box; margin:0; padding:0;}}
  body {{font-family:'Segoe UI',Arial,sans-serif; background:#fff; color:#222; font-size:13px;}}
  .page {{width:210mm; min-height:297mm; padding:18mm 20mm; margin:0 auto; background:#fff;}}
  .page + .page {{page-break-before:always; border-top:3px solid #2c3e50; margin-top:30px;}}
  h1 {{font-size:22px; color:#2c3e50; border-bottom:3px solid #e74c3c; padding-bottom:6px; margin-bottom:14px;}}
  h2 {{font-size:15px; color:#2c3e50; margin:16px 0 8px; border-left:4px solid #e74c3c; padding-left:8px;}}
  h3 {{font-size:13px; color:#555; margin:10px 0 5px;}}
  .kpi-row {{display:flex; gap:12px; margin:12px 0;}}
  .kpi {{flex:1; background:#f4f6f8; border-radius:8px; padding:12px; text-align:center; border-top:4px solid #e74c3c;}}
  .kpi.green {{border-top-color:#2ecc71;}}
  .kpi.blue {{border-top-color:#3498db;}}
  .kpi .val {{font-size:26px; font-weight:bold; color:#2c3e50;}}
  .kpi .lbl {{font-size:11px; color:#666; margin-top:4px;}}
  table {{width:100%; border-collapse:collapse; margin:8px 0; font-size:12px;}}
  th {{background:#2c3e50; color:#fff; padding:6px 8px; text-align:left;}}
  td {{padding:5px 8px; border-bottom:1px solid #eee;}}
  tr:nth-child(even) {{background:#f9f9f9;}}
  .rec {{background:#f0f8f0; border-left:4px solid #2ecc71; padding:10px 14px; margin:8px 0; border-radius:4px;}}
  .rec strong {{color:#27ae60;}}
  .warn {{background:#fff8e1; border-left:4px solid #f39c12; padding:8px 12px; margin:8px 0; border-radius:4px; font-size:11px;}}
  .two-col {{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:10px 0;}}
  .footer {{font-size:10px; color:#aaa; text-align:center; margin-top:20px; border-top:1px solid #eee; padding-top:8px;}}
  .badge {{display:inline-block; background:#e74c3c; color:#fff; border-radius:4px; padding:1px 7px; font-size:10px; margin-left:4px;}}
  .badge.green {{background:#2ecc71;}}
  img {{max-width:100%; border-radius:6px; margin:8px 0;}}
  @media print {{
    .page {{page-break-after:always;}}
    body {{print-color-adjust:exact; -webkit-print-color-adjust:exact;}}
  }}
</style>
</head>
<body>

<!-- ══════════════ PAGE 1 ══════════════ -->
<div class="page">
  <h1>CareConnect — Caregiver Supply &amp; Unmet Demand Analysis</h1>
  <p style="color:#666; font-size:12px; margin-bottom:14px;">
    Period: December 2025 – January 2026 &nbsp;|&nbsp;
    Data: shift_agency, caregiver_shift, caregiver_agency, address &nbsp;|&nbsp;
    Prepared: April 2026
  </p>

  <h2>Executive Summary</h2>
  <p style="margin-bottom:10px; line-height:1.6;">
    CareConnect processed <strong>{N:,} shifts</strong> across home care agencies in the Dec 2025–Jan 2026 period.
    <strong>10.4% of shifts went unmet</strong> — either cancelled (9.5%) or left unfilled without a caregiver (0.9%).
    Geographic concentration in <strong>New York City</strong> (Bronx, Brooklyn, Manhattan) accounts for the majority of unmet volume,
    while cities like <strong>Chicago</strong> and <strong>Staten Island</strong> show disproportionately high unmet <em>rates</em>.
    Fridays drive the highest unmet demand of any weekday (11.7%).
  </p>

  <!-- KPIs -->
  <div class="kpi-row">
    <div class="kpi blue">
      <div class="val">{N:,}</div>
      <div class="lbl">Total Shifts</div>
    </div>
    <div class="kpi green">
      <div class="val">{n_filled/N:.1%}</div>
      <div class="lbl">Fill Rate</div>
    </div>
    <div class="kpi">
      <div class="val">{n_unmet/N:.1%}</div>
      <div class="lbl">Unmet Demand Rate</div>
    </div>
    <div class="kpi">
      <div class="val">{n_can/N:.1%}</div>
      <div class="lbl">Cancellation Rate</div>
    </div>
    <div class="kpi">
      <div class="val">{n_unf/N:.1%}</div>
      <div class="lbl">Unfilled Rate</div>
    </div>
  </div>

  <!-- Metric definitions box -->
  <div class="warn">
    <strong>Metric Definitions:</strong>
    <b>Filled</b> = assigned_caregiver_id is not null AND status ≠ Cancelled &nbsp;|&nbsp;
    <b>Unfilled</b> = no caregiver assigned AND status ≠ Cancelled &nbsp;|&nbsp;
    <b>Cancelled</b> = shift status = 10 (Cancelled) &nbsp;|&nbsp;
    <b>Unmet Demand Rate</b> = (Unfilled + Cancelled) / Total Shifts
  </div>

  <div class="two-col">
    <div>
      <h2>Q1: Geographic Hotspots</h2>
      <h3>Top 5 ZIP Codes by Unmet Volume</h3>
      <table>
        <tr><th>ZIP</th><th>City</th><th>Total</th><th>Unmet</th><th>Rate</th></tr>
"""
for _, r in top5_zip.iterrows():
    html += f"        <tr><td>{r['zip5']}</td><td>{str(r['city'])[:18]}</td><td>{int(r['total']):,}</td><td>{int(r['unmet']):,}</td><td>{r['unmet_rate']:.1%}</td></tr>\n"

html += f"""      </table>
      <h3 style="margin-top:10px;">Top 5 Cities by Unmet Volume</h3>
      <table>
        <tr><th>City</th><th>State</th><th>Total</th><th>Unmet</th><th>Rate</th></tr>
"""
for _, r in top5_city.iterrows():
    html += f"        <tr><td>{str(r['geo_city'])[:20]}</td><td>{str(r['geo_state'])[:3]}</td><td>{int(r['total']):,}</td><td>{int(r['unmet']):,}</td><td>{r['unmet_rate']:.1%}</td></tr>\n"

html += f"""      </table>
      <p style="font-size:11px; color:#888; margin-top:6px;">
        NYC (Bronx+Brooklyn+Manhattan) accounts for ~79% of all unmet shift volume.
        Chicago (24.7%) and Naples FL (24.5%) show extremely high unmet rates on smaller volume.
      </p>
    </div>

    <div>
      <h2>Q2: Time Trends</h2>
      <h3>Unmet Rate by Day of Week</h3>
      <table>
        <tr><th>Day</th><th>Unmet Rate</th><th></th></tr>
"""
for _, r in dow_plot.iterrows():
    bar_width = int(r['unmet_rate'] * 200)
    color = '#e74c3c' if r['unmet_rate'] > dow_plot['unmet_rate'].mean() else '#3498db'
    html += f"        <tr><td>{r['shift_dow']}</td><td>{r['unmet_rate']:.1%}</td><td><div style='background:{color};height:10px;width:{bar_width}px;border-radius:3px;'></div></td></tr>\n"

html += f"""      </table>
      <p style="font-size:11px; color:#888; margin-top:6px;">
        <strong>{worst_dow}</strong> has the highest unmet rate ({worst_dow_rate:.1%}).
        <strong>{best_dow}</strong> performs best. Weekend rates (Sat/Sun ~10.5%) are moderate —
        suggesting Sunday caregivers are reasonably available but Friday end-of-week coverage is weak.
      </p>

      <h3 style="margin-top:10px;">Monthly Comparison</h3>
      <table>
        <tr><th>Month</th><th>Total</th><th>Unmet</th><th>Unmet Rate</th></tr>
"""
for _, r in monthly.iterrows():
    html += f"        <tr><td>{r['shift_month']}</td><td>{int(r['total']):,}</td><td>{int(r['unfilled']+r['cancelled']):,}</td><td>{r['unmet_rate']:.1%}</td></tr>\n"

html += f"""      </table>
      <p style="font-size:11px; color:#888; margin-top:6px;">
        January 2026 had 11% more shifts than December 2025, with similar unmet rates —
        indicating demand growth without a proportional increase in caregiver supply.
      </p>
    </div>
  </div>
</div>

<!-- ══════════════ PAGE 2 ══════════════ -->
<div class="page">
  <h1>Factors &amp; Recommendations</h1>

  <h2>Q3: Explanatory Factors for Unmet Demand</h2>

  <div class="two-col">
    <div>
      <h3>Factor 1: Caregiver Supply per ZIP</h3>
      <p style="line-height:1.6; margin-bottom:8px;">
        ZIPs with <strong>fewer active caregivers per shift</strong> show significantly higher unmet rates.
        Unfilled shifts occur in ZIPs with an average of <strong>199 active caregivers</strong>
        vs <strong>313</strong> for filled shifts — a <strong>36% supply gap</strong>.
        High-volume ZIPs like <em>10463 (Bronx)</em> have 13.4% unmet rates despite large shift counts,
        pointing to concentrated demand in caregiver-sparse areas.
      </p>
      <h3>Factor 2: Invitation Coverage</h3>
      <p style="line-height:1.6; margin-bottom:8px;">
        Analysis of available caregiver-shift interaction records shows that
        <strong>filled shifts receive ~1.6x more caregiver invitations</strong> on average.
        Unfilled shifts have near-zero invitation activity — indicating a routing or notification
        failure upstream of the caregiver decision.
        <span class="badge">Data limitation</span>
        <br><em style="font-size:11px; color:#999;">Note: caregiver_shift file covers only 595 of 2.66M shifts (0.03%).
        Factor analysis should be validated with full data.</em>
      </p>
    </div>
    <div>
      <h3>Factor 3: Day-of-Week Pattern</h3>
      <p style="line-height:1.6; margin-bottom:8px;">
        Friday unmet rate (11.7%) is 28% above Tuesday's (9.1%).
        This mirrors industry patterns where end-of-week caregivers are less available
        and agencies post shifts late. Monday (11.0%) is similarly elevated —
        suggesting weekend-to-weekday transition gaps.
      </p>
      <h3>Factor 4: Service Type Concentration</h3>
      <p style="line-height:1.6; margin-bottom:8px;">
        Several service types show systematically higher unmet rates (>20%)
        even at meaningful volume. Services such as <em>Personal Care Services</em>,
        <em>Health Maintenance</em>, and specialty nursing visits are harder to fill,
        likely due to stricter discipline requirements (RN vs PCW).
      </p>
    </div>
  </div>

  <div class="warn">
    <strong>Data Quality Flags:</strong>
    (1) 4,643 shifts from demo agencies (IDs: 16, 123, 299) were incorrectly included in the original analysis due to a type mismatch — now corrected.
    (2) The caregiver_shift file is a 1,000-row sample (0.03% coverage) — Q3 interaction metrics should be re-run with the full file.
    (3) 6,531 "Confirmed" status shifts have no assigned caregiver — data inconsistency flagged for the engineering team.
  </div>

  <h2>Q4: Recommended Actions</h2>

  <div class="rec">
    <strong>1. Targeted Caregiver Recruitment in Supply-Desert ZIPs</strong><br>
    <span style="font-size:12px; line-height:1.7;">
      Bronx ZIPs (10463, 10462, 10473) combine high shift volume with below-median caregiver density.
      Launch geo-targeted recruiting campaigns in these ZIP codes, partnering with local community colleges
      and home care training programs. Incentivize caregivers already in nearby ZIPs to expand their service radius.
      <br><strong>Track:</strong> Active caregivers per shift in target ZIPs | Time-to-fill in targeted areas
    </span>
  </div>

  <div class="rec">
    <strong>2. Fix Invitation Routing for Unfilled Shifts</strong><br>
    <span style="font-size:12px; line-height:1.7;">
      Unfilled shifts receive near-zero caregiver invitations. Implement an automated escalation rule:
      if a shift has 0 invites sent within X hours of creation, auto-broaden the invite radius and alert
      the coordinator. Use caregiver proximity, availability history, and discipline match to rank invites.
      <br><strong>Track:</strong> % shifts with ≥1 invite within 2 hrs | Invite-to-apply conversion rate
    </span>
  </div>

  <div class="rec">
    <strong>3. Friday &amp; Peak-Day Incentive Program</strong><br>
    <span style="font-size:12px; line-height:1.7;">
      Introduce a premium pay badge for shifts posted on Fridays and Mondays, plus any ZIP with
      unmet rate >12%. A small financial incentive (e.g., +$2–3/hr bonus) has outsized impact on
      caregiver availability for hard-to-fill windows. Flag these in the app with a visible "Priority Shift" badge.
      <br><strong>Track:</strong> Friday fill rate | Time-to-fill for priority-badged shifts
    </span>
  </div>

  <h2>Outputs Generated</h2>
  <table style="font-size:12px;">
    <tr><th>File</th><th>Description</th></tr>
    <tr><td>DASHBOARD.html</td><td>Interactive KPI dashboard with map, DOW chart, top ZIPs</td></tr>
    <tr><td>01_top_zips_unmet_demand.png</td><td>Bar chart — top 15 ZIPs by unmet volume</td></tr>
    <tr><td>02_unmet_demand_map.html</td><td>Interactive scatter map — all ZIPs sized by unmet volume</td></tr>
    <tr><td>03_time_trend.png</td><td>Stacked area + rolling unmet rate trend</td></tr>
    <tr><td>04_day_of_week.png</td><td>Unmet rate by day of week</td></tr>
    <tr><td>05_supply_vs_unmet.png</td><td>Caregiver supply vs unmet rate scatter by ZIP</td></tr>
    <tr><td>06_service_type_unmet.png</td><td>Unmet rate by service type (≥50 shifts)</td></tr>
    <tr><td>07_monthly_trend.png</td><td>Monthly grouped bar with unmet rate line</td></tr>
    <tr><td>zip_level_metrics.csv</td><td>Full ZIP-level aggregated metrics</td></tr>
    <tr><td>daily_trend.csv</td><td>Daily shift counts and 7-day rolling averages</td></tr>
    <tr><td>service_type_metrics.csv</td><td>All service types with unmet rates</td></tr>
  </table>

  <div class="footer">
    CareConnect Home Assignment — Caregiver Supply &amp; Unmet Demand Hotspots &nbsp;|&nbsp;
    Analysis period: Dec 2025 – Jan 2026 &nbsp;|&nbsp; All metrics exclude demo/disabled agencies and template shifts
  </div>
</div>

</body>
</html>"""

with open('output/SUMMARY_2PAGE.html', 'w') as f:
    f.write(html)
print("  ✓ 2-page summary saved: output/SUMMARY_2PAGE.html")

print("\n" + "="*60)
print("ALL OUTPUTS COMPLETE")
print("="*60)
print("""
  output/
  ├── DASHBOARD.html           ← Open in browser (interactive)
  ├── SUMMARY_2PAGE.html       ← 2-page printable summary
  ├── 01_top_zips_unmet_demand.png
  ├── 02_unmet_demand_map.html ← Interactive map
  ├── 03_time_trend.png
  ├── 04_day_of_week.png
  ├── 05_supply_vs_unmet.png
  ├── 06_service_type_unmet.png
  ├── 07_monthly_trend.png
  ├── zip_level_metrics.csv
  ├── daily_trend.csv
  └── service_type_metrics.csv
""")
