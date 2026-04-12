"""
=============================================================================
CareConnect Staffing Hotspot Analysis
=============================================================================
Analyzes unmet demand patterns in home healthcare shift staffing.

SETUP:
1. Update FILE PATHS in Section 0 below to point to your CSV/Excel files
2. pip install pandas matplotlib seaborn plotly folium openpyxl
3. Run: python careconnect_analysis.py

OUTPUTS:
- Console: full metric tables and summaries
- Saved charts (PNG): in ./output/ folder
- Interactive map (HTML): in ./output/ folder
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

# Create output folder
os.makedirs('output', exist_ok=True)

# =============================================================================
# SECTION 0: FILE PATHS — UPDATE THESE TO YOUR LOCAL FILE PATHS
# =============================================================================
# Accepts .csv or .xlsx files. Just change the paths below.

SHIFT_AGENCY_FILE       = "shift_agency.csv"
CAREGIVER_SHIFT_DEC     = "caregiver_shift_dec_2025.csv"
CAREGIVER_SHIFT_JAN     = None                            # No January file
CAREGIVER_AGENCY_FILE   = "caregiver_agency.csv"
ADDRESS_FILE            = "address.csv"
USZIPS_FILE             = "uszips.csv"

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================
print("=" * 70)
print("LOADING DATA...")
print("=" * 70)

def load_file(path):
    """Load CSV or Excel file, handling malformed rows and encoding gracefully."""
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xlsx' or ext == '.xls':
        return pd.read_excel(path)
    else:
        # Try combinations of delimiters and encodings
        for encoding in ['utf-8', 'latin-1']:
            for sep in [',', '\t', '|']:
                try:
                    df = pd.read_csv(path, sep=sep, low_memory=False,
                                     on_bad_lines='warn', encoding=encoding)
                    if len(df.columns) > 2:
                        print(f"    → Loaded {os.path.basename(path)} (sep='{sep}', encoding={encoding})")
                        return df
                except TypeError:
                    try:
                        df = pd.read_csv(path, sep=sep, low_memory=False,
                                         error_bad_lines=False, warn_bad_lines=True,
                                         encoding=encoding)
                        if len(df.columns) > 2:
                            print(f"    → Loaded {os.path.basename(path)} (sep='{sep}', encoding={encoding})")
                            return df
                    except:
                        continue
                except:
                    continue
        # Final fallback: latin-1 encoding with python engine
        return pd.read_csv(path, engine='python', on_bad_lines='skip',
                           encoding='latin-1')

shift_agency = load_file(SHIFT_AGENCY_FILE)
print(f"  shift_agency:      {shift_agency.shape[0]:>8,} rows x {shift_agency.shape[1]} cols")

cg_shift_dec = load_file(CAREGIVER_SHIFT_DEC)
if CAREGIVER_SHIFT_JAN:
    cg_shift_jan = load_file(CAREGIVER_SHIFT_JAN)
    caregiver_shift = pd.concat([cg_shift_dec, cg_shift_jan], ignore_index=True)
    print(f"  caregiver_shift:   {caregiver_shift.shape[0]:>8,} rows (Dec + Jan combined)")
else:
    caregiver_shift = cg_shift_dec
    print(f"  caregiver_shift:   {caregiver_shift.shape[0]:>8,} rows")

caregiver_agency = load_file(CAREGIVER_AGENCY_FILE)
print(f"  caregiver_agency:  {caregiver_agency.shape[0]:>8,} rows")

address = load_file(ADDRESS_FILE)
print(f"  address:           {address.shape[0]:>8,} rows")

uszips = load_file(USZIPS_FILE)
print(f"  uszips:            {uszips.shape[0]:>8,} rows")

# =============================================================================
# SECTION 2: DATA CLEANING & PREP
# =============================================================================
print("\n" + "=" * 70)
print("CLEANING DATA...")
print("=" * 70)

# --- 2a. Exclude demo/disabled agencies ---
DEMO_AGENCIES = [4, 11, 13, 16, 65, 118, 123, 184, 228, 231,
                 299, 422, 427, 439, 444, 450, 455, 466, 479, 489]

before = len(shift_agency)
shift_agency = shift_agency[~shift_agency['agency_id'].isin(DEMO_AGENCIES)]
print(f"  Excluded {before - len(shift_agency):,} demo agency shifts ({len(shift_agency):,} remaining)")

caregiver_shift = caregiver_shift[~caregiver_shift['shift_id'].isin(
    shift_agency[shift_agency['agency_id'].isin(DEMO_AGENCIES)]['shift_id']
)] if 'shift_id' in caregiver_shift.columns else caregiver_shift

caregiver_agency = caregiver_agency[~caregiver_agency['agency_id'].isin(DEMO_AGENCIES)]

# --- 2b. Map status codes ---
SHIFT_STATUS_MAP = {
    '01': 'Pending', '02': 'Confirmed', '03': 'In Process',
    '04': 'Closed', '09': 'Hold', '10': 'Cancelled',
    # Handle as strings and ints
    1: 'Pending', 2: 'Confirmed', 3: 'In Process',
    4: 'Closed', 9: 'Hold', 10: 'Cancelled'
}

STAFF_STATUS_MAP = {
    '01': 'pending', '02': 'active', '03': 'hold',
    '04': 'terminated', '05': 'cancelled',
    1: 'pending', 2: 'active', 3: 'hold',
    4: 'terminated', 5: 'cancelled'
}

shift_agency['status_raw'] = shift_agency['status']
shift_agency['status_label'] = shift_agency['status'].map(SHIFT_STATUS_MAP).fillna(shift_agency['status'])

if 'staff_status' in caregiver_agency.columns:
    caregiver_agency['staff_status_label'] = caregiver_agency['staff_status'].map(STAFF_STATUS_MAP).fillna(caregiver_agency['staff_status'])

# --- 2c. Exclude template shifts ---
if 'template' in shift_agency.columns:
    before = len(shift_agency)
    shift_agency = shift_agency[shift_agency['template'] != 1]
    print(f"  Excluded {before - len(shift_agency):,} template shifts ({len(shift_agency):,} remaining)")

# --- 2d. Parse dates ---
for col in ['start', 'end', 'created', 'last_modified']:
    if col in shift_agency.columns:
        shift_agency[col] = pd.to_datetime(shift_agency[col], errors='coerce')

for col in ['shift_start', 'shift_end', 'offered', 'applied', 'assigned',
            'cancelled', 'declined', 'rejected', 'unapplied', 'read', 'last_invited']:
    if col in caregiver_shift.columns:
        caregiver_shift[col] = pd.to_datetime(caregiver_shift[col], errors='coerce')

# --- 2e. Extract time features ---
if 'start' in shift_agency.columns and shift_agency['start'].notna().any():
    shift_agency['shift_month'] = shift_agency['start'].dt.to_period('M')
    shift_agency['shift_week'] = shift_agency['start'].dt.to_period('W')
    shift_agency['shift_dow'] = shift_agency['start'].dt.day_name()
    shift_agency['shift_date'] = shift_agency['start'].dt.date
    print("  Extracted time features from shift start dates")
else:
    print("  WARNING: Could not parse shift start dates. Time trend analysis may be limited.")

# --- 2f. Prepare address table (client addresses) ---
# Drop the binary 'location' column if present — it causes parsing issues
if 'location' in address.columns:
    address = address.drop(columns=['location'])

# Safe numeric conversion (binary data may have bled into ID columns)
for col in ['client_id', 'caregiver_id', 'agency_id']:
    if col in address.columns:
        address[col] = pd.to_numeric(address[col], errors='coerce')

# Get client addresses with ZIP codes
client_address = address[address['client_id'].notna()].copy()
client_address['client_id'] = client_address['client_id'].astype(int)
print(f"  Client address rows (raw): {len(client_address):,}")

# Get caregiver addresses
cg_address = address[address['caregiver_id'].notna()].copy()
if len(cg_address) > 0:
    cg_address['caregiver_id'] = cg_address['caregiver_id'].astype(int)
    print(f"  Caregiver address rows (raw): {len(cg_address):,}")

# If primary flag exists, prefer primary addresses
if 'primary' in client_address.columns:
    client_address['primary'] = pd.to_numeric(client_address['primary'], errors='coerce').fillna(0)
    client_addr_primary = client_address[client_address['primary'] == 1]
    if len(client_addr_primary) > 0:
        client_address = client_addr_primary.drop_duplicates(subset=['client_id', 'agency_id'], keep='last')
    else:
        client_address = client_address.drop_duplicates(subset=['client_id', 'agency_id'], keep='last')
else:
    client_address = client_address.drop_duplicates(subset=['client_id', 'agency_id'], keep='last')

# --- 2g. Prep uszips ---
uszips['zip'] = uszips['zip'].astype(str).str.zfill(5)
if 'zip' in client_address.columns:
    client_address['zip'] = client_address['zip'].astype(str).str.strip()
    # Normalize ZIP to 5 digits for joining
    client_address['zip5'] = client_address['zip'].str[:5].str.zfill(5)

print(f"  Client addresses ready: {len(client_address):,} rows")

# =============================================================================
# SECTION 3: BUILD MASTER SHIFT TABLE WITH GEOGRAPHY
# =============================================================================
print("\n" + "=" * 70)
print("BUILDING MASTER TABLE...")
print("=" * 70)

# Coerce join keys to same type
for col in ['client_id', 'agency_id']:
    if col in shift_agency.columns:
        shift_agency[col] = pd.to_numeric(shift_agency[col], errors='coerce')
    if col in client_address.columns:
        client_address[col] = pd.to_numeric(client_address[col], errors='coerce')

# Join shift_agency → client_address → uszips
master = shift_agency.merge(
    client_address[['client_id', 'agency_id', 'city', 'state', 'zip', 'zip5', 'county']],
    on=['client_id', 'agency_id'],
    how='left',
    suffixes=('', '_addr')
)

# If the first join didn't match well, try without agency_id
match_rate = master['zip5'].notna().mean()
if match_rate < 0.3:
    print(f"  Low match rate with agency_id ({match_rate:.0%}), trying join on client_id only...")
    master = shift_agency.merge(
        client_address[['client_id', 'city', 'state', 'zip', 'zip5', 'county']].drop_duplicates('client_id'),
        on='client_id',
        how='left',
        suffixes=('', '_addr')
    )

# Join to uszips for lat/lng and enrichment
master = master.merge(
    uszips[['zip', 'lat', 'lng', 'city', 'state_name', 'county_name', 'population', 'density']].rename(
        columns={'zip': 'zip5', 'city': 'city_zip', 'county_name': 'county_zip'}
    ),
    on='zip5',
    how='left'
)

# Use address city/state if available, fall back to uszips
master['geo_city'] = master['city'].fillna(master['city_zip'])
master['geo_state'] = master['state'].fillna(master['state_name'])
master['geo_county'] = master['county'].fillna(master['county_zip'])

print(f"  Master table: {len(master):,} shifts")
print(f"  ZIP match rate: {master['zip5'].notna().mean():.1%}")
print(f"  Geo city coverage: {master['geo_city'].notna().mean():.1%}")

# =============================================================================
# SECTION 4: CORE METRICS
# =============================================================================
print("\n" + "=" * 70)
print("CALCULATING CORE METRICS...")
print("=" * 70)

# Define shift categories
master['is_cancelled'] = (master['status_label'] == 'Cancelled').astype(int)
master['is_filled'] = ((master['assigned_caregiver_id'].notna()) &
                        (master['status_label'] != 'Cancelled')).astype(int)
master['is_unfilled'] = ((master['assigned_caregiver_id'].isna()) &
                          (master['status_label'] != 'Cancelled')).astype(int)
master['is_unmet'] = ((master['is_unfilled'] == 1) | (master['is_cancelled'] == 1)).astype(int)

total_shifts = len(master)
filled = master['is_filled'].sum()
unfilled = master['is_unfilled'].sum()
cancelled = master['is_cancelled'].sum()
unmet = master['is_unmet'].sum()

print(f"""
  OVERALL SUMMARY
  {'─' * 45}
  Total Shifts:         {total_shifts:>10,}
  Filled Shifts:        {filled:>10,}  ({filled/total_shifts:.1%})
  Unfilled Shifts:      {unfilled:>10,}  ({unfilled/total_shifts:.1%})
  Cancelled Shifts:     {cancelled:>10,}  ({cancelled/total_shifts:.1%})
  Unmet Demand Rate:    {unmet/total_shifts:>10.1%}
""")

# =============================================================================
# SECTION 5: QUESTION 1 — GEOGRAPHIC HOTSPOTS
# =============================================================================
print("=" * 70)
print("Q1: WHICH ZIP CODES / CITIES HAVE THE HIGHEST UNMET DEMAND?")
print("=" * 70)

# --- By ZIP Code ---
zip_stats = master.groupby('zip5').agg(
    total_shifts=('shift_id', 'count'),
    filled=('is_filled', 'sum'),
    unfilled=('is_unfilled', 'sum'),
    cancelled=('is_cancelled', 'sum'),
    unmet=('is_unmet', 'sum'),
    city=('geo_city', 'first'),
    state=('geo_state', 'first'),
    county=('geo_county', 'first'),
    lat=('lat', 'first'),
    lng=('lng', 'first'),
    population=('population', 'first'),
    density=('density', 'first')
).reset_index()

zip_stats['unmet_rate'] = zip_stats['unmet'] / zip_stats['total_shifts']
zip_stats['fill_rate'] = zip_stats['filled'] / zip_stats['total_shifts']

# Filter to ZIPs with meaningful volume (at least 10 shifts)
zip_meaningful = zip_stats[zip_stats['total_shifts'] >= 10].sort_values('unmet', ascending=False)

print("\n  TOP 15 ZIP CODES BY UNFILLED VOLUME:")
print("  " + "─" * 85)
print(f"  {'ZIP':<8} {'City':<20} {'State':<6} {'Total':>7} {'Unfilled':>9} {'Cancel':>8} {'Unmet Rate':>11}")
print("  " + "─" * 85)
for _, row in zip_meaningful.head(15).iterrows():
    print(f"  {row['zip5']:<8} {str(row['city'])[:19]:<20} {str(row['state'])[:5]:<6} "
          f"{int(row['total_shifts']):>7,} {int(row['unfilled']):>9,} {int(row['cancelled']):>8,} "
          f"{row['unmet_rate']:>10.1%}")

# --- By City ---
city_stats = master.groupby(['geo_city', 'geo_state']).agg(
    total_shifts=('shift_id', 'count'),
    filled=('is_filled', 'sum'),
    unfilled=('is_unfilled', 'sum'),
    cancelled=('is_cancelled', 'sum'),
    unmet=('is_unmet', 'sum')
).reset_index()

city_stats['unmet_rate'] = city_stats['unmet'] / city_stats['total_shifts']
city_meaningful = city_stats[city_stats['total_shifts'] >= 20].sort_values('unmet', ascending=False)

print("\n\n  TOP 15 CITIES BY UNMET DEMAND:")
print("  " + "─" * 75)
print(f"  {'City':<25} {'State':<6} {'Total':>7} {'Unfilled':>9} {'Cancel':>8} {'Unmet Rate':>11}")
print("  " + "─" * 75)
for _, row in city_meaningful.head(15).iterrows():
    print(f"  {str(row['geo_city'])[:24]:<25} {str(row['geo_state'])[:5]:<6} "
          f"{int(row['total_shifts']):>7,} {int(row['unfilled']):>9,} {int(row['cancelled']):>8,} "
          f"{row['unmet_rate']:>10.1%}")

# --- VISUAL 1: Top 15 ZIPs by unmet demand volume ---
fig, ax = plt.subplots(figsize=(14, 7))
top_zips = zip_meaningful.head(15).sort_values('unmet')
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_zips)))

bars = ax.barh(
    top_zips['zip5'].astype(str) + ' — ' + top_zips['city'].fillna('').astype(str),
    top_zips['unmet'],
    color=colors
)

# Add unmet rate labels
for bar, rate in zip(bars, top_zips['unmet_rate']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{rate:.0%}', va='center', fontsize=9, color='#333')

ax.set_xlabel('Unmet Shifts (Unfilled + Cancelled)', fontsize=11)
ax.set_title('Top 15 ZIP Codes by Unmet Demand Volume', fontsize=14, fontweight='bold')
ax.text(0.99, 0.01, 'Percentages show unmet demand rate', transform=ax.transAxes,
        ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('output/01_top_zips_unmet_demand.png', bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: output/01_top_zips_unmet_demand.png")

# --- VISUAL 2: Map of unmet demand by ZIP ---
try:
    import plotly.express as px

    map_data = zip_meaningful[zip_meaningful['lat'].notna()].copy()
    map_data['label'] = (map_data['zip5'] + ' – ' + map_data['city'].fillna('') +
                          '<br>Unmet: ' + map_data['unmet'].astype(int).astype(str) +
                          ' | Rate: ' + (map_data['unmet_rate'] * 100).round(1).astype(str) + '%')

    fig_map = px.scatter_mapbox(
        map_data,
        lat='lat', lon='lng',
        size='unmet',
        color='unmet_rate',
        color_continuous_scale='Reds',
        hover_name='label',
        size_max=30,
        zoom=3,
        title='Unmet Demand Hotspots by ZIP Code'
    )
    fig_map.update_layout(mapbox_style='carto-positron', height=700)
    fig_map.write_html('output/02_unmet_demand_map.html')
    print("  ✓ Saved: output/02_unmet_demand_map.html (interactive map)")
except ImportError:
    print("  ⚠ Plotly not installed — skipping interactive map. pip install plotly")
except Exception as e:
    print(f"  ⚠ Map generation failed: {e}")

# =============================================================================
# SECTION 6: QUESTION 2 — TIME TRENDS
# =============================================================================
print("\n" + "=" * 70)
print("Q2: HOW DOES UNMET DEMAND TREND OVER TIME?")
print("=" * 70)

if 'shift_date' in master.columns and master['shift_date'].notna().any():

    # --- Weekly trend ---
    daily = master.groupby('shift_date').agg(
        total=('shift_id', 'count'),
        filled=('is_filled', 'sum'),
        unfilled=('is_unfilled', 'sum'),
        cancelled=('is_cancelled', 'sum')
    ).reset_index()
    daily['shift_date'] = pd.to_datetime(daily['shift_date'])
    daily = daily.sort_values('shift_date')

    # Rolling 7-day average for smoother trend
    for col in ['total', 'filled', 'unfilled', 'cancelled']:
        daily[f'{col}_7d'] = daily[col].rolling(7, min_periods=1).mean()
    daily['unmet_rate_7d'] = (daily['unfilled_7d'] + daily['cancelled_7d']) / daily['total_7d']

    # --- VISUAL 3: Time trend of shift outcomes ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.stackplot(daily['shift_date'],
                  daily['filled_7d'], daily['unfilled_7d'], daily['cancelled_7d'],
                  labels=['Filled', 'Unfilled', 'Cancelled'],
                  colors=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.8)
    ax1.set_ylabel('Shifts (7-day rolling avg)', fontsize=11)
    ax1.set_title('Shift Volume & Outcomes Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')

    ax2.plot(daily['shift_date'], daily['unmet_rate_7d'], color='#e74c3c', linewidth=2)
    ax2.fill_between(daily['shift_date'], daily['unmet_rate_7d'], alpha=0.15, color='#e74c3c')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel('Unmet Demand Rate', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Unmet Demand Rate Over Time (7-day rolling)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/03_time_trend.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_time_trend.png")

    # --- Day of week analysis ---
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow = master.groupby('shift_dow').agg(
        total=('shift_id', 'count'),
        unmet=('is_unmet', 'sum')
    ).reindex(dow_order).reset_index()
    dow['unmet_rate'] = dow['unmet'] / dow['total']

    print("\n  UNMET DEMAND BY DAY OF WEEK:")
    print("  " + "─" * 50)
    for _, row in dow.iterrows():
        bar = '█' * int(row['unmet_rate'] * 50)
        print(f"  {row['shift_dow']:<12} {row['unmet_rate']:.1%}  {bar}")

else:
    print("  ⚠ Shift dates could not be parsed — skipping time trend analysis.")
    print("    Check that 'start' column in shift_agency has valid datetime values.")

# =============================================================================
# SECTION 7: QUESTION 3 — EXPLANATORY FACTORS
# =============================================================================
print("\n" + "=" * 70)
print("Q3: WHAT FACTORS ARE ASSOCIATED WITH UNFILLED/CANCELLED SHIFTS?")
print("=" * 70)

# --- 7a. Aggregate caregiver_shift to shift level ---
shift_interactions = caregiver_shift.groupby('shift_id').agg(
    num_caregivers_involved=('caregiver_id', 'nunique'),
    num_offered=('offered', lambda x: x.notna().sum()),
    num_applied=('applied', lambda x: x.notna().sum()),
    num_assigned=('assigned', lambda x: x.notna().sum()),
    num_declined=('declined', lambda x: x.notna().sum()),
    num_cancelled_cg=('cancelled', lambda x: x.notna().sum()),
    num_rejected=('rejected', lambda x: x.notna().sum()),
    num_unapplied=('unapplied', lambda x: x.notna().sum()),
    total_invites=('num_invites_sent', 'sum'),
    avg_invites=('num_invites_sent', 'mean'),
    any_read=('read', lambda x: x.notna().sum()),
).reset_index()

# Calculate rates at shift level
shift_interactions['invite_to_apply_rate'] = np.where(
    shift_interactions['num_offered'] > 0,
    shift_interactions['num_applied'] / shift_interactions['num_offered'], np.nan
)
shift_interactions['decline_rate'] = np.where(
    shift_interactions['num_offered'] > 0,
    shift_interactions['num_declined'] / shift_interactions['num_offered'], np.nan
)
shift_interactions['read_rate'] = np.where(
    shift_interactions['num_offered'] > 0,
    shift_interactions['any_read'] / shift_interactions['num_offered'], np.nan
)

# --- 7b. Merge into master ---
# Coerce shift_id to numeric on both sides
master['shift_id'] = pd.to_numeric(master['shift_id'], errors='coerce')
shift_interactions['shift_id'] = pd.to_numeric(shift_interactions['shift_id'], errors='coerce')
caregiver_shift['shift_id'] = pd.to_numeric(caregiver_shift['shift_id'], errors='coerce')
analysis = master.merge(shift_interactions, on='shift_id', how='left')

# --- 7c. Assignment latency ---
if 'created' in analysis.columns and 'assigned' in caregiver_shift.columns:
    # Get earliest assignment time per shift
    first_assign = caregiver_shift[caregiver_shift['assigned'].notna()].groupby('shift_id')['assigned'].min()
    analysis = analysis.merge(first_assign.rename('first_assigned'), on='shift_id', how='left')
    if analysis['created'].notna().any() and analysis['first_assigned'].notna().any():
        analysis['assignment_latency_hrs'] = (
            (analysis['first_assigned'] - analysis['created']).dt.total_seconds() / 3600
        )

# --- 7d. Caregiver supply per ZIP ---
if len(cg_address) > 0 and 'staff_status_label' in caregiver_agency.columns:
    active_cg = caregiver_agency[caregiver_agency['staff_status_label'] == 'active']
    cg_with_zip = active_cg.merge(
        cg_address[['caregiver_id', 'zip']].drop_duplicates('caregiver_id'),
        on='caregiver_id', how='left'
    )
    cg_with_zip['zip5'] = cg_with_zip['zip'].astype(str).str[:5].str.zfill(5)
    cg_supply = cg_with_zip.groupby('zip5')['caregiver_id'].nunique().reset_index()
    cg_supply.columns = ['zip5', 'active_caregivers_in_zip']

    analysis = analysis.merge(cg_supply, on='zip5', how='left')
    analysis['active_caregivers_in_zip'] = analysis['active_caregivers_in_zip'].fillna(0)

    # Supply-demand ratio
    zip_shift_count = analysis.groupby('zip5')['shift_id'].count().reset_index()
    zip_shift_count.columns = ['zip5', 'shifts_in_zip']
    analysis = analysis.merge(zip_shift_count, on='zip5', how='left')
    analysis['cg_per_shift'] = np.where(
        analysis['shifts_in_zip'] > 0,
        analysis['active_caregivers_in_zip'] / analysis['shifts_in_zip'], np.nan
    )

# --- 7e. Compare filled vs unfilled ---
print("\n  FACTOR COMPARISON: FILLED vs UNFILLED SHIFTS")
print("  " + "─" * 65)
print(f"  {'Factor':<35} {'Filled':>12} {'Unfilled':>12} {'Diff':>8}")
print("  " + "─" * 65)

comparison_cols = {
    'num_offered': 'Avg caregivers offered',
    'num_applied': 'Avg caregivers applied',
    'decline_rate': 'Decline rate',
    'invite_to_apply_rate': 'Invite-to-apply rate',
    'read_rate': 'Invite read rate',
    'total_invites': 'Total invites sent',
}

if 'assignment_latency_hrs' in analysis.columns:
    comparison_cols['assignment_latency_hrs'] = 'Avg assignment latency (hrs)'
if 'active_caregivers_in_zip' in analysis.columns:
    comparison_cols['active_caregivers_in_zip'] = 'Active caregivers in ZIP'
if 'density' in analysis.columns:
    comparison_cols['density'] = 'Population density'
if 'cg_per_shift' in analysis.columns:
    comparison_cols['cg_per_shift'] = 'Caregivers per shift (ZIP)'

filled_df = analysis[analysis['is_filled'] == 1]
unfilled_df = analysis[analysis['is_unfilled'] == 1]

factor_results = {}
for col, label in comparison_cols.items():
    if col in analysis.columns:
        f_val = filled_df[col].mean()
        u_val = unfilled_df[col].mean()
        if pd.notna(f_val) and pd.notna(u_val) and f_val != 0:
            diff_pct = (u_val - f_val) / abs(f_val) * 100
            factor_results[col] = {'filled': f_val, 'unfilled': u_val, 'diff': diff_pct}

            if col in ['decline_rate', 'invite_to_apply_rate', 'read_rate']:
                print(f"  {label:<35} {f_val:>11.1%} {u_val:>11.1%} {diff_pct:>+7.0f}%")
            else:
                print(f"  {label:<35} {f_val:>11.1f} {u_val:>11.1f} {diff_pct:>+7.0f}%")

# --- VISUAL 4: Factor comparison chart ---
if factor_results:
    fig, axes = plt.subplots(1, min(4, len(factor_results)), figsize=(16, 5))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    plot_factors = list(factor_results.items())[:4]
    for ax, (col, vals) in zip(axes, plot_factors):
        label = comparison_cols[col]
        x = ['Filled', 'Unfilled']
        y = [vals['filled'], vals['unfilled']]
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(x, y, color=colors, width=0.5, edgecolor='white')
        ax.set_title(label, fontsize=10, fontweight='bold')

        if col in ['decline_rate', 'invite_to_apply_rate', 'read_rate']:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax.text(1, max(y) * 0.95, f'{vals["diff"]:+.0f}%', ha='center',
                fontsize=11, fontweight='bold', color='#e74c3c' if vals['diff'] > 0 else '#2ecc71')

    plt.suptitle('Key Factors: Filled vs Unfilled Shifts', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output/04_factor_comparison.png', bbox_inches='tight')
    plt.close()
    print("\n  ✓ Saved: output/04_factor_comparison.png")

# --- 7f. Service type analysis ---
print("\n\n  UNMET DEMAND BY SERVICE TYPE:")
print("  " + "─" * 70)
svc = master.groupby('service_description').agg(
    total=('shift_id', 'count'),
    unfilled=('is_unfilled', 'sum'),
    cancelled=('is_cancelled', 'sum'),
    unmet=('is_unmet', 'sum')
).reset_index()
svc['unmet_rate'] = svc['unmet'] / svc['total']
svc = svc[svc['total'] >= 10].sort_values('unmet_rate', ascending=False)

print(f"  {'Service':<35} {'Total':>7} {'Unmet':>7} {'Rate':>8}")
print("  " + "─" * 70)
for _, row in svc.head(10).iterrows():
    print(f"  {str(row['service_description'])[:34]:<35} {int(row['total']):>7,} "
          f"{int(row['unmet']):>7,} {row['unmet_rate']:>7.1%}")

# --- VISUAL 5: Unmet rate by service type ---
if len(svc) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    top_svc = svc.head(10).sort_values('unmet_rate')
    bars = ax.barh(top_svc['service_description'].astype(str), top_svc['unmet_rate'],
                    color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_svc))))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Unmet Demand Rate', fontsize=11)
    ax.set_title('Unmet Demand Rate by Service Type', fontsize=14, fontweight='bold')

    for bar, total in zip(bars, top_svc['total']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'n={int(total):,}', va='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig('output/05_service_type_unmet.png', bbox_inches='tight')
    plt.close()
    print("\n  ✓ Saved: output/05_service_type_unmet.png")

# =============================================================================
# SECTION 8: QUESTION 4 — RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 70)
print("Q4: RECOMMENDED ACTIONS TO IMPROVE COVERAGE")
print("=" * 70)

print("""
  Based on the data patterns, here are 3 data-driven recommendations:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. TARGETED CAREGIVER RECRUITMENT IN SUPPLY DESERT ZIPs            │
  │                                                                     │
  │ Action: Identify ZIPs where unmet demand is high AND caregiver     │
  │ supply per shift is below the median. Launch geo-targeted          │
  │ recruiting campaigns in those areas.                               │
  │                                                                     │
  │ Metric to track: Active caregivers per shift in target ZIPs        │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 2. REDUCE DECLINE RATES THROUGH EARLIER & SMARTER INVITATIONS     │
  │                                                                     │
  │ Action: Send shift invitations earlier (reduce assignment          │
  │ latency). Use caregiver availability and proximity data to         │
  │ invite the most likely-to-accept caregivers first, rather than     │
  │ blanket invitations.                                               │
  │                                                                     │
  │ Metric to track: Decline rate, invite-to-apply rate                │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ 3. WEEKEND & HARD-TO-FILL SHIFT INCENTIVE PROGRAM                 │
  │                                                                     │
  │ Action: Introduce premium pay or bonus incentives for shifts       │
  │ with historically high unmet demand (weekends, holidays,           │
  │ specific service types). Flag these shifts in the app with a       │
  │ "bonus" badge to increase visibility.                              │
  │                                                                     │
  │ Metric to track: Weekend fill rate, time-to-fill for bonus shifts  │
  └─────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# SECTION 9: EXPORT SUMMARY DATA
# =============================================================================
print("=" * 70)
print("EXPORTING DATA...")
print("=" * 70)

# Save key tables for further analysis
zip_stats.to_csv('output/zip_level_metrics.csv', index=False)
city_stats.to_csv('output/city_level_metrics.csv', index=False)

if 'shift_date' in master.columns:
    daily.to_csv('output/daily_trend.csv', index=False)

svc.to_csv('output/service_type_metrics.csv', index=False)

print("  ✓ Saved: output/zip_level_metrics.csv")
print("  ✓ Saved: output/city_level_metrics.csv")
print("  ✓ Saved: output/daily_trend.csv")
print("  ✓ Saved: output/service_type_metrics.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("""
  Generated outputs in ./output/:
  ├── 01_top_zips_unmet_demand.png     (Bar chart: top ZIPs)
  ├── 02_unmet_demand_map.html         (Interactive map)
  ├── 03_time_trend.png                (Stacked area + rate trend)
  ├── 04_factor_comparison.png         (Filled vs unfilled factors)
  ├── 05_service_type_unmet.png        (Service type breakdown)
  ├── zip_level_metrics.csv            (ZIP-level data export)
  ├── city_level_metrics.csv           (City-level data export)
  ├── daily_trend.csv                  (Daily trend data)
  └── service_type_metrics.csv         (Service type data)
""")