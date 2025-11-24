import pandas as pd
import numpy as np
from itertools import combinations
from snowflake.snowpark.context import get_active_session
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from scipy.optimize import minimize
import re
import random
from decimal import Decimal, ROUND_HALF_UP


def round_half_up(value, decimals=2):
    """
    Round using 'round half up' to match PRF official tool.
    Python's built-in round() uses banker's rounding (12.675 -> 12.67).
    PRF Tool uses round half up (12.675 -> 12.68).
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0
    d = Decimal(str(value))
    if decimals == 0:
        quantize_to = Decimal('1')
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')
    return float(d.quantize(quantize_to, rounding=ROUND_HALF_UP))


def calculate_protection(county_base_value, coverage_level, productivity_factor, decimals=2):
    """
    Calculate dollar protection with proper precision using Decimal arithmetic.
    Prevents issues like 16.90 * 0.75 * 1.0 = 12.674999999999999
    """
    cbv = Decimal(str(county_base_value))
    cov = Decimal(str(coverage_level))
    prod = Decimal(str(productivity_factor))
    result = cbv * cov * prod
    if decimals == 0:
        quantize_to = Decimal('1')
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')
    return float(result.quantize(quantize_to, rounding=ROUND_HALF_UP))

# === GLOBAL CONSTANTS ===
INTERVAL_ORDER_11 = ['Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
                     'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec']
INTERVAL_INDICES = {name: i for i, name in enumerate(INTERVAL_ORDER_11)}

# Z-Score to Plain Language Mapping
HISTORICAL_CONTEXT_MAP = {
    'Dry': (-999, -0.25),
    'Normal': (-0.25, 0.25),
    'Wet': (0.25, 999)
}
TREND_MAP = {
    'Get Drier': (-999, -0.2),
    'Stay Stable': (-0.2, 0.2),
    'Get Wetter': (0.2, 999)
}

# === ALLOCATION CONSTRAINTS ===
MIN_ALLOCATION = 0.10  # 10% minimum per active interval
MAX_ALLOCATION = 0.50  # 50% maximum per active interval
ALLOCATION_INCREMENT = 0.01  # 1% increments

# === FULL COVERAGE STAGGERED PATTERNS ===
# Pattern A: 6 non-adjacent intervals (even indices)
PATTERN_A_INTERVALS = [0, 2, 4, 6, 8, 10]  # Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec
# Pattern B: 5 non-adjacent intervals (odd indices)
PATTERN_B_INTERVALS = [1, 3, 5, 7, 9]  # Feb-Mar, Apr-May, Jun-Jul, Aug-Sep, Oct-Nov

# === GLOBAL SORTING MAPS ===
SORT_METRIC_DB_MAP = {
    'Portfolio Return': 'Cumulative_ROI',
    'Risk-Adjusted Return': 'Risk_Adjusted_Return',
    'Median ROI': 'Median_ROI',
    'Win Rate': 'Win_Rate'
}
SORT_METRIC_DISPLAY_MAP = {v: k for k, v in SORT_METRIC_DB_MAP.items()}

st.set_page_config(layout="wide", page_title="Layer 3 - PRF Systematic Positioning")

# =============================================================================
# === HELPER FUNCTIONS ===
# =============================================================================
def extract_numeric_grid_id(grid_id_display):
    """Extract numeric grid ID from display format like '9128 (Jim Wells - TX)'"""
    if isinstance(grid_id_display, (int, float)):
        return int(grid_id_display)
    
    match = re.match(r'(\d+)', str(grid_id_display))
    if match:
        return int(match.group(1))
    return None

def format_grid_display(grid_id_numeric, county_state=None):
    """Format grid ID for display"""
    if county_state:
        return f"{grid_id_numeric} ({county_state})"
    return str(grid_id_numeric)

def highlight_greater_than_zero(val):
    """Style helper for DataFrames"""
    if isinstance(val, (int, float)) and val > 0.001:
        return 'background-color: #DFF0D8'
    return ''

# =============================================================================
# === CACHED DATA LOADING ===
# =============================================================================
@st.cache_data(ttl=3600)
def load_distinct_grids(_session):
    """Fetches all available Grid IDs with county/state information"""
    query = """
        SELECT DISTINCT GRID_ID
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.COUNTY_BASE_VALUES_PLATINUM
        ORDER BY GRID_ID
    """
    df = _session.sql(query).to_pandas()
    return df['GRID_ID'].tolist()

@st.cache_data(ttl=3600)
def load_all_indices(_session, grid_id_numeric):
    """Fetches all historical data for a grid (using numeric ID)"""
    query = f"""
        SELECT
            YEAR, INTERVAL_NAME, INDEX_VALUE, INTERVAL_CODE,
            INTERVAL_MAPPING_TS_TEXT, INTERVAL_MAPPING_TS_NUMBER,
            OPTICAL_MAPPING_CPC, ONI_VALUE,
            SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD,
            SEQUENTIAL_Z_SCORE_5P,
            SEQUENTIAL_Z_SCORE_11P
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.RAIN_INDEX_PLATINUM_ENHANCED
        WHERE GRID_ID = {grid_id_numeric}
        ORDER BY YEAR, INTERVAL_CODE
    """
    df = _session.sql(query).to_pandas()
    df['INDEX_VALUE'] = pd.to_numeric(df['INDEX_VALUE'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_5P'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_5P'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_11P'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_11P'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['INTERVAL_MAPPING_TS_NUMBER'].astype(str).str.split('.').str[0], format='%Y-%m', errors='coerce')
    return df

@st.cache_data(ttl=3600)
def load_county_base_value(_session, grid_id_display):
    """Fetches county base value using the full Grid ID display format"""
    query = f"""
        SELECT AVG(COUNTY_BASE_VALUE)
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.COUNTY_BASE_VALUES_PLATINUM
        WHERE GRID_ID = '{grid_id_display}'
    """
    result = _session.sql(query).to_pandas()
    if result.empty or result.iloc[0, 0] is None:
        return 0.0
    return float(result.iloc[0, 0])

@st.cache_data(ttl=3600)
def get_current_rate_year(_session):
    """Gets most recent premium rate year"""
    return int(_session.sql("SELECT MAX(YEAR) FROM PRF_PREMIUM_RATES").to_pandas().iloc[0, 0])

@st.cache_data(ttl=3600)
def load_premium_rates(_session, grid_id_numeric, use, coverage_level, year):
    """Fetches premium rates (using numeric Grid ID)"""
    cov_string = f"{coverage_level:.0%}"
    query = f"""
        SELECT INDEX_INTERVAL_NAME, PREMIUMRATE
        FROM PRF_PREMIUM_RATES
        WHERE GRID_ID = {grid_id_numeric}
          AND INTENDED_USE = '{use}'
          AND COVERAGE_LEVEL = '{cov_string}'
          AND YEAR = {year}
    """
    df = _session.sql(query).to_pandas()
    df['PREMIUMRATE'] = pd.to_numeric(df['PREMIUMRATE'], errors='coerce')
    return df.set_index('INDEX_INTERVAL_NAME')['PREMIUMRATE'].to_dict()

@st.cache_data(ttl=3600)
def load_subsidy(_session, plan_code, coverage_level):
    """Fetches subsidy percentage"""
    query = f"""
        SELECT SUBSIDY_PERCENT
        FROM SUBSIDYPERCENT_YTD_PLATINUM
        WHERE INSURANCE_PLAN_CODE = {plan_code}
          AND COVERAGE_LEVEL_PERCENT = {coverage_level}
        LIMIT 1
    """
    return float(_session.sql(query).to_pandas().iloc[0, 0])

# =============================================================================
# === CORE HELPER FUNCTIONS & METRICS ===
# =============================================================================
def is_adjacent(interval1, interval2):
    """Check if two intervals are adjacent (excluding Nov-Dec/Jan-Feb wrap)"""
    try:
        idx1 = INTERVAL_ORDER_11.index(interval1)
        idx2 = INTERVAL_ORDER_11.index(interval2)
    except ValueError:
        return False
    
    diff = abs(idx1 - idx2)
    return diff == 1

def has_adjacent_intervals(intervals_list):
    """
    Check if any two intervals in a list are adjacent.
    """
    if len(intervals_list) < 2:
        return False
    
    for i in range(len(intervals_list)):
        for j in range(i + 1, len(intervals_list)):
            interval1 = intervals_list[i]
            interval2 = intervals_list[j]
            
            if (interval1 == 'Nov-Dec' and interval2 == 'Jan-Feb') or \
               (interval1 == 'Jan-Feb' and interval2 == 'Nov-Dec'):
                continue
            
            if is_adjacent(interval1, interval2):
                return True
    
    return False

def generate_allocations(intervals_to_use, num_intervals):
    """Generate allocation percentages respecting 10% min and 50% max rule"""
    allocations = []
    if num_intervals == 1:
        return []
    elif num_intervals == 2:
        allocations.append({intervals_to_use[0]: 0.5, intervals_to_use[1]: 0.5})
    elif num_intervals == 3:
        splits = [(0.50, 0.30, 0.20), (0.50, 0.25, 0.25), (0.40, 0.30, 0.30)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(3)})
    elif num_intervals == 4:
        splits = [(0.25, 0.25, 0.25, 0.25), (0.50, 0.20, 0.20, 0.10), (0.40, 0.30, 0.20, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(4)})
    elif num_intervals == 5:
        splits = [(0.20, 0.20, 0.20, 0.20, 0.20), (0.50, 0.15, 0.15, 0.10, 0.10), (0.40, 0.20, 0.20, 0.10, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(5)})
    elif num_intervals == 6:
        splits = [(0.20, 0.20, 0.15, 0.15, 0.15, 0.15), (0.50, 0.10, 0.10, 0.10, 0.10, 0.10), (0.30, 0.20, 0.15, 0.15, 0.10, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(6)})
    elif num_intervals == 7:
        splits = [(0.20, 0.15, 0.15, 0.15, 0.12, 0.12, 0.11), (0.40, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(7)})
    elif num_intervals == 8:
        splits = [(0.15, 0.15, 0.13, 0.13, 0.12, 0.11, 0.11, 0.10), (0.35, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(8)})
    elif num_intervals == 9:
        splits = [(0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(9)})
    elif num_intervals == 10:
        splits = [(0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(10)})
    
    return allocations

def filter_years_by_market_view(df, regime, hist_context, trend):
    """
    Filter years (not intervals) based on market view conditions.
    """
    hist_min, hist_max = HISTORICAL_CONTEXT_MAP[hist_context]
    trend_min, trend_max = TREND_MAP[trend]

    matching_years = []

    for year in df['YEAR'].unique():
        year_data = df[df['YEAR'] == year]

        if len(year_data) < 11:
            continue

        phase_counts = year_data['OPTICAL_MAPPING_CPC'].value_counts()
        dominant_phase = phase_counts.idxmax() if len(phase_counts) > 0 else None
        phase_intervals = phase_counts.max() if len(phase_counts) > 0 else 0

        if dominant_phase != regime or phase_intervals < 5:
            continue

        year_avg_hist_z = year_data['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'].mean()

        if pd.isna(year_avg_hist_z) or year_avg_hist_z < hist_min or year_avg_hist_z >= hist_max:
            continue

        first_interval = year_data.iloc[0]
        last_interval = year_data.iloc[-1]

        if 'SEQUENTIAL_Z_SCORE_5P' not in last_interval.index or 'SEQUENTIAL_Z_SCORE_11P' not in first_interval.index:
            continue

        z_5p_end = last_interval['SEQUENTIAL_Z_SCORE_5P']
        z_11p_start = first_interval['SEQUENTIAL_Z_SCORE_11P']

        if pd.isna(z_5p_end) or pd.isna(z_11p_start):
            continue

        delta = z_5p_end - z_11p_start

        if delta < trend_min or delta >= trend_max:
            continue

        matching_years.append({
            'year': year,
            'dominant_phase': dominant_phase,
            'phase_intervals': phase_intervals,
            'year_avg_hist_z': year_avg_hist_z,
            'year_start_11p': z_11p_start,
            'year_end_5p': z_5p_end,
            'trajectory_delta': delta
        })

    return matching_years


@st.cache_data(ttl=3600, show_spinner=False)
def filter_years_by_market_view_cached(_session, grid_id_numeric, regime, hist_context, trend):
    """
    Cached version of filter_years_by_market_view.
    Uses grid_id to load data internally for cacheability.
    _session prefix excludes it from cache key.
    """
    df = load_all_indices(_session, grid_id_numeric)
    return filter_years_by_market_view(df, regime, hist_context, trend)


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_single_grid_cached(
    grid_id_display,
    grid_id_numeric,
    matching_years_info_tuple,  # tuple of tuples for hashability
    index_data_tuple,  # tuple of tuples: (year, interval, index_value)
    static_data_tuple,  # tuple of (key, value) pairs
    coverage_level,
    min_intervals,
    max_intervals
):
    """
    Cached function to analyze all ROI combinations for a single grid.
    Returns (combinations_list, year_details_list)
    """
    # Reconstruct matching_years_info from tuple
    matching_years_info = [
        {
            'year': y[0],
            'dominant_phase': y[1],
            'phase_intervals': y[2],
            'year_avg_hist_z': y[3],
            'year_start_11p': y[4],
            'year_end_5p': y[5],
            'trajectory_delta': y[6]
        }
        for y in matching_years_info_tuple
    ]

    # Reconstruct static_data from tuple (premiums is a nested tuple)
    static_data = {}
    for key, value in static_data_tuple:
        if key == 'premiums':
            static_data[key] = dict(value)  # Convert premiums tuple back to dict
        else:
            static_data[key] = value

    # Reconstruct index data lookup: {(year, interval): index_value}
    index_lookup = {(row[0], row[1]): row[2] for row in index_data_tuple}

    matching_years = [y['year'] for y in matching_years_info]

    # Generate allocation candidates
    all_intervals = INTERVAL_ORDER_11
    candidates = []
    for num_intervals in range(min_intervals, max_intervals + 1):
        for combo in combinations(all_intervals, num_intervals):
            if has_adjacent_intervals(list(combo)):
                continue
            allocations = generate_allocations(list(combo), num_intervals)
            candidates.extend(allocations)

    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    all_combinations = []
    all_year_details = []

    for allocation in unique_candidates:
        year_rois = []
        year_indemnities = []
        year_premiums = []

        for year_info in matching_years_info:
            year = year_info['year']

            # Build year_data dict for calculate_yearly_roi_from_index_lookup
            year_indices = {}
            for interval in INTERVAL_ORDER_11:
                idx_val = index_lookup.get((year, interval))
                if idx_val is not None:
                    year_indices[interval] = idx_val

            if len(year_indices) >= 11:
                roi, indemnity, premium = calculate_yearly_roi_from_dict(
                    year_indices, allocation, static_data, coverage_level
                )

                if roi is not None and not np.isinf(roi):
                    year_rois.append(roi)
                    year_indemnities.append(indemnity)
                    year_premiums.append(premium)

                    net_return = indemnity - premium
                    alloc_str_detail = ", ".join([
                        f"{k}: {v*100:.0f}%" for k, v in sorted(
                            allocation.items(), key=lambda x: x[1], reverse=True
                        ) if v > 0
                    ])

                    all_year_details.append({
                        'Grid': grid_id_display,
                        'Year': year,
                        'Allocation': alloc_str_detail,
                        'Phase': year_info['dominant_phase'],
                        'Phase_Intervals': year_info['phase_intervals'],
                        'Year_Avg_Hist_Z': year_info['year_avg_hist_z'],
                        'SOY_11P': year_info['year_start_11p'],
                        'EOY_5P': year_info['year_end_5p'],
                        'Trajectory_Delta': year_info['trajectory_delta'],
                        'Coverage': coverage_level,
                        'ROI': roi,
                        'Indemnity': indemnity,
                        'Producer_Premium': premium,
                        'Net_Return': net_return
                    })

        if year_rois:
            avg_roi = np.mean(year_rois)
            median_roi = np.median(year_rois)
            std_dev = np.std(year_rois)
            win_rate = np.sum(1 for r in year_rois if r > 0) / len(year_rois)

            total_indemnity = np.sum(year_indemnities)
            total_premium = np.sum(year_premiums)
            cumulative_roi = (total_indemnity - total_premium) / total_premium if total_premium > 0 else 0.0
            risk_adjusted_return = avg_roi / std_dev if std_dev > 0 else 0.0

            alloc_str = ", ".join([
                f"{k}: {v*100:.0f}%" for k, v in sorted(
                    allocation.items(), key=lambda x: x[1], reverse=True
                ) if v > 0
            ])

            max_roi = np.max(year_rois)
            min_roi = np.min(year_rois)

            all_combinations.append({
                'Grid_ID': grid_id_display,
                'Allocation': alloc_str,
                'Num_Intervals': sum(1 for v in allocation.values() if v > 0),
                'Occurrences': len(year_rois),
                'Average_ROI': avg_roi,
                'Cumulative_ROI': cumulative_roi,
                'Median_ROI': median_roi,
                'Risk_Adjusted_Return': risk_adjusted_return,
                'Win_Rate': win_rate,
                'Std_Dev': std_dev,
                'Max_ROI': max_roi,
                'Min_ROI': min_roi,
                'Best_Worst_Range': max_roi - min_roi
            })

    return all_combinations, all_year_details


def calculate_yearly_roi_from_dict(year_indices, allocation, static_data, coverage_level):
    """
    Calculate ROI for a single year given allocation, using a dict of index values.
    This is a helper for the cached analysis function.
    """
    subsidy = static_data.get('subsidy', 0.5)
    if subsidy > 1.0:
        subsidy = subsidy / 100.0

    premiums = static_data.get('premiums', {})

    dollar_protection = calculate_protection(
        static_data.get('county_base_value', 0),
        coverage_level,
        static_data.get('prod_factor', 1)
    )
    total_protection = round_half_up(dollar_protection * static_data.get('acres', 1), 0)

    total_indemnity = 0
    total_producer_premium = 0

    for interval, pct in allocation.items():
        if pct <= 0:
            continue

        premium_rate = premiums.get(interval, 0)

        if premium_rate <= 0:
            continue

        interval_protection = round_half_up(total_protection * pct, 0)
        total_premium = round_half_up(interval_protection * premium_rate, 0)
        premium_subsidy = round_half_up(total_premium * subsidy, 0)
        producer_premium = total_premium - premium_subsidy

        total_producer_premium += max(0, producer_premium)

        index_value = year_indices.get(interval)
        if index_value is None or pd.isna(index_value):
            continue

        trigger = coverage_level * 100
        shortfall_pct = max(0, (trigger - index_value) / trigger)
        indemnity = round_half_up(shortfall_pct * interval_protection, 0)

        total_indemnity += indemnity

    if total_producer_premium == 0:
        roi = 0.0
    else:
        roi = (total_indemnity - total_producer_premium) / total_producer_premium

    return roi, total_indemnity, total_producer_premium

def get_year_features(year_data_df):
    """Helper to get dominant phase and avg hist z for a single year's df"""
    if len(year_data_df) < 11:
        return {'dominant_phase': None, 'avg_hist_z': np.nan}
        
    phase_counts = year_data_df['OPTICAL_MAPPING_CPC'].value_counts()
    dominant_phase = phase_counts.idxmax() if len(phase_counts) > 0 else 'Neutral'
    
    avg_hist_z = year_data_df['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'].mean()
    
    return {'dominant_phase': dominant_phase, 'avg_hist_z': avg_hist_z}

def calculate_yearly_roi(year_indices_df, allocation, static_data, coverage_level):
    """
    Calculate ROI for a single year given allocation.
    Note: Returns per-acre values if acres=1 in static_data.
    Uses round_half_up for proper rounding to match PRF official tool.
    """
    subsidy = static_data.get('subsidy', 0.5)
    if subsidy > 1.0:
        subsidy = subsidy / 100.0

    premiums = static_data.get('premiums', {})

    # Use calculate_protection for proper decimal arithmetic
    dollar_protection = calculate_protection(
        static_data.get('county_base_value', 0),
        coverage_level,
        static_data.get('prod_factor', 1)
    )
    total_protection = round_half_up(dollar_protection * static_data.get('acres', 1), 0)

    total_indemnity = 0
    total_producer_premium = 0

    for interval, pct in allocation.items():
        if pct <= 0:
            continue

        premium_rate = premiums.get(interval, 0)

        if premium_rate <= 0:
            continue

        # Round each intermediate dollar value to whole dollars
        interval_protection = round_half_up(total_protection * pct, 0)
        total_premium = round_half_up(interval_protection * premium_rate, 0)
        premium_subsidy = round_half_up(total_premium * subsidy, 0)
        producer_premium = total_premium - premium_subsidy

        # Premium is always charged based on allocation, regardless of data availability
        total_producer_premium += max(0, producer_premium)

        # Indemnity only calculated if index data exists for this year
        row = year_indices_df[year_indices_df['INTERVAL_NAME'] == interval]
        if row.empty:
            continue

        try:
            index_value = float(row['INDEX_VALUE'].iloc[0])
            if pd.isna(index_value):
                continue
        except (ValueError, TypeError):
            continue

        trigger = coverage_level * 100
        shortfall_pct = max(0, (trigger - index_value) / trigger)
        indemnity = round_half_up(shortfall_pct * interval_protection, 0)

        total_indemnity += indemnity

    if total_producer_premium == 0:
        if total_indemnity > 0:
            roi = 0.0
        else:
            roi = 0.0
    else:
        roi = (total_indemnity - total_producer_premium) / total_producer_premium

    return roi, total_indemnity, total_producer_premium

def create_download_button(fig, filename, key):
    """Create download button for matplotlib figure with high DPI"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    st.download_button(
        label="📥 Download Chart",
        data=buf,
        file_name=filename,
        mime="image/png",
        key=key
    )

def calculate_grid_quality_score(row):
    """
    Score grids on 1-10 scale combining:
    - Analog years available (60% weight)
    - Portfolio Return (Cumulative_ROI) (40% weight)
    """
    occurrences = row['Occurrences']
    if occurrences >= 10:
        year_score = 60.0
    elif occurrences >= 5:
        year_score = (2.0 * occurrences) + 40.0
    elif occurrences >= 1:
        year_score = 10.0 * occurrences
    else:
        year_score = 0.0

    cumulative_roi_decimal = row['Cumulative_ROI']

    # Normalize ROI to 0-1 scale for scoring
    # Design: 0% → 0.0, 100% → 0.5, 300% → 0.85, 500%+ → 1.0
    roi_score_normalized = max(0, min(cumulative_roi_decimal / 5.0, 1.0))
    roi_score = roi_score_normalized * 40

    raw_score = year_score + roi_score
    scaled_score = (raw_score / 100.0) * 9.0 + 1.0
    return scaled_score

def calculate_portfolio_metrics(perf_df):
    """
    Calculates a dict of metrics from a *yearly portfolio performance dataframe*.
    perf_df must have columns: 'roi', 'indemnity', 'premium'
    """
    if perf_df.empty:
        return {
            'Analog_Years': 0, 'Avg_ROI': 0, 'Cumulative_ROI': 0,
            'Median_ROI': 0, 'Risk_Adjusted_Return': 0, 'Win_Rate': 0, 'Std_Dev': 0,
            'Max_ROI': 0, 'Min_ROI': 0, 'Range': 0, 'Total_Indemnity': 0, 'Total_Premium': 0
        }
    
    rois_list = perf_df['roi'].replace([np.inf, -np.inf], np.nan).dropna().tolist()
    if not rois_list:
        rois_list = [0]
        
    avg_roi = np.mean(rois_list)
    std_dev = np.std(rois_list)
    
    total_indemnity = perf_df['indemnity'].sum()
    total_premium = perf_df['premium'].sum()
    
    if total_premium > 0:
        cumulative_roi = (total_indemnity - total_premium) / total_premium
    else:
        cumulative_roi = 0.0 if total_indemnity == 0 else 0.0 
    
    return {
        'Analog_Years': len(rois_list),
        'Avg_ROI': avg_roi,
        'Cumulative_ROI': cumulative_roi,
        'Median_ROI': np.median(rois_list),
        'Risk_Adjusted_Return': avg_roi / std_dev if std_dev > 0 else 0,
        'Win_Rate': np.sum(1 for r in rois_list if r > 0) / len(rois_list),
        'Std_Dev': std_dev,
        'Max_ROI': np.max(rois_list),
        'Min_ROI': np.min(rois_list),
        'Range': np.max(rois_list) - np.min(rois_list),
        'Total_Indemnity': total_indemnity,
        'Total_Premium': total_premium
    }

@st.cache_data
def calculate_naive_allocation(selected_grids_tuple, best_strategies_df):
    """
    Calculates the naive allocation dataframe (average of each grid's best strategy).
    """
    if best_strategies_df.empty or not selected_grids_tuple:
        return pd.DataFrame()

    selected_grids = list(selected_grids_tuple)
    grid_strategies_for_table = []
    
    filtered_strategies = best_strategies_df[best_strategies_df['Grid_ID'].isin(selected_grids)].copy()

    for grid in selected_grids:
        strategy_row = filtered_strategies[filtered_strategies['Grid_ID'] == grid]
        if strategy_row.empty:
            continue
            
        strategy_str = strategy_row.iloc[0]['Allocation']
        alloc_dict = {k: 0.0 for k in INTERVAL_ORDER_11}
        for item in strategy_str.split(', '):
            try:
                interval, pct = item.split(': ')
                alloc_dict[interval] = float(pct.strip('%')) / 100.0
            except ValueError:
                continue
        
        alloc_dict['Grid'] = grid
        grid_strategies_for_table.append(alloc_dict)
        
    if not grid_strategies_for_table:
        return pd.DataFrame()
        
    coverage_df = pd.DataFrame(grid_strategies_for_table)
    coverage_df = coverage_df.set_index('Grid')
    
    average_alloc = coverage_df[INTERVAL_ORDER_11].mean(axis=0)
    
    alloc_sum = average_alloc.sum()
    if alloc_sum > 0:
        average_alloc = average_alloc / alloc_sum
    
    avg_row_data = average_alloc.to_dict()
    avg_row_data['Row Sum'] = average_alloc.sum()
    
    final_df = pd.concat([
        coverage_df,
        pd.DataFrame([avg_row_data], index=['AVERAGE'])
    ])
    
    final_df['Row Sum'] = final_df[INTERVAL_ORDER_11].sum(axis=1)
    
    return final_df.drop(columns=[col for col in final_df.columns if col not in INTERVAL_ORDER_11 + ['Row Sum']], errors='ignore')


@st.cache_data(ttl=3600)
def fetch_and_process_all_grid_histories(_session, selected_grids, best_strategies, common_params, grid_acres_tuple):
    """
    Fetches full history for *all* selected grids.
    Uses actual acres from grid_acres to ensure proper precision before rounding.
    grid_acres_tuple: tuple of (grid_id, acres) pairs for cache hashability
    """
    # Convert tuple back to dict
    grid_acres = dict(grid_acres_tuple)

    all_year_data = []

    filtered_strategies = best_strategies[best_strategies['Grid_ID'].isin(selected_grids)].copy()

    naive_alloc_df = calculate_naive_allocation(tuple(selected_grids), best_strategies)
    if 'AVERAGE' not in naive_alloc_df.index:
        return pd.DataFrame()

    avg_interval_weights = naive_alloc_df.loc['AVERAGE', INTERVAL_ORDER_11].to_dict()

    for grid in selected_grids:
        grid_numeric = extract_numeric_grid_id(grid)
        if grid_numeric is None:
            continue

        df = load_all_indices(_session, grid_numeric)
        if df.empty:
            continue

        current_rate_year = get_current_rate_year(_session)
        coverage_level = common_params['coverage_level']

        # Use actual acres from grid_acres to ensure proper precision before rounding
        actual_acres = grid_acres.get(grid, 1.0)

        static_data = {
            'county_base_value': load_county_base_value(_session, grid),
            'premiums': load_premium_rates(_session, grid_numeric, common_params['intended_use'],
                                           coverage_level, current_rate_year),
            'subsidy': load_subsidy(_session, common_params['plan_code'], coverage_level),
            'prod_factor': common_params['productivity_factor'],
            'acres': actual_acres
        }

        for year in df['YEAR'].unique():
            year_data = df[df['YEAR'] == year]
            if len(year_data) < 11:
                continue

            roi, indemnity, premium = calculate_yearly_roi(year_data, avg_interval_weights, static_data, coverage_level)

            if roi is None or pd.isna(roi) or np.isinf(roi):
                continue

            features = get_year_features(year_data)
            if pd.isna(features['avg_hist_z']):
                continue

            index_values = {interval: float(year_data[year_data['INTERVAL_NAME'] == interval]['INDEX_VALUE'].iloc[0])
                            for interval in INTERVAL_ORDER_11 if not year_data[year_data['INTERVAL_NAME'] == interval].empty}

            all_year_data.append({
                'grid': grid,
                'year': year,
                'roi': roi,
                'indemnity': indemnity,
                'premium': premium,
                'index_values': index_values,
                'static_data': static_data,
                'coverage_level': coverage_level,
                'dominant_phase': features['dominant_phase'],
                'avg_hist_z': features['avg_hist_z']
            })

    if not all_year_data:
        return pd.DataFrame()
    all_year_df = pd.DataFrame(all_year_data)

    return all_year_df

# =============================================================================
# === BUDGET CONSTRAINT FUNCTIONS ===
# =============================================================================

def calculate_annual_premium_cost(allocation_weights, selected_grids, grid_acres, session, common_params):
    """
    Calculate total annual premium cost using 2025 rates with validation.
    Uses round_half_up for proper rounding to match PRF official tool.
    Returns: (total_cost, grid_breakdown_dict)
    """
    current_rate_year = get_current_rate_year(session)
    coverage_level = common_params['coverage_level']

    grid_costs = {}
    total_cost = 0.0

    for grid_id in selected_grids:
        grid_numeric = extract_numeric_grid_id(grid_id)

        # Load data with validation
        cbv = load_county_base_value(session, grid_id)
        premiums = load_premium_rates(session, grid_numeric, common_params['intended_use'],
                                      coverage_level, current_rate_year)
        subsidy = load_subsidy(session, common_params['plan_code'], coverage_level)
        if subsidy > 1.0:
            subsidy /= 100.0

        acres = grid_acres.get(grid_id, 0)
        prod_factor = common_params['productivity_factor']

        # Validate inputs
        if cbv is None or cbv <= 0 or np.isnan(cbv):
            continue  # Skip grids with invalid CBV

        if acres <= 0 or np.isnan(acres) or np.isinf(acres):
            continue  # Skip grids with invalid acres

        # Use calculate_protection for proper decimal arithmetic
        dollar_protection_per_acre = calculate_protection(cbv, coverage_level, prod_factor)

        grid_cost = 0.0
        for i, interval in enumerate(INTERVAL_ORDER_11):
            interval_weight = allocation_weights[i]
            if interval_weight > 0.001:
                premium_rate = premiums.get(interval, 0)
                if premium_rate > 0 and not np.isnan(premium_rate):
                    # Round each intermediate dollar value to whole dollars
                    interval_protection = round_half_up(dollar_protection_per_acre * acres * interval_weight, 0)
                    total_premium = round_half_up(interval_protection * premium_rate, 0)
                    premium_subsidy = round_half_up(total_premium * subsidy, 0)
                    interval_premium = total_premium - premium_subsidy

                    # Validate interval premium
                    if not np.isnan(interval_premium) and not np.isinf(interval_premium):
                        grid_cost += interval_premium

        # Validate grid cost before adding
        if not np.isnan(grid_cost) and not np.isinf(grid_cost) and grid_cost >= 0:
            grid_costs[grid_id] = grid_cost
            total_cost += grid_cost
        else:
            grid_costs[grid_id] = 0.0

    # Final validation
    if np.isnan(total_cost) or np.isinf(total_cost) or total_cost < 0:
        total_cost = 0.0

    return total_cost, grid_costs

def apply_budget_constraint(grid_acres, total_cost, budget_limit):
    """
    Scale down acres proportionally if over budget.
    Allocations must always sum to 100%, so we scale acres instead.
    Returns: (scaled_grid_acres_dict, scale_factor)
    """
    if total_cost <= budget_limit:
        return grid_acres, 1.0
    
    scale_factor = budget_limit / total_cost
    
    scaled_acres = {grid: acres * scale_factor for grid, acres in grid_acres.items()}
    
    return scaled_acres, scale_factor

# === OPTIMIZATION HELPER FUNCTIONS (VECTORIZED & CACHED) ===

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_vectorized_data(base_data_df, grid_acres):
    """
    Converts DataFrame into numpy arrays for fast vectorized optimization.
    Uses round_half_up for proper rounding to match PRF official tool.
    Calculates with actual acres BEFORE rounding to preserve precision.
    Returns: (M_indemnity_base, M_premium_base, years_indices, num_grids, acres_vector)
    """
    records = []
    years = []
    acres_list = []

    for _, row in base_data_df.iterrows():
        static = row['static_data']
        cbv = static['county_base_value']
        prems = static['premiums']
        sub = static.get('subsidy', 0.5)
        if sub > 1.0: sub /= 100.0

        cov_level = row['coverage_level']
        prod_factor = static['prod_factor']

        # Get actual acres for this grid - use from static_data if available, else from grid_acres
        actual_acres = static.get('acres', grid_acres.get(row['grid'], 1.0))

        # Use calculate_protection for proper decimal arithmetic (per-acre)
        dollar_prot_per_acre = calculate_protection(cbv, cov_level, prod_factor)
        # Calculate total protection with actual acres, then round
        total_prot = round_half_up(dollar_prot_per_acre * actual_acres, 0)
        trigger = cov_level * 100

        row_indemnity = np.zeros(11)
        row_premium = np.zeros(11)

        for i, interval in enumerate(INTERVAL_ORDER_11):
            idx_val = row['index_values'].get(interval)
            pr_rate = prems.get(interval, 0)

            if idx_val is not None and not pd.isna(idx_val) and pr_rate > 0:
                # Calculate with total protection (includes acres), then round
                # This matches the PRF tool calculation order
                raw_prem = round_half_up(total_prot * pr_rate, 0)
                prem_subsidy = round_half_up(raw_prem * sub, 0)
                prod_prem = raw_prem - prem_subsidy
                row_premium[i] = max(0, prod_prem)

                shortfall = max(0, (trigger - idx_val) / trigger)
                row_indemnity[i] = round_half_up(shortfall * total_prot, 0)

        records.append((row_indemnity, row_premium))
        years.append(row['year'])
        acres_list.append(actual_acres)

    M_ind = np.vstack([r[0] for r in records])
    M_prem = np.vstack([r[1] for r in records])
    years_arr = np.array(years)
    acres_arr = np.array(acres_list).reshape(-1, 1)

    return M_ind, M_prem, years_arr, acres_arr

def calculate_vectorized_roi(weights, M_ind, M_prem, years_arr, acres_arr, optimization_goal, min_allocation, interval_range, full_coverage=False):
    """
    Fast vectorized calculation of portfolio ROI.
    weights: (11,) array
    min_allocation: minimum allocation per active interval
    interval_range: tuple (min_intervals, max_intervals) - RANGE of allowed intervals
    full_coverage: if True, all 11 intervals must be active
    """
    if full_coverage:
        if np.any(weights < min_allocation):
            return 1e10
        num_active = 11
    else:
        active_mask = weights >= min_allocation
        num_active = np.sum(active_mask)
        
        # Check if num_active is within the allowed RANGE
        min_intervals, max_intervals = interval_range
        if num_active < min_intervals or num_active > max_intervals:
            return 1e10
        
        for i in range(11):
            if weights[i] > 0.001:
                if weights[i] < min_allocation or weights[i] > MAX_ALLOCATION:
                    return 1e10
    
    if not full_coverage:
        active_indices = np.where(weights >= min_allocation)[0]
        if len(active_indices) > 1:
            sorted_idx = np.sort(active_indices)
            diffs = np.diff(sorted_idx)
            for d_idx in range(len(diffs)):
                if diffs[d_idx] == 1:
                    interval_a = sorted_idx[d_idx]
                    interval_b = sorted_idx[d_idx + 1]
                    if not ((interval_a == 0 and interval_b == 10) or (interval_a == 10 and interval_b == 0)):
                        return 1e10
            
    if abs(np.sum(weights) - 1.0) > 0.01:
        return 1e10

    # M_ind and M_prem now contain total values (already includes acres from prepare_vectorized_data)
    rec_indemnity_total = M_ind @ weights
    rec_premium_total = M_prem @ weights

    df_temp = pd.DataFrame({'year': years_arr, 'ind': rec_indemnity_total, 'prem': rec_premium_total})
    year_sums = df_temp.groupby('year').sum()
    
    yearly_prems = year_sums['prem'].values
    yearly_inds = year_sums['ind'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        yearly_rois = (yearly_inds - yearly_prems) / yearly_prems
        yearly_rois = np.nan_to_num(yearly_rois, nan=0.0, posinf=0.0, neginf=0.0)

    if optimization_goal == 'Risk-Adjusted Return':
        mean_ret = np.mean(yearly_rois)
        std_dev = np.std(yearly_rois)
        if std_dev < 1e-6: return 1e10 
        sharpe = mean_ret / std_dev
        return -sharpe
        
    elif optimization_goal == 'Cumulative ROI':
        return -np.mean(yearly_rois)

def generate_random_valid_allocation(min_allocation, interval_range, full_coverage=False):
    """
    Generates random valid weights with BUG FIX for randrange error.
    interval_range: tuple (min_intervals, max_intervals)
    """
    if full_coverage:
        weights = np.full(11, min_allocation)
        remaining = 1.0 - (11 * min_allocation)
        
        if remaining < 0:
            weights = np.ones(11) / 11.0
            return weights
        
        attempts = 0
        while remaining > 0.001 and attempts < 100:
            attempts += 1
            idx = random.randint(0, 10)
            max_add = min(remaining, MAX_ALLOCATION - weights[idx])
            
            if max_add > 0.001:
                # BUG FIX: Check if max_add * 100 is at least 1
                max_add_cents = int(max_add * 100)
                if max_add_cents >= 1:
                    add_amount = min(random.randint(1, max_add_cents) / 100.0, max_add)
                else:
                    add_amount = max_add
                    
                weights[idx] += add_amount
                remaining -= add_amount
        
        weights = weights / np.sum(weights)
        weights = np.round(weights / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
        
        sum_diff = 1.0 - np.sum(weights)
        if abs(sum_diff) > 0.001:
            max_idx = np.argmax(weights)
            weights[max_idx] += sum_diff
        
        return weights
    
    else:
        min_intervals, max_intervals = interval_range
        
        # Pick a random target within the RANGE
        target_active = random.randint(min_intervals, max_intervals)
        
        indices = list(range(11))
        random.shuffle(indices)
        
        selected = []
        for idx in indices:
            is_valid = True
            for s in selected:
                if abs(idx - s) == 1:
                    if not ((idx == 0 and s == 10) or (idx == 10 and s == 0)):
                        is_valid = False
                        break
            if is_valid:
                selected.append(idx)
                if len(selected) >= target_active:
                    break
        
        if len(selected) < min_intervals:
            for idx in indices:
                if idx not in selected:
                    is_valid = True
                    for s in selected:
                        if abs(idx - s) == 1:
                            if not ((idx == 0 and s == 10) or (idx == 10 and s == 0)):
                                is_valid = False
                                break
                    if is_valid:
                        selected.append(idx)
                        if len(selected) >= min_intervals:
                            break
        
        weights = np.zeros(11)
        
        for idx in selected:
            weights[idx] = min_allocation
        
        remaining = 1.0 - (len(selected) * min_allocation)
        
        attempts = 0
        while remaining > 0.001 and attempts < 100:
            attempts += 1
            idx = random.choice(selected)
            
            max_add = min(remaining, MAX_ALLOCATION - weights[idx])
            
            if max_add > 0.001:
                # BUG FIX: Check if max_add * 100 is at least 1
                max_add_cents = int(max_add * 100)
                if max_add_cents >= 1:
                    add_amount = min(random.randint(1, max_add_cents) / 100.0, max_add)
                else:
                    add_amount = max_add
                    
                weights[idx] += add_amount
                remaining -= add_amount
        
        weights = weights / np.sum(weights)
        
        weights = np.round(weights / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
        
        sum_diff = 1.0 - np.sum(weights)
        if abs(sum_diff) > 0.001:
            max_idx = np.argmax(weights)
            weights[max_idx] += sum_diff
        
        return weights

def generate_pattern_allocation(pattern_intervals, min_allocation=MIN_ALLOCATION):
    """
    Generate a valid allocation for a specific pattern (A or B).
    pattern_intervals: list of interval indices to allocate across
    Returns: 11-element array with allocations only at pattern_intervals
    """
    n_intervals = len(pattern_intervals)
    weights = np.zeros(11)
    
    # Start with minimum allocation for each interval in the pattern
    for idx in pattern_intervals:
        weights[idx] = min_allocation
    
    remaining = 1.0 - (n_intervals * min_allocation)
    
    # Randomly distribute remaining weight
    attempts = 0
    while remaining > 0.001 and attempts < 100:
        attempts += 1
        idx = random.choice(pattern_intervals)
        
        max_add = min(remaining, MAX_ALLOCATION - weights[idx])
        
        if max_add > 0.001:
            max_add_cents = int(max_add * 100)
            if max_add_cents >= 1:
                add_amount = min(random.randint(1, max_add_cents) / 100.0, max_add)
            else:
                add_amount = max_add
                
            weights[idx] += add_amount
            remaining -= add_amount
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Round to increment
    weights = np.round(weights / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
    
    # Fix any rounding errors
    sum_diff = 1.0 - np.sum(weights)
    if abs(sum_diff) > 0.001:
        # Add difference to the interval with max allocation
        active_indices = [i for i in pattern_intervals if weights[i] > 0]
        if active_indices:
            max_idx = max(active_indices, key=lambda i: weights[i])
            weights[max_idx] += sum_diff
    
    return weights

def generate_pattern_allocation(pattern_intervals, min_allocation=MIN_ALLOCATION):
    """
    Generate a valid allocation for a specific pattern (A or B).
    pattern_intervals: list of interval indices to allocate across
    Returns: 11-element array with allocations only at pattern_intervals
    """
    n_intervals = len(pattern_intervals)
    weights = np.zeros(11)
    
    # Start with minimum allocation for each interval in the pattern
    for idx in pattern_intervals:
        weights[idx] = min_allocation
    
    remaining = 1.0 - (n_intervals * min_allocation)
    
    # Randomly distribute remaining weight
    attempts = 0
    while remaining > 0.001 and attempts < 100:
        attempts += 1
        idx = random.choice(pattern_intervals)
        
        max_add = min(remaining, MAX_ALLOCATION - weights[idx])
        
        if max_add > 0.001:
            max_add_cents = int(max_add * 100)
            if max_add_cents >= 1:
                add_amount = min(random.randint(1, max_add_cents) / 100.0, max_add)
            else:
                add_amount = max_add
                
            weights[idx] += add_amount
            remaining -= add_amount
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Round to increment
    weights = np.round(weights / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
    
    # Fix any rounding errors
    sum_diff = 1.0 - np.sum(weights)
    if abs(sum_diff) > 0.001:
        # Add difference to the interval with max allocation
        active_indices = [i for i in pattern_intervals if weights[i] > 0]
        if active_indices:
            max_idx = max(active_indices, key=lambda i: weights[i])
            weights[max_idx] += sum_diff
    
    return weights

def calculate_staggered_portfolio_returns(M_ind, M_prem, years_arr, acres_arr, grid_list,
                                          grid_assignments, pattern_A_weights, pattern_B_weights):
    """
    Calculate yearly returns for staggered portfolio.
    grid_assignments: list where 0=Pattern A, 1=Pattern B for each grid
    """
    # Calculate indemnity and premium for each record based on grid's pattern
    rec_indemnities = []
    rec_premiums = []
    
    # Assume records are ordered: all years for grid1, then all years for grid2, etc.
    num_records = len(M_ind)
    num_grids = len(grid_list)
    records_per_grid = num_records // num_grids if num_grids > 0 else num_records
    
    for rec_idx in range(num_records):
        grid_idx = min(rec_idx // records_per_grid, num_grids - 1) if records_per_grid > 0 else 0

        pattern = grid_assignments[grid_idx]
        weights = pattern_A_weights if pattern == 0 else pattern_B_weights

        # M_ind and M_prem now contain total values (already includes acres)
        rec_ind = M_ind[rec_idx] @ weights
        rec_prem = M_prem[rec_idx] @ weights

        rec_indemnities.append(rec_ind)
        rec_premiums.append(rec_prem)
    
    # Aggregate by year
    df_temp = pd.DataFrame({
        'year': years_arr,
        'ind': rec_indemnities,
        'prem': rec_premiums
    })
    year_sums = df_temp.groupby('year').sum()
    
    yearly_prems = year_sums['prem'].values
    yearly_inds = year_sums['ind'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        yearly_rois = (yearly_inds - yearly_prems) / yearly_prems
        yearly_rois = np.nan_to_num(yearly_rois, nan=0.0, posinf=0.0, neginf=0.0)
    
    return yearly_rois, yearly_inds, yearly_prems

def evaluate_staggered_portfolio(M_ind, M_prem, years_arr, acres_arr, grid_list, 
                                  grid_assignments, pattern_A_weights, pattern_B_weights, optimization_goal):
    """
    Evaluate a staggered portfolio configuration.
    Returns score (lower is better).
    """
    yearly_rois, _, _ = calculate_staggered_portfolio_returns(
        M_ind, M_prem, years_arr, acres_arr, grid_list,
        grid_assignments, pattern_A_weights, pattern_B_weights
    )
    
    if optimization_goal == 'Risk-Adjusted Return':
        mean_ret = np.mean(yearly_rois)
        std_dev = np.std(yearly_rois)
        if std_dev < 1e-6:
            return 1e10
        sharpe = mean_ret / std_dev
        return -sharpe
    elif optimization_goal == 'Cumulative ROI':
        return -np.mean(yearly_rois)
    
    return 1e10

@st.cache_data(ttl=3600, show_spinner=False)
def run_staggered_optimization_core(M_ind, M_prem, years_arr, acres_arr, grid_list, optimization_goal):
    """
    Optimize staggered portfolio with Pattern A and Pattern B grids.
    Returns: (pattern_A_weights, pattern_B_weights, grid_pattern_assignment, yearly_rois, yearly_inds, yearly_prems)
    """
    num_grids = len(grid_list)
    
    # Initialize best solution
    best_score = 1e10
    best_pattern_A_weights = None
    best_pattern_B_weights = None
    best_grid_assignments = None
    
    iterations = 2000  # Reduced since we have more complexity
    
    # Try different grid assignment strategies
    for iteration in range(iterations):
        # Random assignment: at least 1 grid in each pattern
        if num_grids == 2:
            # With 2 grids, one gets each pattern
            grid_assignments = [random.choice([0, 1]) for _ in range(2)]
            if grid_assignments[0] == grid_assignments[1]:
                grid_assignments = [0, 1]  # Force different patterns
        else:
            # Randomly assign grids, ensuring at least 1 in each pattern
            grid_assignments = [random.randint(0, 1) for _ in range(num_grids)]
            if sum(grid_assignments) == 0:  # All Pattern A
                grid_assignments[random.randint(0, num_grids-1)] = 1
            elif sum(grid_assignments) == num_grids:  # All Pattern B
                grid_assignments[random.randint(0, num_grids-1)] = 0
        
        # Generate random allocations for each pattern
        pattern_A_weights = generate_pattern_allocation(PATTERN_A_INTERVALS)
        pattern_B_weights = generate_pattern_allocation(PATTERN_B_INTERVALS)
        
        # Calculate portfolio performance with this configuration
        score = evaluate_staggered_portfolio(
            M_ind, M_prem, years_arr, acres_arr, grid_list,
            grid_assignments, pattern_A_weights, pattern_B_weights, optimization_goal
        )
        
        if score < best_score:
            best_score = score
            best_pattern_A_weights = pattern_A_weights.copy()
            best_pattern_B_weights = pattern_B_weights.copy()
            best_grid_assignments = grid_assignments.copy()
    
    # Refine the best solution with local search
    current_assignments = best_grid_assignments.copy()
    current_pattern_A = best_pattern_A_weights.copy()
    current_pattern_B = best_pattern_B_weights.copy()
    current_score = best_score
    
    shift_amounts = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    improved = True
    step = 0
    while improved and step < 30:
        improved = False
        step += 1
        
        # Try improving Pattern A allocation
        for i in PATTERN_A_INTERVALS:
            for j in PATTERN_A_INTERVALS:
                if i == j:
                    continue
                for amt in shift_amounts:
                    new_i = current_pattern_A[i] - amt
                    new_j = current_pattern_A[j] + amt
                    
                    if new_i < MIN_ALLOCATION or new_j > MAX_ALLOCATION:
                        continue
                    
                    cand_A = current_pattern_A.copy()
                    cand_A[i] = new_i
                    cand_A[j] = new_j
                    cand_A = cand_A / np.sum(cand_A)
                    
                    score = evaluate_staggered_portfolio(
                        M_ind, M_prem, years_arr, acres_arr, grid_list,
                        current_assignments, cand_A, current_pattern_B, optimization_goal
                    )
                    
                    if score < current_score - 1e-6:
                        current_pattern_A = cand_A
                        current_score = score
                        improved = True
                        break
                if improved: break
            if improved: break
        
        # Try improving Pattern B allocation
        if not improved:
            for i in PATTERN_B_INTERVALS:
                for j in PATTERN_B_INTERVALS:
                    if i == j:
                        continue
                    for amt in shift_amounts:
                        new_i = current_pattern_B[i] - amt
                        new_j = current_pattern_B[j] + amt
                        
                        if new_i < MIN_ALLOCATION or new_j > MAX_ALLOCATION:
                            continue
                        
                        cand_B = current_pattern_B.copy()
                        cand_B[i] = new_i
                        cand_B[j] = new_j
                        cand_B = cand_B / np.sum(cand_B)
                        
                        score = evaluate_staggered_portfolio(
                            M_ind, M_prem, years_arr, acres_arr, grid_list,
                            current_assignments, current_pattern_A, cand_B, optimization_goal
                        )
                        
                        if score < current_score - 1e-6:
                            current_pattern_B = cand_B
                            current_score = score
                            improved = True
                            break
                    if improved: break
                if improved: break
    
    # Round final weights
    final_pattern_A = np.round(current_pattern_A / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
    final_pattern_A = final_pattern_A / np.sum(final_pattern_A)
    
    final_pattern_B = np.round(current_pattern_B / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT
    final_pattern_B = final_pattern_B / np.sum(final_pattern_B)
    
    # Calculate final performance metrics
    yearly_rois, yearly_inds, yearly_prems = calculate_staggered_portfolio_returns(
        M_ind, M_prem, years_arr, acres_arr, grid_list,
        current_assignments, final_pattern_A, final_pattern_B
    )
    
    return final_pattern_A, final_pattern_B, current_assignments, yearly_rois, yearly_inds, yearly_prems

@st.cache_data(ttl=3600, show_spinner=False)
def run_fast_optimization_core(M_ind, M_prem, years_arr, acres_arr, num_grids, naive_weights, optimization_goal, min_allocation, interval_range, full_coverage=False):
    """
    Core optimization logic with BUG FIX.
    interval_range: tuple (min_intervals, max_intervals)
    """
    best_weights = naive_weights
    best_score = calculate_vectorized_roi(naive_weights, M_ind, M_prem, years_arr, acres_arr, optimization_goal, min_allocation, interval_range, full_coverage)
    
    iterations = 3000
    
    for _ in range(iterations):
        cand = generate_random_valid_allocation(min_allocation, interval_range, full_coverage)
        score = calculate_vectorized_roi(cand, M_ind, M_prem, years_arr, acres_arr, optimization_goal, min_allocation, interval_range, full_coverage)
        if score < best_score:
            best_score = score
            best_weights = cand
            
    current_weights = best_weights.copy()
    current_score = best_score
    
    shift_amounts = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    improved = True
    step = 0
    while improved and step < 50:
        improved = False
        step += 1
        
        for i in range(11):
            if current_weights[i] < min_allocation:
                continue
                
            for j in range(11):
                if i == j: continue
                
                if full_coverage:
                    pass
                else:
                    prev, next_idx = j-1, j+1
                    viol = False
                    
                    if prev >= 0 and prev != i and current_weights[prev] >= min_allocation:
                        if not ((j == 0 and prev == 10) or (j == 10 and prev == 0)):
                            viol = True
                    if next_idx < 11 and next_idx != i and current_weights[next_idx] >= min_allocation:
                        if not ((j == 0 and next_idx == 10) or (j == 10 and next_idx == 0)):
                            viol = True
                    if viol: 
                        continue
                
                for amt in shift_amounts:
                    new_i = current_weights[i] - amt
                    new_j = current_weights[j] + amt
                    
                    if full_coverage:
                        if new_i < min_allocation or new_j > MAX_ALLOCATION:
                            continue
                    else:
                        if new_i < min_allocation and new_i > 0.001:
                            continue
                        
                        if new_j > MAX_ALLOCATION:
                            continue
                        
                        if current_weights[j] < 0.001 and new_j < min_allocation:
                            continue
                    
                    cand = current_weights.copy()
                    cand[i] = new_i
                    cand[j] = new_j
                    
                    if not full_coverage:
                        cand = np.where(cand < 0.001, 0, cand)
                    
                    cand_sum = np.sum(cand)
                    if cand_sum > 0:
                        cand = cand / cand_sum
                    else:
                        continue
                    
                    score = calculate_vectorized_roi(cand, M_ind, M_prem, years_arr, acres_arr, optimization_goal, min_allocation, interval_range, full_coverage)
                    if score < current_score - 1e-6:
                        current_weights = cand
                        current_score = score
                        improved = True
                        break
                if improved: break
            if improved: break
            
    final_weights = current_weights

    final_weights = np.round(final_weights / ALLOCATION_INCREMENT) * ALLOCATION_INCREMENT

    final_weights = final_weights / np.sum(final_weights)

    # M_ind and M_prem now contain total values (already includes acres)
    rec_indemnity_total = M_ind @ final_weights
    rec_premium_total = M_prem @ final_weights

    df_res = pd.DataFrame({'year': years_arr, 'ind': rec_indemnity_total, 'prem': rec_premium_total})
    year_sums = df_res.groupby('year').sum()
    
    yearly_prems = year_sums['prem'].values
    yearly_inds = year_sums['ind'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        yearly_rois = (yearly_inds - yearly_prems) / yearly_prems
        yearly_rois = np.nan_to_num(yearly_rois, nan=0.0, posinf=0.0, neginf=0.0)
        
    return final_weights, yearly_rois, yearly_inds, yearly_prems

def run_portfolio_optimization_wrapper(base_data_df, naive_allocation, optimization_goal, selected_grids, grid_acres, interval_range, full_coverage=False):
    """Wrapper to handle data prep and UI formatting, while delegating core calc to cached function"""
    
    M_ind, M_prem, years_arr, acres_arr = prepare_vectorized_data(base_data_df, grid_acres)
    
    num_grids = len(selected_grids)
    
    if full_coverage:
        # Use staggered optimization with Pattern A and Pattern B
        pattern_A_weights, pattern_B_weights, grid_assignments, yearly_rois, yearly_inds, yearly_prems = run_staggered_optimization_core(
            M_ind, M_prem, years_arr, acres_arr, selected_grids, optimization_goal
        )
        
        # Build detailed allocation dataframe showing pattern for each grid
        grid_data_list = []
        for idx, grid_id in enumerate(selected_grids):
            row_dict = {'Grid': grid_id}
            pattern = grid_assignments[idx]
            weights = pattern_A_weights if pattern == 0 else pattern_B_weights
            
            for i, name in enumerate(INTERVAL_ORDER_11):
                row_dict[name] = weights[i]
            row_dict['Row Sum'] = sum(weights)
            row_dict['Pattern'] = 'A (6 intervals)' if pattern == 0 else 'B (5 intervals)'
            grid_data_list.append(row_dict)
        
        # Add average row - showing combined coverage
        avg_row = {'Grid': 'PORTFOLIO AVERAGE'}
        pattern_A_count = sum(1 for p in grid_assignments if p == 0)
        pattern_B_count = len(grid_assignments) - pattern_A_count
        
        # Weighted average of patterns based on grid count
        if pattern_A_count + pattern_B_count > 0:
            for i, name in enumerate(INTERVAL_ORDER_11):
                avg_row[name] = (pattern_A_weights[i] * pattern_A_count + pattern_B_weights[i] * pattern_B_count) / len(grid_assignments)
        else:
            for i, name in enumerate(INTERVAL_ORDER_11):
                avg_row[name] = 0.0
        avg_row['Row Sum'] = sum(avg_row[name] for name in INTERVAL_ORDER_11)
        avg_row['Pattern'] = f'{pattern_A_count} grids Pattern A, {pattern_B_count} grids Pattern B'
        grid_data_list.append(avg_row)
        
        detailed_alloc_df = pd.DataFrame(grid_data_list).set_index('Grid')
        
        # Build change dataframe
        change_data_list = []
        for idx, grid_id in enumerate(selected_grids):
            row_dict = {'Grid': grid_id}
            pattern = grid_assignments[idx]
            opt_weights = pattern_A_weights if pattern == 0 else pattern_B_weights
            
            total_change = 0
            for i, name in enumerate(INTERVAL_ORDER_11):
                change = opt_weights[i] - naive_allocation[name]
                row_dict[name] = change
                total_change += change
            row_dict['Net Change'] = total_change
            change_data_list.append(row_dict)
        
        # Average shift
        avg_change_row = {'Grid': 'AVERAGE SHIFT'}
        for i, name in enumerate(INTERVAL_ORDER_11):
            if pattern_A_count + pattern_B_count > 0:
                avg_weight = (pattern_A_weights[i] * pattern_A_count + pattern_B_weights[i] * pattern_B_count) / len(grid_assignments)
            else:
                avg_weight = 0.0
            avg_change_row[name] = avg_weight - naive_allocation[name]
        avg_change_row['Net Change'] = 0.0
        change_data_list.append(avg_change_row)
        
        detailed_change_df = pd.DataFrame(change_data_list).set_index('Grid')
        
        # Calculate metrics
        perf_df = pd.DataFrame({
            'roi': yearly_rois,
            'indemnity': yearly_inds,
            'premium': yearly_prems
        })
        opt_metrics = calculate_portfolio_metrics(perf_df)
        
        # Return combined weights for budget calculations (use weighted average)
        final_weights = np.zeros(11)
        for i in range(11):
            if pattern_A_count + pattern_B_count > 0:
                final_weights[i] = (pattern_A_weights[i] * pattern_A_count + pattern_B_weights[i] * pattern_B_count) / len(grid_assignments)
        
        return detailed_alloc_df, detailed_change_df, opt_metrics, final_weights
        
    else:
        # Original single-allocation optimization
        naive_weights = naive_allocation[INTERVAL_ORDER_11].values
        min_allocation = MIN_ALLOCATION
        
        final_weights, yearly_rois, yearly_inds, yearly_prems = run_fast_optimization_core(
            M_ind, M_prem, years_arr, acres_arr, num_grids, naive_weights, optimization_goal, min_allocation, interval_range, full_coverage
        )
        
        perf_df = pd.DataFrame({
            'roi': yearly_rois,
            'indemnity': yearly_inds, 
            'premium': yearly_prems   
        })
        opt_metrics = calculate_portfolio_metrics(perf_df)
        
        grid_data_list = []
        for grid_id in selected_grids:
            row_dict = {'Grid': grid_id}
            for i, name in enumerate(INTERVAL_ORDER_11):
                row_dict[name] = final_weights[i]
            row_dict['Row Sum'] = sum(final_weights)
            grid_data_list.append(row_dict)
            
        avg_row = {'Grid': 'OPTIMIZED AVERAGE'}
        for i, name in enumerate(INTERVAL_ORDER_11):
            avg_row[name] = final_weights[i]
        avg_row['Row Sum'] = sum(final_weights)
        grid_data_list.append(avg_row)
        
        detailed_alloc_df = pd.DataFrame(grid_data_list).set_index('Grid')
        
        change_data_list = []
        for grid_id in selected_grids:
            row_dict = {'Grid': grid_id}
            total_change = 0
            for i, name in enumerate(INTERVAL_ORDER_11):
                change = final_weights[i] - naive_allocation[name]
                row_dict[name] = change
                total_change += change
            row_dict['Net Change'] = total_change
            change_data_list.append(row_dict)
            
        avg_change_row = {'Grid': 'AVERAGE SHIFT'}
        for i, name in enumerate(INTERVAL_ORDER_11):
            avg_change_row[name] = final_weights[i] - naive_allocation[name]
        avg_change_row['Net Change'] = 0.0
        change_data_list.append(avg_change_row)
        
        detailed_change_df = pd.DataFrame(change_data_list).set_index('Grid')
        
        return detailed_alloc_df, detailed_change_df, opt_metrics, final_weights

# === GRID ALLOCATION OPTIMIZER (MEAN-VARIANCE) ===
def optimize_grid_allocation(base_data_df, interval_weights, initial_acres_per_grid,
                            annual_budget, session, common_params, selected_grids, risk_aversion=1.0):
    """
    Two-stage optimization with robust error handling:
    Stage 1: Find the maximum total acres that fit within budget
    Stage 2: Optimize distribution of those acres for best risk-adjusted returns

    Args:
        initial_acres_per_grid: Dict of {grid_id: acres} - STARTING point, not hard cap
        annual_budget: Maximum annual premium cost

    Returns:
        (optimized_acres_dict, roi_df) or raises ValueError on failure
    """
    grids = list(base_data_df['grid'].unique())
    num_grids = len(grids)

    # Calculate ROI series for each grid
    grid_roi_series = {}
    for grid_id in grids:
        grid_data = base_data_df[base_data_df['grid'] == grid_id]
        M_ind, M_prem, years, _ = prepare_vectorized_data(grid_data, {grid_id: 1.0})

        rec_ind = M_ind @ interval_weights
        rec_prem = M_prem @ interval_weights

        df_g = pd.DataFrame({'year': years, 'ind': rec_ind, 'prem': rec_prem})
        g_sums = df_g.groupby('year').sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            rois = (g_sums['ind'] - g_sums['prem']) / g_sums['prem']
            rois = np.nan_to_num(rois, nan=0.0)

        grid_roi_series[grid_id] = pd.Series(rois, index=g_sums.index)

    roi_df = pd.DataFrame(grid_roi_series).fillna(0)
    expected_returns = roi_df.mean()
    cov_matrix = roi_df.cov()

    # Get initial acres array (use as reference, not hard cap)
    initial_acres_array = np.array([initial_acres_per_grid.get(grid, 100.0) for grid in grids])

    # Set reasonable upper bounds (10x initial acres or 10,000, whichever is larger)
    max_acres_array = np.maximum(initial_acres_array * 10.0, 10000.0)

    # STAGE 1: Find maximum total acres within budget using binary search
    def calculate_cost_for_allocation(acres_allocation):
        """Helper to calculate total cost for a given allocation"""
        try:
            acres_dict = dict(zip(grids, acres_allocation))
            total_cost, _ = calculate_annual_premium_cost(
                interval_weights, selected_grids, acres_dict,
                session, common_params
            )
            # Validate cost
            if np.isnan(total_cost) or np.isinf(total_cost) or total_cost < 0:
                return 1e10  # Return very high cost if invalid
            return total_cost
        except Exception:
            return 1e10  # Return very high cost on error

    # Start with proportional allocation based on expected returns
    # Give more acres to better-performing grids
    if expected_returns.sum() > 0:
        weights = expected_returns.values / expected_returns.sum()
        weights = np.maximum(weights, 0.05)  # Ensure minimum 5% per grid
        weights = weights / weights.sum()  # Re-normalize
    else:
        weights = np.ones(num_grids) / num_grids

    # Binary search to find the scale factor that gets us closest to budget
    low_scale = 0.1  # Start at 10% of max acres
    high_scale = 100.0  # Allow up to 100x scaling

    # First check if max acres is under budget
    test_alloc = max_acres_array.copy()
    max_cost = calculate_cost_for_allocation(test_alloc)

    if max_cost <= annual_budget:
        # If max acres is within budget, use all max acres!
        optimal_acres = max_acres_array.copy()
    else:
        # Binary search for the right scale factor
        best_scale = 1.0
        best_alloc = initial_acres_array.copy()

        for iteration in range(60):  # 60 iterations for precision
            mid_scale = (low_scale + high_scale) / 2.0

            # Scale initial allocation proportionally by returns-based weights
            test_alloc = initial_acres_array * mid_scale * weights / weights.mean()
            test_alloc = np.minimum(test_alloc, max_acres_array)  # Respect max per grid
            test_alloc = np.maximum(test_alloc, 1.0)  # Ensure at least 1 acre per grid

            test_cost = calculate_cost_for_allocation(test_alloc)

            if test_cost < annual_budget * 0.98:  # Target 98% of budget
                low_scale = mid_scale
                best_scale = mid_scale
                best_alloc = test_alloc.copy()
            elif test_cost > annual_budget * 1.02:  # Over budget
                high_scale = mid_scale
            else:  # Within 2% of budget - good enough!
                best_alloc = test_alloc.copy()
                break

        optimal_acres = best_alloc

    # Validate Stage 1 result
    if np.any(np.isnan(optimal_acres)) or np.any(np.isinf(optimal_acres)) or np.any(optimal_acres < 0):
        # Fallback: equal distribution within budget
        optimal_acres = np.ones(num_grids) * (annual_budget / (num_grids * 100.0))

    # STAGE 2: Fine-tune distribution for better risk-adjusted returns
    def objective_stage2(acres_allocation):
        """Optimize for utility while staying near budget target"""
        try:
            total_acres_used = np.sum(acres_allocation)
            if total_acres_used < 0.01:
                return 1e10

            weights_alloc = acres_allocation / total_acres_used
            port_ret = np.sum(expected_returns.values * weights_alloc)
            port_var = np.dot(weights_alloc.T, np.dot(cov_matrix.values, weights_alloc))
            utility = port_ret - (risk_aversion * port_var)

            if np.isnan(utility) or np.isinf(utility):
                return 1e10

            return -utility
        except Exception:
            return 1e10

    def budget_constraint_stage2(acres_allocation):
        """Must stay within budget"""
        try:
            total_cost = calculate_cost_for_allocation(acres_allocation)
            return annual_budget - total_cost
        except Exception:
            return -1e10  # Violated constraint

    def budget_target_constraint(acres_allocation):
        """Encourage using at least 90% of budget (relaxed from 95%)"""
        try:
            total_cost = calculate_cost_for_allocation(acres_allocation)
            return total_cost - (annual_budget * 0.90)
        except Exception:
            return -1e10  # Violated constraint

    bounds = tuple((1.0, float(max_acres_array[i])) for i in range(num_grids))

    constraints = [
        {'type': 'ineq', 'fun': budget_constraint_stage2},  # Cost <= budget
        {'type': 'ineq', 'fun': budget_target_constraint}   # Cost >= 90% budget
    ]

    # Try Stage 2 optimization with error handling
    try:
        result = minimize(
            objective_stage2,
            optimal_acres,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-6}
        )

        if result.success and result.x is not None:
            # Validate result
            if not (np.any(np.isnan(result.x)) or np.any(np.isinf(result.x)) or np.any(result.x < 0)):
                final_acres = result.x
                final_cost = calculate_cost_for_allocation(final_acres)

                # If cost is valid and within budget, use Stage 2 result
                if not np.isnan(final_cost) and 0 < final_cost <= annual_budget:
                    optimal_acres = final_acres
                # else: fall back to Stage 1
        # else: fall back to Stage 1
    except Exception as e:
        # Stage 2 failed, use Stage 1 result
        pass

    # Final validation before returning
    if np.any(np.isnan(optimal_acres)) or np.any(np.isinf(optimal_acres)) or np.any(optimal_acres < 0):
        raise ValueError("Optimization produced invalid acres allocation. Check grid data (CBV, premium rates).")

    return dict(zip(grids, optimal_acres)), roi_df

# =============================================================================
# === VISUALIZATION HELPER FUNCTIONS ===
# =============================================================================

def create_naive_allocation_heatmap(coverage_df):
    """
    Creates a static matplotlib/seaborn heatmap for the naive allocation table.
    Returns a matplotlib figure.
    """
    plot_data = coverage_df[INTERVAL_ORDER_11].copy()
    
    # Calculate figure height based on number of rows (minimum 4, add 0.6 per row)
    fig_height = max(4.5, len(plot_data) * 0.7)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=100)
    
    sns.heatmap(
        plot_data,
        annot=True,
        fmt='.0%',
        cmap='Greens',
        cbar=False,
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=0.5,
        ax=ax,
        annot_kws={"fontsize": 10, "weight": "bold"}
    )
    
    ax.set_xlabel('Interval', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Grid ID', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Naive Equal-Weight Portfolio Allocation', fontsize=13, fontweight='bold', pad=20)
    
    # Adjust tick labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    if 'AVERAGE' in plot_data.index:
        avg_idx = list(plot_data.index).index('AVERAGE')
        ax.add_patch(plt.Rectangle((0, avg_idx), len(INTERVAL_ORDER_11), 1, 
                                   fill=False, edgecolor='black', lw=3))
    
    plt.tight_layout(pad=2.0)
    return fig

def create_optimized_allocation_table(detailed_alloc_df, grid_acres=None, grid_costs=None, budget_enabled=False):
    """
    Creates a clean, clear table for optimized allocation.
    Returns a matplotlib figure.
    """
    display_df = detailed_alloc_df.copy()

    cols_to_show = INTERVAL_ORDER_11 + ['Row Sum']
    # Always show Acres and Premium columns if they exist in the dataframe for full transparency
    if 'Acres' in display_df.columns:
        cols_to_show.append('Acres')
    if 'Annual Premium ($)' in display_df.columns:
        cols_to_show.append('Annual Premium ($)')

    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    plot_data = display_df[cols_to_show]

    # Create display labels for column headers (abbreviate long names)
    col_display_labels = []
    for col in cols_to_show:
        if col == 'Annual Premium ($)':
            col_display_labels.append('Annual\nPremium ($)')
        else:
            col_display_labels.append(col)

    n_cols = len(cols_to_show)
    n_rows = len(plot_data)
    fig_width = max(17, n_cols * 1.3)
    fig_height = max(4.5, n_rows * 0.75 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.axis('tight')
    ax.axis('off')

    cell_text = []
    cell_colors = []

    for idx, row in plot_data.iterrows():
        row_text = []
        row_colors = []

        for col in cols_to_show:
            val = row[col]

            if col in INTERVAL_ORDER_11 or col == 'Row Sum':
                text = f'{val:.0%}'
                if col in INTERVAL_ORDER_11 and val > 0.001:
                    row_colors.append('#C8E6C9')
                elif col == 'Row Sum':
                    row_colors.append('#E0E0E0')
                else:
                    row_colors.append('white')
            elif col == 'Pattern':
                text = str(val)
                row_colors.append('#E1BEE7')  # Purple tint
            elif col == 'Acres':
                text = f'{val:,.0f}'
                row_colors.append('#BBDEFB')
            elif col == 'Annual Premium ($)':
                text = f'${val:,.0f}'
                row_colors.append('#FFF59D')
            else:
                text = str(val)
                row_colors.append('white')

            row_text.append(text)

        cell_text.append(row_text)

        if 'OPTIMIZED AVERAGE' in idx or 'PORTFOLIO AVERAGE' in idx:
            cell_colors.append(['#BDBDBD'] * len(cols_to_show))
        else:
            cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=plot_data.index,
        colLabels=col_display_labels,
        cellColours=cell_colors,
        cellLoc='center',
        rowLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)  # Increased row height
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', fontsize=11, color='white')
            cell.set_facecolor('#424242')
            cell.set_height(0.1)
        elif j == -1:
            cell.set_text_props(weight='bold', fontsize=10)
            cell.set_facecolor('#E0E0E0')
        
        if i > 0 and ('OPTIMIZED AVERAGE' in plot_data.index[i-1] or 'PORTFOLIO AVERAGE' in plot_data.index[i-1]):
            cell.set_text_props(weight='bold', fontsize=11)
        
        cell.set_edgecolor('#757575')
        cell.set_linewidth(0.8)
    
    plt.tight_layout(pad=2.0)
    return fig

def create_allocation_changes_table(optimized_alloc_df, naive_alloc_df, selected_grids, grid_acres_original=None, grid_acres_final=None, budget_enabled=False):
    """
    Creates a table showing allocation changes for each grid.
    Compares each grid's INDIVIDUAL naive allocation to that SAME grid's INDIVIDUAL optimized allocation.
    Grids as rows, intervals as columns, showing percentage changes.
    Returns a matplotlib figure.
    """
    change_rows = []

    for grid in selected_grids:
        row_dict = {'Grid': grid}

        # Get this grid's naive allocation
        if grid in naive_alloc_df.index:
            grid_naive = naive_alloc_df.loc[grid, INTERVAL_ORDER_11]
        else:
            grid_naive = naive_alloc_df.loc['AVERAGE', INTERVAL_ORDER_11]

        # Get this grid's optimized allocation (NOT the average!)
        if grid in optimized_alloc_df.index:
            grid_optimized = optimized_alloc_df.loc[grid, INTERVAL_ORDER_11]
        else:
            # Fallback to optimized average if grid not found
            if 'OPTIMIZED AVERAGE' in optimized_alloc_df.index:
                grid_optimized = optimized_alloc_df.loc['OPTIMIZED AVERAGE', INTERVAL_ORDER_11]
            elif 'PORTFOLIO AVERAGE' in optimized_alloc_df.index:
                grid_optimized = optimized_alloc_df.loc['PORTFOLIO AVERAGE', INTERVAL_ORDER_11]
            else:
                # Last resort: use naive as fallback
                grid_optimized = grid_naive

        net_change = 0
        for interval in INTERVAL_ORDER_11:
            change = grid_optimized[interval] - grid_naive[interval]
            row_dict[interval] = change
            net_change += change

        row_dict['Net Change'] = net_change
        change_rows.append(row_dict)

    # Add average shift row (comparing portfolio averages)
    avg_naive = naive_alloc_df.loc['AVERAGE', INTERVAL_ORDER_11]

    # Handle different row names for average based on optimization mode
    if 'OPTIMIZED AVERAGE' in optimized_alloc_df.index:
        avg_optimized = optimized_alloc_df.loc['OPTIMIZED AVERAGE', INTERVAL_ORDER_11]
    elif 'PORTFOLIO AVERAGE' in optimized_alloc_df.index:
        avg_optimized = optimized_alloc_df.loc['PORTFOLIO AVERAGE', INTERVAL_ORDER_11]
    else:
        # Fallback: calculate average from grid data
        grid_only_df = optimized_alloc_df[optimized_alloc_df.index.isin(selected_grids)]
        avg_optimized = grid_only_df[INTERVAL_ORDER_11].mean()

    avg_row = {'Grid': 'AVERAGE SHIFT'}
    net_change = 0
    for interval in INTERVAL_ORDER_11:
        change = avg_optimized[interval] - avg_naive[interval]
        avg_row[interval] = change
        net_change += change
    avg_row['Net Change'] = net_change
    change_rows.append(avg_row)
    
    change_df = pd.DataFrame(change_rows).set_index('Grid')
    
    plot_data = change_df[INTERVAL_ORDER_11 + ['Net Change']].copy()
    
    cols_to_show = INTERVAL_ORDER_11 + ['Net Change']
    if budget_enabled and grid_acres_original is not None and grid_acres_final is not None:
        acres_changes = []
        for idx in plot_data.index:
            if idx == 'AVERAGE SHIFT':
                total_original = sum(grid_acres_original.values())
                total_final = sum(grid_acres_final.values())
                acres_changes.append(total_final - total_original)
            else:
                original_acres = grid_acres_original.get(idx, 0)
                final_acres = grid_acres_final.get(idx, 0)
                acres_changes.append(final_acres - original_acres)
        
        plot_data['Acres Change'] = acres_changes
        cols_to_show.append('Acres Change')

    n_cols = len(cols_to_show)
    n_rows = len(plot_data)
    fig_width = max(17, n_cols * 1.3)
    fig_height = max(4.5, n_rows * 0.75 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.axis('off')
    
    cell_text = []
    cell_colors = []
    
    for idx, row in plot_data.iterrows():
        row_text = []
        row_colors = []
        
        for col in cols_to_show:
            val = row[col]
            
            if col in INTERVAL_ORDER_11 or col == 'Net Change':
                text = f'{val:+.0%}' if abs(val) > 0.001 else '0%'
                
                if col in INTERVAL_ORDER_11:
                    if val > 0.001:
                        row_colors.append('#C8E6C9')
                    elif val < -0.001:
                        row_colors.append('#FFCDD2')
                    else:
                        row_colors.append('#F5F5F5')
                else:
                    row_colors.append('#E0E0E0')
                    
            elif col == 'Acres Change':
                if abs(val) < 0.5:
                    text = '0'
                    row_colors.append('#F5F5F5')
                else:
                    text = f'{val:+,.0f}'
                    if val > 0:
                        row_colors.append('#C8E6C9')
                    else:
                        row_colors.append('#FFCDD2')
            else:
                text = str(val)
                row_colors.append('white')
            
            row_text.append(text)
        
        cell_text.append(row_text)
        
        if idx == 'AVERAGE SHIFT':
            cell_colors.append(['#BDBDBD'] * len(cols_to_show))
        else:
            cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=plot_data.index,
        colLabels=cols_to_show,
        cellColours=cell_colors,
        cellLoc='center',
        rowLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)  # Increased row height
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', fontsize=11, color='white')
            cell.set_facecolor('#424242')
            cell.set_height(0.1)
        elif j == -1:
            cell.set_text_props(weight='bold', fontsize=10)
            cell.set_facecolor('#E0E0E0')
        
        if i > 0 and plot_data.index[i-1] == 'AVERAGE SHIFT':
            cell.set_text_props(weight='bold', fontsize=11)

        cell.set_edgecolor('#757575')
        cell.set_linewidth(0.8)

    # Remove any potential overlapping text elements
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    fig.texts.clear()

    plt.tight_layout(pad=2.0)
    return fig

# =============================================================================
# === LOCATION ANALYSIS TAB (Tab 1) ===
# =============================================================================
def render_location_tab(session, all_grid_ids, common_params):
    """Location and Timing Analysis: Discover which grid + allocation combinations perform best"""
    st.subheader("Location and Timing Analysis")
    st.markdown("Discover the best performing grid and allocation combinations based on your market view.")
    
    st.markdown("### 📍 Grid Selection")

    # King Ranch Grids Preset Button
    king_ranch_grids = [
        "9128 (Kleberg - TX)",
        "9129 (Kleberg - TX)",
        "9130 (Kleberg - TX)",
        "9131 (Kleberg - TX)",
        "8828 (Kleberg - TX)",
        "8829 (Kleberg - TX)",
        "8830 (Kleberg - TX)",
        "8831 (Kleberg - TX)",
        "8528 (Brooks - TX)",
        "8228 (Brooks - TX)",
        "8229 (Brooks - TX)",
        "8529 (Brooks - TX)",
        "8829 (Kenedy - TX)",
        "7929 (Kenedy - TX)",
        "8230 (Kenedy - TX)",
        "7930 (Kenedy - TX)",
        "8231 (Kenedy - TX)",
        "7931 (Kenedy - TX)"
    ]

    # Filter to only include grids that exist in the available list
    available_king_ranch_grids = [g for g in king_ranch_grids if g in all_grid_ids]

    if st.button("King Ranch Grids Preset", key="king_ranch_preset_btn"):
        if available_king_ranch_grids:
            st.session_state['loc_grid_mode'] = 'Multiple Grids'
            st.session_state['loc_multi_grid'] = available_king_ranch_grids

            # Set preset values using intermediate keys (not widget-bound keys)
            st.session_state['preset_coverage'] = 0.75  # 75% Coverage Level
            st.session_state['preset_prod_factor'] = 1.35  # 135% Productivity Factor (decimal)

            st.rerun()
        else:
            st.warning("No King Ranch grids found in the available grid list.")

    st.caption("💡 **King Ranch Preset:** Automatically sets grids, **Coverage Level: 75%**, and **Productivity Factor: 135%**.")

    grid_mode = st.radio(
        "Analyze:",
        ['All Grids', 'Single Grid', 'Multiple Grids'],
        horizontal=True,
        key='loc_grid_mode'
    )
    
    with st.form(key='analysis_configuration_form'):
        
        selected_grids = []
        
        if grid_mode == 'All Grids':
            st.info(f"Will analyze all {len(all_grid_ids)} grids")
            selected_grids = all_grid_ids
            
        elif grid_mode == 'Single Grid':
            selected_grid = st.selectbox("Select Grid", all_grid_ids, index=0, key='loc_single_grid')
            if selected_grid:
                selected_grids = [selected_grid]
                
        else:
            # Only set default if there's no existing selection in session state
            if 'loc_multi_grid' not in st.session_state:
                selected_grids = st.multiselect(
                    "Select Grids",
                    all_grid_ids,
                    default=all_grid_ids[:2] if len(all_grid_ids) >= 2 else all_grid_ids,
                    key='loc_multi_grid'
                )
            else:
                selected_grids = st.multiselect(
                    "Select Grids",
                    all_grid_ids,
                    key='loc_multi_grid'
                )
        st.divider()
        
        st.markdown("### 🎯 Your Market View")
        col1, col2, col3 = st.columns(3)
        
        regime = col1.selectbox(
            "1. Expected Regime",
            ['La Nina', 'El Nino', 'Neutral'],
            key='loc_regime'
        )
        
        hist_context = col2.selectbox(
            "2. In the context of the past 75+ years, I think the period we are in is:",
            list(HISTORICAL_CONTEXT_MAP.keys()),
            index=1,
            key='loc_hist'
        )
        
        trend = col3.selectbox(
            "3. I expect conditions throughout the year to:",
            list(TREND_MAP.keys()),
            index=1,
            key='loc_trend'
        )
        
        st.divider()
        
        st.markdown("### 🛠️ Analysis Constraints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            interval_range = st.slider(
                "Number of Intervals to Combine",
                min_value=2,
                max_value=6,
                value=(2, 6),
                help="Filter strategies by the number of 2-month intervals used (e.g., 2-4 intervals)."
            )
        
        with col2:
            selected_sort_metric_display = st.selectbox(
                "Rank Combinations By",
                options=list(SORT_METRIC_DB_MAP.keys()),
                index=0,
                key='loc_sort_metric',
                help="Choose the metric to sort the final results."
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            enable_year_cutoff = st.checkbox(
                "Exclude older analog years",
                value=False,
                key='loc_enable_year_cutoff',
                help="Filter out analog years before a specified cutoff year"
            )

        with col2:
            if enable_year_cutoff:
                cutoff_year = st.number_input(
                    "Cutoff Year (exclude years before this)",
                    min_value=1948,
                    max_value=2024,
                    value=1990,
                    step=1,
                    key='loc_cutoff_year',
                    help="Analog years before this year will be excluded from analysis"
                )
            else:
                cutoff_year = None

        st.divider()

        submitted = st.form_submit_button("🚀 Discover Best Combinations", type="primary")
    
    if 'tab2_run' not in st.session_state:
        st.session_state.tab2_run = False

    if submitted:
        st.session_state.tab2_run = True

        # Clear cached results if key parameters have changed
        if 'tab2_results' in st.session_state and st.session_state.tab2_results:
            cached_params = st.session_state.tab2_results
            params_changed = False
            change_reasons = []

            if cached_params.get('interval_range') != interval_range:
                params_changed = True
                change_reasons.append(f"Interval range: {cached_params.get('interval_range')} → {interval_range}")

            if cached_params.get('regime') != regime:
                params_changed = True
                change_reasons.append(f"Regime: {cached_params.get('regime')} → {regime}")

            if cached_params.get('hist_context') != hist_context:
                params_changed = True
                change_reasons.append(f"Historical context changed")

            if cached_params.get('trend') != trend:
                params_changed = True
                change_reasons.append(f"Trend changed")

            cached_cutoff = cached_params.get('cutoff_year')
            current_cutoff = cutoff_year if enable_year_cutoff else None
            if cached_cutoff != current_cutoff:
                params_changed = True
                if cached_cutoff is None and current_cutoff is not None:
                    change_reasons.append(f"Year cutoff enabled: ≥ {current_cutoff}")
                elif cached_cutoff is not None and current_cutoff is None:
                    change_reasons.append(f"Year cutoff disabled")
                else:
                    change_reasons.append(f"Year cutoff: {cached_cutoff} → {current_cutoff}")

            if params_changed:
                st.session_state.tab2_results = None
                st.info(f"🔄 Parameters changed. Clearing cached results:\n" + "\n".join(f"• {r}" for r in change_reasons))

        if not selected_grids:
            st.error("⚠️ Please select at least one grid to analyze")
            return
        sort_metric = SORT_METRIC_DB_MAP[st.session_state.loc_sort_metric]
        
        st.info(f"🔍 Discovering best grid + allocation combinations for {len(selected_grids)} grid(s)...")

        try:
            current_rate_year = get_current_rate_year(session)
            coverage_level = common_params['coverage_level']

            all_combinations = []
            all_year_details = []

            min_intervals, max_intervals = interval_range

            # Progress bar for grid analysis
            progress_bar = st.progress(0, text="Initializing analysis...")
            total_grids = len(selected_grids)

            for grid_idx, grid_id_display in enumerate(selected_grids):
                grid_id_numeric = extract_numeric_grid_id(grid_id_display)

                # Update progress
                progress_pct = (grid_idx + 1) / total_grids
                progress_bar.progress(progress_pct, text=f"Analyzing grid {grid_idx + 1}/{total_grids}: {grid_id_display}")

                # Use cached filter function
                matching_years_info = filter_years_by_market_view_cached(
                    session, grid_id_numeric, regime, hist_context, trend
                )

                # Apply year cutoff filter if enabled
                if enable_year_cutoff and cutoff_year is not None:
                    matching_years_info = [y for y in matching_years_info if y['year'] >= cutoff_year]

                if not matching_years_info:
                    continue

                matching_years = [y['year'] for y in matching_years_info]

                # Load static data (these are already cached)
                static_data = {
                    'county_base_value': load_county_base_value(session, grid_id_display),
                    'premiums': load_premium_rates(session, grid_id_numeric, common_params['intended_use'],
                                                   coverage_level, current_rate_year),
                    'subsidy': load_subsidy(session, common_params['plan_code'], coverage_level),
                    'prod_factor': common_params['productivity_factor'],
                    'acres': common_params['total_insured_acres']
                }

                # Load index data for matching years
                df = load_all_indices(session, grid_id_numeric)

                # Convert data to hashable formats for caching
                matching_years_info_tuple = tuple(
                    (y['year'], y['dominant_phase'], y['phase_intervals'],
                     y['year_avg_hist_z'], y['year_start_11p'], y['year_end_5p'],
                     y['trajectory_delta'])
                    for y in matching_years_info
                )

                # Extract index data for matching years as tuple of tuples
                index_data_list = []
                for year in matching_years:
                    year_data = df[df['YEAR'] == year]
                    for _, row in year_data.iterrows():
                        if not pd.isna(row['INDEX_VALUE']):
                            index_data_list.append((year, row['INTERVAL_NAME'], row['INDEX_VALUE']))
                index_data_tuple = tuple(index_data_list)

                # Convert static_data to hashable tuple (handle premiums dict specially)
                premiums_tuple = tuple(sorted(static_data['premiums'].items()))
                static_data_tuple = (
                    ('county_base_value', static_data['county_base_value']),
                    ('premiums', premiums_tuple),
                    ('subsidy', static_data['subsidy']),
                    ('prod_factor', static_data['prod_factor']),
                    ('acres', static_data['acres'])
                )

                # Use cached grid analysis function
                grid_combinations, grid_year_details = analyze_single_grid_cached(
                    grid_id_display,
                    grid_id_numeric,
                    matching_years_info_tuple,
                    index_data_tuple,
                    static_data_tuple,
                    coverage_level,
                    min_intervals,
                    max_intervals
                )

                if grid_combinations:
                    all_combinations.extend(grid_combinations)
                    all_year_details.extend(grid_year_details)
                elif matching_years_info:
                    # If grid has matching years but no valid strategy, add placeholder
                    all_combinations.append({
                        'Grid_ID': grid_id_display,
                        'Allocation': f'No valid {min_intervals}-{max_intervals} interval strategy',
                        'Num_Intervals': 0,
                        'Occurrences': len(matching_years_info),
                        'Average_ROI': 0.0,
                        'Cumulative_ROI': 0.0,
                        'Median_ROI': 0.0,
                        'Risk_Adjusted_Return': 0.0,
                        'Win_Rate': 0.0,
                        'Std_Dev': 0.0,
                        'Max_ROI': 0.0,
                        'Min_ROI': 0.0,
                        'Best_Worst_Range': 0.0
                    })

            # Clear progress bar
            progress_bar.empty()

            if not all_combinations:
                st.session_state.tab2_results = None
            else:
                results_df = pd.DataFrame(all_combinations).sort_values(sort_metric, ascending=False)
                details_df = pd.DataFrame(all_year_details)

                st.session_state.tab2_results = {
                    'df': results_df,
                    'details': details_df,
                    'regime': regime,
                    'hist_context': hist_context,
                    'trend': trend,
                    'coverage_level': coverage_level,
                    'num_grids': len(selected_grids),
                    'grid_summary_with_selection': None,
                    'sort_metric': sort_metric,
                    'interval_range': interval_range,
                    'cutoff_year': cutoff_year if enable_year_cutoff else None
                }

            if st.session_state.tab2_results is None:
                st.warning("No valid strategies found for any of the selected grids. Try different conditions.")
        except Exception as e:
            st.error(f"❌ An error occurred during analysis")
            st.error(f"**Error details:** {str(e)}")
            import traceback
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
            st.session_state.tab2_results = None
    
    if 'tab2_results' in st.session_state and st.session_state.tab2_results:
        try:
            results = st.session_state.tab2_results
            results_df = results['df']
            sort_metric = results['sort_metric']
            
            st.success("✅ Analysis Complete")
            
            col1, col2 = st.columns(2)
            with col1:
                csv_combinations = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Combinations (CSV)",
                    data=csv_combinations,
                    file_name=f"prf_combinations_{results['regime']}_{results['hist_context'].replace(' ', '_')}_{results['trend'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="download_combinations_csv"
                )
            with col2:
                if 'details' in results and not results['details'].empty:
                    csv_details = results['details'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Year-by-Year Details (CSV)",
                        data=csv_details,
                        file_name=f"prf_year_details_{results['regime']}_{results['hist_context'].replace(' ', '_')}_{results['trend'].replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_details_csv"
                    )
            
            st.divider()
            
            st.markdown("### 🎯 Grid Suitability for Portfolio")
            st.caption("Grids ranked by data quality and performance for this market view. Check the 'Select' box to send grids to the Portfolio Backtest tab.")
            
            best_by_grid_for_suitability = results_df.loc[results_df.groupby('Grid_ID')[sort_metric].idxmax()].copy()
            best_by_grid_for_suitability['Quality_Score'] = best_by_grid_for_suitability.apply(calculate_grid_quality_score, axis=1)
            best_by_grid_for_suitability = best_by_grid_for_suitability.sort_values('Quality_Score', ascending=False)
            
            def get_suitability(score):
                if score >= 7.5: return "Excellent"
                elif score >= 6.0: return "Good"
                elif score >= 4.5: return "Fair"
                else: return "Limited Data"
                
            best_by_grid_for_suitability['Portfolio_Suitability'] = best_by_grid_for_suitability['Quality_Score'].apply(get_suitability)
            
            if st.session_state.tab2_results['grid_summary_with_selection'] is not None:
                old_selections = st.session_state.tab2_results['grid_summary_with_selection'][['Grid_ID', 'Select']]
                best_by_grid_for_suitability = best_by_grid_for_suitability.merge(old_selections, on='Grid_ID', how='left').fillna(False)
            else:
                best_by_grid_for_suitability.insert(0, 'Select', False)
            
            percent_cols = ['Average_ROI', 'Cumulative_ROI', 'Median_ROI', 'Win_Rate',
                            'Std_Dev', 'Max_ROI', 'Min_ROI', 'Best_Worst_Range']
            for col in percent_cols:
                if col in best_by_grid_for_suitability.columns:
                     best_by_grid_for_suitability[col] = best_by_grid_for_suitability[col] * 100.0
            
            with st.form(key="grid_selection_form"):
                edited_df = st.data_editor(
                    best_by_grid_for_suitability,
                    use_container_width=True,
                    column_config={
                        'Select': st.column_config.CheckboxColumn("Select", width="small"),
                        'Grid_ID': st.column_config.TextColumn('Grid', width='medium', disabled=True),
                        'Allocation': st.column_config.TextColumn('Allocation', width='large', disabled=True),
                        'Num_Intervals': st.column_config.NumberColumn('# Intervals', width='small', disabled=True, format="%.0f"),
                        'Average_ROI': st.column_config.NumberColumn('Avg ROI', width='small', disabled=True, format="%.1f%%"),
                        'Occurrences': st.column_config.NumberColumn('Analog Years', width='small', disabled=True, format="%.0f"),
                        'Best_Worst_Range': st.column_config.NumberColumn('Range', width='small', disabled=True, format="%.1f%%"),
                        'Quality_Score': st.column_config.NumberColumn('Quality Score', width='small', disabled=True, format="%.1f"),
                        'Portfolio_Suitability': st.column_config.TextColumn('Rating', width='small', disabled=True),
                        'Cumulative_ROI': st.column_config.NumberColumn('Portfolio Return', width='small', disabled=True, format="%.1f%%"),
                        'Median_ROI': st.column_config.NumberColumn('Median ROI', width='small', disabled=True, format="%.1f%%"),
                        'Risk_Adjusted_Return': st.column_config.NumberColumn('Risk Adj. Ratio', width='small', disabled=True, format="%.2f"),
                        'Win_Rate': st.column_config.NumberColumn('Win %', width='small', disabled=True, format="%.1f%%"),
                        'Max_ROI': st.column_config.NumberColumn('Max ROI', width='small', disabled=True, format="%.1f%%"),
                        'Min_ROI': st.column_config.NumberColumn('Min ROI', width='small', disabled=True, format="%.1f%%"),
                        'Std_Dev': st.column_config.NumberColumn('Std Dev', width='small', disabled=True, format="%.1f%%"),
                    },
                    column_order=[
                        "Quality_Score", "Portfolio_Suitability", "Select", "Grid_ID", "Allocation", "Num_Intervals", "Occurrences",
                        "Cumulative_ROI", "Median_ROI", "Risk_Adjusted_Return", "Win_Rate",
                        "Max_ROI", "Min_ROI", "Average_ROI", "Best_Worst_Range"
                    ],
                    disabled=best_by_grid_for_suitability.columns.drop('Select')
                )
                
                if st.form_submit_button("Confirm Selections for Backtest"):
                    edited_df_to_save = edited_df.copy()
                    for col in percent_cols:
                          if col in edited_df_to_save.columns:
                            edited_df_to_save[col] = edited_df_to_save[col] / 100.0
                    st.session_state.tab2_results['grid_summary_with_selection'] = edited_df_to_save
                    st.success("Selections confirmed! You can now move to the 'Portfolio Backtest' tab.")
            st.caption("💡 **Quality Score** combines Portfolio Return (40%) and Analog Year history (60%). ROI reaches full points at 500%+. Analog year scoring sharply penalizes years $<5$ and grants full points at $\ge 10$ years.")
            
            st.divider()
            
            st.markdown(f"### 🏆 Top Grid + Allocation Combinations (Ranked by {SORT_METRIC_DISPLAY_MAP[sort_metric]})")
            cutoff_text = f" | **Years ≥ {results['cutoff_year']}**" if results.get('cutoff_year') else ""
            st.caption(f"**Market View:** {results['regime']} | {results['hist_context']} | {results['trend']}{cutoff_text}")
            st.caption(f"**Coverage:** {results['coverage_level']:.0%} | **Grids Analyzed:** {results['num_grids']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Combinations Found", f"{len(results_df)}")
            metric_format = "{:.1%}" if sort_metric in ['Average_ROI', 'Cumulative_ROI', 'Median_ROI', 'Win_Rate'] else "{:.2f}"
            col2.metric(f"Best {SORT_METRIC_DISPLAY_MAP[sort_metric]}", metric_format.format(results_df[sort_metric].max()))
            col3.metric("Top Grid", f"{results_df.iloc[0]['Grid_ID']}")
            
            best_by_grid = results_df.loc[results_df.groupby('Grid_ID')[sort_metric].idxmax()].copy()
            best_by_grid = best_by_grid.sort_values(sort_metric, ascending=False)
            display_df = best_by_grid
            
            download_suffix = f"best_by_grid_{sort_metric}"
            
            all_grids_in_results = sorted(display_df['Grid_ID'].unique())
            with st.expander("🔍 Filter by Grid ID (Optional)"):
                selected_grids_filter = st.multiselect(
                    "Select specific grids to view:",
                    options=all_grids_in_results,
                    default=all_grids_in_results,
                    key='loc_grid_filter'
                )
                if selected_grids_filter:
                    display_df = display_df[display_df['Grid_ID'].isin(selected_grids_filter)]
            
            st.dataframe(
                display_df.style.format({
                    'Num_Intervals': '{:.0f}',
                    'Occurrences': '{:.0f}',
                    'Average_ROI': '{:.1%}',
                    'Cumulative_ROI': '{:.1%}',
                    'Median_ROI': '{:.1%}',
                    'Risk_Adjusted_Return': '{:.2f}',
                    'Win_Rate': '{:.1%}',
                    'Std_Dev': '{:.1%}',
                    'Max_ROI': '{:.1%}',
                    'Min_ROI': '{:.1%}',
                    'Best_Worst_Range': '{:.1%}',
                    'Quality_Score': '{:.1f}'
                }),
                use_container_width=True,
                column_config={
                    'Grid_ID': st.column_config.TextColumn('Grid', width='medium'),
                    'Allocation': st.column_config.TextColumn('Allocation', width='large'),
                    'Num_Intervals': st.column_config.NumberColumn('# Int', width='small'),
                    'Occurrences': st.column_config.NumberColumn('Years', width='small'),
                    'Cumulative_ROI': st.column_config.NumberColumn('Portfolio Return', width='small'),
                    'Median_ROI': st.column_config.NumberColumn('Med ROI', width='small'),
                    'Risk_Adjusted_Return': st.column_config.NumberColumn('Risk Adj. Ratio', width='small'),
                    'Win_Rate': st.column_config.NumberColumn('Win %', width='small'),
                    'Std_Dev': st.column_config.NumberColumn('Std Dev', width='small'),
                    'Max_ROI': st.column_config.NumberColumn('Max ROI', width='small'),
                    'Min_ROI': st.column_config.NumberColumn('Min ROI', width='small'),
                    'Average_ROI': st.column_config.NumberColumn('Avg ROI', width='small'),
                    'Best_Worst_Range': st.column_config.NumberColumn('Range', width='small'),
                    'Quality_Score': st.column_config.NumberColumn('Quality', width='small'),
                    'Portfolio_Suitability': st.column_config.TextColumn('Rating', width='small')
                },
                column_order=[
                    "Grid_ID", "Allocation", "Num_Intervals", "Occurrences",
                    "Cumulative_ROI", "Median_ROI", "Risk_Adjusted_Return", "Win_Rate",
                    "Std_Dev", "Max_ROI", "Min_ROI", "Average_ROI",
                    "Best_Worst_Range", "Quality_Score", "Portfolio_Suitability"
                ]
            )
            
            st.markdown(f"### 📈 Performance Comparison by Grid (Ranked by {SORT_METRIC_DISPLAY_MAP[sort_metric]})")
            st.caption(f"Showing best performance for each grid based on {sort_metric}")
            
            best_by_grid_chart = results_df.loc[results_df.groupby('Grid_ID')[sort_metric].idxmax()].copy()
            best_by_grid_chart = best_by_grid_chart.sort_values(sort_metric, ascending=False)
            
            if selected_grids_filter:
                best_by_grid_chart = best_by_grid_chart[best_by_grid_chart['Grid_ID'].isin(selected_grids_filter)]
            
            chart_data = best_by_grid_chart.head(15)
            fig, ax = plt.subplots(figsize=(16, 8))
            labels = [str(row['Grid_ID']) for _, row in chart_data.iterrows()]
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in chart_data[sort_metric]]
            bars = ax.bar(labels, chart_data[sort_metric], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Grid ID', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Best {SORT_METRIC_DISPLAY_MAP[sort_metric]}', fontsize=12, fontweight='bold')
            ax.set_title(f"Performance by Grid", fontsize=14, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            if sort_metric in ['Average_ROI', 'Cumulative_ROI', 'Median_ROI', 'Win_Rate', 'Std_Dev', 'Max_ROI', 'Min_ROI', 'Best_Worst_Range']:
                 ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            else:
                 ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            for bar in bars:
                height = bar.get_height()
                label_format = '{:.1%}' if sort_metric in ['Average_ROI', 'Cumulative_ROI', 'Median_ROI', 'Win_Rate', 'Std_Dev', 'Max_ROI', 'Min_ROI', 'Best_Worst_Range'] else '{:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        label_format.format(height),
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold', fontsize=9)
            plt.tight_layout(pad=2.0)
            st.pyplot(fig)
            create_download_button(fig, f"location_analysis_{results['regime']}_{download_suffix}.png", "download_loc_chart")
            
            st.divider()
            
            with st.expander("📋 View All Results"):
                st.markdown("**Filter and explore all combinations**")
                all_grids = sorted(results_df['Grid_ID'].unique())
                selected_grids_full = st.multiselect("Filter by Grid ID:", options=all_grids, default=all_grids, key='loc_full_grid_filter')
                filtered_full_df = results_df[results_df['Grid_ID'].isin(selected_grids_full)] if selected_grids_full else results_df
                st.caption(f"Showing {len(filtered_full_df)} of {len(results_df)} total combinations")
                st.dataframe(
                    filtered_full_df.style.format({
                        'Num_Intervals': '{:.0f}', 'Occurrences': '{:.0f}', 'Average_ROI': '{:.1%}',
                        'Cumulative_ROI': '{:.1%}', 'Median_ROI': '{:.1%}', 'Risk_Adjusted_Return': '{:.2f}',
                        'Win_Rate': '{:.1%}', 'Std_Dev': '{:.1%}', 'Max_ROI': '{:.1%}', 'Min_ROI': '{:.1%}', 'Best_Worst_Range': '{:.1%}'
                    }),
                    use_container_width=True, height=400
                )
                # CSV export button for View All Results
                csv_all_results = filtered_full_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export All Results to CSV",
                    data=csv_all_results,
                    file_name=f"all_results_{results['regime']}_{results['hist_context'].replace(' ', '_')}_{results['trend'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="download_all_results_csv"
                )
            
            with st.expander("🔍 Show Year-by-Year Calculation Details (PROOF)"):
                st.markdown("### Year-by-Year Performance & Market View Validation")
                if 'details' in results and not results['details'].empty:
                    details_df = results['details'].copy().sort_values(['Grid', 'Year'], ascending=[True, False])
                    all_grids_details = sorted(details_df['Grid'].unique())
                    selected_grids_details = st.multiselect("Filter by Grid ID:", options=all_grids_details, default=all_grids_details, key='loc_details_grid_filter')
                    filtered_details_df = details_df[details_df['Grid'].isin(selected_grids_details)] if selected_grids_details else details_df
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Matching Years", f"{filtered_details_df['Year'].nunique()}")
                    col2.metric("Grids Shown", f"{filtered_details_df['Grid'].nunique()}")
                    col3.metric("Avg ROI", f"{filtered_details_df['ROI'].mean():.1%}")

                    # Create display copy with ROI converted to percentage (multiply by 100)
                    # The ROI values are stored as decimals (e.g., 1.45 for 145%)
                    display_details_df = filtered_details_df.copy()
                    display_details_df['ROI'] = display_details_df['ROI'] * 100
                    # Remove Coverage column (always shows 1%, adds no value)
                    display_details_df = display_details_df.drop(columns=['Coverage'], errors='ignore')

                    st.dataframe(
                        display_details_df,
                        use_container_width=True,
                        height=600,
                        column_config={
                            'Grid': st.column_config.TextColumn('Grid ID', width='medium'),
                            'Year': st.column_config.NumberColumn('Year', width='small', format="%.0f"),
                            'Allocation': st.column_config.TextColumn('Strategy', width='medium'),
                            'Phase': st.column_config.TextColumn('ENSO Phase', width='small'),
                            'Phase_Intervals': st.column_config.NumberColumn('# Intervals', width='small', format="%.0f", help='Number of 2-month intervals this phase was dominant (must be ≥5)'),
                            'Year_Avg_Hist_Z': st.column_config.NumberColumn('Hist Z', width='small', format="%.2f", help='Year Average Historical Z-Score'),
                            'SOY_11P': st.column_config.NumberColumn('SOY 11P', width='small', format="%.2f", help='Start-of-Year 11-period Z-Score (baseline)'),
                            'EOY_5P': st.column_config.NumberColumn('EOY 5P', width='small', format="%.2f", help='End-of-Year 5-period Z-Score (ending trend)'),
                            'Trajectory_Delta': st.column_config.NumberColumn('Trajectory Δ', width='small', format="%.2f", help='EOY 5P minus SOY 11P (intra-year evolution)'),
                            'ROI': st.column_config.NumberColumn('ROI %', width='small', format="%.2f%%"),
                            'Indemnity': st.column_config.NumberColumn('Indemnity Paid', width='medium', format="$%.0f"),
                            'Producer_Premium': st.column_config.NumberColumn('Premium Cost', width='medium', format="$%.0f"),
                            'Net_Return': st.column_config.NumberColumn('Net Profit/Loss', width='medium', format="$%.0f")
                        }
                    )
                    # CSV export button for Year-by-Year Calculation Details
                    csv_year_details = display_details_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Year-by-Year Details to CSV",
                        data=csv_year_details,
                        file_name=f"year_by_year_details_{results['regime']}_{results['hist_context'].replace(' ', '_')}_{results['trend'].replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_year_details_csv"
                    )
                else:
                    st.info("No detailed results available. Run analysis first.")
            
        except Exception as e:
            st.error(f"Error displaying results: {e}")
            st.exception(e)
            st.session_state.tab2_results = None
    
    elif not st.session_state.tab2_run:
        st.info("👆 Configure your market view and click 'Discover Best Combinations' to see results")

# =============================================================================
# === PORTFOLIO BACKTEST TAB (Tab 2) ===
# =============================================================================

def render_portfolio_tab(session, all_grid_ids, common_params):
    """Portfolio Backtest (Optimized with better caching)"""
    st.subheader("Portfolio Backtest")
    st.markdown("Build a portfolio from your Location Analysis strategies and backtest it against historical scenarios.")
    
    if 'tab2_results' not in st.session_state or not st.session_state.tab2_results:
        st.info("👈 First run the **Location and Timing Analysis** (Tab 1) to generate grid strategies and select grids.")
        st.markdown("""
        ### How This Works:
        
        1. **Run Location Analysis** (Tab 1) to find the best strategy for each grid.
        2. **Return here** and select the grids to include in your portfolio (using the checkboxes in Tab 1's Grid Suitability Table).
        3. The app will create a **'Naive Equal-Weight Portfolio'** by averaging the best strategies.
        4. **Select one scenario** (e.g., 'Historically Dry Years') to backtest.
        5. **Run Backtest** to see how that single allocation would have performed across all selected grids during those specific types of years.
        """)
        return
    
    tab3_results = st.session_state.tab2_results
    results_df = tab3_results['df']
    sort_metric = tab3_results['sort_metric']
    
    st.success(f"✅ Using best strategies from Location Analysis (Ranked by **{SORT_METRIC_DISPLAY_MAP[sort_metric]}**)")
    st.divider()
    
    st.markdown("### 1. Select Grids for Portfolio")
    
    available_grids = sorted(results_df['Grid_ID'].unique())
    
    default_grids = []
    if 'grid_summary_with_selection' in tab3_results and tab3_results['grid_summary_with_selection'] is not None:
        selection_df = tab3_results['grid_summary_with_selection']
        default_grids = selection_df[selection_df['Select'] == True]['Grid_ID'].tolist()
    
    if not default_grids:
        if 'grid_summary_with_selection' in tab3_results and tab3_results['grid_summary_with_selection'] is not None:
            grid_summary = tab3_results['grid_summary_with_selection']
            qualified_grids = grid_summary[grid_summary['Quality_Score'] >= 6.0]['Grid_ID'].tolist()
            default_grids = [g for g in qualified_grids if g in available_grids][:5]
    if not default_grids:
        default_grids = [g for g in available_grids][:5]
    
    selected_grids = st.multiselect(
        "Choose grids to average into the portfolio (2-20 grids):",
        options=available_grids,
        default=default_grids,
        key='portfolio_grids'
    )
    
    if len(selected_grids) < 2:
        st.warning("⚠️ Please select at least 2 grids to run a backtest and optimization.")
        return
    
    if len(selected_grids) > 20:
        st.warning("⚠️ Maximum 20 grids allowed. Using first 20.")
        selected_grids = selected_grids[:20]

    st.divider()
    
    st.markdown("### 2. Naive Equal-Weight Portfolio Allocation")
    
    best_strategies = results_df.loc[results_df.groupby('Grid_ID')[sort_metric].idxmax()]
    
    grid_tuple_key = tuple(sorted(selected_grids))
    if 'cached_naive_allocation_key' not in st.session_state or st.session_state.cached_naive_allocation_key != grid_tuple_key:
        coverage_df = calculate_naive_allocation(tuple(selected_grids), best_strategies)
        
        if coverage_df.empty:
            st.error("No valid strategies found for the selected grids. The base strategies may have issues.")
            return
            
        fig = create_naive_allocation_heatmap(coverage_df)
        
        st.session_state.cached_naive_allocation_key = grid_tuple_key
        st.session_state.cached_naive_allocation_fig = fig
        st.session_state.cached_naive_coverage_df = coverage_df
    
    st.caption("This table shows the best strategy for each selected grid (from Tab 1) and the resulting **Naive Equal-Weight Allocation** (AVERAGE row).")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(st.session_state.cached_naive_allocation_fig, use_container_width=True)
    with col2:
        st.write("")  # Spacer
        st.write("")
        create_download_button(st.session_state.cached_naive_allocation_fig, "naive_allocation_heatmap.png", "download_naive_alloc")
        
        # CSV download
        csv_naive = st.session_state.cached_naive_coverage_df.to_csv()
        st.download_button(
            label="📥 Download CSV",
            data=csv_naive,
            file_name="naive_allocation.csv",
            mime="text/csv",
            key="download_naive_csv"
        )
    
    st.divider()
    
    st.markdown("### 3. Acreage Allocation")
    acreage_mode = st.radio("Acreage Allocation Method:", ["Uniform Acreage", "Customize Acreage"], horizontal=True, key='acreage_mode_radio')
    
    st.divider()
    
    st.markdown("### 4. Scenario Selection and Backtest")
    
    with st.form(key='portfolio_config_backtest_form'):
        
        grid_acres = {}
        
        if acreage_mode == "Uniform Acreage":
            per_grid = common_params['total_insured_acres'] / len(selected_grids)
            for g in selected_grids:
                grid_acres[g] = per_grid
            st.info(f"Allocating **{per_grid:,.0f} acres** per grid (Total: {common_params['total_insured_acres']:,})")
            
        else:
            default_val = float(int(common_params['total_insured_acres'] / max(1, len(selected_grids))))
            
            st.markdown("**Enter acreage for each grid:**")
            st.caption("💡 Adjust the acreage allocation for each grid below. Total will be calculated automatically.")
            
            cols_per_row = 3
            for i in range(0, len(selected_grids), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, grid_id in enumerate(selected_grids[i:i+cols_per_row]):
                    with cols[j]:
                        grid_acres[grid_id] = st.number_input(
                            f"📍 {grid_id}",
                            min_value=0.0,
                            value=default_val,
                            step=10.0,
                            key=f"acres_{grid_id}",
                            help=f"Acres for {grid_id}"
                        )
            
            total_custom_acres = sum(grid_acres.values())
            st.success(f"✅ **Total Portfolio Acres:** {total_custom_acres:,.0f}")
        
        st.divider()
        
        st.markdown("#### Scenario Definition")
        
        scenario_options = [
            'All Years (except Current Year)',
            'Historically Dry Years (< -0.25 Z)',
            'Historically Normal Years (-0.25 to 0.25 Z)',
            'Historically Wet Years (> 0.25 Z)',
            'ENSO Phase: La Niña',
            'ENSO Phase: El Niño',
            'ENSO Phase: Neutral'
        ]
        
        selected_scenario = st.radio(
            "Select one scenario to backtest:",
            options=scenario_options,
            index=0,
            key='portfolio_scenario_select_form'
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            enable_year_cutoff_portfolio = st.checkbox(
                "Exclude older analog years",
                value=False,
                key='portfolio_enable_year_cutoff',
                help="Filter out analog years before a specified cutoff year"
            )

        with col2:
            if enable_year_cutoff_portfolio:
                cutoff_year_portfolio = st.number_input(
                    "Cutoff Year (exclude years before this)",
                    min_value=1948,
                    max_value=2024,
                    value=1990,
                    step=1,
                    key='portfolio_cutoff_year',
                    help="Analog years before this year will be excluded from backtest"
                )
            else:
                cutoff_year_portfolio = None

        st.divider()

        submitted_backtest = st.form_submit_button("📊 Run Naive Backtest", type="primary")
        
    if submitted_backtest:
        
        st.session_state.naive_backtest_results = None
        st.session_state.optimization_results = None
        st.session_state.portfolio_base_data = None
        
        with st.spinner("Fetching all grid histories and running backtest..."):
            # Convert grid_acres dict to hashable tuple for caching
            grid_acres_tuple = tuple(sorted(grid_acres.items()))

            all_year_df = fetch_and_process_all_grid_histories(
                session, selected_grids, best_strategies, common_params, grid_acres_tuple
            )

            if all_year_df.empty:
                st.error("Could not retrieve any historical data for the selected grids.")
                return

            portfolio_avg_z_by_year = all_year_df.groupby('year')['avg_hist_z'].mean()
            portfolio_avg_phase_by_year = all_year_df.groupby('year')['dominant_phase'].apply(lambda x: x.mode()[0] if not x.empty else 'Neutral')

            analog_years_list = []
            if selected_scenario == 'All Years (except Current Year)':
                # Exclude current year (2025) and any future years - only use historical data
                analog_years_list = [yr for yr in portfolio_avg_z_by_year.index.tolist() if yr < 2025]
            elif selected_scenario == 'Historically Dry Years (< -0.25 Z)':
                analog_years_list = [yr for yr in portfolio_avg_z_by_year[portfolio_avg_z_by_year < -0.25].index.tolist() if yr < 2025]
            elif selected_scenario == 'Historically Normal Years (-0.25 to 0.25 Z)':
                analog_years_list = [yr for yr in portfolio_avg_z_by_year[(portfolio_avg_z_by_year >= -0.25) & (portfolio_avg_z_by_year <= 0.25)].index.tolist() if yr < 2025]
            elif selected_scenario == 'Historically Wet Years (> 0.25 Z)':
                analog_years_list = [yr for yr in portfolio_avg_z_by_year[portfolio_avg_z_by_year > 0.25].index.tolist() if yr < 2025]
            elif selected_scenario == 'ENSO Phase: La Niña':
                analog_years_list = [yr for yr in portfolio_avg_phase_by_year[portfolio_avg_phase_by_year == 'La Nina'].index.tolist() if yr < 2025]
            elif selected_scenario == 'ENSO Phase: El Niño':
                analog_years_list = [yr for yr in portfolio_avg_phase_by_year[portfolio_avg_phase_by_year == 'El Nino'].index.tolist() if yr < 2025]
            elif selected_scenario == 'ENSO Phase: Neutral':
                analog_years_list = [yr for yr in portfolio_avg_phase_by_year[portfolio_avg_phase_by_year == 'Neutral'].index.tolist() if yr < 2025]

            # Apply year cutoff filter if enabled
            if enable_year_cutoff_portfolio and cutoff_year_portfolio is not None:
                analog_years_list = [yr for yr in analog_years_list if yr >= cutoff_year_portfolio]

            if not analog_years_list:
                st.warning(f"No analog years found for '{selected_scenario}'" +
                          (f" with cutoff year {cutoff_year_portfolio}" if enable_year_cutoff_portfolio and cutoff_year_portfolio else "") +
                          ". Try a different scenario or earlier cutoff.")
                return

            filtered_df = all_year_df[all_year_df['year'].isin(analog_years_list)].copy()

            st.session_state.portfolio_base_data = filtered_df

            # Indemnity and premium are already calculated with actual acres, just sum them
            yearly_portfolio_performance_naive_df = filtered_df.groupby('year').agg(
                indemnity=('indemnity', 'sum'),
                premium=('premium', 'sum')
            )
            
            yearly_portfolio_performance_naive_df['roi'] = (yearly_portfolio_performance_naive_df['indemnity'] - yearly_portfolio_performance_naive_df['premium']) / yearly_portfolio_performance_naive_df['premium']
            yearly_portfolio_performance_naive_df.replace([np.inf, -np.inf], 0, inplace=True)
            
            naive_metrics = calculate_portfolio_metrics(yearly_portfolio_performance_naive_df)
            naive_metrics['Scenario'] = selected_scenario
            naive_metrics['Cutoff_Year'] = cutoff_year_portfolio if enable_year_cutoff_portfolio else None

            st.session_state.naive_backtest_results = pd.DataFrame([naive_metrics])
            st.session_state.current_grid_acres = grid_acres

    if 'naive_backtest_results' in st.session_state and not st.session_state.naive_backtest_results.empty:
        display_scenario = st.session_state.naive_backtest_results.iloc[0]['Scenario']
        display_cutoff = st.session_state.naive_backtest_results.iloc[0].get('Cutoff_Year')
        cutoff_display = f" (Years ≥ {int(display_cutoff)})" if display_cutoff and not pd.isna(display_cutoff) else ""
        st.markdown(f"### 5. Naive Backtest Result for: *{display_scenario}*{cutoff_display}")
        
        results_table_df = st.session_state.naive_backtest_results
        metrics = results_table_df.iloc[0]
        
        st.markdown("#### Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Analog Years", f"{metrics['Analog_Years']:.0f}")
        col2.metric("Win Rate", f"{metrics['Win_Rate']:.1%}")
        col3.metric("Median ROI", f"{metrics['Median_ROI']:.1%}")
        col4.metric("Risk-Adjusted Return", f"{metrics['Risk_Adjusted_Return']:.2f}")
        
        st.markdown("#### Cumulative Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        total_ind = metrics['Total_Indemnity']
        total_prem = metrics['Total_Premium']
        net_ret = total_ind - total_prem
        
        col1.metric("Total Indemnity", f"${total_ind:,.0f}")
        col2.metric("Total Producer Premium", f"${total_prem:,.0f}")
        col3.metric("Net Return", f"${net_ret:,.0f}")
        col4.metric("Total ROI", f"{metrics['Cumulative_ROI']:.1%}")
        
        st.markdown("#### Annual Averages")
        analog_years = metrics['Analog_Years']
        if analog_years > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Annual Indemnity", f"${total_ind / analog_years:,.0f}")
            col2.metric("Avg Annual Premium", f"${total_prem / analog_years:,.0f}")
            col3.metric("Avg Annual Net Return", f"${net_ret / analog_years:,.0f}")
        
        total_acres_display = sum(st.session_state.current_grid_acres.values())
        st.caption(f"Note: Dollar values are based on **{total_acres_display:,.0f} Total Acres** across selected grids.")
        
        st.markdown("#### Range")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Year ROI", f"{metrics['Max_ROI']:.1%}")
        col2.metric("Worst Year ROI", f"{metrics['Min_ROI']:.1%}")
        col3.metric("Std. Deviation", f"{metrics['Std_Dev']:.1%}")
        
        st.divider()

        st.markdown("### 6. Optimization Controls")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            enable_budget = st.checkbox(
                "Enable Budget Constraint",
                value=False,
                key='enable_budget',
                help="Limit total annual premium spending"
            )

        with col2:
            if enable_budget:
                annual_budget = st.number_input(
                    "Maximum Annual Premium Budget ($)",
                    min_value=1000,
                    value=50000,
                    step=1000,
                    key='annual_budget',
                    help="Maximum producer premium (after subsidy) you want to spend per year"
                )
            else:
                annual_budget = None

        if enable_budget:
            st.markdown("**Acreage Allocation Method:**")
            allocation_method = st.radio(
                "How should acres be distributed to meet the budget?",
                ["Optimize Distribution", "Equal Scaling"],
                index=0,
                key='allocation_method',
                horizontal=True,
                help="Optimize Distribution: Use mean-variance optimization to find ideal distribution that maximizes returns within budget. Equal Scaling: Scale all grids proportionally to use full budget."
            )
        else:
            allocation_method = None

        st.divider()
        
        optimization_goal = st.radio(
            "Select Optimization Goal:",
            ['Cumulative ROI', 'Risk-Adjusted Return'],
            index=0,
            key='optimization_goal_select',
            horizontal=True,
            help="Choose whether to maximize the simple total return (Cumulative ROI) or maximize Sharpe Ratio (Risk-Adjusted Return) over the scenario."
        )
        
        st.markdown("**Diversification Constraints:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            require_full_coverage = st.checkbox(
                "📅 Ensure Full Calendar Coverage",
                value=False,
                key='require_full_coverage',
                help="Creates staggered portfolio with Pattern A (6 intervals) and Pattern B (5 intervals) to cover all 11 intervals while maintaining non-adjacency. Requires at least 2 grids."
            )
            
            if require_full_coverage and len(selected_grids) < 2:
                st.error("⚠️ Full calendar coverage requires at least 2 grids. Please select more grids.")
                require_full_coverage = False
        
        with col2:
            if not require_full_coverage:
                interval_range_opt = st.slider(
                    "Number of Active Intervals Range",
                    min_value=2,
                    max_value=6,
                    value=(2, 6),
                    key='interval_range_slider',
                    help="Optimizer can choose ANY number of intervals within this range. Each interval must have 10%-50%."
                )
            else:
                interval_range_opt = (11, 11)
        
        if not require_full_coverage:
            min_int, max_int = interval_range_opt
            st.caption(f"💡 Optimizer can use **{min_int} to {max_int} intervals** (each ≥{MIN_ALLOCATION:.0%}, ≤{MAX_ALLOCATION:.0%}).")
            st.info(f"ℹ️ **Allocation Constraints:** Each active interval must have **{MIN_ALLOCATION:.0%} to {MAX_ALLOCATION:.0%}** in {ALLOCATION_INCREMENT:.0%} increments. **Non-adjacent** intervals (except Jan-Feb/Nov-Dec wrap).")
        else:
            st.info("ℹ️ **Full Coverage Mode:** Portfolio will use a staggered approach:\n"
                   "- **Pattern A** grids: Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec (6 intervals)\n"
                   "- **Pattern B** grids: Feb-Mar, Apr-May, Jun-Jul, Aug-Sep, Oct-Nov (5 intervals)\n"
                   "- Each interval: 10%-50% allocation\n"
                   "- Optimizer decides which grids use which pattern")
        
        if st.button(f"✨ Optimize Portfolio (Max {optimization_goal})", key="run_optimization_btn", type="primary"):
            # Clear cached optimization results if key parameters have changed
            if 'optimization_results' in st.session_state and st.session_state.optimization_results:
                cached_opt_params = st.session_state.optimization_results
                params_changed = False
                change_reasons = []

                if cached_opt_params.get('interval_range') != interval_range_opt:
                    params_changed = True
                    change_reasons.append(f"Interval range: {cached_opt_params.get('interval_range')} → {interval_range_opt}")

                if cached_opt_params.get('optimization_goal') != optimization_goal:
                    params_changed = True
                    change_reasons.append(f"Optimization goal: {cached_opt_params.get('optimization_goal')} → {optimization_goal}")

                if cached_opt_params.get('full_coverage') != require_full_coverage:
                    params_changed = True
                    change_reasons.append(f"Full coverage: {cached_opt_params.get('full_coverage')} → {require_full_coverage}")

                if params_changed:
                    st.session_state.optimization_results = None
                    st.info(f"🔄 Optimization parameters changed. Clearing cached results:\n" + "\n".join(f"• {r}" for r in change_reasons))

            if 'portfolio_base_data' in st.session_state and not st.session_state.portfolio_base_data.empty:
                
                naive_interval_weights = st.session_state.cached_naive_coverage_df.loc['AVERAGE', INTERVAL_ORDER_11]
                grid_acres = st.session_state.current_grid_acres
                
                with st.spinner(f"Running Monte Carlo Simulation and optimizing for the **{display_scenario}** scenario..."):
                    
                    detailed_alloc_df, detailed_change_df, opt_metrics, optimized_weights_raw = run_portfolio_optimization_wrapper(
                        st.session_state.portfolio_base_data, naive_interval_weights, optimization_goal, selected_grids, grid_acres, interval_range_opt, require_full_coverage
                    )
                    
                    if detailed_alloc_df is not None:
                        # Calculate TRUE initial cost using NAIVE weights (before optimization)
                        # This ensures "started with" matches the naive backtest cost
                        naive_weights_array = naive_interval_weights[INTERVAL_ORDER_11].values
                        initial_cost, initial_grid_cost_breakdown = calculate_annual_premium_cost(
                            naive_weights_array, selected_grids, grid_acres, session, common_params
                        )

                        # Validate initial cost
                        if np.isnan(initial_cost) or initial_cost < 0:
                            st.error("⚠️ Initial cost calculation failed. Check grid data (County Base Values, Premium Rates).")
                            st.stop()

                        total_annual_cost = initial_cost
                        grid_cost_breakdown = initial_grid_cost_breakdown
                        final_grid_acres = grid_acres.copy()
                        budget_scale_factor = 1.0
                        budget_applied = False
                        allocation_method_used = "None (Budget Disabled)"

                        if enable_budget and annual_budget is not None and annual_budget > 0:
                            if allocation_method == "Equal Scaling":
                                # Equal Scaling: ALWAYS run to optimize budget usage
                                budget_applied = True
                                allocation_method_used = "Equal Scaling"

                                if total_annual_cost > annual_budget:
                                    # Scale DOWN if over budget
                                    final_grid_acres, budget_scale_factor = apply_budget_constraint(
                                        grid_acres, total_annual_cost, annual_budget
                                    )
                                else:
                                    # Scale UP if under budget to use full budget
                                    target_scale = annual_budget / total_annual_cost if total_annual_cost > 0 else 1.0
                                    final_grid_acres = {grid: acres * target_scale for grid, acres in grid_acres.items()}
                                    budget_scale_factor = target_scale

                                total_annual_cost, grid_cost_breakdown = calculate_annual_premium_cost(
                                    optimized_weights_raw, selected_grids, final_grid_acres, session, common_params
                                )

                            else:
                                # Optimized Distribution: ALWAYS run to maximize objective
                                # Whether over or under budget, optimize allocation
                                budget_applied = True
                                allocation_method_used = "Optimized Distribution"

                                try:
                                    grid_opt_results, _ = optimize_grid_allocation(
                                        st.session_state.portfolio_base_data,
                                        optimized_weights_raw,
                                        initial_acres_per_grid=grid_acres,
                                        annual_budget=annual_budget,
                                        session=session,
                                        common_params=common_params,
                                        selected_grids=selected_grids,
                                        risk_aversion=1.0
                                    )

                                    # Validate optimization results
                                    if not isinstance(grid_opt_results, dict):
                                        raise TypeError(f"Expected dict, got {type(grid_opt_results)}")

                                    for grid, acres in grid_opt_results.items():
                                        if np.isnan(acres) or np.isinf(acres) or acres < 0:
                                            raise ValueError(f"Invalid acres for {grid}: {acres}")

                                    final_grid_acres = grid_opt_results
                                    budget_scale_factor = 1.0

                                    total_annual_cost, grid_cost_breakdown = calculate_annual_premium_cost(
                                        optimized_weights_raw, selected_grids, final_grid_acres, session, common_params
                                    )

                                    # Validate cost calculation
                                    if np.isnan(total_annual_cost) or total_annual_cost <= 0:
                                        raise ValueError("Cost calculation failed. Check grid data (County Base Values, Premium Rates).")

                                except Exception as e:
                                    st.error(f"⚠️ Optimization failed: {str(e)}. Using fallback allocation.")
                                    # Fallback to Equal Scaling if over budget, else keep original
                                    if initial_cost > annual_budget:
                                        final_grid_acres, budget_scale_factor = apply_budget_constraint(
                                            grid_acres, initial_cost, annual_budget
                                        )
                                        allocation_method_used = "Equal Scaling (Fallback)"
                                        total_annual_cost, grid_cost_breakdown = calculate_annual_premium_cost(
                                            optimized_weights_raw, selected_grids, final_grid_acres, session, common_params
                                        )

                                        # Validate fallback cost
                                        if np.isnan(total_annual_cost) or total_annual_cost < 0:
                                            total_annual_cost = initial_cost
                                            final_grid_acres = grid_acres.copy()
                                            allocation_method_used = "None (Optimization Failed)"
                                    else:
                                        # Under budget but optimization failed - keep original
                                        final_grid_acres = grid_acres.copy()
                                        total_annual_cost = initial_cost
                                        allocation_method_used = "None (Optimization Failed)"

                        opt_metrics['Scenario'] = display_scenario
                        st.session_state.optimization_results = {
                            'allocation_detailed': detailed_alloc_df,
                            'change_detailed': detailed_change_df,
                            'metrics': pd.DataFrame([opt_metrics]),
                            'naive_metrics': metrics,
                            'optimization_goal': optimization_goal,
                            'raw_weights': optimized_weights_raw,
                            'final_grid_acres': final_grid_acres,
                            'annual_cost': total_annual_cost,
                            'grid_costs': grid_cost_breakdown,
                            'budget_applied': budget_applied,
                            'budget_scale_factor': budget_scale_factor,
                            'budget_enabled': enable_budget,
                            'allocation_method_used': allocation_method_used,
                            'full_coverage': require_full_coverage,
                            'interval_range': interval_range_opt,
                            'initial_cost': initial_cost
                        }

                        if budget_applied and allocation_method_used == "Optimized Distribution":
                            # Calculate acres change with defensive formatting
                            try:
                                initial_total_acres = sum(grid_acres.values())
                                final_total_acres = sum(final_grid_acres.values())
                                acres_change = final_total_acres - initial_total_acres
                                acres_change_pct = (acres_change / initial_total_acres * 100) if initial_total_acres > 0 else 0

                                # Defensive formatting for costs
                                if np.isnan(total_annual_cost) or np.isnan(annual_budget) or annual_budget <= 0:
                                    st.error("⚠️ **Cost Calculation Error:** Unable to calculate budget utilization. Check grid data.")
                                else:
                                    budget_utilization = total_annual_cost / annual_budget

                                    # Format the change message
                                    if abs(acres_change) < 1:
                                        change_msg = "maintained"
                                    elif acres_change > 0:
                                        change_msg = f"added {acres_change:,.0f} acres (+{acres_change_pct:.1f}%)"
                                    else:
                                        change_msg = f"removed {abs(acres_change):,.0f} acres ({acres_change_pct:.1f}%)"

                                    if budget_utilization >= 0.95:
                                        st.success(
                                            f"✅ **Budget Fully Utilized:**\n\n"
                                            f"Started with {initial_total_acres:,.0f} acres (${initial_cost:,.0f})\n\n"
                                            f"Optimized to {final_total_acres:,.0f} acres ({change_msg})\n\n"
                                            f"Using ${total_annual_cost:,.0f}"
                                        )
                                    elif budget_utilization >= 0.70:
                                        st.info(
                                            f"ℹ️ **Budget Optimization:**\n\n"
                                            f"Started with {initial_total_acres:,.0f} acres\n\n"
                                            f"Optimized to {final_total_acres:,.0f} acres ({change_msg})\n\n"
                                            f"Using ${total_annual_cost:,.0f}"
                                        )
                                    else:
                                        st.warning(
                                            f"⚠️ **Low Budget Utilization:**\n\n"
                                            f"Started with {initial_total_acres:,.0f} acres\n\n"
                                            f"Optimized to {final_total_acres:,.0f} acres ({change_msg})\n\n"
                                            f"Using ${total_annual_cost:,.0f}"
                                        )
                            except Exception as e:
                                st.warning(f"⚠️ Budget optimization completed but summary calculation failed: {str(e)}")

                        elif budget_applied and allocation_method_used == "Equal Scaling":
                            try:
                                initial_total_acres = sum(grid_acres.values())
                                final_total_acres = sum(final_grid_acres.values())
                                acres_change = final_total_acres - initial_total_acres

                                if acres_change > 0:
                                    st.success(
                                        f"✅ **Equal Scaling Applied:**\n\n"
                                        f"Acres scaled up by {budget_scale_factor:.1%} to use full ${annual_budget:,.0f} budget\n\n"
                                        f"Total acres: {initial_total_acres:,.0f} → {final_total_acres:,.0f} (added {acres_change:,.0f} acres)"
                                    )
                                elif acres_change < 0:
                                    st.warning(
                                        f"⚠️ **Equal Scaling Applied:**\n\n"
                                        f"Acres scaled down by {budget_scale_factor:.1%} to meet ${annual_budget:,.0f} budget\n\n"
                                        f"Total acres: {initial_total_acres:,.0f} → {final_total_acres:,.0f} (removed {abs(acres_change):,.0f} acres)"
                                    )
                                else:
                                    st.info(
                                        f"ℹ️ **Equal Scaling:**\n\n"
                                        f"Acres already optimal for ${annual_budget:,.0f} budget ({initial_total_acres:,.0f} acres)"
                                    )
                            except Exception:
                                if budget_scale_factor > 1.0:
                                    st.success(f"✅ **Equal Scaling Applied:** Acres scaled up by {budget_scale_factor:.1%} to use full budget.")
                                else:
                                    st.warning(f"⚠️ **Equal Scaling Applied:** Acres scaled down by {budget_scale_factor:.1%} to meet budget.")

                        elif budget_applied and "Fallback" in allocation_method_used:
                            st.info("ℹ️ Budget optimization encountered errors and used fallback proportional scaling.")

                        st.success(f"✅ Portfolio Optimization Complete!")
                    else:
                        st.error("Optimization failed to find a valid solution. Try a different scenario/grid combination.")
            else:
                st.warning("Please run the Naive Backtest first to generate the scenario data needed for optimization.")

    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        opt_results = st.session_state.optimization_results
        opt_detailed_df = opt_results['allocation_detailed']
        opt_change_df = opt_results['change_detailed']
        opt_metrics_df = opt_results['metrics']
        opt_metrics = opt_metrics_df.iloc[0]
        naive_metrics = opt_results['naive_metrics']
        optimization_goal = opt_results['optimization_goal']
        optimized_weights_raw = opt_results['raw_weights']
        full_coverage = opt_results.get('full_coverage', False)
        interval_range_used = opt_results.get('interval_range', (2, 6))
        
        st.divider()
        
        with st.expander("📊 **Grid Correlation Matrix** (Click to expand)", expanded=False):
            st.caption("This heatmap shows how the monthly ROIs of your selected grids correlated during the historical years matching your chosen scenario. Low correlation is good for risk diversification.")
            
            roi_pivot = st.session_state.portfolio_base_data.pivot_table(index='year', columns='grid', values='roi')
            
            corr_matrix = roi_pivot.corr(min_periods=3)
            
            if len(corr_matrix) > 1:
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                if (upper_tri > 0.99).any().any():
                    st.info("ℹ️ **Note on High Correlations:** You have selected multiple grids with correlations of **1.00**. This typically happens when different counties share the same underlying **Grid ID** (rainfall data source). Because they use identical weather data, their returns are mathematically identical.")

            fig_size = min(10, max(6, len(corr_matrix) * 0.8))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                annot_kws={"fontsize": 9}
            )
            ax.set_title(f"ROI Correlation Matrix (Scenario: {display_scenario})", fontsize=10, fontweight='bold', pad=10)
            plt.yticks(rotation=0, fontsize=9)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            create_download_button(fig, f"correlation_matrix_{display_scenario}.png", "download_corr_matrix")
        
        st.divider()
        
        st.markdown(f"### 7. Optimized Interval Allocation (Max {optimization_goal})")
        
        if full_coverage:
            pattern_A_grids = [g for i, g in enumerate(selected_grids) if 'Pattern' in opt_detailed_df.columns and g in opt_detailed_df.index and 'A' in str(opt_detailed_df.loc[g, 'Pattern'])]
            pattern_B_grids = [g for i, g in enumerate(selected_grids) if 'Pattern' in opt_detailed_df.columns and g in opt_detailed_df.index and 'B' in str(opt_detailed_df.loc[g, 'Pattern'])]
            st.caption(f"✅ **Staggered Full Calendar Coverage**:\n"
                      f"- **Pattern A** ({len(pattern_A_grids)} grids): Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec (6 intervals, each 10%-50%)\n"
                      f"- **Pattern B** ({len(pattern_B_grids)} grids): Feb-Mar, Apr-May, Jun-Jul, Aug-Sep, Oct-Nov (5 intervals, each 10%-50%)")
        else:
            min_alloc_used = MIN_ALLOCATION
            num_active_intervals = np.sum(optimized_weights_raw >= min_alloc_used)
            min_int, max_int = interval_range_used
            st.caption(f"✅ Optimized to **{num_active_intervals} active intervals** | Allowed range: **{min_int}-{max_int} intervals** | Each interval: {MIN_ALLOCATION:.0%}-{MAX_ALLOCATION:.0%}")
        
        if opt_results.get('budget_enabled', False):
            method_used = opt_results.get('allocation_method_used', 'None')
            st.caption(f"✅ **Budget-Constrained Allocation** | Annual Premium: **${opt_results['annual_cost']:,.0f}** using 2025 rates | Method: **{method_used}**")
            if opt_results.get('budget_applied', False):
                if method_used == "Equal Scaling":
                    st.info(f"ℹ️ **Acres were scaled equally by {opt_results['budget_scale_factor']:.1%}** to meet your budget constraint. Interval allocations remain at 100%.")
                elif method_used == "Optimized Distribution":
                    st.info(f"ℹ️ **Acres were optimally distributed** using mean-variance optimization to maximize risk-adjusted returns while meeting your budget constraint. Interval allocations remain at 100%.")
        else:
            st.caption(f"The interval weights (which apply equally to all chosen grids) have been shifted from the Naive Average to maximize the **{optimization_goal}** over the selected scenario.")

        display_alloc_df = opt_detailed_df.copy()
        display_acres = opt_results.get('final_grid_acres', st.session_state.current_grid_acres)

        # ALWAYS calculate costs for full transparency, regardless of budget settings
        if opt_results.get('budget_enabled', False):
            # Budget enabled: use pre-calculated values from optimization
            annual_cost = opt_results['annual_cost']
            grid_costs = opt_results['grid_costs']
        else:
            # Budget not enabled: calculate costs for transparency
            annual_cost, grid_costs = calculate_annual_premium_cost(
                optimized_weights_raw, selected_grids, display_acres, session, common_params
            )

        # ALWAYS add Acres and Annual Premium columns for transparency
        acres_list = []
        for idx in display_alloc_df.index:
            # Handle both possible average row names
            if idx in ['OPTIMIZED AVERAGE', 'PORTFOLIO AVERAGE']:
                acres_list.append(sum(display_acres.values()))
            else:
                acres_list.append(display_acres.get(idx, 0))

        display_alloc_df['Acres'] = acres_list

        cost_list = []
        for idx in display_alloc_df.index:
            # Handle both possible average row names
            if idx in ['OPTIMIZED AVERAGE', 'PORTFOLIO AVERAGE']:
                cost_list.append(annual_cost)
            else:
                cost_list.append(grid_costs.get(idx, 0))

        display_alloc_df['Annual Premium ($)'] = cost_list

        # ALWAYS pass acres and costs to table for display
        alloc_fig = create_optimized_allocation_table(
            display_alloc_df,
            grid_acres=display_acres,
            grid_costs=grid_costs,
            budget_enabled=opt_results.get('budget_enabled', False)
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.pyplot(alloc_fig, use_container_width=True)
        with col2:
            st.write("")  # Spacer
            st.write("")
            create_download_button(alloc_fig, f"optimized_allocation_{display_scenario}.png", "download_opt_alloc")
            
            # CSV download
            csv_opt_alloc = display_alloc_df.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv_opt_alloc,
                file_name=f"optimized_allocation_{display_scenario}.csv",
                mime="text/csv",
                key="download_opt_alloc_csv"
            )
        
        if opt_results.get('budget_enabled', False) and opt_results.get('allocation_method_used') == "Optimized Distribution":
            st.caption("⚠️ **Important:** The optimizer respects your specified maximum acres per grid. Each grid received between 0 and its custom max (from section 1b).")

        if opt_results.get('budget_enabled', False):
            st.markdown("---")
            st.markdown("### 💰 Annual Cost Estimate (2025 Premium Rates)")
            
            display_acres = opt_results.get('final_grid_acres', st.session_state.current_grid_acres)
            total_display_acres = sum(display_acres.values())
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Annual Premium", f"${opt_results['annual_cost']:,.0f}")
            col2.metric("Total Acres", f"{total_display_acres:,.0f}")
            avg_cost = opt_results['annual_cost'] / total_display_acres if total_display_acres > 0 else 0
            col3.metric("Avg Cost per Acre", f"${avg_cost:.2f}")
            
            st.caption("💡 Premium includes subsidy and represents actual out-of-pocket producer cost based on current 2025 rates.")
        
        st.divider()
        
        st.markdown("### 8. Allocation Changes by Grid and Interval")
        st.caption("Shows the change (+/-) for each grid's individual naive allocation → optimized allocation. Green = increase, Red = decrease, Gray = no change.")
        
        grid_acres_original = st.session_state.current_grid_acres
        grid_acres_final = opt_results.get('final_grid_acres', st.session_state.current_grid_acres)
        
        naive_alloc_df = st.session_state.cached_naive_coverage_df
        
        changes_fig = create_allocation_changes_table(
            optimized_alloc_df=opt_detailed_df,
            naive_alloc_df=naive_alloc_df,
            selected_grids=selected_grids,
            grid_acres_original=grid_acres_original if opt_results.get('budget_enabled', False) else None,
            grid_acres_final=grid_acres_final if opt_results.get('budget_enabled', False) else None,
            budget_enabled=opt_results.get('budget_enabled', False)
        )
        
        if changes_fig:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(changes_fig, use_container_width=True)
            with col2:
                st.write("")  # Spacer
                st.write("")
                create_download_button(changes_fig, f"allocation_changes_{display_scenario}.png", "download_changes_table")
                
                # CSV download
                naive_alloc_df = st.session_state.cached_naive_coverage_df

                # Recreate the changes dataframe for CSV
                change_rows = []
                for grid in selected_grids:
                    row_dict = {'Grid': grid}

                    # Get this grid's naive allocation
                    if grid in naive_alloc_df.index:
                        grid_naive = naive_alloc_df.loc[grid, INTERVAL_ORDER_11]
                    else:
                        grid_naive = naive_alloc_df.loc['AVERAGE', INTERVAL_ORDER_11]

                    # Get this grid's optimized allocation
                    if grid in opt_detailed_df.index:
                        grid_optimized = opt_detailed_df.loc[grid, INTERVAL_ORDER_11]
                    else:
                        # Fallback to optimized average if grid not found
                        if 'OPTIMIZED AVERAGE' in opt_detailed_df.index:
                            grid_optimized = opt_detailed_df.loc['OPTIMIZED AVERAGE', INTERVAL_ORDER_11]
                        elif 'PORTFOLIO AVERAGE' in opt_detailed_df.index:
                            grid_optimized = opt_detailed_df.loc['PORTFOLIO AVERAGE', INTERVAL_ORDER_11]
                        else:
                            grid_optimized = grid_naive

                    net_change = 0
                    for interval in INTERVAL_ORDER_11:
                        change = grid_optimized[interval] - grid_naive[interval]
                        row_dict[interval] = change
                        net_change += change

                    row_dict['Net Change'] = net_change
                    if opt_results.get('budget_enabled', False):
                        row_dict['Acres Change'] = grid_acres_final.get(grid, 0) - grid_acres_original.get(grid, 0)
                    change_rows.append(row_dict)
                
                changes_csv_df = pd.DataFrame(change_rows)
                csv_changes = changes_csv_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_changes,
                    file_name=f"allocation_changes_{display_scenario}.csv",
                    mime="text/csv",
                    key="download_changes_csv"
                )
        else:
            st.warning("Could not generate allocation changes table.")
        
        if opt_results.get('budget_enabled', False):
            st.caption("💡 **Acres Change** shows how many acres were added/removed from each grid to meet your budget constraint.")
        
        st.divider()
        
        st.markdown("### 9. Optimized vs. Naive Performance Comparison")

        comparison_data = []
        metrics_to_compare = [
            'Cumulative_ROI', 'Risk_Adjusted_Return', 'Win_Rate', 'Std_Dev'
        ]
        metric_labels = [
            'Total ROI', 'Risk-Adjusted Return (Sharpe Ratio)', 'Win Rate', 'Standard Deviation (Risk)'
        ]
        metric_formats = [
            '{:.1%}', '{:.2f}', '{:.1%}', '{:.1%}'
        ]

        # Calculate average annual premiums
        naive_avg_premium = naive_metrics['Total_Premium'] / naive_metrics['Analog_Years'] if naive_metrics['Analog_Years'] > 0 else 0
        optimized_annual_premium = opt_results.get('annual_cost', 0)
        premium_change = optimized_annual_premium - naive_avg_premium
        premium_change_pct = (premium_change / naive_avg_premium * 100) if naive_avg_premium > 0 else 0

        # Add premium comparison row
        comparison_data.append({
            'Metric': 'Estimated Annual Premium',
            'Naive Portfolio': f'${naive_avg_premium:,.0f}',
            'Optimized Portfolio': f'${optimized_annual_premium:,.0f}',
            'Change': f'${premium_change:+,.0f} ({premium_change_pct:+.1f}%)'
        })

        for metric, label, fmt in zip(metrics_to_compare, metric_labels, metric_formats):
            naive_val = naive_metrics.get(metric, 0)
            opt_val = opt_metrics.get(metric, 0)

            diff = opt_val - naive_val
            change = f"{diff:+.1%}" if metric in ['Cumulative_ROI', 'Win_Rate', 'Std_Dev'] else f"{diff:+.2f}"

            comparison_data.append({
                'Metric': label,
                'Naive Portfolio': fmt.format(naive_val),
                'Optimized Portfolio': fmt.format(opt_val),
                'Change': change
            })

        comparison_df = pd.DataFrame(comparison_data).set_index('Metric')
        st.table(comparison_df)
        
        st.divider()
        

def main():
    st.title("Layer 3 - PRF Systematic Positioning")
    st.caption("Identify optimal grids and intervals based on climate regime and market conditions")
    
    session = get_active_session()
    
    try:
        valid_grids = load_distinct_grids(session)
    except Exception as e:
        st.sidebar.error("Fatal Error: Could not load Grid ID list.")
        st.error(f"Could could not load Grid ID list: {e}")
        st.stop()
    
    st.sidebar.header("⚙️ Common Parameters")
    
    if 'coverage_level' not in st.session_state:
        st.session_state.coverage_level = 0.80
    if 'productivity_factor' not in st.session_state:
        st.session_state.productivity_factor = 1.0
    if 'total_insured_acres' not in st.session_state:
        st.session_state.total_insured_acres = 1000
    if 'intended_use' not in st.session_state:
        st.session_state.intended_use = 'Grazing'
    if 'insurance_plan_code' not in st.session_state:
        st.session_state.insurance_plan_code = 13
    
    # Determine coverage level index - use preset if available
    coverage_options = [0.70, 0.75, 0.80, 0.85, 0.90]
    if 'preset_coverage' in st.session_state:
        try:
            coverage_index = coverage_options.index(st.session_state['preset_coverage'])
        except ValueError:
            coverage_index = 2  # Default to 80%
    else:
        coverage_index = 2  # Default to 80%

    coverage_level = st.sidebar.selectbox(
        "Coverage Level",
        coverage_options,
        index=coverage_index,
        format_func=lambda x: f"{x:.0%}",
        key='sidebar_coverage'
    )

    # Determine productivity factor index - use preset if available
    prod_options = list(range(60, 151))
    prod_options_formatted = [f"{x}%" for x in prod_options]

    if 'preset_prod_factor' in st.session_state:
        try:
            preset_prod_pct = int(st.session_state['preset_prod_factor'] * 100)
            current_prod_index = prod_options.index(preset_prod_pct)
        except ValueError:
            current_prod_index = 40  # Default to 100%
    else:
        try:
            current_prod_index = prod_options.index(int(st.session_state.productivity_factor * 100))
        except ValueError:
            current_prod_index = 40

    selected_prod_str = st.sidebar.selectbox(
        "Productivity Factor",
        options=prod_options_formatted,
        index=current_prod_index,
        key='sidebar_prod'
    )
    productivity_factor = int(selected_prod_str.replace('%', '')) / 100.0
    
    total_insured_acres = st.sidebar.number_input(
        "Total Insured Acres (if customization not used)",
        value=st.session_state.total_insured_acres,
        step=10,
        key='sidebar_acres'
    )
    
    intended_use = st.sidebar.selectbox(
        "Intended Use",
        ['Grazing', 'Haying'],
        index=0 if st.session_state.intended_use == 'Grazing' else 1,
        key='sidebar_use'
    )
    
    plan_code = st.sidebar.number_input(
        "Insurance Plan Code",
        value=13,
        disabled=True,
        key='sidebar_plan'
    )
    
    with st.sidebar.expander("ℹ️ Z-Score Translation Key"):
        st.markdown("**Historical Context (Year Avg Z-Score):**")
        st.markdown(f"- **Dry:** Year avg Z $< -0.25$ ($\\approx 30^\\text{{th}}$ percentile or drier)")
        st.markdown(f"- **Normal:** Year avg Z $-0.25$ to $+0.25$ ($\\approx 30^\\text{{th}}$ to $70^\\text{{th}}$ percentile)")
        st.markdown(f"- **Wet:** Year avg Z $> +0.25$ ($\\approx 70^\\text{{th}}$ percentile or wetter)")
        st.markdown("---")
        st.markdown("💡 **Portfolio Backtest: Dry Year Definition**")
        st.markdown(f"A **Historically Dry Year** for the backtest is one where the **average annual Z-Score across all selected grids** is **$< -0.25$**.")
        st.markdown("---")
        st.markdown("**Expected Trajectory (SOY 11P vs EOY 5P):**")
        st.markdown("Δ = End-of-year 5P Z-score minus Start-of-year 11P Z-score")
        st.markdown(f"- **Get Drier:** $\\Delta < -0.2$ ($\\approx 1$-in-4 chance of shift)")
        st.markdown(f"- **Stay Stable:** $\\Delta -0.2$ to $+0.2$ ($\\approx 1$-in-2 chance of stability)")
        st.markdown(f"- **Get Wetter:** $\\Delta > +0.2$ ($\\approx 1$-in-4 chance of shift)")
    
    st.sidebar.divider()
    
    common_params = {
        'coverage_level': coverage_level,
        'productivity_factor': productivity_factor,
        'total_insured_acres': total_insured_acres,
        'intended_use': intended_use,
        'plan_code': plan_code
    }
    
    tab1, tab2 = st.tabs([
        "📍 Location and Timing Analysis",
        "💼 Portfolio Backtest"
    ])
    
    with tab1:
        render_location_tab(session, valid_grids, common_params)
    
    with tab2:
        render_portfolio_tab(session, valid_grids, common_params)

if __name__ == "__main__":
    main()