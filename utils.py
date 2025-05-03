import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import re
from collections import Counter
from rapidfuzz import fuzz, process
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message="DataFrame.applymap has been deprecated"
)


def set_visualization_style():
    """Set consistent visualization style for all plots in the notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("deep")
    return sns.color_palette("deep")


def load_data(file_path):
    """Load the dataset from specified file path and display basic information about structure and content."""
    df = pd.read_excel(file_path)

    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


def standardize_column_names(df):
    """Convert all column names to lowercase with underscores for consistency."""
    df.columns = [
        col.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        for col in df.columns
    ]
    print("\nStandardized column names:")
    print(df.columns.tolist())
    return df


def explore_categorical_distributions(df, categorical_cols):
    """Examine the distribution of values in categorical columns of the dataset."""
    for col in categorical_cols:
        if col in df.columns:
            print(f"\nUnique values in {col}: {df[col].nunique()}")
            if df[col].nunique() < 10:
                print(df[col].value_counts())
    return None


def plot_fiscal_year_distribution(df):
    """Create a bar chart showing the distribution of contracts by fiscal year."""
    plt.figure(figsize=(10, 6))
    year_counts = df["fiscal_year"].value_counts().sort_index()
    sns.barplot(x=year_counts.index, y=year_counts.values)
    plt.title("Distribution of Contracts by Fiscal Year")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Number of Contracts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return None


def plot_award_amount_distribution(df):
    """Create visualizations showing the distribution of contract award amounts."""
    plt.figure(figsize=(12, 6))

    amount_data = df["awarded_amount"].dropna()
    q1, q3 = amount_data.quantile(0.01), amount_data.quantile(0.99)
    filtered_amounts = amount_data[(amount_data >= q1) & (amount_data <= q3)]

    sns.histplot(filtered_amounts, bins=30, kde=True)
    plt.title("Distribution of Award Amounts (1st to 99th percentile)")
    plt.xlabel("Award Amount ($)")
    plt.ylabel("Frequency")
    plt.axvline(
        filtered_amounts.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: ${filtered_amounts.mean():,.2f}",
    )
    plt.axvline(
        filtered_amounts.median(),
        color="g",
        linestyle="--",
        label=f"Median: ${filtered_amounts.median():,.2f}",
    )
    plt.ticklabel_format(style="plain", axis="x")
    plt.xticks(rotation=45)
    formatter = plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(
        amount_data[amount_data > 0], bins=30, kde=False, log_scale=(True, False)
    )
    plt.title("Distribution of Award Amounts (Log Scale)")
    plt.xlabel("Award Amount ($, log scale)")
    plt.ylabel("Frequency")
    formatter = plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
    return None


def identify_program(title):
    """Determine which ground combat vehicle program a contract belongs to based on its title."""
    title = str(title).lower()

    if any(
        term in title
        for term in ["abrams", "m1a1", "m1a2", "m1 tank", "tank m1", "m1 abrams"]
    ):
        return "Abrams"

    elif any(
        term in title
        for term in ["bradley", "m2a1", "m2a2", "m2a3", "fighting vehicle", "bfv"]
    ):
        return "Bradley"

    elif ("stryker" in title or "m1130" in title) and not any(
        term in title
        for term in [
            "medical",
            "hospital",
            "surgical",
            "patient",
            "maintenance",
            "neptune",
            "bed",
            "implant",
        ]
    ):
        return "Stryker"
    else:
        return "Unknown"


def categorize_programs(df):
    """Classify contracts into ground combat vehicle programs based on contract titles."""
    df["program"] = df["title"].apply(identify_program)

    program_counts = df["program"].value_counts()
    print("\nContracts by Program:")
    print(program_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=program_counts.index, y=program_counts.values)
    plt.title("Number of Contracts by Program")
    plt.xlabel("Program")
    plt.ylabel("Number of Contracts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    unknown_contracts = df[df["program"] == "Unknown"]
    if not unknown_contracts.empty:
        print(f"\nFound {len(unknown_contracts)} contracts with unidentified programs.")
        print("Sample of unidentified contracts:")
        print(unknown_contracts[["title", "awarded_amount"]].head(10))

    return df


def clean_vendor_name(name):
    """Standardize vendor names by removing legal entity types and extraneous punctuation."""
    if pd.isna(name):
        return name

    name = str(name).lower().strip()

    entity_types = [
        " inc",
        " llc",
        " corporation",
        " corp",
        " co",
        " company",
        " ltd",
        " limited",
        " lp",
        " l.p.",
        " l.l.c.",
        " inc.",
    ]
    for entity in entity_types:
        name = re.sub(f"{entity}$|{entity} ", " ", name)

    name = re.sub(r"[,\.&]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_vendors(vendor_list, threshold=90):
    """Use fuzzy matching to normalize vendor names that are similar but not identical."""
    normalized_vendors = {}

    vendor_counts = Counter(vendor_list)
    sorted_vendors = sorted(
        vendor_counts.keys(), key=lambda x: vendor_counts[x], reverse=True
    )

    normalized_vendors[sorted_vendors[0]] = sorted_vendors[0]
    reference_list = [sorted_vendors[0]]

    for vendor in sorted_vendors[1:]:
        if pd.isna(vendor) or vendor == "":
            continue

        best_match, score, _ = process.extractOne(
            vendor, reference_list, scorer=fuzz.token_sort_ratio
        )

        if score >= threshold:
            normalized_vendors[vendor] = best_match
        else:
            normalized_vendors[vendor] = vendor
            reference_list.append(vendor)

    return normalized_vendors


def normalize_vendor_names(df):
    """Apply vendor name normalization to standardize company names across the dataset."""
    # Check if 'vendor_name' exists, otherwise try to find the correct vendor column
    vendor_col = "vendor_name"
    if vendor_col not in df.columns:
        possible_cols = [col for col in df.columns if "vendor" in col.lower()]
        if possible_cols:
            vendor_col = possible_cols[0]
            print(f"Using '{vendor_col}' instead of 'vendor_name'")
        else:
            raise ValueError("No vendor column found in the dataset")

    unique_vendors = df[vendor_col].unique()
    print(f"\nTotal unique vendor names before normalization: {len(unique_vendors)}")

    df["cleaned_vendor_name"] = df[vendor_col].apply(clean_vendor_name)

    unique_cleaned_vendors = df["cleaned_vendor_name"].unique()
    print(f"Unique vendor names after basic cleaning: {len(unique_cleaned_vendors)}")

    vendor_mapping = normalize_vendors(df["cleaned_vendor_name"].dropna().tolist())

    df["normalized_vendor_name"] = df["cleaned_vendor_name"].map(
        lambda x: vendor_mapping.get(x, x)
    )

    unique_normalized_vendors = df["normalized_vendor_name"].unique()
    print(f"Unique vendor names after normalization: {len(unique_normalized_vendors)}")

    top_vendors_before = df[vendor_col].value_counts().head(10)
    top_vendors_after = df["normalized_vendor_name"].value_counts().head(10)

    print("\nTop 10 vendors before normalization:")
    print(top_vendors_before)

    print("\nTop 10 vendors after normalization:")
    print(top_vendors_after)

    return df, unique_vendors, unique_normalized_vendors


def clean_data_types_and_missing_values(df):
    """Ensure appropriate data types and handle missing values in the dataset."""
    if df["fiscal_year"].dtype != "int64":
        df["fiscal_year"] = df["fiscal_year"].astype("int")

    df["awarded_amount"] = pd.to_numeric(df["awarded_amount"], errors="coerce")

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Column {col} has {missing_count} missing values.")

            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            elif pd.api.types.is_numeric_dtype(df[col]):
                if col == "awarded_amount":
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())

    print("\nMissing values after handling:")
    print(df.isnull().sum())

    return df


def analyze_program_spending(program_df):
    """Calculate and display total spending by program and fiscal year."""
    yearly_program_spend = (
        program_df.groupby(["fiscal_year", "program"])["awarded_amount"]
        .sum()
        .reset_index()
    )
    yearly_program_spend = yearly_program_spend.pivot(
        index="fiscal_year", columns="program", values="awarded_amount"
    )
    yearly_program_spend.fillna(0, inplace=True)

    print("Total spending by program and fiscal year (in millions of dollars):")
    formatted_df = yearly_program_spend.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"${x/1000000:.2f}M")
    print(formatted_df)

    total_by_program = (
        program_df.groupby("program")["awarded_amount"]
        .sum()
        .sort_values(ascending=False)
    )
    print("\nTotal spending by program (FY2016-2021):")
    for program, amount in total_by_program.items():
        print(f"{program}: ${amount:,.2f} (${amount/1000000:.2f}M)")

    return yearly_program_spend, total_by_program


def visualize_program_spending(yearly_program_spend, total_by_program, colors):
    """Create visualizations showing spending patterns by program over time."""
    fig, ax = plt.subplots(figsize=(12, 8))
    yearly_program_spend.plot(kind="bar", stacked=True, ax=ax, color=colors[:3])
    plt.title("Spending by Program and Fiscal Year", fontsize=15)
    plt.xlabel("Fiscal Year", fontsize=12)
    plt.ylabel("Total Award Amount ($)", fontsize=12)
    plt.xticks(rotation=0)

    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.yaxis.set_major_formatter(formatter)
    plt.legend(title="Program", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    total_by_program.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:3],
        ax=ax,
        textprops={"fontsize": 12},
    )
    plt.title("Share of Total Spending by Program (FY2016-2021)", fontsize=15)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    return None


def analyze_contract_trends(program_df):
    """Analyze trends in contract counts and average contract size over time."""
    contract_counts = (
        program_df.groupby(["fiscal_year", "program"]).size().unstack().fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    contract_counts.plot(kind="line", marker="o", ax=ax)
    plt.title("Number of Contracts by Program Over Time", fontsize=15)
    plt.xlabel("Fiscal Year", fontsize=12)
    plt.ylabel("Number of Contracts", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title="Program", fontsize=10)
    plt.tight_layout()
    plt.show()

    avg_contract_size = (
        program_df.groupby(["fiscal_year", "program"])["awarded_amount"]
        .mean()
        .unstack()
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_contract_size.plot(kind="line", marker="o", ax=ax)
    plt.title("Average Contract Size by Program Over Time", fontsize=15)
    plt.xlabel("Fiscal Year", fontsize=12)
    plt.ylabel("Average Award Amount ($)", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.yaxis.set_major_formatter(formatter)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title="Program", fontsize=10)
    plt.tight_layout()
    plt.show()

    return contract_counts, avg_contract_size


def analyze_contract_size_distribution(program_df, total_by_program):
    """Visualize the distribution of contract sizes within each program."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, program in enumerate(total_by_program.index):
        program_data = program_df[program_df["program"] == program]["awarded_amount"]
        sns.histplot(
            program_data[
                (program_data > 0) & (program_data < program_data.quantile(0.99))
            ],
            ax=axes[i],
            kde=True,
            bins=30,
        )
        axes[i].set_title(f"{program} Contract Size Distribution")
        axes[i].set_xlabel("Award Amount ($)")
        axes[i].set_ylabel("Frequency")
        formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
        axes[i].xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
    return None


def analyze_vendor_concentration(program_df):
    """Analyze vendor market share and concentration within each program."""
    vendor_spend = (
        program_df.groupby(["program", "normalized_vendor_name"])["awarded_amount"]
        .sum()
        .reset_index()
    )

    top_vendors_by_program = {}
    market_concentration = {}

    for program in vendor_spend["program"].unique():
        program_vendors = vendor_spend[vendor_spend["program"] == program].sort_values(
            "awarded_amount", ascending=False
        )
        top_vendors_by_program[program] = program_vendors.head(10)

        total_program_spend = program_vendors["awarded_amount"].sum()

        top_3_share = (
            program_vendors.head(3)["awarded_amount"].sum() / total_program_spend
        )
        top_5_share = (
            program_vendors.head(5)["awarded_amount"].sum() / total_program_spend
        )
        top_10_share = (
            program_vendors.head(10)["awarded_amount"].sum() / total_program_spend
        )
        hhi = sum(
            (vendor_amount / total_program_spend) ** 2
            for vendor_amount in program_vendors["awarded_amount"]
        )

        market_concentration[program] = {
            "Total Vendors": len(program_vendors),
            "Top 3 Vendor Share": top_3_share,
            "Top 5 Vendor Share": top_5_share,
            "Top 10 Vendor Share": top_10_share,
            "HHI Index": hhi,
        }

    for program, vendors in top_vendors_by_program.items():
        print(f"\nTop 10 vendors for {program} program:")
        formatted_vendors = vendors.set_index("normalized_vendor_name")[
            "awarded_amount"
        ].apply(lambda x: f"${x:,.2f}")
        print(formatted_vendors)

    print("\nMarket Concentration Metrics:")
    concentration_df = pd.DataFrame(market_concentration).T
    concentration_df["Top 3 Vendor Share"] = concentration_df["Top 3 Vendor Share"].map(
        lambda x: f"{x:.2%}"
    )
    concentration_df["Top 5 Vendor Share"] = concentration_df["Top 5 Vendor Share"].map(
        lambda x: f"{x:.2%}"
    )
    concentration_df["Top 10 Vendor Share"] = concentration_df[
        "Top 10 Vendor Share"
    ].map(lambda x: f"{x:.2%}")
    concentration_df["HHI Index"] = concentration_df["HHI Index"].map(
        lambda x: f"{x:.4f}"
    )
    print(concentration_df)

    return top_vendors_by_program, market_concentration


def analyze_cross_program_vendors(program_df):
    """Identify vendors that work across multiple ground combat vehicle programs."""
    cross_program_vendors = (
        program_df.groupby("normalized_vendor_name")["program"]
        .nunique()
        .reset_index()
        .rename(columns={"program": "program_count"})
    )
    cross_program_vendors = cross_program_vendors[
        cross_program_vendors["program_count"] > 1
    ]

    vendor_program_matrix = program_df.pivot_table(
        index="normalized_vendor_name",
        columns="program",
        values="awarded_amount",
        aggfunc="sum",
        fill_value=0,
    )

    cross_program_amount = vendor_program_matrix.loc[
        cross_program_vendors["normalized_vendor_name"]
    ].sum(axis=1)
    top_cross_program_vendors = cross_program_vendors[
        cross_program_vendors["normalized_vendor_name"].isin(
            cross_program_amount[cross_program_amount > 1000000].index
        )
    ]

    if not top_cross_program_vendors.empty:
        print(f"\nVendors working across multiple programs (min $1M total awards):")
        top_vendors_matrix = vendor_program_matrix.loc[
            top_cross_program_vendors["normalized_vendor_name"]
        ]

        formatted_matrix = top_vendors_matrix.copy()
        for col in formatted_matrix.columns:
            formatted_matrix[col] = formatted_matrix[col].apply(
                lambda x: f"${x:,.2f}" if x > 0 else "-"
            )
        print(formatted_matrix)

    return cross_program_vendors, top_cross_program_vendors


def visualize_vendor_concentration(
    top_vendors_by_program, market_concentration, colors
):
    """Create visualizations showing vendor market share and concentration."""
    fig, axes = plt.subplots(len(top_vendors_by_program), 1, figsize=(12, 15))
    for i, (program, data) in enumerate(top_vendors_by_program.items()):
        data_plot = data.set_index("normalized_vendor_name")[
            "awarded_amount"
        ].sort_values()
        data_plot.plot(kind="barh", ax=axes[i], color=colors[i])
        axes[i].set_title(f"Top 10 Vendors for {program}")
        axes[i].set_xlabel("Award Amount ($)")
        formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
        axes[i].xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()

    vendor_share_fig, axes = plt.subplots(1, len(market_concentration), figsize=(18, 6))
    for i, (program, metrics) in enumerate(market_concentration.items()):
        sizes = [
            metrics["Top 3 Vendor Share"],
            metrics["Top 5 Vendor Share"] - metrics["Top 3 Vendor Share"],
            metrics["Top 10 Vendor Share"] - metrics["Top 5 Vendor Share"],
            1 - metrics["Top 10 Vendor Share"],
        ]
        labels = ["Top 3 Vendors", "Vendors 4-5", "Vendors 6-10", "All Other Vendors"]
        axes[i].pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=sns.color_palette("YlOrRd", 4),
        )
        axes[i].set_title(f"Market Share Distribution - {program}")
    plt.tight_layout()
    plt.show()

    return None


def analyze_geographic_distribution(program_df):
    """Analyze the geographic distribution of contracts by state."""
    geo_df = program_df[program_df["place_of_performance_state"] != "Unknown"].copy()

    state_spend = (
        geo_df.groupby(["place_of_performance_state", "program"])["awarded_amount"]
        .sum()
        .reset_index()
    )
    state_pivot = state_spend.pivot(
        index="place_of_performance_state", columns="program", values="awarded_amount"
    )
    state_pivot.fillna(0, inplace=True)
    state_pivot["Total"] = state_pivot.sum(axis=1)
    state_pivot_sorted = state_pivot.sort_values("Total", ascending=False)

    print("Top 10 states by total contract value (across all programs):")
    formatted_state_pivot = state_pivot_sorted.head(10).copy()
    for col in formatted_state_pivot.columns:
        formatted_state_pivot[col] = formatted_state_pivot[col].apply(
            lambda x: f"${x/1000000:.2f}M"
        )
    print(formatted_state_pivot)

    state_program_counts = (
        geo_df.groupby(["place_of_performance_state", "program"])
        .size()
        .unstack(fill_value=0)
    )
    state_program_counts["Total"] = state_program_counts.sum(axis=1)
    state_counts_sorted = state_program_counts.sort_values("Total", ascending=False)

    print("\nTop 10 states by contract count (across all programs):")
    print(state_counts_sorted.head(10))

    return geo_df, state_pivot_sorted, state_counts_sorted


def visualize_geographic_distribution(state_pivot_sorted, colors):
    """Create visualizations showing the geographic distribution of contract spending."""
    fig, ax = plt.subplots(figsize=(14, 8))
    state_pivot_sorted.head(10)["Total"].plot(kind="bar", ax=ax, color="darkblue")
    plt.title("Top 10 States by Total Award Amount", fontsize=15)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Total Award Amount ($)", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 8))
    top_states_data = state_pivot_sorted.head(10).drop("Total", axis=1)
    top_states_data.plot(kind="bar", stacked=True, ax=ax, color=colors[:3])
    plt.title("Program Distribution in Top 10 States by Spending", fontsize=15)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Award Amount ($)", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.legend(title="Program", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return None


def analyze_city_distribution(geo_df):
    """Analyze contract distribution at the city level."""
    city_state_spend = (
        geo_df.groupby(
            ["place_of_performance_state", "place_of_performance_city", "program"]
        )["awarded_amount"]
        .sum()
        .reset_index()
    )
    city_state_spend["location"] = (
        city_state_spend["place_of_performance_city"]
        + ", "
        + city_state_spend["place_of_performance_state"]
    )
    city_spend = (
        city_state_spend.groupby(["location", "program"])["awarded_amount"]
        .sum()
        .reset_index()
    )
    city_pivot = city_spend.pivot(
        index="location", columns="program", values="awarded_amount"
    )
    city_pivot.fillna(0, inplace=True)
    city_pivot["Total"] = city_pivot.sum(axis=1)
    city_pivot_sorted = city_pivot.sort_values("Total", ascending=False)

    print("\nTop 10 cities by total contract value (across all programs):")
    formatted_city_pivot = city_pivot_sorted.head(10).copy()
    for col in formatted_city_pivot.columns:
        formatted_city_pivot[col] = formatted_city_pivot[col].apply(
            lambda x: f"${x/1000000:.2f}M"
        )
    print(formatted_city_pivot)

    return city_pivot_sorted


def visualize_city_distribution(city_pivot_sorted, colors):
    """Create visualizations showing contract spending distribution across cities."""
    fig, ax = plt.subplots(figsize=(14, 8))
    top_cities_data = city_pivot_sorted.head(10).drop("Total", axis=1)
    top_cities_data.plot(kind="bar", stacked=True, ax=ax, color=colors[:3])
    plt.title("Program Distribution in Top 10 Cities by Spending", fontsize=15)
    plt.xlabel("City, State", fontsize=12)
    plt.ylabel("Award Amount ($)", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.legend(title="Program", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    return None


def analyze_contracting_offices(program_df):
    """Analyze spending patterns by contracting office and agency."""
    office_spend = (
        program_df.groupby(["contracting_office", "program"])["awarded_amount"]
        .sum()
        .reset_index()
    )
    office_pivot = office_spend.pivot(
        index="contracting_office", columns="program", values="awarded_amount"
    )
    office_pivot.fillna(0, inplace=True)
    office_pivot["Total"] = office_pivot.sum(axis=1)
    office_pivot_sorted = office_pivot.sort_values("Total", ascending=False)

    print("Top 10 contracting offices by total contract value:")
    formatted_office_pivot = office_pivot_sorted.head(10).copy()
    for col in formatted_office_pivot.columns:
        formatted_office_pivot[col] = formatted_office_pivot[col].apply(
            lambda x: f"${x/1000000:.2f}M"
        )
    print(formatted_office_pivot)

    agency_spend = (
        program_df.groupby(["contracting_agency", "program"])["awarded_amount"]
        .sum()
        .reset_index()
    )
    agency_pivot = agency_spend.pivot(
        index="contracting_agency", columns="program", values="awarded_amount"
    )
    agency_pivot.fillna(0, inplace=True)
    agency_pivot["Total"] = agency_pivot.sum(axis=1)
    agency_pivot_sorted = agency_pivot.sort_values("Total", ascending=False)

    print("\nContracting agencies by total contract value:")
    formatted_agency_pivot = agency_pivot_sorted.copy()
    for col in formatted_agency_pivot.columns:
        formatted_agency_pivot[col] = formatted_agency_pivot[col].apply(
            lambda x: f"${x/1000000:.2f}M"
        )
    print(formatted_agency_pivot)

    return office_pivot_sorted, agency_pivot_sorted


def visualize_contracting_offices(office_pivot_sorted):
    """Create visualizations for contracting office analysis."""
    fig, ax = plt.subplots(figsize=(14, 10))
    office_pivot_sorted.head(10)["Total"].plot(kind="barh", ax=ax, color="darkgreen")
    plt.title("Top 10 Contracting Offices by Total Award Amount", fontsize=15)
    plt.xlabel("Total Award Amount ($)", fontsize=12)
    plt.ylabel("Contracting Office", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    ax.xaxis.set_major_formatter(formatter)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    return None


def analyze_office_specialization(program_df):
    """Analyze the degree of specialization among contracting offices."""
    program_office_count = program_df.groupby("contracting_office")["program"].nunique()
    office_contract_count = program_df.groupby("contracting_office").size()
    office_program_specialist = program_office_count[program_office_count == 1].index
    office_program_generalist = program_office_count[program_office_count > 1].index

    office_metrics = pd.DataFrame(
        {
            "Contract Count": office_contract_count,
            "Programs Managed": program_office_count,
            "Total Spending": program_df.groupby("contracting_office")[
                "awarded_amount"
            ].sum(),
        }
    )
    office_metrics["Average Contract Size"] = (
        office_metrics["Total Spending"] / office_metrics["Contract Count"]
    )
    office_metrics["Specialized"] = office_metrics.index.isin(office_program_specialist)
    office_metrics_sorted = office_metrics.sort_values(
        "Total Spending", ascending=False
    )

    print("\nContracting office metrics (top 10 by spending):")
    metrics_display = office_metrics_sorted.head(10).copy()
    metrics_display["Total Spending"] = metrics_display["Total Spending"].apply(
        lambda x: f"${x/1000000:.2f}M"
    )
    metrics_display["Average Contract Size"] = metrics_display[
        "Average Contract Size"
    ].apply(lambda x: f"${x:,.2f}")
    print(metrics_display)

    print(
        f"\nSpecialized offices (managing only one program): {len(office_program_specialist)}"
    )
    print(
        f"Generalist offices (managing multiple programs): {len(office_program_generalist)}"
    )

    return office_metrics, office_program_specialist, office_program_generalist


def visualize_office_specialization(office_metrics):
    """Create visualizations showing contracting office specialization patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    specialist_spend = office_metrics[office_metrics["Specialized"] == True][
        "Total Spending"
    ].sum()
    generalist_spend = office_metrics[office_metrics["Specialized"] == False][
        "Total Spending"
    ].sum()

    axes[0].pie(
        [specialist_spend, generalist_spend],
        labels=["Specialized Offices", "Generalist Offices"],
        autopct="%1.1f%%",
        colors=["#66c2a5", "#fc8d62"],
    )
    axes[0].set_title(
        "Share of Spending: Specialized vs. Generalist Offices", fontsize=14
    )

    avg_contract_by_office_type = [
        office_metrics[office_metrics["Specialized"] == True][
            "Average Contract Size"
        ].mean(),
        office_metrics[office_metrics["Specialized"] == False][
            "Average Contract Size"
        ].mean(),
    ]

    axes[1].bar(
        ["Specialized Offices", "Generalist Offices"],
        avg_contract_by_office_type,
        color=["#66c2a5", "#fc8d62"],
    )
    axes[1].set_title(
        "Average Contract Size: Specialized vs. Generalist Offices", fontsize=14
    )
    axes[1].set_ylabel("Average Contract Size ($)")
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000:,.0f}K")
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
    return None


def analyze_contract_modifications(program_df):
    """Analyze contract modification rates by program."""
    program_df["has_modification"] = ~pd.isna(program_df["modification_number"]) & (
        program_df["modification_number"] != "Unknown"
    )

    modification_rate_by_program = program_df.groupby("program")[
        "has_modification"
    ].mean()

    print("\nContract modification rates by program:")
    for program, rate in modification_rate_by_program.items():
        print(f"{program}: {rate:.2%} of contracts have modifications")

    return modification_rate_by_program


def visualize_contract_modifications(modification_rate_by_program, colors):
    """Create visualizations for contract modification analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    modification_rate_by_program.plot(kind="bar", ax=ax, color=colors[:3])
    plt.title("Contract Modification Rate by Program", fontsize=15)
    plt.xlabel("Program", fontsize=12)
    plt.ylabel("Percentage of Contracts with Modifications", fontsize=12)
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    return None


def generate_analysis_summary(
    df,
    program_df,
    total_by_program,
    market_concentration,
    cross_program_vendors,
    state_pivot_sorted,
    office_program_specialist,
    office_program_generalist,
    modification_rate_by_program,
):
    """Generate a comprehensive summary of key metrics from the analysis."""
    years = df["fiscal_year"].unique()
    years.sort()
    years_str = ", ".join([f"FY{year}" for year in years])

    print(f"Summary of Ground Combat Systems Contract Analysis ({years_str})")
    print("=" * 70)

    total_spend = program_df["awarded_amount"].sum()
    print(f"\nTotal Spending: ${total_spend:,.2f} (${total_spend/1000000:.2f}M)")

    for program, amount in total_by_program.items():
        pct = amount / total_spend * 100
        print(f"  - {program}: ${amount:,.2f} (${amount/1000000:.2f}M, {pct:.1f}%)")

    print("\nVendor Concentration:")
    for program, metrics in market_concentration.items():
        print(
            f"  - {program}: {metrics['Total Vendors']} vendors, top 3 account for {metrics['Top 3 Vendor Share']}"
        )

    cross_program_vendor_count = len(cross_program_vendors)
    print(f"\nVendors working across multiple programs: {cross_program_vendor_count}")

    top_states = state_pivot_sorted.head(5).index.tolist()
    print("\nTop 5 states by award amount:")
    for state in top_states:
        amount = state_pivot_sorted.loc[state, "Total"]
        pct = amount / total_spend * 100
        print(f"  - {state}: ${amount:,.2f} (${amount/1000000:.2f}M, {pct:.1f}%)")

    print("\nContracting Office Analysis:")
    print(f"  - Specialized offices (one program): {len(office_program_specialist)}")
    print(
        f"  - Generalist offices (multiple programs): {len(office_program_generalist)}"
    )

    print("\nContract Modification Rates:")
    for program, rate in modification_rate_by_program.items():
        print(f"  - {program}: {rate:.1%}")

    avg_contract_sizes = program_df.groupby("program")["awarded_amount"].mean()
    print("\nAverage Contract Size by Program:")
    for program, avg in avg_contract_sizes.items():
        print(f"  - {program}: ${avg:,.2f}")

    print("\nKey Insights:")
    print("  1. [First key insight based on analysis]")
    print("  2. [Second key insight based on analysis]")
    print("  3. [Third key insight based on analysis]")

    return None


def analyze_psc_codes(program_df):
    """Analyze Product Service Codes distribution by program."""
    psc_spend = (
        program_df.groupby(["program", "psc_name"])["awarded_amount"]
        .sum()
        .reset_index()
    )

    top_psc_by_program = {}
    psc_diversity = {}

    for program in psc_spend["program"].unique():
        if program == "Unknown":
            continue

        program_psc = psc_spend[psc_spend["program"] == program].sort_values(
            "awarded_amount", ascending=False
        )
        total_program_spend = program_psc["awarded_amount"].sum()

        top_5_psc_share = (
            program_psc.head(5)["awarded_amount"].sum() / total_program_spend
        )
        psc_count = len(program_psc)

        top_psc_by_program[program] = program_psc.head(10)
        psc_diversity[program] = {
            "PSC Count": psc_count,
            "Top 5 PSC Share": top_5_psc_share,
        }

    print("PSC Diversity Metrics:")
    diversity_df = pd.DataFrame(psc_diversity).T
    diversity_df["Top 5 PSC Share"] = diversity_df["Top 5 PSC Share"].map(
        lambda x: f"{x:.2%}"
    )
    print(diversity_df)

    for program, psc_data in top_psc_by_program.items():
        print(f"\nTop 10 PSC codes for {program} program:")
        formatted_psc = psc_data.set_index("psc_name")["awarded_amount"].apply(
            lambda x: f"${x:,.2f}"
        )
        print(formatted_psc)

    return top_psc_by_program, psc_diversity


def extract_key_technologies(program_df):
    """Extract and analyze key technologies mentioned in contract titles."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    def preprocess_text(text):
        if pd.isna(text) or text == "Unknown":
            return []

        text = str(text).lower()

        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 3]

        return tokens

    program_technologies = {}
    program_bigrams = {}

    for program in program_df["program"].unique():
        if program == "Unknown":
            continue

        program_data = program_df[program_df["program"] == program]
        program_titles = program_data["title"].tolist()

        all_tokens = []
        for title in program_titles:
            all_tokens.extend(preprocess_text(title))

        token_freq = Counter(all_tokens)

        bigrams = []
        for title in program_titles:
            tokens = preprocess_text(title)
            if len(tokens) > 1:
                bigrams.extend(list(zip(tokens, tokens[1:])))

        bigram_freq = Counter(bigrams)

        program_technologies[program] = token_freq
        program_bigrams[program] = bigram_freq

    for program, tech_freq in program_technologies.items():
        print(f"\nTop 15 technology terms for {program} program:")
        for term, count in tech_freq.most_common(15):
            print(f"{term}: {count}")

    print("\nTop 10 technology term pairs (bigrams) by program:")
    for program, bg_freq in program_bigrams.items():
        print(f"\n{program}:")
        for bg, count in bg_freq.most_common(10):
            print(f"{bg[0]} {bg[1]}: {count}")

    return program_technologies, program_bigrams


def identify_program_lifecycle_status(program_df):
    """Identify the lifecycle status of each program based on contract patterns."""
    lifecycle_indicators = {}

    for program in sorted(program_df["program"].unique()):
        if program == "Unknown":
            continue

        program_data = program_df[program_df["program"] == program]

        yearly_spend = program_data.groupby("fiscal_year")["awarded_amount"].sum()

        if len(yearly_spend) >= 3:
            recent_years = sorted(yearly_spend.index)[-3:]
            first_year_spend = yearly_spend[recent_years[0]]
            last_year_spend = yearly_spend[recent_years[-1]]

            if last_year_spend > first_year_spend * 1.2:
                spend_trend = "Increasing"
            elif last_year_spend < first_year_spend * 0.8:
                spend_trend = "Decreasing"
            else:
                spend_trend = "Stable"
        else:
            spend_trend = "Insufficient data"

        yearly_avg_size = program_data.groupby("fiscal_year")["awarded_amount"].mean()
        if len(yearly_avg_size) >= 3:
            recent_years = sorted(yearly_avg_size.index)[-3:]
            first_year_avg = yearly_avg_size[recent_years[0]]
            last_year_avg = yearly_avg_size[recent_years[-1]]

            if last_year_avg > first_year_avg * 1.2:
                size_trend = "Increasing"
            elif last_year_avg < first_year_avg * 0.8:
                size_trend = "Decreasing"
            else:
                size_trend = "Stable"
        else:
            size_trend = "Insufficient data"

        psc_categories = {
            "Research": ["research", "development", "test", "evaluation"],
            "Production": [
                "production",
                "manufacturing",
                "construction",
                "procurement",
            ],
            "Maintenance": ["maintenance", "repair", "overhaul", "support", "spare"],
        }

        psc_spend = {}
        for category, keywords in psc_categories.items():
            mask = (
                program_data["psc_name"]
                .str.lower()
                .str.contains("|".join(keywords), na=False)
            )
            psc_spend[category] = program_data.loc[mask, "awarded_amount"].sum()

        total_categorized = sum(psc_spend.values())
        if total_categorized > 0:
            psc_share = {k: v / total_categorized for k, v in psc_spend.items()}
            primary_psc = max(psc_share, key=psc_share.get)
            primary_psc_share = psc_share[primary_psc]
        else:
            primary_psc = "Unknown"
            primary_psc_share = 0

        if primary_psc == "Research" and spend_trend in ["Increasing", "Stable"]:
            lifecycle_phase = "Early (Research & Development)"
        elif primary_psc == "Production" and spend_trend == "Increasing":
            lifecycle_phase = "Growth (Production Ramp-up)"
        elif primary_psc == "Production" and spend_trend == "Stable":
            lifecycle_phase = "Mature (Full Production)"
        elif primary_psc == "Maintenance" or spend_trend == "Decreasing":
            lifecycle_phase = "Late (Sustainment & Modernization)"
        else:
            lifecycle_phase = "Indeterminate"

        lifecycle_indicators[program] = {
            "Spend Trend": spend_trend,
            "Contract Size Trend": size_trend,
            "Primary PSC Category": primary_psc,
            "Primary PSC Share": primary_psc_share,
            "Lifecycle Phase": lifecycle_phase,
        }

    lifecycle_df = pd.DataFrame(lifecycle_indicators).T
    lifecycle_df["Primary PSC Share"] = lifecycle_df["Primary PSC Share"].apply(
        lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x
    )

    print("Program Lifecycle Assessment:")
    print(lifecycle_df)

    return lifecycle_df


def project_future_acquisitions(program_df, lifecycle_df):
    """Project future acquisition patterns based on program lifecycle status."""
    future_projections = {}

    for program, lifecycle in lifecycle_df.iterrows():
        current_phase = lifecycle["Lifecycle Phase"]
        spend_trend = lifecycle["Spend Trend"]

        if current_phase == "Early (Research & Development)":
            if spend_trend == "Increasing":
                projected_trend = "Continued growth transitioning to production"
                spending_outlook = "Significant increase expected"
                acquisition_focus = (
                    "Prototype development, testing, and initial production"
                )
            else:
                projected_trend = "Potential transition to production"
                spending_outlook = "Moderate increase likely"
                acquisition_focus = "Testing and low-rate initial production"

        elif current_phase == "Growth (Production Ramp-up)":
            projected_trend = "Continued high volume production"
            spending_outlook = "Sustained high spending"
            acquisition_focus = "Full-rate production and support systems"

        elif current_phase == "Mature (Full Production)":
            projected_trend = "Stable production with potential modernization"
            spending_outlook = "Stable to slight decrease"
            acquisition_focus = "Production, spares, and incremental upgrades"

        elif current_phase == "Late (Sustainment & Modernization)":
            projected_trend = "Declining production, increasing maintenance"
            spending_outlook = "Gradual decrease with maintenance floor"
            acquisition_focus = "Spares, maintenance, and targeted modernization"

        else:
            projected_trend = "Indeterminate based on available data"
            spending_outlook = "Uncertain"
            acquisition_focus = "Depends on program decisions"

        future_projections[program] = {
            "Projected Trend": projected_trend,
            "Spending Outlook": spending_outlook,
            "Acquisition Focus": acquisition_focus,
        }

    projection_df = pd.DataFrame(future_projections).T
    print("\nFuture Acquisition Projections:")
    print(projection_df)

    return projection_df


def analyze_vendor_dependence_risk(program_df, market_concentration):
    """Analyze vendor dependence risks across programs."""
    vendor_risk_metrics = {}

    for program in program_df["program"].unique():
        if program == "Unknown":
            continue

        program_data = program_df[program_df["program"] == program]
        vendor_spend = (
            program_data.groupby("normalized_vendor_name")["awarded_amount"]
            .sum()
            .sort_values(ascending=False)
        )

        total_program_spend = vendor_spend.sum()

        if total_program_spend == 0:
            continue

        top_vendor = vendor_spend.index[0] if not vendor_spend.empty else "None"
        top_vendor_share = (
            vendor_spend.iloc[0] / total_program_spend if not vendor_spend.empty else 0
        )

        hhi = sum((amount / total_program_spend) ** 2 for amount in vendor_spend)

        significant_vendors = vendor_spend[
            vendor_spend > total_program_spend * 0.01
        ].count()

        if top_vendor_share > 0.7:
            vendor_concentration_risk = "Very High"
        elif top_vendor_share > 0.5:
            vendor_concentration_risk = "High"
        elif top_vendor_share > 0.3:
            vendor_concentration_risk = "Moderate"
        else:
            vendor_concentration_risk = "Low"

        if hhi > 0.5:
            market_structure_risk = "Very High (Near Monopoly)"
        elif hhi > 0.25:
            market_structure_risk = "High (Highly Concentrated)"
        elif hhi > 0.15:
            market_structure_risk = "Moderate (Moderately Concentrated)"
        else:
            market_structure_risk = "Low (Competitive Market)"

        vendor_risk_metrics[program] = {
            "Top Vendor": top_vendor,
            "Top Vendor Share": top_vendor_share,
            "HHI Index": hhi,
            "Significant Vendors": significant_vendors,
            "Vendor Concentration Risk": vendor_concentration_risk,
            "Market Structure Risk": market_structure_risk,
        }

    risk_df = pd.DataFrame(vendor_risk_metrics).T

    risk_df["Top Vendor Share"] = risk_df["Top Vendor Share"].apply(
        lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x
    )
    risk_df["HHI Index"] = risk_df["HHI Index"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
    )

    print("Vendor Dependence Risk Analysis:")
    print(risk_df)

    return risk_df


def analyze_budget_consistency_risk(program_df):
    """Analyze budget consistency and predictability risks."""
    budget_risk_metrics = {}

    for program in program_df["program"].unique():
        if program == "Unknown":
            continue

        program_data = program_df[program_df["program"] == program]
        yearly_spend = program_data.groupby("fiscal_year")["awarded_amount"].sum()

        if len(yearly_spend) < 3:
            budget_risk_metrics[program] = {
                "Year-to-Year Volatility": "Insufficient data",
                "Trend Consistency": "Insufficient data",
                "Budget Risk Assessment": "Indeterminate",
            }
            continue

        pct_changes = yearly_spend.pct_change().dropna().abs()
        avg_volatility = pct_changes.mean() if not pct_changes.empty else 0

        years = sorted(yearly_spend.index)
        is_consistent_trend = True
        is_increasing = yearly_spend[years[1]] > yearly_spend[years[0]]

        for i in range(2, len(years)):
            current_direction = yearly_spend[years[i]] > yearly_spend[years[i - 1]]
            if current_direction != is_increasing:
                is_consistent_trend = False
                break

        has_outliers = any(pct_changes > 0.5)

        if avg_volatility > 0.3:
            volatility_risk = "High"
        elif avg_volatility > 0.15:
            volatility_risk = "Moderate"
        else:
            volatility_risk = "Low"

        if not is_consistent_trend and has_outliers:
            consistency_risk = "High"
        elif not is_consistent_trend or has_outliers:
            consistency_risk = "Moderate"
        else:
            consistency_risk = "Low"

        if volatility_risk == "High" or consistency_risk == "High":
            overall_risk = "High"
        elif volatility_risk == "Moderate" or consistency_risk == "Moderate":
            overall_risk = "Moderate"
        else:
            overall_risk = "Low"

        budget_risk_metrics[program] = {
            "Year-to-Year Volatility": (
                f"{avg_volatility:.2%}"
                if isinstance(avg_volatility, (int, float))
                else avg_volatility
            ),
            "Consistent Trend": "Yes" if is_consistent_trend else "No",
            "Has Significant Outliers": "Yes" if has_outliers else "No",
            "Volatility Risk": volatility_risk,
            "Consistency Risk": consistency_risk,
            "Overall Budget Risk": overall_risk,
        }

    budget_risk_df = pd.DataFrame(budget_risk_metrics).T

    print("\nBudget Consistency Risk Analysis:")
    print(budget_risk_df)

    return budget_risk_df


def analyze_supply_chain_risk(program_df):
    """Analyze potential supply chain vulnerabilities."""
    supply_chain_risks = {}

    for program in program_df["program"].unique():
        if program == "Unknown":
            continue

        program_data = program_df[program_df["program"] == program]

        state_spend = program_data.groupby("place_of_performance_state")[
            "awarded_amount"
        ].sum()
        total_spend = state_spend.sum()

        if total_spend == 0:
            continue

        top_state = state_spend.idxmax() if not state_spend.empty else "Unknown"
        top_state_share = (
            state_spend.max() / total_spend if not state_spend.empty else 0
        )

        significant_states = state_spend[state_spend > total_spend * 0.05].count()

        geo_hhi = sum((amount / total_spend) ** 2 for amount in state_spend)

        vendor_count = program_data["normalized_vendor_name"].nunique()
        avg_vendors_per_state = (
            program_data.groupby("place_of_performance_state")["normalized_vendor_name"]
            .nunique()
            .mean()
        )

        if top_state_share > 0.7:
            geo_concentration_risk = "Very High"
        elif top_state_share > 0.5:
            geo_concentration_risk = "High"
        elif top_state_share > 0.3:
            geo_concentration_risk = "Moderate"
        else:
            geo_concentration_risk = "Low"

        if vendor_count <= 5:
            vendor_diversity_risk = "High"
        elif vendor_count <= 15:
            vendor_diversity_risk = "Moderate"
        else:
            vendor_diversity_risk = "Low"

        if geo_concentration_risk in [
            "Very High",
            "High",
        ] and vendor_diversity_risk in ["High", "Moderate"]:
            overall_supply_chain_risk = "High"
        elif (
            geo_concentration_risk == "Moderate" or vendor_diversity_risk == "Moderate"
        ):
            overall_supply_chain_risk = "Moderate"
        else:
            overall_supply_chain_risk = "Low"

        supply_chain_risks[program] = {
            "Top State": top_state,
            "Top State Share": top_state_share,
            "Significant States": significant_states,
            "Geographic HHI": geo_hhi,
            "Vendor Count": vendor_count,
            "Avg Vendors per State": avg_vendors_per_state,
            "Geographic Concentration Risk": geo_concentration_risk,
            "Vendor Diversity Risk": vendor_diversity_risk,
            "Overall Supply Chain Risk": overall_supply_chain_risk,
        }

    supply_chain_df = pd.DataFrame(supply_chain_risks).T

    supply_chain_df["Top State Share"] = supply_chain_df["Top State Share"].apply(
        lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x
    )
    supply_chain_df["Geographic HHI"] = supply_chain_df["Geographic HHI"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
    )
    supply_chain_df["Avg Vendors per State"] = supply_chain_df[
        "Avg Vendors per State"
    ].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)

    print("\nSupply Chain Risk Analysis:")
    print(supply_chain_df)

    return supply_chain_df


def visualize_risk_assessment(vendor_risk_df, budget_risk_df, supply_chain_df):
    """Create visualizations for risk assessment."""
    risk_levels = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

    consolidated_risk = pd.DataFrame(index=vendor_risk_df.index)

    consolidated_risk["Vendor Concentration Risk"] = vendor_risk_df[
        "Vendor Concentration Risk"
    ].map(lambda x: risk_levels.get(x, 0) if isinstance(x, str) else 0)

    consolidated_risk["Budget Volatility Risk"] = budget_risk_df[
        "Overall Budget Risk"
    ].map(lambda x: risk_levels.get(x, 0) if isinstance(x, str) else 0)

    consolidated_risk["Supply Chain Risk"] = supply_chain_df[
        "Overall Supply Chain Risk"
    ].map(lambda x: risk_levels.get(x, 0) if isinstance(x, str) else 0)

    consolidated_risk["Overall Risk Score"] = consolidated_risk.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    consolidated_risk["Overall Risk Score"].sort_values(ascending=False).plot(
        kind="bar", ax=ax, color="darkred"
    )
    plt.title("Overall Program Risk Assessment", fontsize=15)
    plt.xlabel("Program", fontsize=12)
    plt.ylabel("Risk Score (1=Low, 4=Very High)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax.axhline(y=3, color="red", linestyle="--", alpha=0.7, label="High Risk Threshold")
    ax.axhline(
        y=2, color="orange", linestyle="--", alpha=0.7, label="Moderate Risk Threshold"
    )
    ax.axhline(
        y=1, color="green", linestyle="--", alpha=0.7, label="Low Risk Threshold"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return consolidated_risk


def visualize_psc_analysis(top_psc_by_program, colors):
    """Create visualizations for Product Service Code analysis."""
    plt.figure(figsize=(12, 15))
    for i, (program, data) in enumerate(top_psc_by_program.items()):
        plt.subplot(len(top_psc_by_program), 1, i + 1)
        data = data.copy()
        data["psc_short_name"] = data["psc_name"].str.slice(0, 40) + "..."
        data_plot = (
            data.set_index("psc_short_name")["awarded_amount"].sort_values().tail(10)
        )
        data_plot.plot(kind="barh", color=colors[i])
        plt.title(f"Top 10 Product Service Codes for {program}")
        plt.xlabel("Award Amount ($)")
        formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
        plt.gca().xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()
    return None


def visualize_lifecycle_assessment(lifecycle_df):
    """Create visualization showing program lifecycle assessment."""
    plt.figure(figsize=(12, 6))
    lifecycle_order = {
        "Early (Research & Development)": 1,
        "Growth (Production Ramp-up)": 2,
        "Mature (Full Production)": 3,
        "Late (Sustainment & Modernization)": 4,
        "Indeterminate": 5,
    }
    phase_colors = {
        "Early (Research & Development)": "lightgreen",
        "Growth (Production Ramp-up)": "limegreen",
        "Mature (Full Production)": "darkgreen",
        "Late (Sustainment & Modernization)": "orange",
        "Indeterminate": "gray",
    }
    sorted_programs = lifecycle_df.sort_values(
        by="Lifecycle Phase", key=lambda x: x.map(lifecycle_order)
    ).index

    for i, program in enumerate(sorted_programs):
        phase = lifecycle_df.loc[program, "Lifecycle Phase"]
        plt.scatter(
            lifecycle_order[phase],
            i,
            s=300,
            color=phase_colors[phase],
            edgecolor="black",
        )
        plt.text(
            lifecycle_order[phase],
            i,
            program,
            va="center",
            ha="center",
            fontweight="bold",
        )

    plt.yticks([])
    plt.xticks(list(lifecycle_order.values()))
    plt.gca().set_xticklabels(list(lifecycle_order.keys()), rotation=45, ha="right")
    plt.xlim(0.5, 5.5)
    plt.ylim(-0.5, len(sorted_programs) - 0.5)
    plt.title("Program Lifecycle Assessment", fontsize=15)

    plt.tight_layout()
    plt.show()
    return sorted_programs


def visualize_spending_trajectories(program_df, sorted_programs, lifecycle_df):
    """Create visualizations showing program spending trajectories over time."""
    plt.figure(figsize=(12, 6))
    for program in sorted_programs:
        program_data = program_df[program_df["program"] == program]
        yearly_spend = program_data.groupby("fiscal_year")["awarded_amount"].sum()
        phase = lifecycle_df.loc[program, "Lifecycle Phase"]
        plt.plot(
            yearly_spend.index,
            yearly_spend.values,
            marker="o",
            linewidth=2,
            label=f"{program} ({phase})",
        )

    plt.title("Program Spending Trajectories", fontsize=15)
    plt.xlabel("Fiscal Year", fontsize=12)
    plt.ylabel("Total Spending ($)", fontsize=12)
    formatter = mtick.FuncFormatter(lambda x, p: f"${x/1000000:.1f}M")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Programs")

    plt.tight_layout()
    plt.show()
    return None


def visualize_risk_radar(consolidated_risk):
    """Create radar chart visualizations for program risk assessment."""
    plt.figure(figsize=(15, 5))
    risk_metrics = [
        "Vendor Concentration Risk",
        "Budget Volatility Risk",
        "Supply Chain Risk",
    ]
    N = len(risk_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for i, program in enumerate(consolidated_risk.index):
        plt.subplot(1, 3, i + 1, polar=True)
        values = consolidated_risk.loc[program, risk_metrics].tolist()
        values += values[:1]
        plt.plot(angles, values, "o-", linewidth=2, label=program)
        plt.fill(angles, values, alpha=0.25)
        plt.xticks(angles[:-1], risk_metrics)
        plt.ylim(0, 4)
        plt.title(
            f"{program}\nOverall Risk: {consolidated_risk.loc[program, 'Overall Risk Score']:.1f}"
        )

    plt.tight_layout()
    plt.show()
    return None


def run_technology_analysis(program_df, colors):
    """Run the complete technology analysis workflow."""
    print("\n### Technology and Product Analysis ###\n")
    top_psc_by_program, psc_diversity = analyze_psc_codes(program_df)
    program_technologies, program_bigrams = extract_key_technologies(program_df)
    visualize_psc_analysis(top_psc_by_program, colors)
    return top_psc_by_program, psc_diversity, program_technologies, program_bigrams


def run_lifecycle_assessment(program_df):
    """Run the complete program lifecycle assessment workflow."""
    print("\n### Lifecycle Assessment ###\n")
    lifecycle_df = identify_program_lifecycle_status(program_df)
    projection_df = project_future_acquisitions(program_df, lifecycle_df)
    sorted_programs = visualize_lifecycle_assessment(lifecycle_df)
    visualize_spending_trajectories(program_df, sorted_programs, lifecycle_df)
    return lifecycle_df, projection_df


def run_risk_analysis(program_df, market_concentration):
    """Run the complete risk analysis workflow."""
    print("\n### Risk Analysis ###\n")
    vendor_risk_df = analyze_vendor_dependence_risk(program_df, market_concentration)
    budget_risk_df = analyze_budget_consistency_risk(program_df)
    supply_chain_df = analyze_supply_chain_risk(program_df)
    consolidated_risk = visualize_risk_assessment(
        vendor_risk_df, budget_risk_df, supply_chain_df
    )
    visualize_risk_radar(consolidated_risk)
    return vendor_risk_df, budget_risk_df, supply_chain_df, consolidated_risk
