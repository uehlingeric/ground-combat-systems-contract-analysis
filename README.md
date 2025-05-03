# Ground Combat Systems Contract Analysis

![DoD Contracts](https://img.shields.io/badge/Domain-Defense_Contracts-blue)
![Python](https://img.shields.io/badge/Language-Python_3-green)
![Data Analysis](https://img.shields.io/badge/Type-Data_Analysis-orange)

## Overview

This repository contains a comprehensive analysis of Department of Defense (DoD) ground combat systems contracts from FY2016-2020, focusing on three major platforms:

- Abrams tank
- Bradley fighting vehicle
- Stryker combat vehicle

The analysis uncovers spending patterns, vendor concentration, geographic distribution, and program lifecycle stages to identify risks and opportunities for improved acquisition planning.

## Key Findings

1. **High Vendor Concentration**: The top three vendors account for 85.9% of Abrams, 93.6% of Bradley, and 91.8% of Stryker program spending, creating significant single-source supplier risks.

2. **Geographic Concentration**: Michigan accounts for 71.4% of total program spending, with Sterling Heights alone receiving nearly $6.3 billion, posing substantial geographic risk.

3. **Late Lifecycle Status**: All three systems are in the sustainment and modernization phase with high modification rates (>96%), requiring different contracting approaches from new systems acquisition.

4. **Budget Volatility**: All programs show high year-to-year budget volatility, complicating long-term planning for both DoD and vendors.

## Repository Contents

- `gcsca.ipynb`: Jupyter notebook containing the full data analysis workflow organized in three phases:
  - Phase 1: Data Exploration and Cleaning
  - Phase 2: Exploratory Data Analysis 
  - Phase 3: Advanced Analysis
- `utils.py`: Python utility functions for data processing and visualization
- `report.pdf`: Comprehensive report with findings and recommendations

## Technologies Used

- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **RapidFuzz** - Fuzzy string matching for vendor name normalization

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - rapidfuzz
  - openpyxl (for Excel file handling)
  
### Data Requirements

The analysis requires DoD contract data with the following structure:
- Contract award information (fiscal year, award amount)
- Vendor information
- Program details
- Geographic data (state, city)
- Contracting office information

### Installation

```bash
# Clone this repository
git clone https://github.com/uehlingeric/ground-combat-systems-contract-analysis.git

# Navigate to the project directory
cd ground-combat-systems-contract-analysis

# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## Analysis Structure

The analysis is structured in three phases:

### Phase 1: Data Exploration and Cleaning
- Loading and initial exploration of the dataset
- Standardizing column names and data types
- Exploring distributions of fiscal years and award amounts
- Classifying contracts by program (Abrams, Bradley, Stryker)
- Normalizing vendor names using fuzzy matching
- Handling missing values and data quality issues

### Phase 2: Exploratory Data Analysis
- Program-level spending analysis across fiscal years
- Vendor concentration analysis and market share calculations
- Geographic distribution of contracts at state and city levels
- Contracting office/agency patterns and specialization
- Contract modification rate analysis

### Phase 3: Advanced Analysis
- Technology and Product Service Code (PSC) analysis
- Program lifecycle assessment and future projections
- Risk analysis (vendor dependence, budget consistency, supply chain)

## Key Recommendations

1. **Vendor Diversification Strategy**: Implement targeted contract awards to second and third-tier vendors to reduce extreme concentration risk.

2. **Geographic Risk Mitigation**: Incentivize contractors to develop additional production and maintenance capabilities in different regions.

3. **Lifecycle-Appropriate Contracting**: Adapt contracting approaches to match the late lifecycle phase of these systems.

4. **Contract Modification Reduction**: Implement more rigorous upfront requirements analysis to reduce costly modifications.

5. **Data-Driven Decision Making**: Improve vendor data standardization and contract categorization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Eric Uehling