import gradio as gr
import pandas as pd
from itertools import permutations, product
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple

# ---------------------------
# @title DATA RETRIEVAL MODULE
# ---------------------------

FIELDS = {
    "Faction": [
        "Corporation", "Cult", "Empire", "Guardian",
        "Kingdom", "Tribe", "Undying"
    ],
    "Job": [
        "Arcanist", "Assassin", "Barbarian", "Druid",
        "Duelist", "Huntsman", "Inquisitor", "Lord",
        "Mystic", "Necromancer", "Paladin", "Protector",
        "Scientist", "Shaman", "Soldier", "Sorcerer", "Warden"
    ],
    "Element": [
        "Air", "Chaos", "Darkness", "Fire",
        "Light", "Nature", "Order", "Technology", "Water"
    ],
    "Weapon": [
        "Blunderbuss", "Crossbow", "Dagger", "Greataxe",
        "Greatsword", "Halberd", "Katana", "Mystical horn",
        "Rapier", "Runic scimitar", "Scepter", "Sickle",
        "Spatha", "Spear", "Spellbook", "Voodoo doll"
    ],
    "Quirk": [
        "Agile", "Blind", "Corporate", "Disciple",
        "Cursed", "Demon slayer", "Sandguard",
        "Evil", "Lucky", "Masked", "One eyed",
        "Quick", "Runic", "Sacred", "Strong",
        "Veteran", "Vigilante", "Zealous"
    ]
}

def clean_text(text):
    """Clean the text by removing extra whitespace and unwanted artifacts."""
    return ' '.join(text.split())

def extract_text_bs4(url):
    """Extracts and cleans text from the given URL using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {e}"

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return clean_text(text)

def extract_between(text, start_seq, stop_seq):
    """Extracts text between two sequences."""
    start_idx = text.lower().find(start_seq.lower())
    if start_idx == -1:
        return f"Start sequence '{start_seq}' not found."

    start_idx += len(start_seq)
    stop_idx = text.lower().find(stop_seq.lower(), start_idx)
    if stop_idx == -1:
        return f"Stop sequence '{stop_seq}' not found."

    return text[start_idx:stop_idx].strip()

def parse_extracted_data(extracted_data, fields, field_values):
    """Parses the extracted data string and maps it to the specified fields."""
    words = extracted_data.split()
    data_dict = {field: [] for field in fields}
    current_field = None

    fields_lower = {field.lower(): field for field in fields}
    field_values_lower = {
        field: {value.lower(): value for value in values} for field, values in field_values.items()
    }

    i = 0
    n = len(words)

    while i < n:
        word = words[i].lower()
        if word in fields_lower:
            current_field = fields_lower[word]
            i += 1
            continue

        if current_field:
            max_words = max(len(value.split()) for value in field_values[current_field])
            match_found = False

            for window in range(max_words, 0, -1):
                if i + window > n:
                    continue
                candidate = ' '.join(words[i:i + window]).lower()
                if candidate in field_values_lower[current_field]:
                    data_dict[current_field].append(field_values_lower[current_field][candidate])
                    i += window
                    match_found = True
                    break

            if not match_found:
                i += 1
        else:
            i += 1

    return data_dict

def handle_quirk_duplications(quirks):
    """
    Handles quirks based on their count:
    - If two quirks: No changes.
    - If three quirks: If the first two are duplicates, delete the first.
      The second becomes the first with a '1' suffix, and the third becomes the second.
    """
    if len(quirks) == 3 and quirks[0].lower() == quirks[1].lower():
        quirks[1] = f"{quirks[1]}1"
        return [quirks[1], quirks[2]]
    return quirks

def handle_weapon_element_duplications(weapon, element):
    """
    Handles weapon and element based on their count:
    - If three items: No changes.
    - If four items: If the second and third are duplicates, delete the third,
      mark the second with a '1' suffix, and the fourth becomes the third.
    """
    def process_category(category):
        if len(category) == 4 and category[1].lower() == category[2].lower():
            category[1] = f"{category[1]}1"
            return [category[0], category[1], category[3]]
        return category

    weapon = process_category(weapon)
    element = process_category(element)

    return weapon, element

def create_dataframe(data_dict):
    """Creates a DataFrame from the data dictionary, handling quirks, weapons, and elements."""
    if 'Quirk' in data_dict:
        data_dict['Quirk'] = handle_quirk_duplications(data_dict['Quirk'])

    if 'Weapon' in data_dict and 'Element' in data_dict:
        data_dict['Weapon'], data_dict['Element'] = handle_weapon_element_duplications(
            data_dict['Weapon'], data_dict['Element']
        )

    max_length = max(len(v) for v in data_dict.values())
    for field in data_dict:
        if len(data_dict[field]) < max_length:
            data_dict[field].extend([None] * (max_length - len(data_dict[field])))

    return pd.DataFrame(data_dict)

def extract_and_structure_data(url, start_sequence, stop_sequence, fields, field_values):
    """Extracts, parses, and structures data from a URL into a DataFrame."""
    extracted_text = extract_text_bs4(url)
    if extracted_text.startswith("Error"):
        return extracted_text

    specific_data = extract_between(extracted_text, start_sequence, stop_sequence)
    if specific_data.startswith("Error"):
        return specific_data

    parsed_data = parse_extracted_data(specific_data, fields, field_values)
    return create_dataframe(parsed_data)

# ---------------------------
# GENOTYPE CALCULATOR MODULE
# ---------------------------

# Fixed Probabilities for Parent 1 and Parent 2 (identical)
probs_p1 = [
    [0.3, 0.15, 0.05],    # Faction
    [0.3, 0.15, 0.05],    # Job
    [0.3, 0.15, 0.05],    # Weapon
    [0.3, 0.15, 0.05],    # Element
    [0.375, 0.125]        # Quirk
]
probs_p2 = probs_p1.copy()

# Number of Alleles per Locus
perm_lengths = [3, 3, 3, 3, 2]  # First four loci: 3 alleles, fifth locus: 2 alleles

# Helper function to assign unique identifiers
def make_unique_alleles(alleles, parent_label):
    return [f"{allele}_{parent_label}_{i+1}" for i, allele in enumerate(alleles)]

def calculate_rolling_sequence_odds(all_dfs, genotype, allele_name_map):
    """
    Calculates the rolling sequence odds of the child being identical to a given genotype.
    """
    report_data = []  # Store report data for each feature
    overall_chance = 1.0  # Initialize overall chance

    for i in range(len(all_dfs)):
        feature_name = ["Faction", "Job", "Weapon", "Element", "Quirk"][i]
        df = all_dfs[i]

        # Get the alleles from the genotype for the current locus
        locus_name = f"Locus{i+1}"
        alleles = genotype.get(locus_name, [])

        # Handle cases where alleles are in a tuple or list
        if isinstance(alleles, str):
            alleles = [alleles]
        elif isinstance(alleles, tuple):
            alleles = list(alleles)

        # For consistency, make sure we have the correct number of alleles
        perm_length = perm_lengths[i]
        while len(alleles) < perm_length:
            alleles.append("-")

        # Now, find the matching rows in df
        matching_rows = df.copy()
        for j in range(perm_length):
            allele_col = f"Allele{j+1}"
            matching_rows = matching_rows[matching_rows[allele_col] == alleles[j]]

        # Sum the 'Total_Prob' for matching rows
        rolling_chance = matching_rows['Total_Prob'].sum() if not matching_rows.empty else 0.0

        # Update the overall rolling chance
        overall_chance *= rolling_chance

        # Append to report data
        alleles_display = ', '.join(alleles)
        report_data.append([
            feature_name, alleles_display, f"{rolling_chance * 100:.2f}%"
        ])

    # Add the overall rolling chance to the report
    report_data.append(["Overall", "-", f"{overall_chance * 100:.2f}%"])

    # Add the report data to a DataFrame
    rolling_sequence_df = pd.DataFrame(report_data, columns=[
        "Feature", "Genes", "Rolling Chance"
    ])

    return rolling_sequence_df

def report_gene_position_probabilities(all_dfs):
    """
    Reports the chance of each gene to be in each position (dominant, recessive, minor)
    for each category.

    Args:
        all_dfs (List[pd.DataFrame]): List of DataFrames for each category containing
                                      allele combinations and their probabilities.

    Returns:
        pd.DataFrame: DataFrame with columns ['Category', 'Gene', 'Dominant', 'Recessive', 'Minor']
                      and rows for each gene in each category.
    """
    positions = ['dominant', 'recessive', 'minor']
    categories = ['Faction', 'Job', 'Weapon', 'Element', 'Quirk']
    perm_lengths = [3, 3, 3, 3, 2]  # Number of positions per category

    result = []

    for idx, df in enumerate(all_dfs):
        category = categories[idx]
        perm_length = perm_lengths[idx]  # Number of positions for this category
        position_names = positions[:perm_length]  # Positions for this category

        # Get the set of genes in this category
        genes = set()
        for col in ['Allele1', 'Allele2', 'Allele3']:
            if col in df.columns:
                genes.update(df[col].unique())

        # Initialize the gene probabilities
        gene_probs = {gene: {pos: 0.0 for pos in position_names} for gene in genes}

        # Iterate over the rows of the dataframe
        for _, row in df.iterrows():
            total_prob = row['Total_Prob']
            alleles = []
            for col in ['Allele1', 'Allele2', 'Allele3']:
                if col in df.columns:
                    alleles.append(row[col])
            # Now, for each allele (gene) in position, add total_prob to gene_probs[gene][position]
            for pos_idx, gene in enumerate(alleles):
                if pos_idx >= len(position_names):
                    break
                pos_name = position_names[pos_idx]
                gene_probs[gene][pos_name] += total_prob

        # Now, create a DataFrame for this category
        data = []
        for gene, pos_probs in gene_probs.items():
            row = {'Category': category, 'Gene': gene}
            row.update({pos.capitalize(): pos_probs[pos] for pos in position_names})
            data.append(row)

        df_category = pd.DataFrame(data)

        # Sort the DataFrame: First by Dominant descending, then Recessive descending, then Minor descending
        sort_columns = [pos.capitalize() for pos in position_names]
        df_category = df_category.sort_values(by=sort_columns, ascending=False)

        result.append(df_category)

    # Concatenate all category DataFrames
    final_df = pd.concat(result, ignore_index=True)

    # Reorder columns to have: Category, Gene, Dominant, Recessive, Minor
    cols = ['Category', 'Gene'] + [pos.capitalize() for pos in positions]
    existing_cols = [col for col in cols if col in final_df.columns]
    final_df = final_df[existing_cols]

    # Replace NaN with 0
    final_df = final_df.fillna(0)

    return final_df

def genotype_calculator(parent1_df, parent2_df):
    """
    Generates calculation reports including rolling sequence odds, allele combinations, and combined dataframe.
    Includes comprehensive error handling.
    """
    try:
        num_features = 5
        if parent1_df is None or parent2_df is None:
            return "Error: Parent genotype data is missing.", None

        # Validate DataFrames
        required_fields = ["Faction", "Job", "Weapon", "Element", "Quirk"]
        for df, parent_label in zip([parent1_df, parent2_df], ["Parent 1", "Parent 2"]):
            if not all(field in df.columns for field in required_fields):
                return f"Error: {parent_label} DataFrame is missing required fields.", None

        # Assign unique identifiers
        alleles_p1_unique = [
            make_unique_alleles(parent1_df[field].dropna().tolist(), 'p1') for field in required_fields
        ]
        alleles_p2_unique = [
            make_unique_alleles(parent2_df[field].dropna().tolist(), 'p2') for field in required_fields
        ]

        # Create mapping from unique identifiers to base names
        allele_name_map = []
        for locus_idx in range(num_features):
            mapping = {}
            field = required_fields[locus_idx]
            for j, allele in enumerate(alleles_p1_unique[locus_idx]):
                if j < len(parent1_df[field]):
                    mapping[allele] = parent1_df.iloc[j][field]
            for j, allele in enumerate(alleles_p2_unique[locus_idx]):
                if j < len(parent2_df[field]):
                    mapping[allele] = parent2_df.iloc[j][field]
            allele_name_map.append(mapping)

        # STEP 1: GENERATE ALLELE COMBINATIONS WITH PROBABILITIES
        all_dfs = []
        allele_combinations_reports = ""

        for i in range(len(alleles_p1_unique)):
            feature_name = required_fields[i]
            parent_sequence_unique = alleles_p1_unique[i]

            # Handle features with fewer alleles
            if perm_lengths[i] == 2:
                parent_sequence_unique = list(parent_sequence_unique[:2]) + ["-"]  # Replace missing with "-"

            # Combine alleles from both parents
            combined_alleles = alleles_p1_unique[i] + alleles_p2_unique[i]
            combined_probs = probs_p1[i] + probs_p2[i]

            perm_length = perm_lengths[i]

            # Generate all possible sequences of perm_length alleles using permutations
            allele_combos = list(permutations(combined_alleles, perm_length))

            # Create a DataFrame with the allele combinations
            if perm_length == 3:
                df = pd.DataFrame(allele_combos, columns=["Allele1", "Allele2", "Allele3"])

                # Assign probabilities using the unique identifiers
                df["Prob_Allele1"] = df["Allele1"].apply(
                    lambda x: probs_p1[i][alleles_p1_unique[i].index(x)] if x in alleles_p1_unique[i] else probs_p2[i][alleles_p2_unique[i].index(x)]
                )
                df["Prob_Allele2"] = df["Allele2"].apply(
                    lambda x: probs_p1[i][alleles_p1_unique[i].index(x)] if x in alleles_p1_unique[i] else probs_p2[i][alleles_p2_unique[i].index(x)]
                )
                df["Prob_Allele3"] = df["Allele3"].apply(
                    lambda x: probs_p1[i][alleles_p1_unique[i].index(x)] if x in alleles_p1_unique[i] else probs_p2[i][alleles_p2_unique[i].index(x)]
                )

                # Calculate positional probabilities
                df["Pos_Prob1"] = df["Prob_Allele1"]
                df["Pos_Prob2"] = df["Prob_Allele2"] / (1 - df["Prob_Allele1"])
                df["Pos_Prob3"] = df["Prob_Allele3"] / (1 - df["Prob_Allele1"] - df["Prob_Allele2"])

                # Calculate total probability
                df["Total_Prob"] = df["Pos_Prob1"] * df["Pos_Prob2"] * df["Pos_Prob3"]

            elif perm_length == 2:
                df = pd.DataFrame(allele_combos, columns=["Allele1", "Allele2"])

                # Assign probabilities using the unique identifiers
                df["Prob_Allele1"] = df["Allele1"].apply(
                    lambda x: probs_p1[i][alleles_p1_unique[i].index(x)] if x in alleles_p1_unique[i] else probs_p2[i][alleles_p2_unique[i].index(x)]
                )
                df["Prob_Allele2"] = df["Allele2"].apply(
                    lambda x: probs_p1[i][alleles_p1_unique[i].index(x)] if x in alleles_p1_unique[i] else probs_p2[i][alleles_p2_unique[i].index(x)]
                )

                # Calculate positional probabilities
                df["Pos_Prob1"] = df["Prob_Allele1"]
                df["Pos_Prob2"] = df["Prob_Allele2"] / (1 - df["Prob_Allele1"])

                # Calculate total probability
                df["Total_Prob"] = df["Pos_Prob1"] * df["Pos_Prob2"]
                df["Allele3"] = "-"  # Replace missing with "-"

            # Replace unique identifiers with base names for display
            for col in ["Allele1", "Allele2", "Allele3"]:
                if col in df.columns:
                    df[col] = df[col].map(lambda x: allele_name_map[i].get(x, x))

            # Verify that probabilities sum to 1
            total_prob = df["Total_Prob"].sum()
            if not np.isclose(total_prob, 1.0):
                df["Total_Prob"] = df["Total_Prob"] / total_prob
                print(f"Normalized probabilities for {feature_name}. Total Prob was {total_prob}, now normalized to 1.")

            all_dfs.append(df)

            # Allele Combinations Report
            allele_combinations_reports += f"<h3>Allele Combinations for {feature_name}:</h3>"
            allele_combinations_reports += df.to_html(index=False)
            allele_combinations_reports += "<br>"

        # STEP 2: COMBINE ALL LOCI DATA INTO FINAL DATAFRAME
        for idx, df in enumerate(all_dfs, 1):
            df.insert(0, "Locus", f"Locus{idx}")

        final_df = pd.concat(all_dfs, ignore_index=True)

        combined_dataframe_report = "<h3>Final Combined DataFrame:</h3>"
        combined_dataframe_report += final_df.to_html(index=False)
        combined_dataframe_report += "<br>"

        # STEP 3: SELECT TOP N ALLELE COMBINATIONS PER LOCUS WITH AGGREGATED PROBABILITIES
        TOP_N = 5  # Adjust as needed
        grouped = final_df.groupby("Locus")

        top_allele_combos_per_locus = []
        top_combos_report = ""

        for locus, group in grouped:
            allele_columns = [col for col in group.columns if col.startswith('Allele')]

            # Group by allele combinations and sum their probabilities
            grouped_combos = group.groupby(allele_columns)['Total_Prob'].sum().reset_index()

            # Sort by summed probabilities in descending order
            top_combos = grouped_combos.sort_values(by='Total_Prob', ascending=False).head(TOP_N)

            allele_combos = top_combos[allele_columns].values.tolist()
            probabilities = top_combos['Total_Prob'].values.tolist()

            locus_combos = [(tuple(alleles), prob) for alleles, prob in zip(allele_combos, probabilities)]
            top_allele_combos_per_locus.append(locus_combos)

            # Extract locus number to map to category
            locus_num = int(locus.replace("Locus", ""))
            category = required_fields[locus_num - 1]  # Zero-based index

            # Update the report title to include category
            top_combos_report += f"<h3>Top {TOP_N} Allele Combinations for {locus} ({category}):</h3><ul>"
            for combo, prob in locus_combos:
                combo_str = ', '.join([allele if isinstance(allele, str) else 'nan' for allele in combo])
                top_combos_report += f"<li>Alleles: ({combo_str}), Probability: {prob*100:.2f}%</li>"
            top_combos_report += "</ul>"

        # STEP 4: GENERATE CHILD GENOTYPES FROM TOP ALLELE COMBINATIONS
        all_genotype_combinations = product(*top_allele_combos_per_locus)
        child_genotypes = []

        for genotype in all_genotype_combinations:
            genotype_dict = {}
            total_prob = 1.0
            for locus_idx, (alleles, prob) in enumerate(genotype, 1):
                locus_name = f"Locus{locus_idx}"
                genotype_dict[locus_name] = alleles
                total_prob *= prob
            genotype_dict['Total_Probability'] = total_prob
            child_genotypes.append(genotype_dict)

        child_genotypes_df = pd.DataFrame(child_genotypes)

        # STEP 5: SORT CHILD GENOTYPES BY TOTAL PROBABILITY
        child_genotypes_sorted = child_genotypes_df.sort_values(by='Total_Probability', ascending=False).reset_index(drop=True)

        # STEP 6: MERGE DUPLICATED CHILD GENOTYPES AND AGGREGATE PROBABILITIES
        loci_columns = [col for col in child_genotypes_sorted.columns if col.startswith('Locus')]
        grouped_genotypes = child_genotypes_sorted.groupby(loci_columns)
        unique_genotypes = grouped_genotypes['Total_Probability'].sum().reset_index()
        unique_genotypes.rename(columns={'Total_Probability': 'Global_Probability'}, inplace=True)
        unique_genotypes_sorted = unique_genotypes.sort_values(by='Global_Probability', ascending=False).reset_index(drop=True)

        # STEP 7: DISPLAY TOP M MOST PROBABLE UNIQUE CHILD GENOTYPES
        TOP_M = 10  # Number of top genotypes to display
        top_genotypes = unique_genotypes_sorted.head(TOP_M)
        top_genotypes_report = f"<h3>Top {TOP_M} Most Probable Unique Child Genotypes:</h3>"
        top_genotypes_report += top_genotypes.to_html(index=False)
        top_genotypes_report += "<br>"

        # STEP 8: ROLLING SEQUENCE ODDS REPORT OVER MOST PROBABLE CHILD GENOTYPE
        # Determine the most probable genotype
        if top_genotypes.empty:
            return "Error: No genotype combinations available to calculate rolling odds.", None

        most_probable_genotype = top_genotypes.iloc[0].to_dict()

        rolling_sequence_df = calculate_rolling_sequence_odds(all_dfs, most_probable_genotype, allele_name_map)
        rolling_sequence_html = (
            "<h3>Rolling Sequence Odds Report Over Most Probable Child Genotype:</h3>" +
            rolling_sequence_df.to_html(index=False)
        )

        # REPORT GENE POSITION PROBABILITIES
        gene_position_df = report_gene_position_probabilities(all_dfs)
        gene_position_html = (
            "<h3>Chance of Each Gene in Each Position:</h3>" +
            gene_position_df.to_html(index=False)
        )

        # COMBINE ALL REPORTS INTO SINGLE HTML OUTPUT
        # {allele_combinations_reports}
        # {combined_dataframe_report}
        # {child_genotypes_sorted.to_html(index=False)}
        combined_output = f"""
        {top_combos_report}
        {top_genotypes_report}
        {rolling_sequence_html}
        {gene_position_html}
        """

        return combined_output, top_allele_combos_per_locus

    except Exception as e:
        return f"An unexpected error occurred during genotype calculation: {e}", None

# ---------------------------
# GRADIO INTERFACE
# ---------------------------

def process_and_calculate(parent1_url, parent2_url):
    """
    Retrieves and processes parent genotypes from URLs and calculates offspring genotype reports.
    """
    start_sequence = "Dominant Recessive Minor Recessive"
    stop_sequence = "Skills Reactive"
    fields = ["Faction", "Job", "Weapon", "Element", "Quirk"]

    # Extract Parent 1 Data
    parent1_data = extract_and_structure_data(parent1_url, start_sequence, stop_sequence, fields, FIELDS)
    if isinstance(parent1_data, str) and parent1_data.startswith("Error"):
        return f"Parent 1 Data Retrieval Error: {parent1_data}", None

    # Extract Parent 2 Data
    parent2_data = extract_and_structure_data(parent2_url, start_sequence, stop_sequence, fields, FIELDS)
    if isinstance(parent2_data, str) and parent2_data.startswith("Error"):
        return f"Parent 2 Data Retrieval Error: {parent2_data}", None

    # Convert parent DataFrames to HTML tables
    parent1_html = f"<h3>Parent 1 Genotype:</h3>{parent1_data.to_html(index=False)}<br>"
    parent2_html = f"<h3>Parent 2 Genotype:</h3>{parent2_data.to_html(index=False)}<br>"

    # Generate Genotype Reports
    report, top_allele_combos_per_locus = genotype_calculator(parent1_data, parent2_data)

    if isinstance(report, str):
        # If an error message was returned
        full_report = f"{parent1_html}{parent2_html}<h3>Offspring Genotype Report:</h3><p>{report}</p>"
    else:
        # Combine parent genotypes with the offspring report
        full_report = f"{parent1_html}{parent2_html}<h3>Offspring Genotype Report:</h3>{report}"

    return full_report, top_allele_combos_per_locus

# ---------------------------
# GENE STATS CALCULATOR MODULE (from second code)
# -------------------- Gene Data -------------------- #

csvData1 = """
Gene,Vitality,Initiative,Phys. Power,Elem. Power,Phys. Resist.,Elem. Resist.,Crit. Chance,Crit. Multiplier
# FACTION
Corporation,8,,4,2,,,,
Cult,,6,,2,,,,
Empire,,,2,2,,,8,10
Guardian,4,4,,2,,,,
Kingdom,8,2,2,,,,,
Tribe,4,,6,4,,,,
Undying,12,,,2,,,,
# JOB
Arcanist,,4,6,8,,,4,
Assassin,4,6,4,4,,,,10
Barbarian,,,6,6,,,12,10
Druid,16,4,,,,10,,
Duelist,8,4,6,4,,,,
Huntsman,4,4,4,4,,,4,10
Inquisitor,6,6,3,5,,,,
Lord,12,,6,4,10,10,,
Mystic,12,6,,,,10,,
Necromancer,8,2,4,4,,10,,
Paladin,18,,,,25,15,,
Protector,18,,,,15,25,,
Scientist,,2,7,7,,,6,10
Shaman,12,6,2,,,,,
Soldier,14,5,,,10,,,
Sorcerer,,2,6,8,,,8,
Warden,10,4,,,20,10,,
# ELEMENT
Air,4,4,2,,,,,
Chaos,,,2,2,,,8,10
Darkness,8,2,,2,,,,
Fire,4,,4,6,,,,
Light,,2,4,2,,,4,
Nature,12,,,2,,,,
Order,4,2,2,2,10,,,
Technology,8,,2,2,,10,,
Water,8,2,2,,,,,
# WEAPON
Blunderbuss,,,2,,,,6,5
Crossbow,,,2,,,,6,5
Dagger,,2,4,,,,,
Greataxe,,,6,,,,,
Greatsword,6,,3,,,,,
Halberd,,,6,,,,,
Katana,,2,4,,,,,
Mystical horn,,,,6,,,,
Rapier,,2,4,,,,,
Runic scimitar,,2,4,,,,,
Scepter,,,,6,,,,
Sickle,,2,4,,,,,
Spatha,,2,4,,,,,
Spear,,,6,,,,,
Spellbook,,,,6,,,,
Voodoo doll,,,,6,,,,
# QUIRK
Agile,,4,,,,,,
Blind,,,,4,,,,
Corporate,8,,,,,,,
Disciple,8,,,,,,,
Cursed,,,,4,,,,
Demon slayer,,,4,,,,,
Sandguard,,4,,,,,,
Evil,,,,4,,,,
Lucky,,,,,,,8,
Masked,,,,,,,8,
One eyed,,,4,,,,,
Quick,,4,,,,,,
Runic,8,,,,,,,
Sacred,8,,,,,,,
Strong,,,4,,,,,
Veteran,,,4,,,,,
Vigilante,8,,,,,,,
Zealous,,,,4,,,,
"""

csvData2 = """
Gene,Vitality,Initiative,Phys. Power,Elem. Power,Phys. Resist.,Elem. Resist.,Crit. Chance,Crit. Multiplier
# FACTION
Corporation,4,,1,,,,,
Cult,,2,,1,,,,
Empire,,,,,,,4,5
Guardian,,2,,1,,,,
Kingdom,4,,1,,,,,
Tribe,,,3,2,,,,
Undying,4,,,1,,,,
# JOB
Arcanist,,,2,3,,,,
Assassin,,2,,,,,,5
Barbarian,,,2,2,,,,
Druid,6,,,,,,,
Duelist,2,,2,1,,,,
Huntsman,,2,,,,,,5
Inquisitor,,2,,1,,,,
Lord,2,,2,1,,,,
Mystic,2,2,,,,,,
Necromancer,2,,2,2,,,,
Paladin,6,,,,,,,
Protector,6,,,,,,,
Scientist,,,2,2,,,,5
Shaman,4,,1,,,,,
Soldier,4,1,,,,,,
Sorcerer,,,2,3,,,,
Warden,4,1,,,,,,
# ELEMENT
Air,,2,1,,,,,
Chaos,,,,,,,4,5
Darkness,4,,,1,,,,
Fire,,,2,3,,,,
Light,,,2,1,,,2,
Nature,4,,,1,,,,
Order,,2,1,,,,,
Technology,4,,,1,,,,
Water,,2,1,,,,,
# WEAPON
Blunderbuss,,,,,,,4,
Crossbow,,,,,,,4,
Dagger,,2,,,,,,
Greataxe,,,2,,,,,
Greatsword,4,,,,,,,
Halberd,,,2,,,,,
Katana,,2,,,,,,
Mystical horn,,,,2,,,,
Rapier,,2,,,,,,
Runic scimitar,,2,,,,,,
Scepter,,,,2,,,,
Sickle,,2,,,,,,
Spatha,,2,,,,,,
Spear,,,2,,,,,
Spellbook,,,,2,,,,
Voodoo doll,,,,2,,,,
# QUIRK
Agile,,2,,,,,,
Blind,,,,2,,,,
Corporate,4,,,,,,,
Disciple,4,,,,,,,
Cursed,,,,2,,,,
Demon slayer,,,2,,,,,
Sandguard,,2,,,,,,
Evil,,,,2,,,,
Lucky,,,,,,,4,
Masked,,,,,,,4,
One eyed,,,2,,,,,
Quick,,2,,,,,,
Runic,4,,,,,,,
Sacred,4,,,,,,,
Strong,,,2,,,,,
Veteran,,,2,,,,,
Vigilante,4,,,,,,,
Zealous,,,,2,,,,
"""

# -------------------- Parsing Functions -------------------- #

def parse_csv(csv_string: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Parses the CSV data into a dictionary categorized by gene types.

    Args:
        csv_string (str): Multi-line CSV string.

    Returns:
        Dict[str, List[Dict[str, str]]]: Parsed gene data.
    """
    data = {}
    current_category = None
    lines = csv_string.strip().split('\n')
    headers = lines[0].split(',')

    for line in lines[1:]:
        line = line.strip()
        if line.startswith('# '):
            current_category = line[2:].strip().upper()
            data[current_category] = []
        elif current_category and line and not line.startswith('Gene,'):
            values = [v.strip() for v in line.split(',')]
            gene_info = dict(zip(headers, values))
            data[current_category].append(gene_info)
    return data

# Parse the CSV data
parsed_data1 = parse_csv(csvData1)
parsed_data2 = parse_csv(csvData2)

# -------------------- GeneStatsCalculator Class -------------------- #

class GeneStatsCalculator:
    """
    A class to handle gene selection, importing gene sequences,
    calculating stats, and retrieving results as a DataFrame.
    """

    def __init__(self):
        self.parsed_data1 = parsed_data1  # Dominant genes data
        self.parsed_data2 = parsed_data2  # Recessive genes data
        self.selected_genes1: Dict[str, str] = {}  # Dominant genes
        self.selected_genes2: Dict[str, str] = {}  # Recessive genes
        self.selected_genes3: Dict[str, str] = {}  # Minor genes

    def select_gene(self, category: str, gene_type: str, gene_name: str):
        """
        Selects a gene for a given category and gene type.

        Args:
            category (str): The gene category (e.g., FACTION).
            gene_type (str): The gene type ('Dominant', 'Recessive', 'Minor').
            gene_name (str): The name of the gene to select.
        """
        if gene_type == 'Dominant':
            self.selected_genes1[category] = gene_name
        elif gene_type == 'Recessive':
            self.selected_genes2[category] = gene_name
        elif gene_type == 'Minor':
            self.selected_genes3[category] = gene_name

    def import_gene_sequence(self, gene_sequence: str):
        """
        Imports a gene sequence string and sets the selected genes accordingly.

        The gene sequence should have 5 lines corresponding to FACTION, JOB, WEAPON, ELEMENT, and QUIRK.
        Each line should have Dominant, Recessive, and Minor genes separated by commas.
        Use '-' to denote absence of a Dominant gene.

        Args:
            gene_sequence (str): The gene sequence string.
        """
        categories = ['FACTION', 'JOB', 'WEAPON', 'ELEMENT', 'QUIRK']
        lines = gene_sequence.strip().split('\n')
        if len(lines) != 5:
            raise ValueError("Gene sequence must have exactly 5 lines corresponding to FACTION, JOB, WEAPON, ELEMENT, and QUIRK.")

        for i, line in enumerate(lines):
            parts = [part.strip() for part in line.split(',')]
            # Remove empty strings and handle trailing commas
            parts = [part for part in parts if part]
            category = categories[i]
            # Dominant, Recessive, Minor
            dominant = parts[0] if len(parts) >= 1 and parts[0] != '-' else None
            recessive = parts[1] if len(parts) >= 2 and parts[1] != '-' else None
            minor = parts[2] if len(parts) >= 3 and parts[2] != '-' else None

            if dominant:
                self.selected_genes1[category] = dominant
            if recessive:
                self.selected_genes2[category] = recessive
            if minor:
                self.selected_genes3[category] = minor

    def calculate_stats(self) -> pd.DataFrame:
        """
        Calculates the aggregated stats based on the selected genes.

        Returns:
            pd.DataFrame: DataFrame containing individual gene stats and the total aggregated stats.
        """
        # Initialize totalStats with base values
        total_stats = {
            'Gene': 'Total (with base)',
            'Vitality': 44.0,            # Base Vitality
            'Initiative': 0.0,
            'Phys. Power': 0.0,
            'Elem. Power': 0.0,
            'Phys. Resist.': 0.0,
            'Elem. Resist.': 0.0,
            'Crit. Chance': 0.0,
            'Crit. Multiplier': 50.0     # Base Crit. Multiplier
        }

        genes_data = []

        # Helper function to find a gene in a specific data set
        def find_gene(gene_name: str, data: Dict[str, List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
            for category, genes in data.items():
                for gene in genes:
                    if gene['Gene'].lower() == gene_name.lower():
                        return gene
            return None

        # Process selectedGenes1 (Dominant) using data1
        for category, gene_name in self.selected_genes1.items():
            gene = find_gene(gene_name, self.parsed_data1)
            if gene:
                gene_stat = {
                    'Gene': gene['Gene'],
                    'Vitality': float(gene.get('Vitality') or 0),
                    'Initiative': float(gene.get('Initiative') or 0),
                    'Phys. Power': float(gene.get('Phys. Power') or 0),
                    'Elem. Power': float(gene.get('Elem. Power') or 0),
                    'Phys. Resist.': float(gene.get('Phys. Resist.') or 0),
                    'Elem. Resist.': float(gene.get('Elem. Resist.') or 0),
                    'Crit. Chance': float(gene.get('Crit. Chance') or 0),
                    'Crit. Multiplier': float(gene.get('Crit. Multiplier') or 0)
                }
                genes_data.append(gene_stat)
                # Sum the stats
                for key in total_stats:
                    if key != 'Gene' and gene_stat[key] is not None:
                        total_stats[key] += gene_stat[key]

        # Combine Recessive and Minor Genes
        combined_genes2_3 = list(self.selected_genes2.values()) + list(self.selected_genes3.values())

        # Process selectedGenes2 and selectedGenes3 (Recessive and Minor) using data2
        for gene_name in combined_genes2_3:
            gene = find_gene(gene_name, self.parsed_data2)
            if gene:
                gene_stat = {
                    'Gene': gene['Gene'],
                    'Vitality': float(gene.get('Vitality') or 0),
                    'Initiative': float(gene.get('Initiative') or 0),
                    'Phys. Power': float(gene.get('Phys. Power') or 0),
                    'Elem. Power': float(gene.get('Elem. Power') or 0),
                    'Phys. Resist.': float(gene.get('Phys. Resist.') or 0),
                    'Elem. Resist.': float(gene.get('Elem. Resist.') or 0),
                    'Crit. Chance': float(gene.get('Crit. Chance') or 0),
                    'Crit. Multiplier': float(gene.get('Crit. Multiplier') or 0)
                }
                genes_data.append(gene_stat)
                # Sum the stats
                for key in total_stats:
                    if key != 'Gene' and gene_stat[key] is not None:
                        total_stats[key] += gene_stat[key]

        # Append total_stats to genes_data
        genes_data.append(total_stats)

        # Convert to DataFrame
        df = pd.DataFrame(genes_data)
        return df

    def get_selected_genes(self) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Retrieves the currently selected genes.

        Returns:
            Dict[str, Dict[str, Optional[str]]]: Selected genes categorized by gene type.
        """
        categories = ['FACTION', 'JOB', 'WEAPON', 'ELEMENT', 'QUIRK']
        gene_selection = {}
        for category in categories:
            gene_selection[category] = {
                'Dominant': self.selected_genes1.get(category),
                'Recessive': self.selected_genes2.get(category),
                'Minor': self.selected_genes3.get(category)
            }
        return gene_selection

# ---------------------------
# FUNCTION TO GENERATE RANDOM CHILD GENOTYPE
# ---------------------------

def generate_random_child_genotype(top_allele_combos_per_locus):
    """
    Generates a random child genotype from the top allele combinations per locus and calculates its stats.
    """
    if top_allele_combos_per_locus is None:
        return "<h3>Error:</h3><p>Please calculate the offspring genotypes first.</p>"

    child_genotype = []
    for locus_combos in top_allele_combos_per_locus:
        # Unzip allele combinations and their probabilities
        allele_combos, probs = zip(*locus_combos)

        # Normalize probabilities
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        # Randomly select one allele combination based on probabilities
        selected_idx = np.random.choice(len(allele_combos), p=probs)
        selected_alleles = allele_combos[selected_idx]
        child_genotype.append(selected_alleles)

    # Format the child genotype for display and for the stats calculator
    formatted_lines = []
    gene_sequence_lines = []
    for idx, alleles in enumerate(child_genotype):
        # Remove any placeholders like "-"
        alleles = [allele for allele in alleles if allele != '-']
        # Depending on the locus, adjust the formatting
        if idx == 4:  # Quirk has different formatting
            formatted_line = ", ".join(alleles) + ","
            gene_sequence_line = f"{', '.join(alleles)},"
        else:
            formatted_line = ", ".join(alleles) + ","
            gene_sequence_line = f"{', '.join(alleles)},"
        formatted_lines.append(formatted_line)
        gene_sequence_lines.append(gene_sequence_line)

    formatted_genotype = "\n".join(formatted_lines)
    gene_sequence = "\n".join(gene_sequence_lines)

    # Calculate stats using GeneStatsCalculator
    calculator = GeneStatsCalculator()
    try:
        calculator.import_gene_sequence(gene_sequence)
        stats_df = calculator.calculate_stats()
        stats_html = stats_df.to_html(index=False)
    except Exception as e:
        stats_html = f"<p>Error calculating stats: {e}</p>"

    random_child_report = (
        "<h3>Random Child Genotype from Top Combinations:</h3>"
        f"<pre>{formatted_genotype}</pre>"
        "<h3>Calculated Stats:</h3>"
        f"{stats_html}"
    )
    return random_child_report

# ---------------------------
# GRADIO INTERFACE SETUP
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Genotype Calculator with Automated Data Retrieval and Stats Calculation")
    gr.Markdown("This application retrieves parent genotypes from provided URLs, calculates the most probable offspring genotypes, and displays the stats for a randomly generated child genotype.")
    gr.Markdown("A correct URL starts with https://championstactics.ubisoft.com/items/champions/ followed by the number of the champion.")

    with gr.Row():
        with gr.Column():
            parent1_url_input = gr.Textbox(
                label="Parent 1 URL",
                placeholder="Enter the URL of Parent 1's genotype page",
                lines=2
            )
        with gr.Column():
            parent2_url_input = gr.Textbox(
                label="Parent 2 URL",
                placeholder="Enter the URL of Parent 2's genotype page",
                lines=2
            )

    with gr.Row():
        calculate_button = gr.Button("Calculate Offspring Genotypes")

    with gr.Row():
        output_display = gr.HTML(label="Genotype Reports")

    # Increased spacing and clear layout for mobile
    with gr.Row():
        with gr.Column(scale=1):
            generate_random_child_button = gr.Button("Generate Random Child Genotype and Calculate Stats")
        with gr.Column(scale=3):
            output_random_child = gr.HTML(label="Random Child Genotype and Stats")

    top_allele_combos_state = gr.State()

    # Define interactions
    calculate_button.click(
        process_and_calculate,
        inputs=[parent1_url_input, parent2_url_input],
        outputs=[output_display, top_allele_combos_state]
    )

    generate_random_child_button.click(
        generate_random_child_genotype,
        inputs=top_allele_combos_state,
        outputs=output_random_child
    )

    gr.Markdown("""
    ---
    **Instructions:**
    1. Enter the URLs of the genotype pages for **Parent 1** and **Parent 2**.
    2. Click "**Calculate Offspring Genotypes**" to generate the report.
    3. Click "**Generate Random Child Genotype and Calculate Stats**" to generate a random child genotype and see its stats.
    4. Ensure the URLs are correct and accessible.

    **Global Crafting Rules:**
    1. TLDR: Dominant -> 30%, Recessive -> 15%, Minor recessive -> 5%
    2. The chance to get a secondary weapon, element, and quirk is 10% + 10% for each parent with a secondary weapon, element, or quirk.
    3. The chance to get an exalted is 1%
    """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
