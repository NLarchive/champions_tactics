# @title Optimized Child Genotype Calculation with Correct Trait Frequency Calculation

import pandas as pd
from itertools import permutations, product

# -------------------------------------------------------------------
# Step 1: Define Parent Alleles and Transmission Probabilities
# -------------------------------------------------------------------

# Define the parent alleles for each locus (Feature)
alleles_p1 = [
    ["a1", "a2", "a3"],  # Feature 1 for Parent 1
    ["b1", "b2", "b3"],  # Feature 2 for Parent 1
    ["c1", "c2", "c3"],  # Feature 3 for Parent 1
    ["d1", "d2", "d3"],  # Feature 4 for Parent 1
    ["e1", "e2"]          # Feature 5 for Parent 1
]

alleles_p2 = [
    ["a1", "a2", "a3"],  # Feature 1 for Parent 2
    ["b1", "b2", "b3"],  # Feature 2 for Parent 2
    ["c1", "c2", "c3"],  # Feature 3 for Parent 2
    ["d1", "d2", "d3"],  # Feature 4 for Parent 2
    ["e1", "e2"]          # Feature 5 for Parent 2
]

# Define the transmission probabilities for each parent's alleles at each locus
probs_p1 = [
    [0.3, 0.15, 0.05],    # Transmission probabilities for Parent 1's alleles at Feature 1
    [0.3, 0.15, 0.05],    # Feature 2
    [0.3, 0.15, 0.05],    # Feature 3
    [0.3, 0.15, 0.05],    # Feature 4
    [0.375, 0.125]        # Feature 5
]

probs_p2 = [
    [0.3, 0.15, 0.05],    # Transmission probabilities for Parent 2's alleles at Feature 1
    [0.3, 0.15, 0.05],    # Feature 2
    [0.3, 0.15, 0.05],    # Feature 3
    [0.3, 0.15, 0.05],    # Feature 4
    [0.375, 0.125]        # Feature 5
]

# Define the number of alleles to select per locus
perm_lengths = [3, 3, 3, 3, 2]  # First four loci: 3 alleles, fifth locus: 2 alleles

# -------------------------------------------------------------------
# Step 2: Generate Allele Combinations with Probabilities
# -------------------------------------------------------------------

all_dfs = []  # List to store DataFrames for each locus

for i in range(len(alleles_p1)):
    # Combine alleles from both parents for the current locus
    combined_alleles = alleles_p1[i] + alleles_p2[i]
    combined_probs = probs_p1[i] + probs_p2[i]

    # Determine the number of alleles to select for the current locus
    perm_length = perm_lengths[i]

    # Generate all possible sequences of perm_length alleles using permutations
    allele_combos = list(permutations(combined_alleles, perm_length))

    # Function to map an allele to its original transmission probability
    def get_prob(allele):
        if allele in alleles_p1[i]:
            return probs_p1[i][alleles_p1[i].index(allele)]
        elif allele in alleles_p2[i]:
            return probs_p2[i][alleles_p2[i].index(allele)]
        return 0  # Safety fallback

    # Create a DataFrame with the allele combinations
    if perm_length == 3:
        # Dynamic Column Creation for 3 Alleles
        df = pd.DataFrame(allele_combos, columns=["Allele1", "Allele2", "Allele3"])
        # Add the original transmission probabilities for each allele at its respective locus
        df["Prob_Allele1"] = df["Allele1"].apply(get_prob)
        df["Prob_Allele2"] = df["Allele2"].apply(get_prob)
        df["Prob_Allele3"] = df["Allele3"].apply(get_prob)

        # Function to calculate position-based probabilities dynamically
        def calc_pos_probs(row):
            prob1 = row["Prob_Allele1"]
            prob2 = row["Prob_Allele2"]
            prob3 = row["Prob_Allele3"]

            # Calculate position-based probabilities
            pos_prob1 = prob1
            pos_prob2 = prob2 / (1 - prob1) if (1 - prob1) > 0 else 0  # Avoid division by zero
            pos_prob3 = prob3 / (1 - prob1 - prob2) if (1 - prob1 - prob2) > 0 else 0  # Avoid division by zero

            return pd.Series([pos_prob1, pos_prob2, pos_prob3])

        # Apply the function to compute the positional transmission probabilities
        df[["Pos_Prob1", "Pos_Prob2", "Pos_Prob3"]] = df.apply(calc_pos_probs, axis=1)

        # Calculate the total sequence transmission probability as the product of all position-based probabilities
        df["Total_Prob"] = df["Pos_Prob1"] * df["Pos_Prob2"] * df["Pos_Prob3"]

    elif perm_length == 2:
        # Dynamic Column Creation for 2 Alleles
        df = pd.DataFrame(allele_combos, columns=["Allele1", "Allele2"])
        # Add the original transmission probabilities for each allele at its respective locus
        df["Prob_Allele1"] = df["Allele1"].apply(get_prob)
        df["Prob_Allele2"] = df["Allele2"].apply(get_prob)

        # Function to calculate position-based probabilities dynamically
        def calc_pos_probs(row):
            prob1 = row["Prob_Allele1"]
            prob2 = row["Prob_Allele2"]

            # Calculate position-based probabilities
            pos_prob1 = prob1
            pos_prob2 = prob2 / (1 - prob1) if (1 - prob1) > 0 else 0  # Avoid division by zero

            return pd.Series([pos_prob1, pos_prob2])

        # Apply the function to compute the positional transmission probabilities
        df[["Pos_Prob1", "Pos_Prob2"]] = df.apply(calc_pos_probs, axis=1)

        # Calculate the total sequence transmission probability as the product of all position-based probabilities
        df["Total_Prob"] = df["Pos_Prob1"] * df["Pos_Prob2"]

    # Store the DataFrame for each locus
    all_dfs.append(df)

    # Display the first few rows for verification for each locus
    print(f"\nAllele Combinations with Position-Based Transmission Probabilities for Locus Set {i + 1}:")
    print(df.head(10))  # Display only the first 10 rows for brevity

# -------------------------------------------------------------------
# Step 3: Combine All Loci Data into a Final DataFrame
# -------------------------------------------------------------------

# Insert a 'Locus' identifier for each DataFrame
for idx, df in enumerate(all_dfs, 1):
    df.insert(0, "Locus", f"Locus{idx}")

# Concatenate all DataFrames vertically
final_df = pd.concat(all_dfs, ignore_index=True)

# Display the combined DataFrame (optional)
print("\nFinal Combined DataFrame:")
print(final_df.head(20))  # Display the first 20 rows for brevity

# -------------------------------------------------------------------
# Step 4: Optimize Child Genotype Calculation by Selecting Top N Allele Combinations per Locus
# -------------------------------------------------------------------

# Define the number of top allele combinations to select per locus
TOP_N = 5  # Adjust based on your requirements

# Group the final_df by 'Locus'
grouped = final_df.groupby("Locus")

# For each locus, select the top N allele combinations based on 'Total_Prob'
top_allele_combos_per_locus = []
for locus, group in grouped:
    # Sort the group by 'Total_Prob' in descending order and select top N
    top_combos = group.sort_values(by='Total_Prob', ascending=False).head(TOP_N)

    # Extract the allele combinations and their probabilities
    # Ensure only allele name columns are selected
    allele_columns = [col for col in top_combos.columns if col.startswith('Allele')]
    allele_combinations = top_combos[allele_columns].values.tolist()
    probabilities = top_combos['Total_Prob'].values.tolist()

    # Combine alleles and probabilities into separate lists
    # Ensure only allele names are included in the tuple
    locus_combos = [ (tuple(alleles), prob) for alleles, prob in zip(allele_combinations, probabilities) ]
    top_allele_combos_per_locus.append(locus_combos)

    print(f"\nTop {TOP_N} Allele Combinations for {locus}:")
    for combo, prob in locus_combos:
        print(f"  Alleles: {combo}, Probability: {prob:.6f}")

# -------------------------------------------------------------------
# Step 5: Generate Child Genotypes from Top Allele Combinations
# -------------------------------------------------------------------

# Use itertools.product to generate all possible child genotype combinations from top allele combos
all_genotype_combinations = product(*top_allele_combos_per_locus)

# Initialize a list to store child genotypes and their total probabilities
child_genotypes = []

for genotype in all_genotype_combinations:
    genotype_dict = {}
    total_prob = 1.0
    for locus_idx, (alleles, prob) in enumerate(genotype, 1):
        locus_name = f"Locus{locus_idx}"
        genotype_dict[locus_name] = alleles  # Only allele names
        total_prob *= prob
    genotype_dict['Total_Probability'] = total_prob
    child_genotypes.append(genotype_dict)

# Convert the list of genotype dictionaries into a DataFrame
child_genotypes_df = pd.DataFrame(child_genotypes)

# -------------------------------------------------------------------
# Step 6: Sort Child Genotypes by Total Probability
# -------------------------------------------------------------------

# Sort the genotypes by 'Total_Probability' in descending order
child_genotypes_sorted = child_genotypes_df.sort_values(by='Total_Probability', ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------
# Step 7: Merge Duplicated Child Genotypes and Aggregate Probabilities
# -------------------------------------------------------------------

# Define the loci columns for grouping
loci_columns = [col for col in child_genotypes_sorted.columns if col.startswith('Locus')]

# Group by all loci to identify unique genotypes
grouped_genotypes = child_genotypes_sorted.groupby(loci_columns)

# Compute the global probability for each unique genotype by summing their probabilities
unique_genotypes = grouped_genotypes['Total_Probability'].sum().reset_index()
unique_genotypes.rename(columns={'Total_Probability': 'Global_Probability'}, inplace=True)

# Sort the unique genotypes by 'Global_Probability' in descending order
unique_genotypes_sorted = unique_genotypes.sort_values(by='Global_Probability', ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------
# Step 8: Display the Top M Most Probable Unique Child Genotypes
# -------------------------------------------------------------------

# Define the number of top child genotypes to display
TOP_M = 10

print(f"\nTop {TOP_M} Most Probable Unique Child Genotypes:")
print(unique_genotypes_sorted.head(TOP_M))

# -------------------------------------------------------------------
# Step 9: Create Trait Frequency Table
# -------------------------------------------------------------------

# Initialize an empty list to store trait data
trait_data = []

# Populate the list with trait information for each feature
for feature_idx, df in enumerate(all_dfs, 1):
    feature_name = f"Feature{feature_idx}"

    # Calculate the frequency of each allele by summing 'Total_Prob' across all unique combinations where the allele appears
    allele_freq = {}
    for _, row in df.iterrows():
        # Use a set to ensure each allele is counted only once per combination
        unique_alleles = set()
        unique_alleles.add(row['Allele1'])
        unique_alleles.add(row['Allele2'])
        if 'Allele3' in row and pd.notnull(row['Allele3']):
            unique_alleles.add(row['Allele3'])

        for allele in unique_alleles:
            allele_freq[allele] = allele_freq.get(allele, 0) + row['Total_Prob']

    # Sort alleles by frequency in descending order
    sorted_alleles = sorted(allele_freq.items(), key=lambda x: x[1], reverse=True)

    # Prepare the row for the trait frequency table
    row = [feature_name]
    for allele, prob in sorted_alleles[:3]:  # Limit to top 3 alleles
        row.extend([allele, f"{prob*100:.2f}%"])

    # If less than 3 alleles, fill empty cells
    while len(row) < 7:
        row.extend(["", ""])

    trait_data.append(row)

# Create the Trait Frequency Table DataFrame
trait_frequency_df = pd.DataFrame(trait_data, columns=[
    'Feature', 'Trait1', 'Probability1 (%)',
    'Trait2', 'Probability2 (%)', 'Trait3', 'Probability3 (%)'
])

# -------------------------------------------------------------------
# Step 10: Display the Trait Frequency Table
# -------------------------------------------------------------------

print("\nTrait Frequency Table:")
print(trait_frequency_df.to_string(index=False))

# -------------------------------------------------------------------
# Step 11: Additional Fun Facts and Summaries
# -------------------------------------------------------------------

# Function to calculate how often each trait appears in the offspring
def calculate_trait_frequencies(offspring_df):
    trait_frequencies = {}
    total_probability = offspring_df['Global_Probability'].sum()
    for feature_idx in range(1, 6):
        locus = f"Locus{feature_idx}"
        counts = offspring_df.groupby(locus)['Global_Probability'].sum()
        frequencies = counts / total_probability
        feature_name = f"Feature{feature_idx}"
        trait_frequencies[feature_name] = frequencies
    return trait_frequencies

trait_frequencies = calculate_trait_frequencies(unique_genotypes_sorted)

# Display trait frequencies
print("\nHow Often Each Trait Appears in All Possible Pets:")
for feature, freq in trait_frequencies.items():
    print(f"\n{feature}:")
    for trait, frequency in freq.items():
        print(f"  {trait}: {frequency*100:.2f}%")

# Calculate total chance of the top pets
cumulative_chance = unique_genotypes_sorted['Global_Probability'].head(TOP_M).sum()
print(f"\nTotal Chance of Getting One of the Top {TOP_M} Pets: {cumulative_chance*100:.2f}%")

# Number of offspring with a reasonable chance
CHANCE_THRESHOLD = 0.01  # 1%
num_reasonable_pets = (unique_genotypes_sorted['Global_Probability'] >= CHANCE_THRESHOLD / 100).sum()
print(f"\nNumber of Pets with a Chance Greater Than {CHANCE_THRESHOLD}%: {num_reasonable_pets}")

# -------------------------------------------------------------------
# Step 12: Create Trait Position Probability Matrix
# -------------------------------------------------------------------

# Display the trait frequency table as the Trait Position Probability Matrix
print("\nTrait Position Probability Matrix:")
print(trait_frequency_df.to_string(index=False))
