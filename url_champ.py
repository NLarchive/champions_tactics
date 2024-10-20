# @title CLEANED DATA, HANDLE DUPLICATES IF MORE THAN THE NUMBER OF EXPECTED FIELDS
import requests
from bs4 import BeautifulSoup
import pandas as pd

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
        "Light", "Nature", "Order", "Techno", "Water"
    ],
    "Weapon": [
        "Blunderbuss", "Crossbow", "Dagger", "Greataxe",
        "Great sword", "Halberd", "Katana", "Mystical horn",
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
        response = requests.get(url)
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
    specific_data = extract_between(extracted_text, start_sequence, stop_sequence)

    if specific_data.startswith("Error"):
        return specific_data

    parsed_data = parse_extracted_data(specific_data, fields, field_values)
    return create_dataframe(parsed_data)

if __name__ == "__main__":
    urls = [
        "https://championstactics.ubisoft.com/items/champions/19560",
        "https://championstactics.ubisoft.com/items/champions/14032"
    ]

    start_sequence = "Dominant Recessive Minor Recessive"
    stop_sequence = "Skills Reactive"
    fields = ["Faction", "Job", "Weapon", "Element", "Quirk"]

    for url in urls:
        print(f"\nProcessing URL: {url}")
        df = extract_and_structure_data(url, start_sequence, stop_sequence, fields, FIELDS)

        if isinstance(df, str):
            print(df)
        else:
            print("Structured DataFrame:")
            print(df)
