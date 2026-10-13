import pandas as pd
import numpy as np
import sqlite3
import json 

def plant_label_stoy(df, andel=0.1): # ca 10 prosent av dataene skal odelegges, den snur verdiene tilfeldig for en gitt andel av radene.
    df_kopi = df.copy()

    #Velger randomme rader å ødelegge
    antall_rader = int(len(df_kopi) * andel)
    valgte_indekser = df_kopi.sample(n=antall_rader, random_state=42).index

    # Snu verdien 0 blir 1, 1 blir 0
    df_kopi.loc[valgte_indekser, "Diabetes_binary"] = 1 - df_kopi.loc[valgte_indekser, "Diabetes_binary"]


    
    fasit = {
        "type": "label_stoy",
        "andel": andel,
        "antall_rader_endret": antall_rader,
        "radene_som_ble_endret": valgte_indekser.tolist()
    }
    
    return df_kopi, fasit


def plant_urealistisk_bmi(df, andel=0.05, kolonne="BMI"): # ca 5 prosent av radene får BMI satt til 0 eller 999.
    df_kopi = df.copy()

    # Velger randomme rader å ødelegge
    antall_rader = int(len(df_kopi) * andel)
    valgte_indekser = df_kopi.sample(n=antall_rader, random_state=42).index

    # Setter BMI til enten 0 eller 999 tilfeldig
    rng = np.random.RandomState(42)
    df_kopi.loc[valgte_indekser, kolonne] = rng.choice([0, 999], size=antall_rader)

    
    fasit = {
        "type": "urealistisk_bmi",
        "kolonne": kolonne,
        "andel": andel,
        "antall_rader_endret": antall_rader,
        "radene_som_ble_endret": valgte_indekser.tolist()
    }

    return df_kopi, fasit


def plant_manglende_verdier(df, andel=0.15, kolonne="Income"): # ca 15 prosent av radene i en kolonne blir NaN eller null i databasen. 
    df_kopi = df.copy()
    df_kopi[kolonne] = df_kopi[kolonne].astype(float)  # må være float for at NaN skal fungere

    # Velger randomme rader å ødelegge
    antall_rader = int(len(df_kopi) * andel)
    valgte_indekser = df_kopi.sample(n=antall_rader, random_state=42).index

    # Fjerner verdien i de valgte radene
    df_kopi.loc[valgte_indekser, kolonne] = np.nan

    
    fasit = {
        "type": "manglende_verdier",
        "kolonne": kolonne,
        "andel": andel,
        "antall_rader_endret": antall_rader,
        "radene_som_ble_endret": valgte_indekser.tolist()
    }

    return df_kopi, fasit


def plant_kolonneombytting(df, kol_a="BMI", kol_b="PhysHlth"): # bytter innholdet i to kolonner for alle rader
    df_kopi = df.copy()

    # Bytter plass på de to kolonnene
    df_kopi[[kol_a, kol_b]] = df_kopi[[kol_b, kol_a]].values

    
    fasit = {
        "type": "kolonneombytting",
        "kolonner": [kol_a, kol_b],
        "andel": 1.0,
        "antall_rader_endret": len(df_kopi)
    }

    return df_kopi, fasit


# Hver feiltype får sin egen tabell og fasit-fil.
feil = {
    "diabetes_binary_corrupt": ("label_stoy", plant_label_stoy),
    "diabetes_binary_bmi": ("urealistisk_bmi", plant_urealistisk_bmi),
    "diabetes_binary_missing": ("manglende_verdier", plant_manglende_verdier),
    "diabetes_binary_swap": ("kolonneombytting", plant_kolonneombytting),
}

# connecter med diabetes.db
conn = sqlite3.connect("diabetes.db")
df = pd.read_sql("SELECT * FROM diabetes_binary", conn)

for tabellnavn, (feiltype, funksjon) in feil.items():
    # Plant feilen
    df_odelagt, fasit = funksjon(df)

    # Lagre den odelagte tabellen som en ny tabell i databasen
    df_odelagt.to_sql(tabellnavn, conn, if_exists="replace", index=False)

    # Lagre fasiten som en JSON-fil
    with open(f"fasit_{feiltype}.json", "w") as f:
        json.dump(fasit, f, indent=2)

    print(f"Ferdig! {tabellnavn}: {fasit['antall_rader_endret']} rader ble endret.")
    print(f"Fasit lagret i fasit_{feiltype}.json")

conn.close()