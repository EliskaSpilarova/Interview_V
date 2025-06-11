import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# načtení CSV souboru a kontrola
def load_and_prepare_data(filepath):
    df = pd.read_csv("Dummy_practise_SQL.csv", sep=';', encoding='cp1250')
    df.info()
    df.head()

# oprava čárek v souřadnicích, odstranění NaN řádků v souřadnicích, převod fbb na float a hp na integer
    df['latitude'] = df['latitude'].str.replace(',', '.').astype(float)
    df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)
    df['fbb_penetration_desetinne_cislo'] = df['fbb_penetration_desetinne_cislo'].str.replace(',', '.').astype(float)
    df['hp'] = df['hp'].astype(int)
    df = df.dropna(subset=['latitude', 'longitude'])

    return df

# výpočet vzdálenosti z každého řádku k zadaným městům
def compute_distances(df, cities):
    for city, coords in cities.items():
        df[f'distance_to_{city.lower()}'] = df.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), coords).km,
            axis=1
        )
    return df

# filtrování podle parametrů
def filter_city(df, city):
    dist_col = f'distance_to_{city.lower()}'
    return df[
        (df['hp'] >= 50) &
        (df['fbb_penetration_desetinne_cislo'] < 0.3) &
        (df[dist_col] <= 100)
    ].copy()

# clustery pro týmy a počet domácností v clusteru
def cluster_objects(df_filtered, city, n_clusters=5):
    if len(df_filtered) < n_clusters:
        n_clusters = max(1, len(df_filtered))
    coords = df_filtered[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df_filtered['cluster'] = kmeans.fit_predict(coords)

    cluster_hp = df_filtered.groupby('cluster')['hp'].sum().reset_index()
    cluster_hp_filename = f'soucet_hp_{city.lower()}.csv'
    cluster_hp.to_csv(cluster_hp_filename, index=False)
    print(f"\nSoučet domácností (hp) podle clusteru uložen do: {cluster_hp_filename}")
    print(cluster_hp)

    return df_filtered

# export CSV souborů
def export_results(df, city):
    filename = f'vystup_{city.lower()}.csv'
    df.to_csv(filename, index=False)
    print(f"Výsledky pro {city} uloženy do: {filename} ({len(df)} záznamů)")

# hlavní funkce
def main():
    filepath = 'Dummy_practise_SQL.csv'

    cities = {
        'Praha': (50.0755, 14.4378),
        'Liberec': (50.7663, 15.0540),
        'Brno': (49.1951, 16.6068),
        'Ostrava': (49.8209, 18.2625),
    }

    print("Načítání a čištění dat...")
    df = load_and_prepare_data(filepath)

    print("Výpočet vzdáleností...")
    df = compute_distances(df, cities)

    for city in cities:
        print(f"\nZpracování města: {city}")
        df_filtered = filter_city(df, city)
        if df_filtered.empty:
            print(f"Žádné objekty nesplňují podmínky pro {city}.")
            continue
        df_clustered = cluster_objects(df_filtered, city)
        export_results(df_clustered, city)

if __name__ == '__main__':
    main()
