# %% [markdown]
# # Netflix Catalogue: Exploratory Data Analysis
#
# **Question we are trying to answer:** what does Netflix's catalogue actually look like,
# where does it come from, and how has it changed as the platform grew?
#
# **Dataset:** 8,807 titles (movies + TV shows) listed on Netflix, with cast, director,
# country, date added, release year, rating, duration and genre.
#
# **Plan**
# 1. Load and inspect the data
# 2. Clean it (missing values, a few mis-shifted rows, dates, multi-value columns)
# 3. Explore: type split, growth over time, countries, genres, ratings, duration, people
# 4. Pull the findings together
#
# Run top to bottom. Charts are also saved to `figures/` (used in the presentation).

# %%
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = "data/netflix_data.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Palette used across every chart (also used in the slide deck)
PAPER = "#F3EBDC"   # warm off-white background
INK = "#1C1814"     # near-black text
RED = "#B3261E"     # Netflix-ish brick red (movies / highlight)
GOLD = "#D69E2E"    # secondary accent (TV shows)
MUTED = "#8C8173"   # grey-brown for context bars
GRID = "#DCD1BC"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 110,
})


def save(fig, name):
    """Save a figure into figures/ and show it."""
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=200)
    plt.show()


# %% [markdown]
# ## 1. Load and inspect

# %%
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)
raw.head(3)

# %%
raw.info()

# %%
missing = raw.isna().sum().to_frame("missing")
missing["pct"] = (missing["missing"] / len(raw) * 100).round(1)
missing[missing["missing"] > 0].sort_values("missing", ascending=False)

# %% [markdown]
# **What stands out**
# - `director` is missing for ~30% of titles, which is expected (many shows and docs have no single credited director).
# - `cast` and `country` are missing for roughly 9-10% each.
# - `date_added`, `rating`, `duration` have only a handful of gaps.
# - No duplicate `show_id`s, so no duplicate rows to drop.
#
# The `rating` column also has odd values (`74 min`, `84 min`, `66 min`). Let's look.

# %%
print("Duplicate rows:", raw.duplicated().sum(), "| duplicate IDs:", raw["show_id"].duplicated().sum())
raw[raw["duration"].isna()][["title", "type", "rating", "duration"]]

# %% [markdown]
# These three rows (all Louis C.K. specials) have the **duration typed into the rating column**
# and nothing in `duration`. It's a data-entry shift. We move the value across and mark the rating as unknown.
#
# ## 2. Cleaning

# %%
df = raw.copy()

# 2a. Fix the shifted rows
shifted = df["rating"].astype(str).str.contains("min", na=False)
df.loc[shifted, "duration"] = df.loc[shifted, "rating"]
df.loc[shifted, "rating"] = np.nan
print("Rows repaired:", shifted.sum())

# 2b. Text columns: strip stray whitespace
for col in ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in"]:
    df[col] = df[col].astype("string").str.strip()

# 2c. Missing categorical values become an explicit 'Unknown' (so they stay visible, not silently dropped)
for col in ["director", "cast", "country", "rating"]:
    df[col] = df[col].fillna("Unknown")

# 2d. Dates
df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# 2e. Duration -> number. Movies are in minutes, TV shows in seasons.
df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["minutes"] = np.where(df["type"] == "Movie", df["duration_num"], np.nan)
df["seasons"] = np.where(df["type"] == "TV Show", df["duration_num"], np.nan)

# 2f. How long after release did Netflix add the title?
df["years_to_netflix"] = df["year_added"] - df["release_year"]

# 2g. Group the ratings into audience bands (easier to read than 14 codes)
audience = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Older Kids", "PG": "Older Kids",
    "TV-14": "Teens", "PG-13": "Teens",
    "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults",
    "NR": "Unrated", "UR": "Unrated", "Unknown": "Unrated",
}
df["audience"] = df["rating"].map(audience)

print("Remaining nulls in key columns:")
print(df[["date_added", "duration", "minutes", "seasons", "audience"]].isna().sum())

# %% [markdown]
# Only 10 titles have no `date_added`; we leave those out of the time-based charts instead of guessing.
#
# Several columns hold **several values in one cell** (`country`, `listed_in`, `cast`, `director`).
# For any "top X" question we split and explode them so every country/genre/person is counted.

# %%
def explode(frame, col):
    out = frame[["show_id", "type", col]].copy()
    out[col] = out[col].str.split(",")
    out = out.explode(col)
    out[col] = out[col].str.strip()
    return out[(out[col] != "") & (out[col] != "Unknown")]


countries = explode(df, "country")
genres = explode(df, "listed_in")
actors = explode(df, "cast")
directors = explode(df, "director")

print(f"{len(countries):,} country tags | {len(genres):,} genre tags | "
      f"{len(actors):,} cast credits | {len(directors):,} director credits")

# %% [markdown]
# ## 3. Exploration
#
# ### 3.1 Movies vs TV shows

# %%
type_counts = df["type"].value_counts()
type_pct = (type_counts / len(df) * 100).round(1)
print(type_counts.to_string(), "\n", type_pct.to_string())

fig, ax = plt.subplots(figsize=(8, 2.6))
ax.grid(False)
left = 0
for label, color in [("Movie", RED), ("TV Show", GOLD)]:
    w = type_counts[label]
    ax.barh(0, w, left=left, color=color, height=0.55)
    ax.text(left + w / 2, 0, f"{label}\n{w:,}  ({w / len(df):.0%})", ha="center", va="center",
            color="white" if label == "Movie" else INK, fontsize=12, fontweight="bold")
    left += w
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Catalogue split: roughly 7 movies for every 3 shows")
save(fig, "01_type_split")

# %% [markdown]
# **Insight:** Movies are about 70% of the catalogue. Netflix is still mainly a movie library by *title count*,
# even though its brand is built on series (a series takes far more watch-time per title, which this data can't show).
#
# ### 3.2 How fast did the catalogue grow?

# %%
added = (df.dropna(subset=["year_added"])
           .groupby(["year_added", "type"]).size().unstack(fill_value=0))
added = added.loc[2012:]          # the years before this are a handful of titles
added["Total"] = added.sum(axis=1)
print(added.rename_axis(index=None))

fig, ax = plt.subplots(figsize=(9, 4.6))
x = added.index.astype(int)
ax.bar(x, added["Movie"], color=RED, label="Movies", width=0.7)
ax.bar(x, added["TV Show"], bottom=added["Movie"], color=GOLD, label="TV Shows", width=0.7)
ax.set_xticks(x)
ax.set_xticklabels([str(i) if i != 2021 else "2021*" for i in x])
ax.set_ylabel("Titles added")
ax.set_title("Titles added to Netflix each year")
ax.legend(frameon=False, loc="upper left")
ax.text(0.99, -0.14, "*2021 only runs to September", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
save(fig, "02_growth")

peak_year = int(added["Total"].idxmax())
print("Peak year:", peak_year, "->", int(added["Total"].max()), "titles")

# %% [markdown]
# **Insight:** Additions were tiny before 2015, jumped from ~80 titles in 2015 to ~430 in 2016 and ~1,190 in 2017,
# and peaked in **2019** at 2,016 titles. 2020 dipped (the pandemic is a plausible cause, though the data can't prove it).
# 2021 had already added ~1,500 by September, so it was on course to match the earlier years.
# TV shows grew later and more slowly than movies.
#
# ### 3.3 Old catalogue or new?

# %%
release_bins = pd.cut(df["release_year"], [1924, 1999, 2009, 2014, 2018, 2021],
                      labels=["Before 2000", "2000-09", "2010-14", "2015-18", "2019-21"])
release_share = release_bins.value_counts(normalize=True).sort_index() * 100
print(release_share.round(1).to_string())

gap = df["years_to_netflix"].dropna()
print(f"\nMedian gap between release and landing on Netflix: {gap.median():.0f} years")
print(f"Added in the same year as release: {(gap == 0).mean():.0%} | within 2 years: {(gap <= 2).mean():.0%}")

fig, ax = plt.subplots(figsize=(9, 4.4))
bins = np.arange(0, 21, 1)
ax.hist(gap.clip(upper=20), bins=bins, color=MUTED, edgecolor=PAPER)
ax.hist(gap[gap <= 1], bins=bins, color=RED, edgecolor=PAPER)
ax.set_xlabel("Years between release and being added to Netflix (20 = 20 or more)")
ax.set_ylabel("Titles")
ax.set_title("Most titles arrive within a year or two of release")
save(fig, "03_release_gap")

# %% [markdown]
# **Insight:** The catalogue is young. About 70% of titles were released in 2015 or later, and the median title
# reaches Netflix just **1 year** after release (63% within 2 years). That fits a mix of fresh licensing and Netflix originals.
# The long tail on the right is the back-catalogue of older films.
#
# ### 3.4 Where does the content come from?

# %%
top_c = countries["country"].value_counts().head(10)
print(top_c.to_string())
share_us = top_c.iloc[0] / df["country"].ne("Unknown").sum()
print(f"\nUS appears in {share_us:.0%} of titles with a known country")

fig, ax = plt.subplots(figsize=(8, 4.8))
colors = [RED if c in ("United States", "India") else MUTED for c in top_c.index]
ax.barh(top_c.index[::-1], top_c.values[::-1], color=colors[::-1])
for i, v in enumerate(top_c.values[::-1]):
    ax.text(v + 25, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("Titles (a co-production counts for each country)")
ax.set_title("Top 10 producing countries")
ax.grid(axis="y", visible=False)
save(fig, "04_countries")

# %%
# Do countries lean towards movies or shows?
top8 = top_c.head(8).index
mix = (countries[countries["country"].isin(top8)]
       .groupby(["country", "type"]).size().unstack(fill_value=0))
mix["tv_share"] = mix["TV Show"] / mix.sum(axis=1) * 100
mix = mix.sort_values("tv_share")
print(mix["tv_share"].round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(mix.index, 100 - mix["tv_share"], color=RED, label="Movies")
ax.barh(mix.index, mix["tv_share"], left=100 - mix["tv_share"], color=GOLD, label="TV Shows")
for i, (c, r) in enumerate(mix.iterrows()):
    ax.text(100 - r["tv_share"] + r["tv_share"] / 2, i, f"{r['tv_share']:.0f}%", ha="center", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("% of country's titles")
ax.set_title("Movies vs TV mix by country")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.grid(False)
save(fig, "05_country_mix")

# %% [markdown]
# **Insight:** The US appears in 46% of titles with a known country, but India is a clear second and is
# almost entirely **movies** (92%). South Korea (74% TV) and Japan (63% TV) show the opposite pattern: K-dramas and anime.
#
# ### 3.5 Genres

# %%
top_g = genres["listed_in"].value_counts().head(10)
print(top_g.to_string())

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(top_g.index[::-1], top_g.values[::-1], color=[RED if i >= 7 else MUTED for i in range(10)])
for i, v in enumerate(top_g.values[::-1]):
    ax.text(v + 30, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("Titles (a title can carry several genres)")
ax.set_title("Top 10 genres")
ax.grid(axis="y", visible=False)
save(fig, "06_genres")

# %%
# Top genres inside each type
g_movie = genres[genres["type"] == "Movie"]["listed_in"].value_counts().head(5)
g_tv = genres[genres["type"] == "TV Show"]["listed_in"].value_counts().head(5)
print("MOVIES\n", g_movie.to_string(), "\n\nTV SHOWS\n", g_tv.to_string())

# %% [markdown]
# **Insight:** "International" is the biggest label for both movies (2,752) and TV (1,351), followed by Dramas and Comedies.
# Beyond those, movies lean to Documentaries and Action, while TV leans to Crime and Kids' TV.
# The dominance of "International" reinforces the global spread seen in 3.4.
#
# ### 3.6 Who is the content made for? (Ratings)

# %%
order = ["Kids", "Older Kids", "Teens", "Adults", "Unrated"]
aud = df["audience"].value_counts().reindex(order)
print((aud / len(df) * 100).round(1).to_string())

top_r = df["rating"].value_counts().head(8)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
axes[0].bar(top_r.index, top_r.values, color=[RED if r == "TV-MA" else MUTED for r in top_r.index])
for i, v in enumerate(top_r.values):
    axes[0].text(i, v + 40, f"{v:,}", ha="center", fontsize=8)
axes[0].set_title("Most common ratings")
axes[0].grid(axis="x", visible=False)
axes[1].barh(aud.index[::-1], aud.values[::-1], color=[MUTED, RED, MUTED, MUTED, MUTED])  # Adults highlighted
for i, v in enumerate(aud.values[::-1]):
    axes[1].text(v + 30, i, f"{v / len(df):.0%}", va="center", fontsize=9)
axes[1].set_title("Grouped by audience")
axes[1].grid(axis="y", visible=False)
save(fig, "07_ratings")

# %%
# Has it become more adult over time?
aud_year = (df.dropna(subset=["year_added"]).query("year_added >= 2015")
            .groupby("year_added")["audience"].value_counts(normalize=True).unstack().fillna(0) * 100)
print(aud_year[["Kids", "Older Kids", "Teens", "Adults"]].round(1).rename_axis(index=None))

# %% [markdown]
# **Insight:** About **three quarters of the library (76%) targets teens or adults**; TV-MA alone is the biggest
# rating (36%). Kids' content is ~10%. The adult share has been stable at ~45% since 2017; the kids' share fell
# after 2015 and has recovered a little since 2020.
#
# ### 3.7 How long are the titles?

# %%
mov = df["minutes"].dropna()
print(mov.describe().round(1).to_string())
print(f"\nMovies between 80 and 120 min: {mov.between(80, 120).mean():.0%}")

seasons = df["seasons"].dropna().astype(int).value_counts().sort_index()
print("\nSeasons per show:\n", seasons.to_string())
print(f"Single-season shows: {seasons.loc[1] / seasons.sum():.0%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(mov, bins=40, color=RED, edgecolor=PAPER)
axes[0].axvline(mov.median(), color=INK, ls="--", lw=1)
axes[0].text(mov.median() + 3, axes[0].get_ylim()[1] * 0.92, f"median {mov.median():.0f} min", fontsize=9)
axes[0].set_xlabel("Minutes"); axes[0].set_ylabel("Movies")
axes[0].set_title("Movie length")
s = seasons[seasons.index <= 10]
axes[1].bar(s.index, s.values, color=GOLD)
for i, v in zip(s.index, s.values):
    axes[1].text(i, v + 20, f"{v:,}", ha="center", fontsize=8)
axes[1].set_xticks(range(1, 11))
axes[1].set_xlabel("Seasons"); axes[1].set_ylabel("TV shows")
axes[1].set_title("Seasons per TV show")
axes[1].grid(axis="x", visible=False)
save(fig, "08_duration")

# %% [markdown]
# **Insight:** The median movie runs 98 minutes and 64% fall in the 80-120 minute window. **Two thirds of TV shows have
# a single season**, so the library is wide rather than deep: lots of short runs, few long-running series.
#
# ### 3.8 When are titles added? (seasonality)

# %%
heat = (df.dropna(subset=["year_added"]).query("2016 <= year_added <= 2020")
        .pivot_table(index="year_added", columns="month_added", values="show_id", aggfunc="count", fill_value=0))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = df["month_added"].value_counts().sort_index()
print("Titles added by calendar month (all years):")
print(pd.Series(by_month.values, index=month_names[: len(by_month)]).to_string())

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.grid(False)
im = ax.imshow(heat.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("n", [PAPER, GOLD, RED]), aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index.astype(int))
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, heat.values[i, j], ha="center", va="center", fontsize=7.5, color=INK)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_title("When titles are added (2016-2020)")
save(fig, "09_monthly")

# %% [markdown]
# **Insight:** Additions are spread fairly evenly through the year. July and December are the busiest months and
# February the quietest (it is also the shortest). The single biggest month is **November 2019 (255 titles)**. There is no
# fixed "release season", which fits a steady always-on pipeline.
#
# ### 3.9 People: directors and cast

# %%
top_d = directors["director"].value_counts().head(10)
top_a = actors["cast"].value_counts().head(10)
print("DIRECTORS\n", top_d.to_string(), "\n\nACTORS\n", top_a.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, series, title, col in [(axes[0], top_d, "Most frequent directors", RED), (axes[1], top_a, "Most frequent cast members", GOLD)]:
    ax.barh(series.index[::-1], series.values[::-1], color=col)
    for i, v in enumerate(series.values[::-1]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
save(fig, "10_people")

# %% [markdown]
# **Insight:** The most frequent cast names are mostly **Indian actors** (Anupam Kher, Shah Rukh Khan, Naseeruddin Shah)
# plus voice actors for dubbed anime and kids' shows (Takahiro Sakurai, Yuki Kaji, Julie Tejwani). The top directors are
# Rajiv Chilaka (children's animation), Jan Suter and Raúl Campos (stand-up specials). This matches the India finding in 3.4.
# Because ~30% of directors are unknown, director rankings come with a caveat.
#
# ## 4. Summary of findings

# %%
summary = {
    "total_titles": int(len(df)),
    "movies": int(type_counts["Movie"]),
    "tv_shows": int(type_counts["TV Show"]),
    "movie_pct": float(type_pct["Movie"]),
    "countries_count": int(countries["country"].nunique()),
    "genres_count": int(genres["listed_in"].nunique()),
    "peak_year": peak_year,
    "peak_titles": int(added["Total"].max()),
    "median_gap_years": float(gap.median()),
    "pct_within_2yrs": float((gap <= 2).mean() * 100),
    "top_country": top_c.index[0], "top_country_n": int(top_c.iloc[0]),
    "india_n": int(top_c["India"]),
    "tv_ma_pct": float(df["rating"].eq("TV-MA").mean() * 100),
    "teen_adult_pct": float(df["audience"].isin(["Teens", "Adults"]).mean() * 100),
    "median_movie_min": float(mov.median()),
    "single_season_pct": float(seasons.loc[1] / seasons.sum() * 100),
    "top_genre": top_g.index[0],
    "missing_director_pct": float(raw["director"].isna().mean() * 100),
    "added_by_year": {int(k): int(v) for k, v in added["Total"].items()},
    "top_countries": {k: int(v) for k, v in top_c.items()},
    "tv_share_by_country": {k: round(float(v), 1) for k, v in mix["tv_share"].items()},
}
with open("summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))

# %% [markdown]
# ### Key takeaways
# 1. **Movie-heavy library.** ~70% movies, ~30% TV shows.
# 2. **Rapid build-up.** Yearly additions went from a few dozen (pre-2015) to a 2,016-title peak in 2019, with a 2020 dip.
# 3. **Young catalogue.** ~70% of titles are from 2015 onward; the median title arrives 1 year after release.
# 4. **Global, led by the US and India.** India is almost all movies; Korea and Japan lean to TV.
# 5. **Grown-up audience.** 76% target teens/adults; TV-MA is the top rating (36%).
# 6. **Wide, not deep.** Movies centre on ~98 minutes; two thirds of shows have just one season.
# 7. **Genres:** International, Dramas, Comedies dominate.
#
# ### Limitations
# - The data is a snapshot (to Sept 2021) and says nothing about **viewership**, so "popular" here means "most listed".
# - Director / cast / country have gaps (30% / 9% / 9%), so people and country rankings are lower bounds.
# - Co-productions count once for every country, so country counts add up to more than the title total.
#
# ### Possible next steps
# - Join with viewing-hours or IMDb ratings to separate "listed" from "watched".
# - Text-mine the `description` field to cluster themes.
# - Forecast yearly additions, accounting for the 2020 shock.

# %%
# Save the cleaned dataset for reuse
df.to_csv("data/netflix_clean.csv", index=False)
print("Saved data/netflix_clean.csv", df.shape)
