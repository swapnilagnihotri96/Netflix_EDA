# Netflix Catalogue: Exploratory Data Analysis

An exploratory analysis of 8,807 Netflix titles (6,131 movies and 2,676 TV shows, added between 2008 and September 2021). The project cleans the data, explores it, and reports what the catalogue looks like, where it comes from and how it grew.

## What's in this folder

| File | What it is |
|---|---|
| `netflix_eda.ipynb` | **Main deliverable.** Documented Jupyter notebook with code, charts and written insights (already executed, so outputs are visible). |
| `netflix_eda.py` | The same analysis as a plain script (cells marked with `# %%`). |
| `Netflix_EDA_Presentation.pptx` | 15-slide presentation of the findings. |
| `build_deck.py` | Script that rebuilds the presentation from the charts and stats. |
| `data/netflix_data.csv` | Original dataset. |
| `data/netflix_clean.csv` | Cleaned dataset (created when the analysis runs). |
| `figures/` | The 10 charts used in the notebook and slides (created when the analysis runs). |
| `summary_stats.json` | Headline numbers used by the deck (created when the analysis runs). |

## How to run

```bash
pip install pandas numpy matplotlib python-pptx pillow jupyter
jupyter notebook netflix_eda.ipynb      # run all cells
# or:
python netflix_eda.py                   # run as a script (charts are saved to figures/)
python build_deck.py                    # rebuild the presentation
```

Run everything from inside this folder: paths are relative (`data/`, `figures/`).

## Dataset columns

`show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` (genres), `description`.

## Method

1. **Inspect:** shape, types, missing values, duplicates (none found).
2. **Clean:**
   - Three rows had their duration typed into the `rating` column; moved to `duration`.
   - Missing director / cast / country / rating labelled `Unknown`.
   - `date_added` parsed to dates; year and month extracted. The 10 rows with no date are left out of time-based charts.
   - `duration` split into numeric `minutes` (movies) and `seasons` (TV).
   - 14 rating codes grouped into 4 audience bands (Kids, Older Kids, Teens, Adults).
   - Multi-value columns (`country`, `listed_in`, `cast`, `director`) split and exploded before counting.
3. **Explore:** type split, growth by year, release-to-Netflix gap, countries, genres, ratings, durations, monthly seasonality, top directors and cast.
4. **Report:** insights written under each chart in the notebook and summarised in the slides.

## Key findings

- **70% movies, 30% TV shows** (by title count).
- **Fast growth:** 82 titles added in 2015, 2,016 in 2019 (the peak), a 7% dip in 2020, and 1,498 already by September 2021.
- **Young catalogue:** the median title arrives 1 year after release; 63% within 2 years; 71% were released in 2015 or later.
- **Global:** the US is involved in 46% of titles with a known country, India is second (1,046), and India is 92% movies while South Korea (74%) and Japan (63%) lean to TV.
- **Genres:** International Movies (2,752), Dramas (2,427) and Comedies (1,674) lead.
- **Audience:** 76% of titles target teens or adults; TV-MA is the top rating (36%).
- **Length:** median movie is 98 minutes; 67% of TV shows have a single season.
- **Timing:** additions are spread across the year; the busiest single month is November 2019 (255 titles).

## Limitations

- The data lists titles only. It has no viewing numbers, so "most" means "most listed", not "most watched".
- Director (30%), cast (9%) and country (9%) have gaps, so people and country rankings are lower bounds.
- Co-productions count once per country, so country totals exceed the number of titles.
- The data ends in September 2021, so 2021 is a partial year.

## Possible next steps

- Join viewing hours or IMDb scores to separate "listed" from "loved".
- Text-mine `description` to cluster themes.
- Forecast yearly additions, allowing for the 2020 shock.

## Presentation fonts

The deck uses Georgia (headlines), Calibri (body) and Courier New (labels). All three ship with Windows and macOS Office, so it should look as designed without installing anything.
