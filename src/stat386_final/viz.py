import matplotlib.pyplot as plt
import seaborn as sns

def print_genre_distribution(sales, genre, area):
    """Prints the distribution of sales in area for genre."""
    fig, ax = plt.subplots()

    sns.histplot(
        data=sales[sales['Genre'].astype(str).str.contains(genre, na=False)],
        x=area,
        bins=50,
        ax=ax
    )

    ax.set_title(f"{genre} – {area.replace('_', ' ')}")
    ax.set_xlabel(area.replace('_', ' '))
    ax.set_ylabel("Count")

    return ax


def print_platform_distribution(sales, platform, area):
    """Prints the distribution of sales in area for platform."""
    fig, ax = plt.subplots()

    sns.histplot(
        data=sales[sales['Platform'].astype(str).str.contains(platform, na=False)],
        x=area,
        bins=50,
        ax=ax
    )

    ax.set_title(f"{platform} – {area.replace('_', ' ')}")
    ax.set_xlabel(area.replace('_', ' '))
    ax.set_ylabel("Count")

    return ax