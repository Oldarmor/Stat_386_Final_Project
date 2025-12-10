import matplotlib.pyplot as plt
import seaborn as sns

def print_genre_distribution(sales, genre, area):
    """Prints the distribution of sales in area for genre."""
    sns.histplot(data=sales[sales['Genre'].astype(str).str.contains(genre, na=False)], x=area, bins=50)
    plt.show()

def print_platform_distribution(sales, platform, area): 
    """Prints the distribution of sales in area for platform."""
    sns.histplot(data=sales[sales['Platform'].astype(str).str.contains(platform, na=False)], x=area, bins=50)
    plt.show()