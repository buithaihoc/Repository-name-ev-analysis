import matplotlib.pyplot as plt
import seaborn as sns

def plot_trend(df):

    trend = df.groupby("year")["ev_share_pct"].mean()

    plt.figure(figsize=(8,5))
    plt.plot(trend.index, trend.values, marker="o")
    plt.title("EV Adoption Trend")
    plt.xlabel("Year")
    plt.ylabel("EV Share (%)")
    plt.show()
