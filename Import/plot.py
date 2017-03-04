def plotFreqDistAsBarChart(fdist,n):
    words = [x[0] for x in fdist.most_common(n)]
    values = [x[1] for x in fdist.most_common(n)]
    matplotlib.style.use('ggplot')
    
    d = {'values': values}
    df = pd.DataFrame(d)

    ax = df.plot(kind='bar',legend=False,title='Word Frequencies for Most Common Words')
    ax.set_xticklabels(words)
    ax.set_xlabel("Words",fontsize=12)
    ax.set_ylabel("Occurrences",fontsize=12)

    plt.show()