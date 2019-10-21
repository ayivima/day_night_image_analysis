
import matplotlib.pyplot as plt

def fsep(features_, sep_coords=([25, 129], [200, 50])):
    """Plots day and night images as clusters in 2D space for given features.
    And, draws a line of separation
    """
    
    night = df2.loc[df2["labels"]==0, features_]
    day = df2.loc[df2["labels"]==1, features_]

    data = night, day
    colors = "red", "green"
    groups = "Night", "Day"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data[features_[0]], data[features_[1]]
        ax.scatter(
            x, 
            y, 
            alpha=0.8,
            c=color,
            edgecolors='white', 
            s=100,
            label=group
        )
        
    x_, y_ = sep_coords
    plt.plot(x_, y_)
    plt.title('Night and Day Images Feature Separation')
    plt.xlabel(features_[0])
    plt.ylabel(features_[1])
    plt.legend(loc=2)
    plt.show()
