---
layout: post
mathjax: true
title:  "Streaks or Luck?"
date:  2019-04-20
categories: baseball
#tags: featured
image: /assets/article_images/2019-04-21-streaks-or-luck/cleveland_streak.jpg
permalink: /:categories/:year/:month/:day/:title
---

When a team goes on an extended winning streak, my gut tells me that something so unlikely is happening that it *proves* sports contests must be more than just weighted random number generators.  <br>
But is that true?  In an attempt to answer that question, I work out the probabilities of winning streaks of different lengths given a winning percentage, and estimate that for each team given their (2018) records.  I compare those odds to the actual lengths of winning streaks for that season. <br>
I admit, what I find is a little surprising.


```python
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn

%matplotlib inline
plt.style.use('ggplot')

sys.setrecursionlimit(20000)
```

I use a recursive function to estimate the probability of a streak of length winStreak during a season numGames long. <br>
Naturally, the probability is expected to increase for:
- shorter streaks
- better teams (defined by winPercent)
- longer seasons


```python
def probability_of_streak(numGames, winStreak, winPercent, saved = None):

        if saved == None: saved = {}

        ID = (numGames, winStreak, winPercent)

        if ID in saved: return saved[ID]
        else:
            if winStreak > numGames or numGames <=0:
                result = 0
            else:
                result = winPercent**winStreak
                for firstWin in xrange(1, winStreak+1):
                    pr = probability_of_streak(numGames-firstWin, winStreak, winPercent, saved)
                    result += (winPercent**(firstWin-1))*(1-winPercent)*pr

        saved[ID] = result

        return result
```


```python
def consecutive(data, stepsize=0):
    sched = np.zeros(np.size(data))
    sched[np.isin(data,'W')] = 1
    return np.split(sched, np.where(np.diff(sched) != stepsize)[0]+1)
```


```python
def group_streaks(sched):
    all_streaks = consecutive(sched)
    longest_streak = np.size(max(all_streaks,key=len))
    wins = np.zeros(longest_streak)
    loss = np.zeros(longest_streak)
    for i in all_streaks:
        ind = np.size(i)-1
        if i[0] == 1:
            wins[ind] += 1
        else:
            loss[ind] += 1

    return [wins, loss]    
```


```python
def winning_percentage(streaks):
    wp = {}
    gp = {}
    for team in streaks:
        total_number_games = np.sum((np.arange(np.size(streaks[team][0]))+1) * streaks[team][0]) + np.sum((np.arange(np.size(streaks[team][1]))+1) * streaks[team][1])
        gp[team] = total_number_games
        wp[team] = np.sum((np.arange(np.size(streaks[team][0]))+1) * streaks[team][0]) / total_number_games
    return wp,gp
```


```python
path_files = '/data/baseball/Baseball-Reference/schedule_results/2018/'
leagues    = ['AL','NL']
```


```python
streaks = {}
for league in leagues:
    path_csv = path_files + league + '/'
    #print path_csv
    fnames = os.listdir(path_csv)
    #print fnames
    for itm in fnames:
        #print itm
        team = itm.split('.')[0].split('_')[0]
        #print team
        pd.read_csv(path_csv+itm)
        df = pd.read_csv(path_csv+itm)
        df['W/L'][df['W/L'] == 'W-wo'] = 'W'
        df['W/L'][df['W/L'] == 'L-wo'] = 'L'
        a = df['W/L'].values
        streaks[team] = group_streaks(a)
```

```python
for t in streaks:
    print streaks[t][0]
```

    [18. 14.  4.  5.  1.  0.  0.  1.  1.]
    [18. 11.  9.  2.  1.  0.  0.  2.]
    [20.  7.  4.  4.  2.  1.  0.  0.]
    [28.  3.  6.  4.  1.]
    [18. 11.  5.  2.  3.  2.]
    [ 7. 12.  5.  6.  1.  1.  0.  1.  1.  1.]
    [27.  7.  2.  3.  1.  0.  0.  0.  0.  0.  0.]
    [22. 10.  4.  0.  0.  1.  1.  0.]
    [25. 13.  3.  2.  0.  0.  0.  0.  1.]
    [27.  5.  2.  1.  0.  0.  0.  0.  0.]
    [15.  7.  5.  4.  2.  1.  1.  1.]
    [19. 10.  3.  5.  1.  4.]
    [25. 11.  3.  1.  0.  0.  1.]
    [22. 12.  3.  2.  0.  0.]
    [27.  8.  4.  4.  2.  0.  0.  1.]
    [22.  8.  3.  1.  4.  0.  0.  0.  0.  0.  1.]
    [20. 14.  4.  3.  2.  1.  1.]
    [19. 14.  5.  2.  2.  0.  0.  1.]
    [23.  9.  7.  0.  2.  2.  1.]
    [22. 10.  4.  2.  0.  0.  0.  0.]
    [12.  7.  7.  1.  3.  3.  1.  0.  0.  0.  0.  1.]
    [18.  8.  6.  4.  1.  0.  0.  0.  0.  0.  0.]
    [13. 15.  8.  3.  1.  1.  1.]
    [23. 10.  4.  2.  1.  2.  0.  0.  0.]
    [29.  9.  5.  1.  0.  0.  0.]
    [12.  9.  8.  7.  2.  0.]
    [16.  6.  4.  2.  4.  1.  0.  2.]
    [17.  9.  4.  5.  0.  1.  1.]
    [25.  7.  3.  1.  0.  1.  0.  0.  0.  0.]
    [19. 10. 10.  2.  1.  0.  0.]



```python
streak_size = 15
win_percent, games_played = winning_percentage(streaks)
real_win_streaks  = np.zeros(streak_size)
cum_win_streaks  = np.zeros(streak_size)
win_predict = np.zeros(streak_size)
for t in streaks:
    x = (np.arange(np.size(streaks[t][0])))
    cumulative_wins = np.cumsum(streaks[t][0][::-1])[::-1]
    ind_non_zero = np.where(streaks[t][0] > 0)
    cind_non_zero = np.where(cumulative_wins > 0)
    real_win_streaks[x[ind_non_zero]] += 1
    cum_win_streaks[x[cind_non_zero]] += 1
    for p in np.arange(streak_size):
        win_predict[p] += probability_of_streak(games_played[t], p+1, win_percent[t])

#plt.step(np.arange(streak_size)+1,win_tally)

#plt.step(np.arange(streak_size)+1,real_win_streaks,linestyle = ':')
plt.step(np.arange(streak_size)+1,cum_win_streaks,linestyle = '--',label='Actual # Teams w/ Streak Length')
plt.step(np.arange(streak_size)+1,win_predict,label='Simulated # Teams w/ Streak Length')
plt.legend()
plt.title('Number of Teams with Winning Streaks of Length W')
plt.xlabel('W - Winning Streak Length')
plt.ylabel('Number of Teams')
#ax.set_xlim(23.5, 28)
#ax.set_ylim(1, 30)
#ax.grid(True)
#plt.yscale('log')
plt.show()
```


![png]({{"/assets/images/Win_Streak_Probabilities_files/Win_Streak_Probabilities_11_0.png"}})



```python
streak_size = 15
lose_percent, games_played = winning_percentage(streaks)
real_lose_streaks  = np.zeros(streak_size)
cum_lose_streaks  = np.zeros(streak_size)
lose_predict = np.zeros(streak_size)
for t in streaks:
    x = (np.arange(np.size(streaks[t][1])))
    cumulative_losses = np.cumsum(streaks[t][1][::-1])[::-1]
    ind_non_zero = np.where(streaks[t][1] > 0)
    cind_non_zero = np.where(cumulative_losses > 0)
    real_lose_streaks[x[ind_non_zero]] += 1
    cum_lose_streaks[x[cind_non_zero]] += 1
    for p in np.arange(streak_size):
        lose_predict[p] += probability_of_streak(games_played[t], p+1, lose_percent[t])

#plt.step(np.arange(streak_size)+1,win_tally)

#plt.step(np.arange(streak_size)+1,real_win_streaks,linestyle = ':')
plt.step(np.arange(streak_size)+1,cum_lose_streaks,linestyle = '--',label='Actual # Teams w/ Streak Length')
plt.step(np.arange(streak_size)+1,lose_predict,label='Simulated # Teams w/ Streak Length')
plt.legend()
plt.title('Number of Teams with Losing Streaks of Length L')
plt.xlabel('L - Losing Streak Length')
plt.ylabel('Number of Teams')
#ax.set_xlim(23.5, 28)
#ax.set_ylim(1, 30)
#ax.grid(True)
#plt.yscale('log')
plt.show()
```


![png](/assets/images/Win_Streak_Probabilities_files/Win_Streak_Probabilities_12_0.png)



```python
print np.arange(14)+1
print real_win_streaks
print cum_win_streaks
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    [30. 30. 30. 28. 21. 14.  8.  7.  3.  1.  1.  1.  0.  0.  0.]
    [30. 30. 30. 30. 26. 21. 16. 10.  5.  3.  2.  1.  0.  0.  0.]



```python
print real_lose_streaks
print cum_lose_streaks
```

    [30. 30. 30. 25. 21. 13.  7.  6.  3.  1.  2.  0.  0.  0.  0.]
    [30. 30. 30. 29. 24. 17. 14. 10.  5.  3.  2.  0.  0.  0.  0.]



```python
np.cumsum(streaks['ARI'][1][::-1])[::-1]
```




    array([42., 18.,  9.,  6.,  2.,  2.,  1.])




```python
win_percent, games_played = winning_percentage(streaks)
loss_tally = np.zeros(12)
real_losing_streaks  = np.zeros(12)
loss_predict = np.zeros(12)
for t in streaks:
    x = (np.arange(np.size(streaks[t][1])))
    loss_tally[x] += streaks[t][1]
    ind_non_zero = np.where(streaks[t][1] > 0)
    real_losing_streaks[x[ind_non_zero]] += 1
    for p in np.arange(12):
        loss_predict[p] += probability_of_streak(games_played[t], p+1, 1-win_percent[t])

#plt.step(np.arange(12)+1,win_tally)
plt.step(np.arange(12)+1,real_losing_streaks,linestyle = ':')
plt.step(np.arange(12)+1,loss_predict)
#ax.set_xlim(23.5, 28)
#ax.set_ylim(1, 30)
#ax.grid(True)
#plt.yscale('log')
plt.show()
```


![png](/assets/images/Win_Streak_Probabilities_files/Win_Streak_Probabilities_16_0.png)



```python
probOfStreak(162,10,.60)
```




    0.3183344242913501




```python

```
