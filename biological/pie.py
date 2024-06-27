import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 32, 'lines.linewidth': 6})
plt.figure(figsize=(16, 12))

y = np.array([65156, 8778, 2623, 19790])
wp = {'linewidth': 6, 'edgecolor':'white'}
#mylabels = ["DRACH, mRNA", "DRACH, ncRNA", "non-DRACH, mRNA", "non-DRACH, ncRNA"]
#explode = (0.05, 0.05, 0.05, 0.05)
colors = ['royalblue', 'lightsteelblue', 'lightsalmon', 'tomato', ]
plt.title('number of m6A sites for different sequence motifs')

plt.pie(y,startangle = 90, colors=colors, wedgeprops=wp)
#centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)

plt.savefig('/extdata4/baeklab/Hyeonseo/m6A/biological/pie_293t_4_div_cutoff75.png')
plt.close()

plt.rcParams.update({'font.size': 32, 'lines.linewidth': 6})
plt.figure(figsize=(16, 12))

y = np.array([71511, 5350, 2041, 23336])
wp = {'linewidth': 6, 'edgecolor':'white'}
#mylabels = ["DRACH, mRNA", "DRACH, ncRNA", "non-DRACH, mRNA", "non-DRACH, ncRNA"]
#explode = (0.05, 0.05, 0.05, 0.05)
colors = ['royalblue', 'lightsteelblue', 'lightsalmon', 'tomato', ]
plt.title('number of m6A sites for different sequence motifs')

plt.pie(y,startangle = 90, colors=colors, wedgeprops=wp)
#centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)

plt.savefig('/extdata4/baeklab/Hyeonseo/m6A/biological/pie_HeLa_4_div_cutoff75.png')
plt.close()