import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(6, 4))
plt.bar(["Before Layoff", "After Layoff"], [nbr_before, nbr_after], color=['green', 'red'])
plt.xlabel("Time Period")
plt.ylabel("Net Brand Reputation (NBR)")
plt.title("Company Reputation Before & After Layoffs")
plt.ylim(-1, 1)
plt.show()
