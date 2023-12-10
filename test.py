import matplotlib.pyplot as plt

pi = 9
print(f"Training/Logistic_Regression_Plot/Logistic Regression PCA - Pi = {pi} - Pi_T = 0.5.pdf")


plt.figure()
plt.xlabel("Lambda")
plt.xscale("log")
plt.ylabel("minDCF")
plt.title(f"Logistic Regression PCA Pi = {pi}  Pi_T = 0.5")

plt.savefig(f"Training/Logistic_Regression_Plot/prova Pi = {pi} Pi_T = 0.5.pdf")