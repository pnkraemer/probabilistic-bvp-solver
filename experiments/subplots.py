import matplotlib.pyplot as plt

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
a0.plot(range(3))
a1.plot(range(3))
plt.show()
