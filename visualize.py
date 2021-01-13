import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import csv


# File containing credit card fraud information.
data_filename = "creditcard.csv"

# Data structures consisting of 30-dimensional data points.
# Positive = positive for fraud.
# Negative = negative for fraud.
data_pts_positive = []
data_pts_negative = []

with open(data_filename, "r") as data_file:
	csvreader = csv.reader(data_file)

	fields = next(csvreader)

	for row in csvreader:
		try:
			next_pt = [float(val) for val in row[1:-1]]
			next_label = int(row[-1])

			if next_label == 1:
				data_pts_positive.append(next_pt)
			else:
				data_pts_negative.append(next_pt)

		except ValueError:
			# invalid data point; skip.
			pass

#print("Number of positive data points = ", len(data_pts_positive))
#print("Number of negative data points = ", len(data_pts_negative))

# For ease of visualization, make number of positive and
#	negative points equal.
data_pts_negative = data_pts_negative[:len(data_pts_positive)]

# Convert to numpy arrays.
data_pts_positive = np.array(data_pts_positive)
data_pts_negative = np.array(data_pts_negative)

# Create TSNE in 2D and 3D.
positive_embedded_2d = TSNE(n_components=2).fit_transform(data_pts_positive)
negative_embedded_2d = TSNE(n_components=2).fit_transform(data_pts_negative)
positive_embedded_3d = TSNE(n_components=3).fit_transform(data_pts_positive)
negative_embedded_3d = TSNE(n_components=3).fit_transform(data_pts_negative)

#print(positive_embedded_2d.shape)
#print(negative_embedded_2d.shape)
#print(positive_embedded_3d.shape)
#print(negative_embedded_3d.shape)

# Plot results of 2D TSNE.
plt.title("Credit card fraud data: t-SNE 2D")
plt.scatter(positive_embedded_2d[:, 0], positive_embedded_2d[:, 1],
	c='g', label="positive")
plt.scatter(negative_embedded_2d[:, 0], negative_embedded_2d[:, 1],
	c='r', label="negative")
plt.legend(loc="lower left")
plt.show()

# Plot results of 3D TSNE.
plt.clf()
fig = plt.figure()
ax = Axes3D(fig)
plt.title("Credit card fraud data: t-SNE 3D")
ax.scatter(positive_embedded_3d[:, 0], positive_embedded_3d[:, 1],
	positive_embedded_3d[:, 2], c='g', label="positive")
ax.scatter(negative_embedded_3d[:, 0], negative_embedded_3d[:, 1],
	negative_embedded_3d[:, 2], c='r', label="negative")
plt.legend(loc="lower left")
plt.show()