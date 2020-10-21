import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.cluster import MiniBatchKMeans, KMeans

ffnet = pd.read_csv('/Users/joanner/Documents/GradSchool/dmfinal/1allcopy.csv',
                    engine='python', sep='\t', error_bad_lines=False, header=0)

ffcl = ffnet.drop_duplicates(keep='first')
cols = ['story', 'author', 'published', 'updated', 'description']
ffcl = ffcl.drop(cols, axis=1)

# fix the commas in the thousands for the numeric columns
ffcl['word_count'] = ffcl['word_count'].str.replace(',', '')
ffcl['fav_count'] = ffcl['fav_count'].str.replace(',', '')
ffcl['follow_count'] = ffcl['follow_count'].str.replace(',', '')

# separation for type conversion
ffcl['review_count'] = pd.to_numeric(ffcl.review_count, errors='coerce')
ffcl['word_count'] = pd.to_numeric(ffcl.word_count, errors='coerce')
ffcl['chapter_count'] = pd.to_numeric(ffcl.chapter_count, errors='coerce')
ffcl['fav_count'] = pd.to_numeric(ffcl.fav_count, errors='coerce')
ffcl['follow_count'] = pd.to_numeric(ffcl.follow_count, errors='coerce')

# replacing null values
ffcl['category'].fillna("None", inplace=True)
ffcl['review_count'].fillna(0, inplace=True)
ffcl['word_count'].fillna(0, inplace=True)
ffcl['chapter_count'].fillna(0, inplace=True)
ffcl['fav_count'].fillna(0, inplace=True)
ffcl['follow_count'].fillna(0, inplace=True)

# floats to int
#ffcl['author_id'] = ffcl['author_id'].astype(int)
ffcl['review_count'] = ffcl['review_count'].astype(int)
ffcl['chapter_count'] = ffcl['chapter_count'].astype(int)
ffcl['word_count'] = ffcl['word_count'].astype(int)
ffcl['fav_count'] = ffcl['fav_count'].astype(int)
ffcl['follow_count'] = ffcl['follow_count'].astype(int)

# removing spam
ffcl = ffcl.drop(ffcl[(ffcl.chapter_count > 2) &
                      (ffcl.word_count < 200)].index)
ffcl = ffcl.drop(ffcl[(ffcl.chapter_count <= 2) &
                      (ffcl.word_count < 100)].index)
ffcl = ffcl.drop(ffcl[ffcl.chapter_count == 0].index)
ffcl = ffcl.drop(ffcl[ffcl.word_count == 0].index)

# removing outliers
ffcl = ffcl.drop(ffcl[ffcl.word_count > 100000].index)
ffcl = ffcl.drop(ffcl[ffcl.review_count > 1000].index)
ffcl = ffcl.drop(ffcl[ffcl.follow_count > 1000].index)
ffcl = ffcl.drop(ffcl[ffcl.fav_count > 1000].index)
ffcl = ffcl.drop(ffcl[ffcl.chapter_count > 50].index)


# TO TEST THE PREPROCESSING
#fftop = ffcl.head(10)
# print(fftop)

# Scatter plots to visualize data

#ax1 = ffcl.plot.scatter(x='review_count', y='word_count', c='DarkBlue')
#ax2 = ffcl.plot.scatter(x='fav_count', y='word_count', c='DarkBlue')
#ax3 = ffcl.plot.scatter(x='follow_count', y='word_count', c='DarkBlue')
#ax4 = ffcl.plot.scatter(x = 'follow_count', y='chapter_count', c='DarkBlue')


# //////////////// CLUSTERING WORK BEGINS ///////////////////

ffsamp = ffcl.sample(n=50000, replace=True)

X = ffsamp[['review_count', 'word_count']]

kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=42, batch_size=1000)
pred_y = kmeans.fit_predict(X)
plt.scatter(X['review_count'], X['word_count'], c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Reviews')
plt.ylabel('# of Words')
plt.show()


# FAVS VS WC

X = ffsamp[['fav_count', 'word_count']]
kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=0, batch_size=1000)
pred_y = kmeans.fit_predict(X)
plt.scatter(X['fav_count'], X['word_count'], c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Favs')
plt.ylabel('# of Words')
plt.show()


# FOLLOW CT VS WC

X = ffsamp[['follow_count', 'word_count']]
kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=0, batch_size=1000)
pred_y = kmeans.fit_predict(X)
plt.scatter(X['follow_count'], X['word_count'], c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Follows')
plt.ylabel('# of Words')
plt.show()


Y = ffsamp[['review_count', 'chapter_count']]
kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=0, batch_size=1000)
pred_y = kmeans.fit_predict(Y)
plt.scatter(Y['review_count'], Y['chapter_count'],
            c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Reviews')
plt.ylabel('# of Chapters')
plt.show()


Y = ffsamp[['follow_count', 'chapter_count']]
kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=0, batch_size=1000)
pred_y = kmeans.fit_predict(Y)
plt.scatter(Y['follow_count'], Y['chapter_count'],
            c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Follows')
plt.ylabel('# of Chapters')
plt.show()


Y = ffsamp[['fav_count', 'chapter_count']]
kmeans = MiniBatchKMeans(n_clusters=10, init='k-means++',
                         max_iter=300, n_init=10, random_state=0, batch_size=1000)
pred_y = kmeans.fit_predict(Y)
plt.scatter(Y['fav_count'], Y['chapter_count'], c=pred_y, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='black', alpha=0.5)
plt.xlabel('# of Favs')
plt.ylabel('# of Chapters')
plt.show()
