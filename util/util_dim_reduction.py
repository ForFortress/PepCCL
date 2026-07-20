from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import umap


def pca(title, data, data_index, data_label, class_num):

    X_pca = PCA(n_components=2).fit_transform(data)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    plt.style.use("default")
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_pca)):
            plt.annotate(data_label[i], xy=(X_pca[:, 0][i], X_pca[:, 1][i]),
                         xytext=(X_pca[:, 0][i] + 0.00, X_pca[:, 1][i] + 0.00))
    plt.title(title, fontdict=font)
    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)

    imagesave = 'path_to_save_imgs' + title + '.png'
    plt.savefig(imagesave, dpi=300)
    plt.show()
    plt.close()


def uumap(title, data, data_index, data_label, class_num):
    data = np.array(data)
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_tsne = reducer.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('coolwarm', class_num))

    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.rcParams['font.family'] = 'Times New Roman'

    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='Digital value',fontsize=15)
        plt.clim(0 - 0.5, class_num - 0.5)

    plt.title(title,fontfamily='Times New Roman',fontsize=16)

    imagesave_svg = 'path_to_save_imgs' + title + '_umap.svg'
    imagesave_png = 'path_to_save_imgs' + title + '_umap.png'
    plt.savefig(imagesave_svg,format='svg',dpi=300)
    plt.savefig(imagesave_png, dpi=300)
    plt.show()
    plt.close()


def t_sne(title, data, data_index, data_label, class_num, kind):
    data = np.array(data)

    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(data)

    plt.figure()

    colors_custom =  ['#4B5CC4', '#E4C6D0']
    cmap_custom = ListedColormap(colors_custom)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=cmap_custom)

    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.rcParams['font.family'] = 'Times New Roman'

    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='Digital value',fontsize=15)
        plt.clim(0 - 0.5, class_num - 0.5)

    plt.title(title,fontfamily='Times New Roman',fontsize=16)

    if kind == 'c1':
        imagesave_svg = 'path_to_save_imgs' + title + '_tsne_c1.svg'
        imagesave_png = 'path_to_save_imgs' + title + '_tsne_c1.png'
    elif kind == 'c2':
        imagesave_svg = 'path_to_save_imgs' + title + '_tsne_c2.svg'
        imagesave_png = 'path_to_save_imgs' + title + '_tsne_c2.png'

    plt.savefig(imagesave_svg,format='svg',dpi=300)
    plt.savefig(imagesave_png, dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    digits = load_digits()

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    X_pca = PCA(n_components=2).fit_transform(digits.data)

    font = {"color": "darkred", "size": 13, "family": "serif"}

    plt.style.use("default")

    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.tight_layout()
    plt.show()
