import pickle
import time
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from models.netvlad.datasets.robotcar_dataset import RobotCarDataset
from models.netvlad.pca import normalize
from models.netvlad.pose_prediction.nearest_neighbor_predictor import NearestNeighborPredictor
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer
from util import html

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from models.netvlad.network import ImageRetrievalModel


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1


model = ComboGANModel(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % (opt.name, opt.phase, opt.which_epoch))
# store images for matrix visualization
vis_buffer = []


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

monte_carlo_samples = opt.monte_carlo_samples

if monte_carlo_samples > 1:
    # Enable dropout during test
    for m in model.netG.encoders + model.netG.decoders:
        enable_dropout(m)

if opt.netvlad:
    print('Load NetVLAD')
    netvlad_model = ImageRetrievalModel(
        opt.netvlad_checkpoint,
        device=opt.gpu_ids[0] if len(opt.gpu_ids) > 0 else -1)
    print('Load pre-computed reference NetVLAD descriptors')
    netvlad_ref_descriptors = np.loadtxt(opt.netvlad_ref_descr)
    print('Load PCA')
    with open(opt.netvlad_pca_dump, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

    print('Load robotcar dataset')
    rd = RobotCarDataset(name='robotcar',
                         root='/net/skoll/storage/datasets/robotcar/robotcar/data_path/robotcar',
                         image_folder='images_temp/',
                         reference_sequences=['overcast-reference'],
                         query_sequences=['night'],  # 'night-rain']
                         nvm_model='/net/skoll/storage/datasets/robotcar/robotcar/data_path/robotcar/databases/all.nvm',
                         triangulation_data_file='/home/poldan/S2DHM/data/triangulation/robotcar_triangulation.npz'
                         )
    robotcar_dataset = rd

    netvlad_mean_images = []
    netvlad_mean_netvlads = []
    mahalanobis_distances = []

    dataset = DataLoader(opt, img_list=robotcar_dataset.data['query_image_names'])
else:
    dataset = DataLoader(opt)

for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test(monte_carlo_samples=monte_carlo_samples, sampling=True)
    visuals = model.get_current_visuals(testing=True)
    img_path = model.get_image_paths()
    paths = visualizer.save_images(webpage, visuals, img_path)

    # Compute NetVLAD embeddings for the images
    if opt.netvlad:
        # Only consider the 'mc_' samples and the 'mc_mean' image
        paths = [path for path in paths if ('real' not in path and 'mc_std' not in path)]
        embeddings = netvlad_model.compute_embedding(paths)
        if not opt.no_pca:
            embeddings = normalize(pca.transform(normalize(embeddings)))

        # NetVLAD of the mean image
        netvlad_mean_images.append(embeddings[['mc_mean' in path for path in paths]])

        # Mean of the samples' NetVLADs
        samples = embeddings[['mc_mean' not in path for path in paths]]
        sample_mean = np.mean(samples, axis=0)
        netvlad_mean_netvlads.append(sample_mean)

        if opt.mahala:
            sample_cov = np.cov(samples, rowvar=False)
            inv_cov = np.linalg.pinv(sample_cov)
            condition_number = np.linalg.norm(sample_cov) * np.linalg.norm(inv_cov)
            # calc mahalanobis distance to every reference vector
            dists = []
            for ref in range(netvlad_ref_descriptors.shape[0]):  # Vectorize!
                y = netvlad_ref_descriptors[ref]
                m_d = distance.mahalanobis(sample_mean, y, inv_cov)
                dists.append(m_d)
            mahalanobis_distances.append(dists)


if opt.netvlad:
    # netvlad mean images matching
    print('Euclidean Ranking: netvlad_mean_images')
    netvlad_mean_images = np.vstack(netvlad_mean_images)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_images.T)
    ranks = np.argsort(-scores, axis=0)
    plt.matshow(scores)
    plt.savefig("euclidean_similarities_images_mean.jpg")

    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_images_mean.txt'))
    pose_predictor.save(pose_predictor.run())

    # netvlad mean images matching
    print('Euclidean Ranking: netvlad_mean_netvlads')
    netvlad_mean_netvlads = np.vstack(netvlad_mean_netvlads)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_netvlads.T)
    ranks = np.argsort(-scores, axis=0)
    plt.matshow(scores)
    plt.savefig("euclidean_similarities_netvlad_mean.jpg")
    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_mean_netvlads.txt'))
    pose_predictor.save(pose_predictor.run())

    if opt.mahala:
        # netvlad mahal.
        mahalanobis_distances = np.array(mahalanobis_distances).T
        print('mahalanobis_distances', mahalanobis_distances.shape)
        # Replace NaNs with large distances
        mahalanobis_distances[np.isnan(mahalanobis_distances)] = np.nanmax(mahalanobis_distances)
        # Convert the distance matrix to a similarity matrix
        reciprocal_similarities = np.nanmax(mahalanobis_distances) / mahalanobis_distances
        plt.matshow((reciprocal_similarities - np.mean(reciprocal_similarities))/np.std(reciprocal_similarities))
        plt.savefig("mahalanobis_similarities.jpg")
        ranks = np.argsort(-reciprocal_similarities, axis=0)
        pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                                  output_filename=os.path.join(opt.results_dir, 'top_1_mahalanobis.txt'))
        pose_predictor.save(pose_predictor.run())

webpage.save()

