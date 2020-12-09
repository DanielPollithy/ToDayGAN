import time
import os
import pickle
import numpy as np
from scipy.spatial import distance

from models.bbb_combogan_model import BBBComboGANModel
from models.netvlad.datasets.robotcar_dataset import RobotCarDataset
from models.netvlad.pca import normalize
from models.netvlad.pose_prediction.nearest_neighbor_predictor import NearestNeighborPredictor
from options.test_options import TestOptions
from data.data_loader import DataLoader
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

import matplotlib.pyplot as plt

from models.netvlad.network import ImageRetrievalModel


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoader(opt)
model = BBBComboGANModel(opt, len(dataset))
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % (opt.name, opt.phase, opt.which_epoch))
# store images for matrix visualization
vis_buffer = []

monte_carlo_samples = opt.monte_carlo_samples

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
    sed_distances = []

    dataset = DataLoader(opt, img_list=robotcar_dataset.data['query_image_names'])
else:
    dataset = DataLoader(opt)

for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test(monte_carlo_samples=monte_carlo_samples)
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

        # Mahalanobis
        if opt.mahala:
            sample_cov = np.cov(samples, rowvar=False)
            inv_cov = np.linalg.pinv(sample_cov)
            condition_number = np.linalg.norm(sample_cov) * np.linalg.norm(inv_cov)

            # calc mahalanobis and SED distance to every reference vector
            mahal_dists = []
            sed_dists = []
            for ref in range(netvlad_ref_descriptors.shape[0]):  # Vectorize!
                m_d = distance.mahalanobis(sample_mean, netvlad_ref_descriptors[ref], inv_cov)
                mahal_dists.append(m_d)
                # Standardized Euclidean Distance (SED)
                sed_dists.append(distance.seuclidean(sample_mean, netvlad_ref_descriptors[ref], np.var(samples, axis=0)))
            mahalanobis_distances.append(mahal_dists)
            sed_distances.append(sed_dists)

if opt.netvlad:
    # netvlad mean images matching
    print('Euclidean Ranking: netvlad_mean_images')
    netvlad_mean_images = np.vstack(netvlad_mean_images)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_images.T)
    ranks = np.argsort(-scores, axis=0)
    plt.matshow(scores)
    plt.savefig(os.path.join(opt.results_dir, "euclidean_similarities_images_mean.jpg"))

    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_images_mean.txt'))
    pose_predictor.save(pose_predictor.run())

    # netvlad mean images matching
    print('Euclidean Ranking: netvlad_mean_netvlads')
    netvlad_mean_netvlads = np.vstack(netvlad_mean_netvlads)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_netvlads.T)
    ranks = np.argsort(-scores, axis=0)
    plt.matshow(scores)
    plt.savefig(os.path.join(opt.results_dir, "euclidean_similarities_netvlad_mean.jpg"))
    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_mean_netvlads.txt'))
    pose_predictor.save(pose_predictor.run())

    # netvlad mahal.
    if opt.mahala:
        mahalanobis_distances = np.array(mahalanobis_distances).T
        print('mahalanobis_distances', mahalanobis_distances.shape)
        # Replace NaNs with large distances
        mahalanobis_distances[np.isnan(mahalanobis_distances)] = np.nanmax(mahalanobis_distances)
        # Convert the distance matrix to a similarity matrix
        reciprocal_similarities = np.nanmax(mahalanobis_distances) / mahalanobis_distances
        plt.matshow((reciprocal_similarities - np.mean(reciprocal_similarities))/np.std(reciprocal_similarities))
        plt.savefig(os.path.join(opt.results_dir, "mahalanobis_similarities.jpg"))
        ranks = np.argsort(-reciprocal_similarities, axis=0)
        pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                                  output_filename=os.path.join(opt.results_dir, 'top_1_mahalanobis.txt'))
        pose_predictor.save(pose_predictor.run())

        # SED distances
        sed_distances = np.array(sed_distances).T
        print('sed_distances', sed_distances.shape)
        # Replace NaNs with large distances
        sed_distances[np.isnan(sed_distances)] = np.nanmax(sed_distances)
        # Convert the distance matrix to a similarity matrix
        reciprocal_similarities = np.nanmax(sed_distances) / sed_distances
        plt.matshow((reciprocal_similarities - np.mean(reciprocal_similarities)) / np.std(reciprocal_similarities))
        plt.savefig(os.path.join(opt.results_dir, "sed_similarities.jpg"))
        ranks = np.argsort(-reciprocal_similarities, axis=0)
        pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                                  output_filename=os.path.join(opt.results_dir, 'top_1_sed.txt'))
        pose_predictor.save(pose_predictor.run())

webpage.save()












# test
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test(monte_carlo_samples=monte_carlo_samples)
    visuals = model.get_current_visuals(testing=True)
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

    if opt.show_matrix:
        vis_buffer.append(visuals)
        if (i+1) % opt.n_domains == 0:
            save_path = os.path.join(web_dir, 'mat_%d.png' % (i//opt.n_domains))
            visualizer.save_image_matrix(vis_buffer, save_path)
            vis_buffer.clear()

webpage.save()

