import time
import os
import pickle
import numpy as np

from models.nll_combogan_model import NLLComboGANModel
from options.test_options import TestOptions
from data.data_loader import DataLoader
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

from models.netvlad.network import ImageRetrievalModel
from models.netvlad.datasets.robotcar_dataset import RobotCarDataset
from models.netvlad.pca import normalize
from models.netvlad.pose_prediction.nearest_neighbor_predictor import NearestNeighborPredictor


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoader(opt)
model = NLLComboGANModel(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % (opt.name, opt.phase, opt.which_epoch))
# store images for matrix visualization
vis_buffer = []

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
    netvlad_blurred_images = []

    dataset = DataLoader(opt, img_list=robotcar_dataset.data['query_image_names'])
else:
    dataset = DataLoader(opt)

for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(testing=True)
    img_path = model.get_image_paths()
    paths = visualizer.save_images(webpage, visuals, img_path)

    # Compute NetVLAD embeddings for the images
    if opt.netvlad:
        # Only consider the 'mc_' samples and the 'mc_mean' image
        paths = [path for path in paths if ('mean' in path or 'blurred' in path)]
        embeddings = netvlad_model.compute_embedding(paths)
        if not opt.no_pca:
            embeddings = normalize(pca.transform(normalize(embeddings)))

        # NetVLAD of the mean image (either single image or mean of flipped)
        netvlad_mean_images.append(embeddings[['mean' in path for path in paths]])

        # NetVLAD of the blurred image
        netvlad_blurred_images.append(embeddings[['blurred' in path for path in paths]])

if opt.netvlad:
    # netvlad mean images matching
    print('Euclidean Ranking: netvlad_mean_images')
    netvlad_mean_images = np.vstack(netvlad_mean_images)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_images.T)
    ranks = np.argsort(-scores, axis=0)
    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_images_mean.txt'))
    pose_predictor.save(pose_predictor.run())

    # netvlad blurred images matching
    print('Euclidean Ranking: images_blur')
    netvlad_mean_netvlads = np.vstack(netvlad_blurred_images)
    scores = np.dot(netvlad_ref_descriptors, netvlad_mean_netvlads.T)
    ranks = np.argsort(-scores, axis=0)
    pose_predictor = NearestNeighborPredictor(dataset=robotcar_dataset, network=None, ranks=ranks, log_images=False,
                                              output_filename=os.path.join(opt.results_dir, 'top_1_images_blur.txt'))
    pose_predictor.save(pose_predictor.run())

webpage.save()

