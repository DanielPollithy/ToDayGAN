from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        self.parser.add_argument('--which_epoch', required=True, type=int, help='which epoch to load for inference?')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc (determines name of folder to load from)')

        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run (if serial_test not enabled)')
        self.parser.add_argument('--serial_test', action='store_true', help='read each image once from folders in sequential order')

        self.parser.add_argument('--autoencode', action='store_true', help='translate images back into its own domain')
        self.parser.add_argument('--reconstruct', action='store_true', help='do reconstructions of images during testing')

        self.parser.add_argument('--show_matrix', action='store_true', help='visualize images in a matrix format as well')

        self.parser.add_argument('--netvlad', action='store_true', help='Compute embeddings')
        self.parser.add_argument('--netvlad_checkpoint', type=str,
                                 default='/home/poldan/S2DHM/checkpoints/netvlad_no_pretraining/checkpoint.pth.tar',
                                 help='Archive containing checkpoint of VGG-16 and NetVLAD')
        self.parser.add_argument('--netvlad_pca_dump', type=str, default='/home/poldan/S2DHM/pca.pkl',
                                 help='Pickle dump of PCA trained on the reference images')
        self.parser.add_argument('--netvlad_ref_descr', type=str,
                                 default='/home/poldan/S2DHM/data/ranks/query_descriptors.tsv',
                                 help='np array of NetVLAD descriptors for the reference images')
