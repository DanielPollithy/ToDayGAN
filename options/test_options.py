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
        self.parser.add_argument('--no_pca', action='store_true', default=False, help='Disable PCA')
        self.parser.add_argument('--flip_export', action='store_true', default=False, help='Flip the sampled images')
        self.parser.add_argument('--mahala', action='store_true', default=False, help='Compute mahalanobis distance')

        self.parser.add_argument('--netvlad_checkpoint', type=str,
                                 default='./netvlad_data/pittsburgh30k/checkpoint.pth.tar',
                                 help='Archive containing checkpoint of VGG-16 and NetVLAD')
        self.parser.add_argument('--netvlad_pca_dump', type=str, default='./netvlad_data/pittsburgh30k/pca.pkl',
                                 help='Pickle dump of PCA trained on the reference images')
        self.parser.add_argument('--netvlad_ref_descr', type=str,
                                 default='./netvlad_data/pittsburgh30k/reference_descriptors.tsv',
                                 help='np array of NetVLAD descriptors for the reference images')

        # blurring for NLL-CycleGAN
        self.parser.add_argument('--blur', action='store_true', default=False, help='Blur uncertain regions')
        self.parser.add_argument('--blur_thresh', type=int, default=-7, help='Value between -5*e and +5*e')
        self.parser.add_argument('--blur_dilat_size', type=int, default=9, help='Size of structuring element for dilatation')  # noqa
        self.parser.add_argument('--blur_gauss_size', type=int, default=7, help='Size of Gaussian matrix')
        self.parser.add_argument('--blur_gauss_sigma', type=int, default=3, help='Standard deviation of Gaussian')
