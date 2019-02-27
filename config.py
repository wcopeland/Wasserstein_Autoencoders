class Config(object):
    # Identifier for
    NAME = None

    # Default device is cpu
    DEVICE = 'cpu'

    # Default batch size
    BATCH_SIZE = 100 #batch_size

    #
    NUM_EPOCHS = 100 #epochs

    #
    LEARNING_RATE = 1e-4 #lr


    # regularization coeff MMD term
    LAMBDA = 10


    # Latent space parameters
    # Dimension of the latent space (n_z)
    LATENT_DIMENSION = 8
    # Variance of the hidden dimension (sigma)
    LATENT_SAMPLING_VARIANCE = 1

    ENCODER_DIMENSION_1 = 128 #dim_h

    def __init__(self):
        return

    def display(self):
        heading = 'CONFIGURATION:'
        print('\n{}'.format(heading))
        print(''.join(['=' for _ in heading]))
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")