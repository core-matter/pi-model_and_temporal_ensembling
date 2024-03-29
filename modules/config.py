"""Constants used in the articled"""

num_classes                         = 10                        # Number of classes.
supervised_ratio                    = 1                         # Rate of labeled data.
rampup_length                       = 80                        # Ramp learning rate and unsupervised loss weight up during first n epochs.
rampdown_length                     = 50                        # Ramp learning rate and Adam beta1 down during last n epochs.
beta1                               = 0.9                       # Default value.
beta2                               = 0.999                     # Default value.
rampdown_beta1_target               = 0.5                       # Target value for Adam beta1 for rampdown.
X_ZCAmin                            = -28.21404101296473        # normalizing value
X_ZCAmax                            = 30.95940075784862         # normalizing value
num_epochs                          = 300                       # Total number of epochs.
minibatch_size                      = 100                       # Number of samples in batch.
noise_stddev                        = 0.15                      # the Gaussian noise added inside dataset during training
batch_normalization_momentum        = 0.999                     # Batch normalization momentum.
learning_rate_max                   = 0.003                     # Maximum learning rate.
augment_translation                 = 0.0625                    # Image translation by augment_translation * img_size pixels
unsup_weight_max                    = 100.0                     # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
start_epoch                         = 0                         # Which epoch to start training from. For continuing a previously trained network.
load_network_filename               = None                      # Set to load a previously saved network.
no_label                            = -1                        # label mask for unlabaled data
