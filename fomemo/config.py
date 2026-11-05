from gpytorch.kernels import RBFKernel


# Model Architecture
MODEL_CONFIG = {
    "embedding_size": 512,
    "hidden_size": 1024,
    "attention_heads": 4,
    "transformer_layers": 12,
    "output_logits_size": 1000,
    "max_encoder_features": 50,
    "max_encoder_objectives": 10,
}


# Data Generation
DATA_CONFIG = {
    "gp_mean": 0.0,
    "gp_kernel": RBFKernel,
    "gp_output_scale": 1.0,
    "gp_noise_variance": 1e-4,
    "lengthscale_concentration": 3.0,
    "lengthscale_rate": 6.0,
    "min_features": 1,
    "max_features": 30,
    "min_objectives": 2,
    "max_objectives": 6,
    "max_sample_length": 256,
    "sigmoid_scale": 1.5,
    "preference_concentration": 1.0,
    "gp_noise_lower_bound": 1e-9,
    "gp_fast_computations": (False, False, False),
}


# Model Training
TRAINING_CONFIG = {
    "batch_size": 16,
    "steps_per_epoch": 2048,
    "warmup_epochs": 5,
    "epochs": 500,
    "learning_rate": 5e-5,
    "checkpoint_interval": 10,
}


# Calibration
CALIBRATION_CONFIG = {
    "dimensions": (1, 5, 10, 15, 20, 25, 30),
    "trials": 200,
    "batch_size": 128,
    "seq_len": 1000,
    "ood_objectives": (7, 8, 9, 10),
}


# Demo
DEMO_CONFIG = {
    "epochs": 2,
    "steps_per_epoch": 2,
    "batch_size": 2,
    "seq_len": 16,
    "warmup_epochs": 0,
    "calibration_dimensions": (1,),
    "calibration_trials": 1,
    "calibration_batch_size": 16,
    "calibration_seq_len": 256,
}
