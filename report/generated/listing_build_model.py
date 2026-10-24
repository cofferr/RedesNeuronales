def build_model(max_iter: int = 1) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(config.N_HIDDEN,),
        activation="relu",
        solver="sgd",
        alpha=0.0,
        batch_size=config.N_TRAIN,
        learning_rate="constant",
        learning_rate_init=config.LEARNING_RATE,
        momentum=0.0,
        nesterovs_momentum=False,
        shuffle=False,
        early_stopping=False,
        max_iter=max_iter,
        tol=0.0,
        n_iter_no_change=config.EPOCHS + 1,
        random_state=config.SEED,
    )
