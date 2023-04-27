# def train_model(
#     across="session",
#     n_sess=24,
#     data_dir="/Users/jrudoler/data/small_scalp_features/",
#     fast_dev_run=False,
# ):
#     # HYPERPARAMETERS
#     learning_rate = 1e-2
#     weight_decay = 0.5
#     batch_size = 256
#     ########################
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     _ = pl.seed_everything(56)
#     subject = "LTP093"
#     test_result = []
#     for sess in range(n_sess):
#         if across == "session":
#             test_file_crit = (
#                 lambda s: s.endswith(".pt")
#                 and s.count(f"sub_{subject}")
#                 and s.count(f"sess_{sess}")
#             )
#             train_file_crit = (
#                 lambda s: s.endswith(".pt")
#                 and s.count(f"sub_{subject}")
#                 and not s.count(f"sess_{sess}")
#             )
#         elif across == "subject":
#             test_file_crit = (
#                 lambda s: s.endswith(".pt")
#                 and s.count(f"sub_{subject}")
#                 and s.count(f"sess_{sess}")
#             )
#             train_file_crit = lambda s: s.endswith(".pt") and not (
#                 s.count(f"sub_{subject}") and s.count(f"sess_{sess}")
#             )
#         else:
#             raise ValueError(
#                 f"across must be 'session' or 'subject', not '{across}'"
#             )
#         try:
#             test_dataset = DatasetFolder(
#                 data_dir,
#                 loader=partial(torch.load),
#                 is_valid_file=test_file_crit,
#             )
#             train_dataset = DatasetFolder(
#                 data_dir,
#                 loader=partial(torch.load),
#                 is_valid_file=train_file_crit,
#             )
#         except FileNotFoundError:
#             print(f"no session {sess}")
#             test_result += [{"subject": subject, "session": sess}]
#             continue
#         # class balancing
#         cls_weights = compute_class_weight(
#             class_weight="balanced",
#             classes=np.unique(train_dataset.targets),
#             y=train_dataset.targets,
#         )
#         weights = cls_weights[train_dataset.targets]
#         sampler = WeightedRandomSampler(
#             weights, len(train_dataset), replacement=True  # type: ignore
#         )
#         # data loaders
#         train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             pin_memory=True,
#             num_workers=N_CPU,
#             prefetch_factor=10,
#             persistent_workers=True,
#         )
#         test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=len(test_dataset),
#             shuffle=False,
#             pin_memory=True,
#         )
#         # create model
#         n_features = train_dataset[0][0].shape[0]
#         model = LitPrecondition(
#             n_features, n_features, learning_rate, weight_decay, batch_size
#         )
#         es = EarlyStopping(
#             "Loss/train", min_delta=1e-3, patience=25, mode="min"
#         )
#         lr_mtr = LearningRateMonitor("epoch")
#         check = ModelCheckpoint(monitor="AUC/train", mode="max")
#         run_dir = f"run_{subject}_{sess}_{timestr}"
#         logger = TensorBoardLogger(
#             save_dir=log_dir,
#             name="precondition",
#             version=run_dir,
#             default_hp_metric=True,
#         )
#         logger.log_hyperparams(
#             {
#                 "learning_rate": learning_rate,
#                 "weight_decay": weight_decay,
#                 "batch_size": batch_size,
#             },
# {"AUC/train": 0, "Loss/train": 0, "AUC/test": 0, "Loss/test":0},
#         )
#         trainer = Trainer(
#             min_epochs=50,
#             max_epochs=500,
#             accelerator="mps",
#             devices=1,
#             callbacks=[lr_mtr, es, check],
#             logger=logger,
#             log_every_n_steps=5,
#             fast_dev_run=fast_dev_run,
#         )
#         trainer.fit(
#             model,
#             train_dataloaders=train_dataloader,
#             val_dataloaders=test_dataloader,
#         )
#         if fast_dev_run:
#             return
#         model = LitPrecondition.load_from_checkpoint(
#             trainer.checkpoint_callback.best_model_path  # type: ignore
#         )  # Load best checkpoint after training
#         test_result += trainer.test(
#             model, dataloaders=test_dataloader, verbose=False
#         )
#         test_result[-1].update({"subject": subject, "session": sess})
#         torch.mps.empty_cache()
#         result_df = pd.DataFrame(test_result)
#         result_df.to_csv(log_dir + f"precond_results_LTP093_{timestr}.csv")
#         result_df.to_csv(f"precond_results_LTP093_{timestr}.csv")
