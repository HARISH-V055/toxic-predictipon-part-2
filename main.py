"""
Main Entry Point for EQ-KA-GCN Scientific Project Pipeline

Coordinates initialization, dataset loading, validation, preprocessing, statistics generation,
graph dataset construction, stratified splitting, DataLoader construction, model training,
and baseline GCN model evaluation with automated text/JSON reports and publication-quality figures.
Emits standard IEEE publication project metadata.
"""

import json
import sys
from pathlib import Path
from typing import List
import torch
from torch_geometric.data import Data

from config import get_config
from datasets import load_dataset, validate_dataset, clean_dataset, dataset_statistics
from graph import (
    smiles_to_graph,
    draw_molecule,
    print_graph_info,
    DatasetBuilder,
    GraphDataset,
    compute_and_log_dataset_statistics,
)
from models import BaselineGCN, get_loss_criterion
from training import (
    split_graph_dataset,
    create_dataloaders,
    create_optimizer,
    create_scheduler,
    EarlyStopping,
    History,
    Trainer,
)
from evaluation import (
    Evaluator,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    generate_json_report,
    generate_text_report,
)
from utils import set_seed, get_device, setup_logger


def log_split_statistics(name: str, graphs: List[Data], logger) -> None:
    """Computes and logs the toxicity class balance and size for a dataset split."""
    total = len(graphs)
    pos = sum(1 for g in graphs if g.y.item() == 1)
    neg = sum(1 for g in graphs if g.y.item() == 0)
    pos_pct = (pos / total) * 100 if total > 0 else 0.0
    neg_pct = (neg / total) * 100 if total > 0 else 0.0

    logger.info("==================================================================")
    logger.info(f"SPLIT STATISTICS - {name.upper()}")
    logger.info("==================================================================")
    logger.info(f"Number of Graphs:    {total}")
    logger.info(f"Positive Samples:    {pos} ({pos_pct:.2f}%)")
    logger.info(f"Negative Samples:    {neg} ({neg_pct:.2f}%)")
    logger.info("==================================================================")


def run_pipeline() -> None:
    """
    Orchestrates the EQ-KA-GCN pipeline:
      Phase 1: Project Initialization (directories, seed, device, logging)
      Phase 2: Dataset Ingestion, Validation, Cleaning, Statistics, and Saving
      Phase 3: Molecular Graph Construction Demonstration
      Phase 4: Full Dataset Graph Compilation, Serialization, & Statistics
      Phase 5: Stratified Train/Val/Test Dataset Splitting & DataLoader Generation
      Phase 7: Baseline GCN Model Training Loop & Metrics Logging
      Phase 8: Baseline GCN Model Test Set Evaluation, Plots, & Reports
    """
    # ─── PHASE 1: INITIALIZATION ─────────────────────────────────────────────
    # 1. Load configuration
    config = get_config()

    # 2. Dynamic directory creation for safety
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup logger
    logger = setup_logger(config.paths.logs_dir)

    logger.info("==================================================================")
    logger.info("Initializing EQ-KA-GCN Scientific Project Pipeline")
    logger.info("==================================================================")

    # 4. Set random seed
    set_seed(config.training.seed)
    logger.info(f"Global random seed set to: {config.training.seed}")

    # 5. Detect and log device
    try:
        device = get_device(config.device)
        logger.info(f"Detected computing device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize computing device: {str(e)}")
        sys.exit(1)

    # 6. Log configuration details
    logger.info("--- Configuration System Settings ---")
    logger.info(f"Project Name:           {config.model.name}")
    logger.info(f"Model Save Target:      {config.paths.checkpoints_dir / config.model.save_filename}")
    logger.info(f"Batch Size:             {config.training.batch_size}")
    logger.info(f"Learning Rate:          {config.training.learning_rate}")
    logger.info(f"Training Epochs:        {config.training.epochs}")
    logger.info(f"QAT Enabled:            {config.quantization.qat_enabled} ({config.quantization.bits}-bit)")
    logger.info(f"Model Hidden Dim:       {config.model.hidden_dim}")
    logger.info(f"Model Dropout:          {config.model.dropout}")
    logger.info(f"Model Layers (GCN):     {config.model.num_gcn_layers}")
    logger.info("==================================================================")
    logger.info("Project initialized successfully. Running Phase 2: Data construction.")

    # ─── PHASE 2: DATA LOADING & PREPROCESSING ──────────────────────────────
    logger.info("Starting Phase 2: Dataset Loading and Preprocessing...")
    
    # Define paths
    raw_csv_path = config.paths.raw_dir / config.data.raw_filename
    processed_csv_path = config.paths.processed_dir / config.data.processed_filename
    
    # 1. Load Dataset
    try:
        df_raw = load_dataset(raw_csv_path)
    except Exception as e:
        logger.error(f"Pipeline terminated. Failed to load raw dataset: {str(e)}")
        sys.exit(1)
        
    # 2. Validate Dataset
    try:
        validate_dataset(
            df_raw, 
            smiles_column=config.data.smiles_column, 
            target_column=config.data.target_column
        )
    except Exception as e:
        logger.error(f"Pipeline terminated. Dataset validation failed: {str(e)}")
        sys.exit(1)
        
    # 3. Clean Dataset
    df_clean = clean_dataset(
        df_raw, 
        smiles_column=config.data.smiles_column, 
        target_column=config.data.target_column
    )
    
    # 4. Generate Statistics
    _ = dataset_statistics(
        raw_df=df_raw,
        clean_df=df_clean,
        dataset_name=config.model.name,
        target_column=config.data.target_column,
        smiles_column=config.data.smiles_column
    )
    
    # 5. Save Clean Dataset
    try:
        logger.info(f"Saving cleaned dataset to: {processed_csv_path}")
        df_clean.to_csv(processed_csv_path, index=False)
        logger.info("Cleaned dataset saved successfully.")
    except Exception as e:
        logger.error(f"Pipeline terminated. Failed to save cleaned dataset: {str(e)}")
        sys.exit(1)
        
    logger.info("==================================================================")
    logger.info("Phase 2 complete. Running Phase 3: Graph Construction Demo.")
    logger.info("==================================================================")

    # ─── PHASE 3: GRAPH CONSTRUCTION & VISUALIZATION DEMONSTRATION ─────────
    logger.info("Starting Phase 3: Graph Construction Demonstration...")

    if df_clean.empty:
        logger.error("Cleaned dataset is empty. Cannot demonstrate Phase 3.")
        sys.exit(1)

    # 1. Load one sample molecule from the cleaned dataset
    sample_row = df_clean.iloc[0]
    sample_smiles = sample_row[config.data.smiles_column]
    sample_label = int(sample_row[config.data.target_column])

    logger.info(f"Loaded sample molecule SMILES: '{sample_smiles}' with label {sample_label}")

    # 2. Convert it into a graph
    graph_data = smiles_to_graph(sample_smiles, sample_label)

    if graph_data is not None:
        # 3. Print graph information
        print_graph_info(graph_data)

        # 4. Draw the molecule
        img = draw_molecule(sample_smiles)
        if img is not None:
            save_img_path = config.paths.figures_dir / "sample_molecule.png"
            img.save(save_img_path)
            logger.info(f"Successfully drew molecule and saved to: {save_img_path}")
        else:
            logger.warning("Failed to render molecular drawing.")
    else:
        logger.error("Failed to build graph from sample molecule.")

    logger.info("==================================================================")
    logger.info("Phase 3 complete. Running Phase 4: Full Dataset Graph Construction.")
    logger.info("==================================================================")

    # ─── PHASE 4: FULL DATASET GRAPH COMPILATION & SERIALIZATION ───────────
    logger.info("Starting Phase 4: Full Dataset Graph Processing...")

    # Define paths
    graphs_pt_path = config.paths.processed_dir / config.data.graphs_filename
    info_json_path = config.paths.processed_dir / config.data.info_filename

    # 1. Generate graph dataset
    builder = DatasetBuilder()
    graphs = builder.build_dataset(
        csv_path=processed_csv_path,
        smiles_column=config.data.smiles_column,
        target_column=config.data.target_column,
    )

    # 2. Save graphs.pt
    builder.save_dataset(graphs, graphs_pt_path)

    # 3. Reload graphs.pt using GraphDataset wrapper
    logger.info(f"Reloading graph dataset from: {graphs_pt_path}")
    graph_dataset = GraphDataset(graphs_path=graphs_pt_path)
    logger.info(f"Loaded Graph Dataset. Graphs count: {len(graph_dataset)}")

    # 4. Run statistics
    stats = compute_and_log_dataset_statistics(
        dataset=graph_dataset,
        skipped_count=builder.skipped_count,
        dataset_name="Tox21",
        target_column=config.data.target_column,
    )

    # 5. Save stats to dataset_info.json
    try:
        logger.info(f"Writing dataset information metadata to: {info_json_path}")
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("dataset_info.json updated successfully.")
    except Exception as e:
        logger.error(f"Failed to write dataset_info.json metadata: {str(e)}")

    # 6. Print one sample graph from reloaded dataset
    if len(graph_dataset) > 0:
        logger.info("Displaying one sample graph from the reloaded GraphDataset:")
        sample_graph = graph_dataset[0]
        print_graph_info(sample_graph)

    logger.info("==================================================================")
    logger.info("Phase 4 complete. Running Phase 5: Stratified Dataset Splitting.")
    logger.info("==================================================================")

    # ─── PHASE 5: STRATIFIED SPLITTING & DATALOADER GENERATION ──────────────
    logger.info("Starting Phase 5: Stratified Dataset Splitting...")

    # Define filenames
    train_graphs_path = config.paths.processed_dir / config.data.train_graphs_filename
    val_graphs_path = config.paths.processed_dir / config.data.val_graphs_filename
    test_graphs_path = config.paths.processed_dir / config.data.test_graphs_filename

    # 1. Split graph list
    train_graphs, val_graphs, test_graphs = split_graph_dataset(
        graphs=graph_dataset.graphs,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=config.training.seed,
    )

    # 2. Save split datasets to disk
    try:
        logger.info(f"Saving train graphs split ({len(train_graphs)} items) to: {train_graphs_path}")
        torch.save(train_graphs, train_graphs_path)

        logger.info(f"Saving validation graphs split ({len(val_graphs)} items) to: {val_graphs_path}")
        torch.save(val_graphs, val_graphs_path)

        logger.info(f"Saving test graphs split ({len(test_graphs)} items) to: {test_graphs_path}")
        torch.save(test_graphs, test_graphs_path)

        logger.info("All stratified splits saved successfully to disk.")
    except Exception as e:
        logger.error(f"Failed to serialize split datasets: {str(e)}")
        sys.exit(1)

    # 3. Create PyTorch Geometric DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        batch_size=config.training.batch_size,
    )

    # 4. Generate and Log Dataset Statistics for each split
    log_split_statistics("Train Split", train_graphs, logger)
    log_split_statistics("Validation Split", val_graphs, logger)
    log_split_statistics("Test Split", test_graphs, logger)

    logger.info("==================================================================")
    logger.info("Phase 5 complete. Running Phase 7: Model Training.")
    logger.info("==================================================================")

    # ─── PHASE 7: BASELINE GCN TRAINING ──────────────────────────────────────
    logger.info("Starting Phase 7: Baseline GCN Model Training Loop...")

    # 1. Instantiate Model
    model = BaselineGCN(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        dropout=config.model.dropout,
    )
    logger.info(f"Model initialized:\n{str(model)}")
    model.to(device)

    # 2. Set Up Training Utilities
    criterion = get_loss_criterion(positive_class_weight=None)
    optimizer = create_optimizer(
        model=model,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    
    # Early Stopping setup to save best_model.pt
    best_model_path = config.paths.checkpoints_dir / config.model.save_filename
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping,
        save_path=str(best_model_path),
    )
    
    # History tracking setup to save history.csv
    history = History()

    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        history=history,
    )

    # 4. Fit/Train GCN
    best_val_loss, best_epoch, training_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
    )

    # 5. Export history logs to outputs/history.csv
    history_csv_path = config.paths.outputs_dir / "history.csv"
    history.export_csv(str(history_csv_path))

    # 6. Print summary statement
    logger.info("==================================================================")
    logger.info("BASELINE GCN TRAINING RUN COMPLETE")
    logger.info("==================================================================")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"Best Epoch:           {best_epoch}")
    logger.info(f"Training Time:        {training_time:.2f} seconds")
    logger.info("==================================================================")

    # ─── PHASE 8: MODEL EVALUATION ───────────────────────────────────────────
    logger.info("Starting Phase 8: Model Evaluation on Held-Out Test Set...")

    # 1. Instantiate Evaluator
    evaluator = Evaluator(model=model, device=device)

    # 2. Load the best checkpoint weights saved during early stopping
    evaluator.load_model(str(best_model_path))

    # 3. Evaluate the model on test DataLoader
    test_metrics, y_true, y_pred, y_prob = evaluator.evaluate(
        loader=test_loader,
        threshold=config.evaluation.threshold,
    )

    # 4. Setup output locations
    outputs_figures_dir = config.paths.outputs_dir / "figures"
    outputs_figures_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = config.paths.outputs_dir / "evaluation_report.json"
    text_report_path = config.paths.outputs_dir / "classification_report.txt"

    loss_curve_path = outputs_figures_dir / "loss_curve.png"
    accuracy_curve_path = outputs_figures_dir / "accuracy_curve.png"
    roc_curve_path = outputs_figures_dir / "roc_curve.png"
    pr_curve_path = outputs_figures_dir / "precision_recall_curve.png"
    cm_path = outputs_figures_dir / "confusion_matrix.png"

    # 5. Save reports
    if config.evaluation.save_reports:
        generate_json_report(
            metrics=test_metrics,
            save_path=str(json_report_path),
            model_name=config.model.name,
            dataset_name="Tox21",
        )
        generate_text_report(
            metrics=test_metrics,
            save_path=str(text_report_path),
        )

    # 6. Save plots
    if config.evaluation.save_plots:
        plot_loss_curve(str(history_csv_path), str(loss_curve_path))
        plot_accuracy_curve(str(history_csv_path), str(accuracy_curve_path))
        plot_roc_curve(y_true, y_prob, str(roc_curve_path))
        plot_precision_recall_curve(y_true, y_prob, str(pr_curve_path))
        plot_confusion_matrix(test_metrics["confusion_matrix"], str(cm_path))

    # 7. Print summary console statement exactly as required
    print("=================================================")
    print("BASELINE GCN EVALUATION")
    print("=================================================")
    print(f"Accuracy            : {test_metrics['accuracy'] * 100:.2f} %")
    print(f"Precision           : {test_metrics['precision'] * 100:.2f} %")
    print(f"Recall              : {test_metrics['recall'] * 100:.2f} %")
    print(f"F1 Score            : {test_metrics['f1_score'] * 100:.2f} %")
    print(f"ROC-AUC             : {test_metrics['roc_auc']:.4f}")
    print(f"Balanced Accuracy   : {test_metrics['balanced_accuracy'] * 100:.2f} %")
    print(f"MCC                 : {test_metrics['mcc']:.4f}")
    print(f"Inference Time      : {test_metrics['inference_time_per_sample_ms']:.2f} ms/sample")
    print("=================================================")
    print("Outputs Saved")
    print("evaluation_report.json")
    print("classification_report.txt")
    print("roc_curve.png")
    print("confusion_matrix.png")
    print("loss_curve.png")
    print("accuracy_curve.png")
    print("precision_recall_curve.png")
    print("=================================================")


if __name__ == "__main__":
    run_pipeline()
