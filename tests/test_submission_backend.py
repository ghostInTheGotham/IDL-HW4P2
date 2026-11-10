"""Exercise the provided submission flow without contacting Kaggle or W&B."""
import json
import pickle
from dataclasses import replace
import zipfile

import pytest
import torch
from torch import nn

from submission import main
from submission.backend_config import HW4P2_BACKEND_CONFIG
from submission.model_metadata import build_model_metadata, ModelMetadataError
from submission.submission_config import SubmissionConfig


def submission_config(tmp_path, **changes):
    notebook = tmp_path / 'completed.ipynb'
    notebook.write_text('{"cells": [], "nbformat": 4}')
    values = dict(model=nn.Linear(3, 2), kaggle_username='qa-user',
                  kaggle_api_key='test-token', wandb_api_key='test-wandb-token',
                  wandb_entity='qa', wandb_project='hw4p2', acknowledged=True,
                  readme='Model and training results.', notebook_path=str(notebook))
    values.update(changes)
    return SubmissionConfig(**values)


def test_final_zip_contains_checklist_and_linked_wandb_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ranked = {'competition': 'pilot', 'score': 12.0, 'score_source': 'public',
              'submission_ref': '72', 'submitted_at': None, 'wandb_run_path': 'qa/hw4p2/run123'}
    monkeypatch.setattr(main, 'export_kaggle_metadata', lambda **kwargs: {
        'kaggle_username': 'qa-user', 'total_submissions': 1, 'ranked_submissions': [ranked]})
    monkeypatch.setattr(main, 'export_wandb_runs', lambda **kwargs: {'run_paths': kwargs['run_paths']})
    backend = replace(HW4P2_BACKEND_CONFIG, param_limit=10, submission_zip='final.zip',
                      required_evidence_archive=None)
    main.create_submission_zip(submission_config(tmp_path), backend)
    with zipfile.ZipFile(tmp_path / 'final.zip') as archive:
        assert {'ACKNOWLEDGEMENT.txt', 'README.txt', 'model_metadata.json',
                'kaggle_data.json', 'wandb_export.pkl', 'completed.ipynb'} <= set(archive.namelist())
        assert json.loads(archive.read('model_metadata.json'))['parameter_count'] == 8
        assert json.loads(archive.read('kaggle_data.json'))['ranked_submissions'][0]['submission_ref'] == '72'
        assert pickle.loads(archive.read('wandb_export.pkl'))['run_paths'] == ['qa/hw4p2/run123']


def test_parameter_limit_includes_frozen_base_weights():
    model = nn.Linear(3, 2)
    model.weight.requires_grad_(False)
    with pytest.raises(ModelMetadataError):
        build_model_metadata(model, param_limit=8)
    metadata = build_model_metadata(model, param_limit=9)
    assert metadata['parameter_count'] == 8
    assert metadata['model']['trainable_parameters'] == 2


def test_acknowledgement_is_required_before_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(main.SubmissionError, match='Acknowledgement'):
        main.create_submission_zip(submission_config(tmp_path, acknowledged=False), HW4P2_BACKEND_CONFIG)
    assert not list(tmp_path.glob('*.zip'))


def test_loaded_checkpoint_keeps_its_kaggle_wandb_message(tmp_path):
    from hw4lib.utils.sft_experiment import load_base_weights
    from hw4lib.trainers.sft_trainer import SFTTrainer
    model = nn.Linear(3, 2)
    tracking = {'entity': 'qa', 'project': 'hw4p2', 'run_id': 'run123', 'run_name': 'full'}
    checkpoint = tmp_path / 'best.pth'
    torch.save({'model_state_dict': model.state_dict(),
                'submission_tracking': {'version': 1, 'wandb_ref': tracking}}, checkpoint)
    load_base_weights(model, checkpoint)
    predictor = object.__new__(SFTTrainer)
    predictor.model = model
    assert predictor.build_kaggle_submission_message('HW4P2') == 'HW4P2|WB=qa/hw4p2/run123'
