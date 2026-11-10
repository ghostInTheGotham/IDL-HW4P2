import zipfile


def test_code_export_keeps_shared_requirements_and_gqa_fixture(tmp_path):
    from hw4lib.utils.sft_experiment import export_submission
    (tmp_path / 'tests').mkdir()
    fixture = tmp_path / 'tests/gqa_reference.json'
    fixture.write_text('{"cases": []}')
    configs = tmp_path / 'submission/configs'
    configs.mkdir(parents=True)
    (configs / 'hw4p2.json').write_text('{"param_limit": 60000000}')
    (configs / 'hw1p2.json').write_text('{}')
    for name in ('requirements.txt', 'requirements-colab.txt',
                 'requirements-hw4p2.txt', 'requirements-hw4p2-colab.txt'):
        (tmp_path / name).write_text('pytest\n')
    notebook = tmp_path / 'notebook.ipynb'
    notebook.write_text('{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}')
    target = tmp_path / 'evidence.zip'
    export_submission(tmp_path, [], notebook, target)
    with zipfile.ZipFile(target) as archive:
        assert archive.read('tests/gqa_reference.json') == fixture.read_bytes()
        assert archive.read('submission/configs/hw4p2.json') == b'{"param_limit": 60000000}'
        assert archive.read('submission/configs/hw1p2.json') == b'{}'
        for name in ('requirements.txt', 'requirements-colab.txt',
                     'requirements-hw4p2.txt', 'requirements-hw4p2-colab.txt'):
            assert archive.read(name) == b'pytest\n'
