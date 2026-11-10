"""Correctness checks for the provided WebNLG preparation and scoring."""
import csv
import io
import tarfile

import pytest


def archive(path, split, entries):
    xml = '<benchmark><entries>' + ''.join(
        '<entry><modifiedtripleset>' + ''.join(f'<mtriple>{t}</mtriple>' for t in triples)
        + '</modifiedtripleset>' + ''.join(f'<lex>{r}</lex>' for r in refs) + '</entry>'
        for triples, refs in entries) + '</entries></benchmark>'
    with tarfile.open(path, 'w:gz') as tar:
        member = tarfile.TarInfo(f'repo/release_v3.0/en/{split}/1triples/a.xml')
        data = xml.encode()
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))


def test_group_references_and_remove_train_overlap(tmp_path):
    from hw4lib.utils.webnlg import prepare_webnlg, load_webnlg_records
    raw = tmp_path / 'raw'
    raw.mkdir()
    archive(raw / 'train.tar.gz', 'train', [(['A | p | B'], ['one']), (['A|p|B'], ['two']), (['C|p|D'], ['held out'])])
    archive(raw / 'dev.tar.gz', 'dev', [(['C | p | D'], ['dev reference'])])
    stats = prepare_webnlg(raw, tmp_path / 'prepared')
    train = load_webnlg_records(tmp_path / 'prepared/train.jsonl')
    assert len(train) == 1
    assert train[0]['references'] == ['one', 'two']
    assert stats['removed_train_groups'] == 1
    assert stats['removed_train_references'] == 1


def test_exact_references_score_100():
    from hw4lib.utils.webnlg import score_webnlg
    text = 'The airport is located in the city.'
    result = score_webnlg([text], [[text]])
    assert result['bleu'] == pytest.approx(100)
    assert result['rouge_l'] == pytest.approx(100)
    assert result['kaggle_score'] == pytest.approx(100)
    assert score_webnlg([''], [[text]])['bleu'] == 0
    assert score_webnlg(['xyz xyz xyz xyz'], [[text]])['bleu'] == 0


def test_variable_reference_counts():
    from hw4lib.utils.webnlg import score_webnlg
    first, second = 'A quiet airport serves this city.', 'This building was built in 1920.'
    result = score_webnlg([first, second], [['An alternative wording.', first], [second]])
    assert result['bleu'] == pytest.approx(100)
    assert result['rouge_l'] == pytest.approx(100)


def test_rouge_l_averages_best_reference_f1_per_input():
    from hw4lib.utils.webnlg import score_webnlg
    # First input: LCS=2, precision=2/2, recall=2/3, F1=0.8.
    # Second input: empty prediction, F1=0. The average is 40/100.
    result = score_webnlg(['ALPHA beta', ''], [
        ['unrelated words', 'alpha beta gamma'], ['delta epsilon']])
    assert result['rouge_l'] == pytest.approx(40)
    assert result['kaggle_score'] == pytest.approx(20)


def test_csv_quotes_commas_and_newlines(tmp_path):
    from hw4lib.utils.webnlg import write_predictions
    text = 'A said, "Hello".\nThen left.'
    path = tmp_path / 'predictions.csv'
    write_predictions([{'id': '0001'}], [text], path)
    with path.open(newline='') as f:
        assert list(csv.DictReader(f)) == [{'id': '0001', 'prediction': text}]
