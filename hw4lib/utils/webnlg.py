"""Provided WebNLG v3.0 English preparation, metrics, and CSV export."""
import csv
import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from itertools import zip_longest
from pathlib import Path, PurePosixPath


def canonical_graph(triples: list[str]) -> tuple[str, ...]:
    """Compare complete triple sets, ignoring input order and extra spaces."""
    if not isinstance(triples, list) or not triples:
        raise ValueError('Each input needs a nonempty list of triples')
    canonical = []
    for triple in triples:
        if not isinstance(triple, str):
            raise ValueError('Triples must be strings')
        parts = [part.strip() for part in triple.split('|')]
        if len(parts) != 3 or not all(parts):
            raise ValueError('Each triple must have subject | relation | object')
        canonical.append(' | '.join(parts))
    return tuple(sorted(set(canonical)))


def format_webnlg_prompt(triples: list[str]) -> str:
    canonical_graph(triples)
    return ('Write an English description of the following RDF triples.\n'
            'Triples:\n' + '\n'.join(triples) + '\nDescription:')


def _read_archive(path: Path, split: str) -> dict:
    groups = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            parts = PurePosixPath(member.name).parts
            if member.isdir():
                continue
            if (not member.isfile() or '..' in parts or member.name.startswith('/')
                    or parts[1:4] != ('release_v3.0', 'en', split)
                    or not member.name.endswith('.xml')):
                raise ValueError('Expected a WebNLG v3.0 English split archive')
            # Read members in memory; never extract archive paths to disk.
            with archive.extractfile(member) as stream:
                root = ET.parse(stream).getroot()
            if root.tag != 'benchmark' or root.find('entries') is None:
                raise ValueError('Invalid WebNLG XML schema')
            for entry in root.findall('./entries/entry'):
                triples = [(node.text or '').strip() for node in entry.findall('./modifiedtripleset/mtriple')]
                key = canonical_graph(triples)
                record = groups.setdefault(key, {'triples': triples, 'references': []})
                record['references'].extend(
                    text for node in entry.findall('lex') if (text := ''.join(node.itertext()).strip()))
    if not groups or any(not row['references'] for row in groups.values()):
        raise ValueError('Empty split or an input without reference descriptions')
    return groups


def prepare_webnlg(raw_dir: Path, output_dir: Path, *, download: bool = False) -> dict:
    """Prepare train/dev JSONL; remove exact train graphs also found in dev.

    Place train.tar.gz and dev.tar.gz under raw_dir, or set download=True.
    Source: https://gitlab.com/shimorina/webnlg-dataset (v3.0, CC BY-NC-SA 4.0).
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Choose a new prepared directory: {output_dir}')
    groups = {}
    for split in ('train', 'dev'):
        path = raw_dir / f'{split}.tar.gz'
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f'Place the official {split}.tar.gz in {raw_dir}, or enable download')
            raw_dir.mkdir(parents=True, exist_ok=True)
            url = ('https://gitlab.com/api/v4/projects/shimorina%2Fwebnlg-dataset/'
                   'repository/archive.tar.gz?path=release_v3.0%2Fen%2F' + split)
            # A failed download must not be mistaken for a complete cached archive.
            with urllib.request.urlopen(url, timeout=120) as response, path.with_suffix('.download').open('wb') as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
            path.with_suffix('.download').replace(path)
        groups[split] = _read_archive(path, split)
    overlap = groups['train'].keys() & groups['dev'].keys()
    counts = {'removed_train_groups': len(overlap),
              'removed_train_references': sum(len(groups['train'][key]['references']) for key in overlap)}
    for key in overlap:
        del groups['train'][key]
    if not groups['train']:
        raise ValueError('No training inputs remain after overlap removal')
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in groups.items():
        with (output_dir / f'{split}.jsonl').open('x', encoding='utf-8') as f:
            for i, record in enumerate(rows.values()):
                f.write(json.dumps({'id': f'{split}-{i:06d}', **record}, ensure_ascii=False) + '\n')
        counts[split + '_groups'] = len(rows)
        counts[split + '_references'] = sum(len(row['references']) for row in rows.values())
    (output_dir / 'data_info.json').write_text(json.dumps(counts, indent=2) + '\n')
    return counts


def load_webnlg_records(path: Path, *, require_references: bool = True) -> list[dict]:
    """Read prepared JSONL or competition id/triples CSV."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as file:
        if path.suffix == ".csv":
            rows = list(csv.DictReader(file))
            for row in rows:
                row["triples"] = json.loads(row["triples"])
            return rows
        return [json.loads(line) for line in file if line.strip()]


def score_webnlg(predictions: list[str], references: list[list[str]]) -> dict:
    """Return corpus BLEU and mean best-reference ROUGE-L F1, both on 0–100."""
    from rouge_score.rouge_scorer import RougeScorer
    from sacrebleu.metrics import BLEU
    if not predictions or len(predictions) != len(references):
        raise ValueError('Predictions and references must have equal nonzero length')
    if any(not isinstance(p, str) for p in predictions):
        raise ValueError('Predictions must be strings')
    if any(not isinstance(rs, list) or not rs or any(not isinstance(r, str) or not r.strip() for r in rs)
           for rs in references):
        raise ValueError('Each input needs nonempty reference strings')
    streams = [list(column) for column in zip_longest(*references, fillvalue=None)]
    bleu = BLEU(tokenize='13a', lowercase=False, smooth_method='exp', effective_order=False)
    rouge = RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = sum(rouge.score_multi(refs, prediction)['rougeL'].fmeasure
                  for prediction, refs in zip(predictions, references)) / len(predictions)
    result = {'bleu': float(bleu.corpus_score(predictions, streams).score),
              'rouge_l': 100.0 * rouge_l}
    result['kaggle_score'] = (result['bleu'] + result['rouge_l']) / 2.0
    result['settings'] = {'bleu': str(bleu.get_signature()),
                          'rouge_l': 'rouge-score 0.1.2; rougeL F1; no stemming; best reference; mean; 0-100'}
    return result


def write_predictions(records: list[dict], predictions: list[str], path: Path) -> None:
    """Write id,prediction, preserving IDs and CSV quoting. Never overwrite."""
    ids = [row['id'] for row in records]
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        writer.writerows(zip(ids, predictions))
