"""One frozen test evaluation: original VLM and the dev-selected clean adapter."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import zipfile

from .audit_dataset import verify_dataset
from .formal_protocol import ROOT, read
from .io_utils import read_jsonl, sha256_file, unique_by_id, write_json


FREEZE = Path('research/final-test-freeze-v1.json')


def verify_files(root: Path, expected: dict[str,str]) -> None:
    for name,digest in expected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path)!=digest:
            raise ValueError(f'Frozen file changed or missing: {name}')


def verify_predictions(root: Path,tag: str,method: str,freeze: dict) -> dict:
    truth=unique_by_id(read_jsonl(root/'data/prepared/formal-clean/test.jsonl'))
    path=root/'runs'/tag/method/'predictions.jsonl'
    rows=unique_by_id(read_jsonl(path));meta=read(path.with_suffix('.run.json'))
    if set(rows)!=set(truth) or len(rows)!=150 or meta['samples']!=150:
        raise ValueError('Final predictions do not cover exactly the frozen test150')
    expected={**freeze['inference'],'manifest_sha256':freeze['test_manifest_sha256']}
    for key,value in expected.items():
        if meta.get(key)!=value:
            raise ValueError(f'Final inference protocol changed: {key}')
    expected_adapter=(root/freeze['selected_adapter_relative']).resolve()
    if method=='q0' and meta['adapter'] is not None:
        raise ValueError('Q0 unexpectedly used an adapter')
    if method=='q1' and (not meta['adapter'] or Path(meta['adapter']).resolve()!=expected_adapter):
        raise ValueError('Q1 did not use the frozen selected adapter')
    if any(r['status'] not in {'ok','truncated'} for r in rows.values()):
        raise ValueError('Runtime errors occurred; retain predictions and recover without changing the freeze')
    return meta


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','claim','check-q0','check-q1','summarize','package'])
    parser.add_argument('--tag',required=True)
    parser.add_argument('--exit-code',type=int,default=0)
    args=parser.parse_args()
    if '/' in args.tag or '\\' in args.tag or not args.tag.startswith('final-test-'):
        raise ValueError('Invalid final run tag')
    freeze=read(ROOT/FREEZE)
    report,run=ROOT/'reports'/args.tag,ROOT/'runs'/args.tag
    if args.action=='verify':
        if freeze['selected_model']!='q1' or freeze['test_samples']!=150:
            raise ValueError('Final model selection is not the registered Q1')
        verify_files(ROOT,freeze['source_sha256'])
        verify_files(ROOT,freeze['bound_files'])
        verify_files(ROOT,freeze['model_files_sha256'])
        for name,version in freeze['gpu_versions'].items():
            if importlib.metadata.version(name)!=version:
                raise ValueError(f'Frozen runtime changed: {name}')
        data=ROOT/'data/prepared/formal-clean'
        if sha256_file(data/'test.jsonl')!=freeze['test_manifest_sha256']:
            raise ValueError('Frozen test manifest changed')
        verify_dataset(data)
        write_json(report/'freeze-binding.json',{'passed':True,'freeze_sha256':sha256_file(ROOT/FREEZE),
            'selected_model':'q1','test_samples':150,'training_ran':False})
        print('FINAL_TEST_FREEZE_VERIFIED')
    elif args.action=='claim':
        marker=ROOT/'runs/final-test-v1-started.json'
        if marker.exists():
            raise RuntimeError(f'Final test was already started: {read(marker)}. Do not launch a new full run.')
        write_json(marker,{'run':args.tag,'freeze_sha256':sha256_file(ROOT/FREEZE)})
    elif args.action.startswith('check-'):
        verify_predictions(ROOT,args.tag,args.action.removeprefix('check-'),freeze)
    elif args.action=='summarize':
        metrics={}
        for method in ['q0','q1']:
            verify_predictions(ROOT,args.tag,method,freeze)
            summary=read(report/method/'summary.json')
            path=run/method/'predictions.jsonl'
            if (summary['samples']!=150 or summary['manifest_sha256']!=freeze['test_manifest_sha256']
                or summary['predictions_sha256']!=sha256_file(path)
                or summary['normalization_version']!=freeze['normalization_version']):
                raise ValueError('Frozen scoring protocol differs')
            metrics[method]={k:summary[k] for k in ['teds','teds_s','valid_html_rate','categories']}
        write_json(report/'comparison.json',{'completed':True,'test_evaluated':True,'training_ran':False,
            'selected_model':'q1','freeze_sha256':sha256_file(ROOT/FREEZE),'metrics':metrics,
            'delta_q1_minus_q0':{k:metrics['q1'][k]-metrics['q0'][k] for k in ['teds','teds_s','valid_html_rate']},
            'note':'Frozen one-seed final test. Report results unchanged; do not tune or reselect using test.'})
        print(json.dumps(read(report/'comparison.json')))
    else:
        report.mkdir(parents=True,exist_ok=True)
        write_json(report/'exit-status.json',{'exit_code':args.exit_code,
            'completed':args.exit_code==0 and (report/'comparison.json').is_file(),'training_ran':False})
        destination=ROOT/f'{args.tag}-results.zip'
        with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for folder in [run,report]:
                if folder.exists():
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and p.suffix in {'.json','.jsonl','.md'}:
                            z.write(p,p.relative_to(ROOT).as_posix())
            z.write(ROOT/FREEZE,FREEZE.as_posix())
            log=ROOT/'logs'/f'{args.tag}.log'
            if log.is_file():z.write(log,log.relative_to(ROOT).as_posix())
        print(f'RESULTS_ZIP={destination}',flush=True)


if __name__=='__main__':
    main()
