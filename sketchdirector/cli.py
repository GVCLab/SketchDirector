import copy
import json
import os
import re
from pathlib import Path

from . import config

def _slugify(text, max_words=10):
    name = '_'.join(text.strip().split()[:max_words]) or 'scene'
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return name.strip('._') or 'scene'


def _resolve_file(configured, env_name, candidates):
    configured = configured or os.getenv(env_name)
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'{env_name} path does not exist: {path}')
        return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not resolve {env_name}. Tried: {candidates}')


def _prepare_args(base_args, concept, base_output, base_assets):
    args = copy.deepcopy(base_args)
    args.concept_to_draw = concept
    scene_name = _slugify(concept)
    if not args.wandb_name or args.wandb_name in {'default', 'defualt'}:
        args.wandb_name = f'{scene_name}_{args.num_strokes}_strokes'
    args.output_dir_target = str(base_output / scene_name)
    args.output_dir = str(base_output / scene_name / args.wandb_name)
    args.path2save = str(base_assets / _slugify(concept, 30))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.path2save).mkdir(parents=True, exist_ok=True)
    log = Path(args.path2save) / 'experiment_log.json'
    if not log.exists():
        log.write_text('[]\n', encoding='utf-8')
    return args


def main(argv=None):
    args = config.parse_arguments(argv)

    from .agent import ExperienceMemory, SketchAgent
    from .generation import SketchApp
    project_root = Path(__file__).resolve().parent.parent
    args.background_path = str(_resolve_file(
        args.background_path, 'SKETCHDIRECTOR_BACKGROUND_PATH', [project_root/'assets'/'blank.png']
    ))

    if args.concept_file:
        concept_file = _resolve_file(args.concept_file, 'SKETCHDIRECTOR_CONCEPT_FILE', [])
        concepts = [line.strip() for line in concept_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        default_file = project_root / 'examples' / 'prompts.txt'
        concepts = [line.strip() for line in default_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if args.concept_to_draw and args.concept_to_draw != 'cat':
            concepts = [args.concept_to_draw]

    base_output = Path(args.output_dir)
    base_assets = Path(args.path2save)
    base_output.mkdir(parents=True, exist_ok=True)
    base_assets.mkdir(parents=True, exist_ok=True)
    memory = ExperienceMemory(args.memory_dir)

    for index, concept in enumerate(concepts, 1):
        print(f'\n[{index}/{len(concepts)}] {concept}')
        scene_args = _prepare_args(args, concept, base_output, base_assets)
        app = SketchApp(scene_args)
        agent = SketchAgent(app, scene_args, memory)
        result = agent.run_loop(concept)

        if result.get('success') and not scene_args.skip_refinement:
            from .refinement.optimizer import RefinementOptimizer
            RefinementOptimizer(scene_args, app).run_optimize_progress(result)
        elif not result.get('success'):
            print(f"Scene failed: {result.get('error', 'unknown error')}")
        memory.save_memory()
    return 0
