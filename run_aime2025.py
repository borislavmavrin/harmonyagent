from minisweagent.run.aime25_harmony import main

if __name__ == "__main__":
    import shutil
    from pathlib import Path

    debug_path = Path("evaluation/aime25_medium/")

    if debug_path.exists():
        shutil.rmtree(debug_path)

    main(
        n_repeats=8,
        slice_spec="0:30",
        filter_spec="",
        output=debug_path,
        model="gpt-oss",
        model_class="vllmraw",
        config_path="src/minisweagent/config/aime25_harmony.yaml",
        redo_existing=True,
        shuffle=False,
        workers=6,
    )
