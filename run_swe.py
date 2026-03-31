from harmonyagent.run.swebench_harmony import main

if __name__ == "__main__":
    # import shutil
    from pathlib import Path

    debug_path = "evaluation/swe_medium/"

    resolution_statuses = main(
        subset="verified",
        split="test",
        slice_spec="",
        filter_spec="",
        instance_names_from_file="",
        output=debug_path,
        model="gpt-oss",
        model_class="vllmraw",
        config_path=Path("src/harmonyagent/config/swebench_harmony.yaml"),
        environment_class="docker",
        redo_existing=False,
        shuffle=True,
        workers=4,
    )
    resolution_score = len([row for row in resolution_statuses if row == "Resolved"]) / len(resolution_statuses)
    print(resolution_statuses)
    print(f"resolution_score: {resolution_score}")
