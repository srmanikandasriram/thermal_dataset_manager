import uvicorn


def main() -> None:
    # Restrict the reload watcher to source code -- watching the repo root
    # (the default) also watches THERMAL_DATA_ROOT's default location
    # (./datasets), so every file an ingest writes triggers a server
    # restart mid-background-task.
    uvicorn.run(
        "thermal_dataset_manager.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
    )


if __name__ == "__main__":
    main()
