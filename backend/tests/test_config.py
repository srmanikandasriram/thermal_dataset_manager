from thermal_dataset_manager.config import Settings


def test_catalog_db_defaults_inside_data_root_when_unset():
    """Regression test: catalog_db previously had a fixed CWD-relative
    default independent of data_root, so setting THERMAL_DATA_ROOT without
    also setting THERMAL_CATALOG_DB caused the catalog file to silently
    move whenever the server was started from a different directory --
    data written to the right place, but not indexed there.
    """
    settings = Settings(data_root="/some/custom/root")
    assert settings.catalog_db == "sqlite:////some/custom/root/catalog.sqlite"


def test_catalog_db_explicit_value_is_preserved():
    settings = Settings(data_root="/some/custom/root", catalog_db="sqlite:///elsewhere.sqlite")
    assert settings.catalog_db == "sqlite:///elsewhere.sqlite"


def test_catalog_db_default_data_root_uses_relative_path():
    settings = Settings()
    assert settings.catalog_db == "sqlite:///datasets/catalog.sqlite"
