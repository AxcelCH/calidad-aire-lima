"""Tests de las reglas de negocio puras (umbrales ECA y limpieza basica)."""
import pandas as pd
import pytest

from src.data.clean import coerce_pollutants, normalize_columns
from src.data.eca import (
    ECA_PM10_24H,
    ECA_PM25_24H,
    exceeds_pm10,
    exceeds_pm25,
    label_exceedance,
)


class TestUmbralesECA:
    def test_pm25_debajo_del_umbral_no_excede(self):
        assert exceeds_pm25(49.9) is False

    def test_pm25_exactamente_en_el_umbral_no_excede(self):
        # El D.S. 003-2017-MINAM define excedencia como estrictamente mayor
        assert exceeds_pm25(ECA_PM25_24H) is False

    def test_pm25_sobre_el_umbral_excede(self):
        assert exceeds_pm25(50.1) is True

    def test_pm10_umbral(self):
        assert exceeds_pm10(ECA_PM10_24H) is False
        assert exceeds_pm10(100.1) is True


class TestEtiquetado:
    def test_label_exceedance_crea_columnas_binarias(self):
        df = pd.DataFrame({"pm25": [10.0, 60.0], "pm10": [120.0, 80.0]})
        out = label_exceedance(df)
        assert out["excede_pm25"].tolist() == [0, 1]
        assert out["excede_pm10"].tolist() == [1, 0]

    def test_label_exceedance_no_modifica_el_original(self):
        df = pd.DataFrame({"pm25": [60.0]})
        label_exceedance(df)
        assert "excede_pm25" not in df.columns


class TestLimpieza:
    def test_normalize_columns_mapea_variantes(self):
        df = pd.DataFrame(columns=["ESTACION", "PM2.5", "PM10", "NO2", "FECHA"])
        out = normalize_columns(df)
        for col in ["estacion", "pm25", "pm10", "no2", "fecha"]:
            assert col in out.columns

    def test_coerce_pollutants_convierte_y_anula_negativos(self):
        df = pd.DataFrame({"pm25": ["12.5", "", "-3", "abc"]})
        out = coerce_pollutants(df)
        assert out["pm25"].iloc[0] == pytest.approx(12.5)
        assert out["pm25"].isna().sum() == 3
