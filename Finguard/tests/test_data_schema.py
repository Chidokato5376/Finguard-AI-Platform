"""Unit tests cho src/data/schema.py."""

from datetime import datetime

import pytest

from src.data.schema import Channel, DataSource, TransactionRecord


def test_valid_record_ieee_cis() -> None:
    record = TransactionRecord(
        account_id="acc_1",
        counterparty_id="acc_2",
        timestamp=datetime(2026, 1, 1),
        amount=100.0,
        channel=Channel.DOMESTIC,
        device_id="dev_1",
        ip_country="VN",
        label=False,
        source=DataSource.IEEE_CIS,
    )
    assert record.amount == 100.0


def test_negative_amount_raises() -> None:
    with pytest.raises(ValueError):
        TransactionRecord(
            account_id="acc_1",
            counterparty_id="acc_2",
            timestamp=datetime(2026, 1, 1),
            amount=-5.0,
            channel=Channel.DOMESTIC,
            device_id=None,
            ip_country=None,
            label=None,
            source=DataSource.PAYSIM,
        )


def test_paysim_device_id_without_synthetic_flag_raises() -> None:
    """PaySim gốc không có device_id — nếu điền giá trị mà không khai báo
    is_synthetic_field, phải raise (xem data/README.md cảnh báo nguồn gốc).
    """
    with pytest.raises(ValueError):
        TransactionRecord(
            account_id="acc_1",
            counterparty_id="acc_2",
            timestamp=datetime(2026, 1, 1),
            amount=100.0,
            channel=Channel.DOMESTIC,
            device_id="dev_1",  # không hợp lệ nếu không khai báo synthetic
            ip_country=None,
            label=None,
            source=DataSource.PAYSIM,
            is_synthetic_field=(),
        )


def test_paysim_device_id_with_synthetic_flag_ok() -> None:
    record = TransactionRecord(
        account_id="acc_1",
        counterparty_id="acc_2",
        timestamp=datetime(2026, 1, 1),
        amount=100.0,
        channel=Channel.DOMESTIC,
        device_id="dev_1",
        ip_country="VN",
        label=None,
        source=DataSource.PAYSIM,
        is_synthetic_field=("device_id", "ip_country"),
    )
    assert record.device_id == "dev_1"
