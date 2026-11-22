"""Unit tests for Module 9 payroll calculations."""

import pytest

from module9 import payroll


@pytest.mark.parametrize(
    ("hours", "pay_rate", "department"),
    [(-1, 15.0, "Sales"), (40, -1.0, "IT"), (40, float("inf"), "Sales")],
)
def test_validate_rejects_invalid_numbers(hours, pay_rate, department):
    """Specification: payroll validation rejects invalid numeric values."""
    with pytest.raises((TypeError, ValueError)):
        payroll.validate_payroll_inputs("1001", hours, pay_rate, department)


def test_payroll_calculates_overtime_bonus_and_taxes():
    """Specification: payroll calculation preserves overtime and tax totals."""
    result = payroll.calculate_payroll("1001", 45, 20.0, "Sales")

    assert result["regular_pay"] == 800.0
    assert result["overtime_pay"] == 150.0
    assert result["gross_pay"] == 950.0
    assert result["bonus"] == 95.0
    assert result["taxes"] == 261.25
    assert result["net_pay"] == 783.75
    assert result["department_code"] == "S"


def test_payroll_rejects_unknown_department():
    """Specification: unsupported payroll departments are rejected."""
    with pytest.raises(ValueError, match="unsupported department"):
        payroll.calculate_payroll("1002", 40, 15.0, "Unknown")


def test_payroll_rounding_preserves_record_invariants():
    """Specification: rounding preserves gross and payable relations."""
    result = payroll.calculate_payroll("1003", 40.25, 19.99, "IT")

    assert (
        result["gross_pay"] == result["regular_pay"] + result["overtime_pay"]
    )
    assert result["taxable_income"] == result["gross_pay"] + result["bonus"]
    assert result["net_pay"] == result["taxable_income"] - result["taxes"]
