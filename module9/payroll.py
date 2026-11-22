"""Calculate payroll information for an employee.

Date: 11/22/2026
Author: Kyle Gortych
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TypedDict


class PayrollRecord(TypedDict):
    employee_id: str
    employee_name: str
    regular_pay: float
    overtime_pay: float
    gross_pay: float
    bonus: float
    taxable_income: float
    taxes: float
    net_pay: float
    department: str
    department_code: str
    status: str


DEPARTMENT_CODES = {"Sales": "S", "IT": "I", "Marketing": "M"}
DEPARTMENT_BONUS_RATES = {"Sales": 0.10, "IT": 0.05}
TAX_RATE = 0.25


def validate_payroll_inputs(
    employee_id: object,
    hours_worked: object,
    pay_rate: object,
    department: object,
) -> None:
    """Validate payroll inputs.

    Precondition: employee ID and department are non-empty strings; hours and
    pay rate are real, finite, non-negative numbers.
    Postcondition: no exception is raised for valid inputs.
    Effect: no state is mutated.
    """
    if not isinstance(employee_id, str) or not employee_id.strip():
        raise ValueError("employee_id must be a non-empty string")
    if not isinstance(department, str) or not department.strip():
        raise ValueError("department must be a non-empty string")
    if department not in DEPARTMENT_CODES:
        raise ValueError(f"unsupported department: {department}")
    if not isinstance(hours_worked, Real) or isinstance(hours_worked, bool):
        raise TypeError("hours_worked must be a number")
    if not isinstance(pay_rate, Real) or isinstance(pay_rate, bool):
        raise TypeError("pay_rate must be a number")
    if not math.isfinite(float(hours_worked)):
        raise ValueError("hours_worked must be finite")
    if float(hours_worked) < 0:
        raise ValueError("hours_worked must be non-negative")
    if not math.isfinite(float(pay_rate)):
        raise ValueError("pay_rate must be finite")
    if float(pay_rate) < 0:
        raise ValueError("pay_rate must be non-negative")


def calculate_payroll(
    employee_id: str,
    hours_worked: Real,
    pay_rate: Real,
    department: str,
) -> PayrollRecord:
    """Calculate a validated employee payroll record.

    Precondition: employee_id and department are recognized non-empty strings;
    hours_worked and pay_rate are finite, non-negative real numbers.
    Postcondition: the record contains consistent gross, tax, and net pay
    values, with overtime limited to hours above 40.
    Invariant: net pay equals taxable income minus taxes; regular hours do not
    exceed 40.
    Effect: no file or external state is accessed or mutated.
    """
    validate_payroll_inputs(
        employee_id,
        hours_worked,
        pay_rate,
        department,
    )

    hours = float(hours_worked)
    rate = float(pay_rate)
    regular_hours = min(hours, 40.0)
    overtime_hours = max(hours - 40.0, 0.0)
    regular_pay = round(regular_hours * rate, 2)
    overtime_pay = round(overtime_hours * rate * 1.5, 2)
    gross_pay = round(regular_pay + overtime_pay, 2)
    bonus = round(
        gross_pay * DEPARTMENT_BONUS_RATES.get(department, 0.0),
        2,
    )
    taxable_income = round(gross_pay + bonus, 2)
    taxes = round(taxable_income * TAX_RATE, 2)
    net_pay = round(taxable_income - taxes, 2)

    return {
        "employee_id": employee_id.strip(),
        "employee_name": f"{employee_id.strip()} Employee",
        "regular_pay": regular_pay,
        "overtime_pay": overtime_pay,
        "gross_pay": gross_pay,
        "bonus": bonus,
        "taxable_income": taxable_income,
        "taxes": taxes,
        "net_pay": net_pay,
        "department": department,
        "department_code": DEPARTMENT_CODES[department],
        "status": "Due" if net_pay > 0 else "No payment due",
    }


if __name__ == "__main__":
    print(calculate_payroll("1001", 40, 15.0, "Sales"))
