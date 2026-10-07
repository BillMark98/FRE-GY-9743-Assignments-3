"""Fixed-accrual and overnight-index cashflow products."""

from math import isfinite
from typing import Optional

import QuantLib as ql

from fixedincomelib.date.basics import Date, TermOrDate
from fixedincomelib.date.utilities import accrued, add_period
from fixedincomelib.market.basics import (
    AccrualBasis, BusinessDayConvention, Currency, HolidayConvention,
)
from fixedincomelib.market.data_conventions import CompoundingMethod
from fixedincomelib.market.registries import IndexRegistry
from fixedincomelib.product.product_interfaces import Product, ProductVisitor
from fixedincomelib.product.utilities import LongOrShort

__all__ = [
    "ProductCashflow", "ProductFixedAccruedCashflow", "ProductOvernightIndexCashflow",
]


def _valid_date(value, name: str) -> Date:
    result = Date(value)
    if not result.is_valid():
        raise ValueError(f"{name} must not be a null date")
    return result


def _check_accrual_dates(effective_date: Date, termination_date: Date) -> None:
    if termination_date < effective_date:
        raise ValueError("termination_date must not precede effective_date")


class ProductCashflow(Product):
    """Abstract cashflow with currency, signed notional and payment date."""

    def __init__(self, currency: Currency, notional: float, payment_date: Date) -> None:
        super().__init__()
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency")
        if not currency.is_valid:
            raise ValueError("Unsupported currency")
        if not isfinite(notional):
            raise ValueError("notional must be finite")
        self.currency_ = currency
        self.notional_ = notional
        self.long_or_short_ = LongOrShort.LONG if notional >= 0 else LongOrShort.SHORT
        self.payment_date_ = _valid_date(payment_date, "payment_date")
        self.last_date_ = self.payment_date_

    @property
    def payment_date(self) -> Date:
        return self.payment_date_


class ProductFixedAccruedCashflow(ProductCashflow):
    """Fixed-accrual cashflow; accrued is a year fraction, not a cash amount."""

    _version = 1
    _product_type = "PRODUCT_FIXED_ACCRUED"

    def __init__(
        self,
        effective_date: Date,
        termination_date: Date,
        currency: Currency,
        notional: float,
        accrual_basis: AccrualBasis,
        payment_date: Optional[Date] = None,
        business_day_convention: Optional[BusinessDayConvention] = None,
        holiday_convention: Optional[HolidayConvention] = None,
    ) -> None:
        effective_date = _valid_date(effective_date, "effective_date")
        termination_date = _valid_date(termination_date, "termination_date")
        _check_accrual_dates(effective_date, termination_date)
        # Defaults.  A cashflow with no explicit payment date pays at its
        # accrual end, and the US government-bond market standard is Following
        # on the USGS calendar.  These two kwargs exist so the same class can
        # describe a non-US cashflow without the factory's defaults.
        if payment_date is None:
            payment_date = termination_date
        if business_day_convention is None:
            business_day_convention = BusinessDayConvention("F")
        if holiday_convention is None:
            holiday_convention = HolidayConvention("USGS")

        # ProductCashflow stores currency, signed notional (and therefore
        # long/short) and the payment date; it also sets last_date = payment_date.
        super().__init__(currency, notional, payment_date)

        self.effective_date_ = effective_date
        self.termination_date_ = termination_date
        self.accrual_basis_ = accrual_basis
        self.business_day_convention_ = business_day_convention
        self.holiday_convention_ = holiday_convention
        # A cashflow begins accruing on its effective date (first_date) and its
        # last event is the payment (already set by the base).  `accrued` adjusts
        # the accrual end by the convention, then measures under the basis; the
        # result is a year fraction, NOT a cash amount.
        self.first_date_ = effective_date
        self.accrued_ = accrued(
            effective_date, termination_date,
            accrual_basis, business_day_convention, holiday_convention)

    @property
    def effective_date(self) -> Date:
        return self.effective_date_

    @property
    def termination_date(self) -> Date:
        return self.termination_date_

    @property
    def accrual_basis(self) -> AccrualBasis:
        return self.accrual_basis_

    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return self.business_day_convention_

    @property
    def holiday_convention(self) -> HolidayConvention:
        return self.holiday_convention_

    @property
    def accrued(self) -> float:
        return self.accrued_

    def accept(self, visitor: ProductVisitor):
        # Visitor double dispatch: the product knows its own concrete type, so
        # it calls the matching visit() overload; singledispatchmethod routes to
        # the ProductFixedAccruedCashflow handler.
        return visitor.visit(self)

    def serialize(self) -> dict:
        return {
            "VERSION": self._version,
            "TYPE": self._product_type,
            "EFFECTIVE_DATE": self.effective_date.ISO(),
            "TERMINATION_DATE": self.termination_date.ISO(),
            "PAYMENT_DATE": self.payment_date.ISO(),
            "ACCRUAL_BASIS": self.accrual_basis.value_str,
            "BUSINESS_DAY_CONVENTION": self.business_day_convention.value_str,
            "HOLIDAY_CONVENTION": self.holiday_convention.value_str,
            "NOTIONAL": self.notional,
            "CURRENCY": self.currency.value_str,
        }

    @classmethod
    def deserialize(cls, input_dict: dict) -> "ProductFixedAccruedCashflow":
        return cls(
            Date(input_dict["EFFECTIVE_DATE"]), Date(input_dict["TERMINATION_DATE"]),
            Currency(input_dict["CURRENCY"]), float(input_dict["NOTIONAL"]),
            AccrualBasis(input_dict["ACCRUAL_BASIS"]), Date(input_dict["PAYMENT_DATE"]),
            BusinessDayConvention(input_dict["BUSINESS_DAY_CONVENTION"]),
            HolidayConvention(input_dict["HOLIDAY_CONVENTION"]),
        )


class ProductOvernightIndexCashflow(ProductCashflow):
    """Overnight-index cashflow terms, with spread as an annual decimal rate."""

    _version = 1
    _product_type = "PRODUCT_OVERNIGHT_INDEX_CASHFLOW"

    def __init__(
        self,
        effective_date: Date,
        term_or_termination_date: TermOrDate,
        on_index: str,
        compounding_method: CompoundingMethod,
        spread: float,
        notional: float,
        payment_date: Optional[Date] = None,
    ) -> None:
        effective_date = _valid_date(effective_date, "effective_date")
        if not isinstance(term_or_termination_date, TermOrDate):
            raise TypeError("term_or_termination_date must be a TermOrDate")
        if not isinstance(compounding_method, CompoundingMethod):
            raise TypeError("compounding_method must be a CompoundingMethod")
        if not isfinite(spread):
            raise ValueError("spread must be finite")
        # Resolve the index through the registry; it fails loudly on an unknown key.
        self.on_index_ = IndexRegistry().get(on_index)
        self.on_index_str_ = on_index

        # The termination date is either given directly or implied by a tenor.
        # For a tenor, roll forward on the INDEX's own calendar and convention --
        # an overnight index knows its fixing calendar, which is the only
        # calendar its tenor can be defined on.  A "3M" SOFR cashflow therefore
        # ends three months later, adjusted by the SOFR fixing calendar.
        if term_or_termination_date.is_term():
            termination_date = add_period(
                effective_date,
                term_or_termination_date.get_term(),
                self.on_index_.businessDayConvention(),
                self.on_index_.fixingCalendar(),
            )
        else:
            termination_date = term_or_termination_date.get_date()

        termination_date = _valid_date(termination_date, "termination_date")
        _check_accrual_dates(effective_date, termination_date)

        # Payment defaults to the accrual end; unlike the fixed cashflow there
        # is no convention override here -- the index fixes both calendar and
        # currency.
        if payment_date is None:
            payment_date = termination_date

        # Currency is carried by the index (SOFR/FF -> USD, SONIA -> GBP), not
        # supplied by the caller.
        super().__init__(Currency(self.on_index_.currency().code()), notional, payment_date)

        self.effective_date_ = effective_date
        self.termination_date_ = termination_date
        self.compounding_method_ = compounding_method
        self.spread_ = spread
        self.first_date_ = effective_date

    @property
    def on_index(self) -> ql.OvernightIndex:
        return self.on_index_

    @property
    def compounding_method(self) -> CompoundingMethod:
        return self.compounding_method_

    @property
    def effective_date(self) -> Date:
        return self.effective_date_

    @property
    def termination_date(self) -> Date:
        return self.termination_date_

    @property
    def spread(self) -> float:
        return self.spread_

    def accept(self, visitor: ProductVisitor):
        # Same double-dispatch as the fixed cashflow; the runtime type selects
        # the ProductOvernightIndexCashflow display handler.
        return visitor.visit(self)

    def serialize(self) -> dict:
        return {
            "VERSION": self._version,
            "TYPE": self._product_type,
            "EFFECTIVE_DATE": self.effective_date.ISO(),
            "TERMINATION_DATE": self.termination_date.ISO(),
            "PAYMENT_DATE": self.payment_date.ISO(),
            "ON_INDEX": self.on_index_str_,
            "SPREAD": self.spread,
            "COMPOUNDING_METHOD": self.compounding_method.to_string().upper(),
            "NOTIONAL": self.notional,
        }

    @classmethod
    def deserialize(cls, input_dict: dict) -> "ProductOvernightIndexCashflow":
        return cls(
            Date(input_dict["EFFECTIVE_DATE"]), TermOrDate(input_dict["TERMINATION_DATE"]),
            input_dict["ON_INDEX"], CompoundingMethod.from_string(input_dict["COMPOUNDING_METHOD"]),
            float(input_dict["SPREAD"]), float(input_dict["NOTIONAL"]),
            Date(input_dict["PAYMENT_DATE"]),
        )
