
from __future__ import annotations


class BrokerError(Exception): ...


class AuthenticationError(BrokerError): ...


class OrderRejectedError(BrokerError): ...


class InsufficientFundsError(BrokerError): ...


class SymbolNotFoundError(BrokerError): ...
