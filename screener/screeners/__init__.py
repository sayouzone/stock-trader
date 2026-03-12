from .position import PositionScreener
from .growth import GrowthScreener
from .value import ValueScreener
from .swing import SwingScreener
from .momentum import MomentumScreener
from .dividend import DividendScreener

ALL_SCREENERS = {
    "position": ("포지션 트레이딩", PositionScreener),
    "growth": ("성장주 트레이딩", GrowthScreener),
    "value": ("가치 트레이딩", ValueScreener),
    "swing": ("스윙 트레이딩", SwingScreener),
    "momentum": ("모멘텀 트레이딩", MomentumScreener),
    "dividend": ("배당 투자", DividendScreener),
}

__all__ = [
    "PositionScreener", "GrowthScreener", "ValueScreener",
    "SwingScreener", "MomentumScreener", "DividendScreener",
    "ALL_SCREENERS",
]
