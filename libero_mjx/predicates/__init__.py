"""Success predicates for LIBERO MJX port.

Spatial suite: distance_to, on_top_of, inside, in_contact
Other suites: on, in_region, is_open, is_closed, is_turned_on
"""

from libero_mjx.predicates.spatial import (
    PredicateFn,
    on_top_of, inside, in_contact, distance_to,
    on, in_region, is_open, is_closed, is_turned_on,
)

__all__ = [
    "PredicateFn",
    "on_top_of", "inside", "in_contact", "distance_to",
    "on", "in_region", "is_open", "is_closed", "is_turned_on",
]