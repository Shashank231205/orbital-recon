"""Helpers shared by route modules."""

from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from orbital_recon.core.exceptions import NotFoundError
from orbital_recon.db.models import Base

ModelT = TypeVar("ModelT", bound=Base)


async def require(
    session: AsyncSession,
    model: type[ModelT],
    identifier: int,
    label: str,
) -> ModelT:
    """Fetch a row by primary key or raise a not-found error.

    Args:
        session: Active database session.
        model: ORM class to load.
        identifier: Primary key value.
        label: Human-readable name used in the error message.

    Raises:
        NotFoundError: If no row has that primary key.
    """
    instance = await session.get(model, identifier)
    if instance is None:
        raise NotFoundError(f"{label} {identifier} does not exist")
    return instance
