import ibis


def type_filter(expr: ibis.Expr, cls: type) -> ibis.Expr:
    """Filter an Ibis expression by the name of the provided class.

    This ensures that only rows of the specified type are processed,
    which is important for backend compatibility and data integrity.

    Args:
        expr: The Ibis expression (table) to filter.
        cls: The class whose __name__ will be used as the filter value.

    Returns:
        A filtered Ibis expression.
    """
    return expr.filter(expr.type == cls.__name__)
