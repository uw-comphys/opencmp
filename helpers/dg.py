from ngsolve import CoefficientFunction, Grad
from ngsolve.comp import ProxyFunction


def jump(n: CoefficientFunction, q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the jump of a field.

    Args:
        n: The unit normal for every facet of the mesh.
        q: The  field.

    Returns:
        ~: The jump of q at every facet of the mesh.
    """

    if q.dim > 1:
        # q is a vector.
        return q - q.Other()
    else:
        # q is a scalar.
        return n * (q - q.Other())


def grad_jump(n: CoefficientFunction, q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the jump of the gradient of a field.

    Args:
        n: The unit normal for every facet of the mesh.
        q: The field.

    Returns:
        ~: The jump of the gradient of q at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        grad_q = Grad(q) - Grad(q.Other())
    else:
        grad_q = Grad(q) - Grad(q).Other()

    if q.dim > 1:
        # q is a vector.
        return grad_q
    else:
        # q is a scalar.
        return n * grad_q


def avg(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of a scalar field.

    Args:
        q (CoefficientFunction): The scalar field.

    Returns:
        ~ (CoefficientFunction): The average of q at every facet of the mesh.
    """

    return 0.5 * (q + q.Other())


def grad_avg(q: CoefficientFunction) -> CoefficientFunction:
    """
    Returns the average of the gradient of a field.

    Args:
        q: The field.

    Returns:
        ~: The average of the gradient of q at every facet of the mesh.
    """

    # Grad must be called differently if q is a trial or testfunction instead of a coefficientfunction/gridfunction.
    if isinstance(q, ProxyFunction):
        return 0.5 * (Grad(q) + Grad(q.Other()))
    else:
        return 0.5 * (Grad(q) + Grad(q).Other())