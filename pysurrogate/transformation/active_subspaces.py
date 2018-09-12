from active_subspaces.domains import BoundedActiveVariableDomain, ActiveVariableMap
from active_subspaces.subspaces import Subspaces

from pysurrogate.transformation.transformation import Transformation


class ActiveSubspaceTransformation(Transformation):

    def __init__(self, X, F) -> None:
        super().__init__()
        self.var_map = None

        subspace = Subspaces()
        subspace.compute(X, F[:, None], sstype='OLS', ptype='EVG')

        domain = BoundedActiveVariableDomain(subspace)
        domain.compute_boundary()

        self.var_map = ActiveVariableMap(domain)

    def forward(self, X):
        _X, _ = self.var_map.forward(X)
        return _X

    def inverse(self, Y):
        return self.var_map.inverse(Y)
