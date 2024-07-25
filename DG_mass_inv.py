from firedrake import *

class DGMassInv(PCBase):

    def initialize(self, pc):

        self.Re = Constant(1)
        self.gamma = Constant(10000)

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        self.massinv = massinv.petscmat
    
    def update(self, pc):
        pass
    
    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        scaling = 1/float(self.Re) + float(self.gamma)
        y.scale(-scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")
